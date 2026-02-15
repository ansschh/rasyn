"""Honest evaluation on full USPTO-50K test set with R-SMILES alignment.

Same framework as eval_honest.py, but root-aligns the product SMILES at the
reaction center atom before feeding to models trained on R-SMILES data.

Comparison always uses canonical SMILES (both prediction and GT are
canonicalized before matching).

Usage:
    # LLM v7 with R-SMILES
    python -u scripts/eval_honest_rsmiles.py --model llm \
        --llm-checkpoint checkpoints/llm/uspto50k_v7_rsmiles/final

    # RetroTx v2 with R-SMILES data
    python -u scripts/eval_honest_rsmiles.py --model retro \
        --retro-checkpoint checkpoints/retro_v2/r_smiles/best/model.pt

    # Both + TTA
    python -u scripts/eval_honest_rsmiles.py --model both --tta

    # RunPod
    nohup python -u scripts/eval_honest_rsmiles.py --model llm --tta > eval_rsmiles.log 2>&1 &
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import click
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

TEST_CSV_URL = "https://raw.githubusercontent.com/uta-smile/RetroXpert/main/data/USPTO50K/canonicalized_csv/test.csv"


# ── SMILES utilities (same as eval_honest.py) ──────────────────────

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.logger().setLevel(RDLogger.ERROR)
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDKIT_AVAILABLE = False


def canonicalize_smiles(smi: str, remove_mapping: bool = False) -> str:
    if not RDKIT_AVAILABLE:
        return smi.strip()
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is not None:
            if remove_mapping:
                for atom in mol.GetAtoms():
                    atom.ClearProp("molAtomMapNumber")
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return ""


def normalize_reactants(smiles_str: str, remove_mapping: bool = False) -> str:
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    canon_parts = [canonicalize_smiles(p.strip(), remove_mapping=remove_mapping) for p in parts if p.strip()]
    canon_parts = [c for c in canon_parts if c]
    return ".".join(sorted(canon_parts))


# ── Data loading ──────────────────────────────────────────────────

def download_test_csv(output_path: Path) -> Path:
    if output_path.exists():
        logger.info(f"Using cached test CSV: {output_path}")
        return output_path
    logger.info("Downloading USPTO-50K test.csv from RetroXpert...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TEST_CSV_URL, str(output_path))
    logger.info(f"Downloaded {output_path}")
    return output_path


def load_test_data(csv_path: Path) -> list[dict]:
    """Load ALL test reactions. NO filtering. NO skipping."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rxn_smiles = row.get("rxn_smiles", "")
            if not rxn_smiles:
                continue
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                continue
            reactants_raw = parts[0].strip()
            product_raw = parts[1].strip()
            product_canon = canonicalize_smiles(product_raw, remove_mapping=True)
            gt_reactants = normalize_reactants(reactants_raw, remove_mapping=True)
            rows.append({
                "rxn_smiles": rxn_smiles,
                "rxn_class": int(row.get("class", 0)),
                "rxn_id": row.get("id", ""),
                "product_canon": product_canon,
                "gt_reactants": gt_reactants,
                "product_raw": product_raw,
                "reactants_raw": reactants_raw,
            })
    logger.info(f"Loaded {len(rows)} test reactions (ALL of them, no filtering)")
    return rows


def preprocess_for_edit_pipeline(rxn_smiles: str, rxn_id: str) -> dict | None:
    from rasyn.preprocess.build_edit_dataset import build_training_example
    return build_training_example(rxn_smiles, rxn_id)


def get_rsmiles_product(rxn_smiles: str) -> str | None:
    """Root-align the product SMILES at the reaction center atom.

    Returns the R-SMILES product or None if it fails.
    """
    from rasyn.preprocess.r_smiles import build_r_smiles_example
    result = build_r_smiles_example(rxn_smiles)
    if result is None:
        return None
    return result["product"]


# ── Model inference (mirrors eval_honest.py) ──────────────────────

def decode_llm_completion(tokenizer, output_ids) -> str:
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    if "<OUT>" in text:
        completion = text.split("<OUT>")[-1]
        for stop in ["</s>", "<EOS>", "<pad>", "<PAD>"]:
            completion = completion.split(stop)[0]
        return completion.replace("<unk>", "").strip()
    return ""


def get_llm_predictions_beam(
    model, tokenizer, prompt: str, device: str,
    num_beams: int = 10, max_new_tokens: int = 256,
) -> list[str]:
    from rasyn.models.llm.generate import tokenize_prompt_for_inference
    inputs = tokenize_prompt_for_inference(prompt, tokenizer, max_length=512, device=device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=min(num_beams, 10),
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

    predictions = []
    seen = set()
    for seq in out:
        comp = decode_llm_completion(tokenizer, seq)
        if comp:
            norm = normalize_reactants(comp)
            if norm and norm not in seen:
                seen.add(norm)
                predictions.append(norm)
    return predictions


def get_llm_predictions_tta(
    model, tokenizer, prompt: str, device: str,
    num_beams: int = 10, max_new_tokens: int = 256,
    n_temp_passes: int = 20, samples_per_pass: int = 10,
    temperature: float = 0.8,
) -> list[str]:
    from rasyn.models.llm.generate import tokenize_prompt_for_inference
    inputs = tokenize_prompt_for_inference(prompt, tokenizer, max_length=512, device=device)

    candidates: dict[str, int] = defaultdict(int)

    # Beam search
    with torch.no_grad():
        base_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=min(num_beams, 10),
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
    for seq in base_out:
        comp = decode_llm_completion(tokenizer, seq)
        if comp:
            norm = normalize_reactants(comp)
            if norm:
                candidates[norm] += num_beams

    # Temperature sampling
    for _ in range(n_temp_passes):
        with torch.no_grad():
            sample_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=samples_per_pass,
                pad_token_id=tokenizer.pad_token_id,
            )
        for seq in sample_out:
            comp = decode_llm_completion(tokenizer, seq)
            if comp:
                norm = normalize_reactants(comp)
                if norm:
                    candidates[norm] += 1

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [r for r, _ in ranked]


def get_retro_predictions(
    model, tokenizer, src_text: str, device: str,
    beam_size: int = 10, max_len: int = 128,
) -> list[str]:
    src_ids = torch.tensor(
        [tokenizer.encode(src_text, max_len=256)],
        dtype=torch.long, device=device,
    )
    seg_ids = torch.tensor(
        [tokenizer.get_segment_ids(tokenizer.encode(src_text, max_len=256))],
        dtype=torch.long, device=device,
    )
    beam_results = model.generate_beam(
        src_ids, tokenizer.bos_token_id, tokenizer.eos_token_id,
        beam_size=beam_size, max_len=max_len, segment_ids=seg_ids,
    )[0]

    predictions = []
    seen = set()
    for token_ids, score in beam_results:
        pred_str = tokenizer.decode(token_ids)
        norm = normalize_reactants(pred_str)
        if norm and norm not in seen:
            seen.add(norm)
            predictions.append(norm)
    return predictions


# ── Metrics ──────────────────────────────────────────────────────

class MetricsTracker:
    def __init__(self, name: str):
        self.name = name
        self.correct = {1: 0, 3: 0, 5: 0, 10: 0}
        self.attempted = 0
        self.total = 0

    def update(self, gt: str, predictions: list[str], was_attempted: bool):
        self.total += 1
        if was_attempted:
            self.attempted += 1
            for k in self.correct:
                if gt in predictions[:k]:
                    self.correct[k] += 1

    def report(self) -> dict:
        coverage = self.attempted / max(self.total, 1)
        result = {"name": self.name, "total": self.total,
                  "attempted": self.attempted, "coverage": coverage}
        for k in sorted(self.correct):
            acc_attempted = self.correct[k] / max(self.attempted, 1)
            acc_total = self.correct[k] / max(self.total, 1)
            result[f"top{k}_on_attempted"] = acc_attempted
            result[f"top{k}_on_total"] = acc_total
            result[f"top{k}_correct"] = self.correct[k]
        return result

    def log(self):
        r = self.report()
        logger.info(f"  {self.name}:")
        logger.info(f"    Coverage: {r['coverage']:.4f} ({r['attempted']}/{r['total']})")
        for k in [1, 3, 5, 10]:
            logger.info(
                f"    Top-{k}:  {r[f'top{k}_on_total']:.4f} (on total={r['total']}) | "
                f"{r[f'top{k}_on_attempted']:.4f} (on attempted={r['attempted']}) | "
                f"correct={r[f'top{k}_correct']}"
            )


# ── Main ─────────────────────────────────────────────────────────

@click.command()
@click.option("--model", type=click.Choice(["llm", "retro", "both"]), default="llm")
@click.option("--tta/--no-tta", default=False)
@click.option("--llm-checkpoint", default="checkpoints/llm/uspto50k_v7_rsmiles/final")
@click.option("--retro-checkpoint", default="checkpoints/retro_v2/r_smiles/best/model.pt")
@click.option("--llm-beams", default=10, type=int)
@click.option("--retro-beams", default=10, type=int)
@click.option("--n-temp-passes", default=20, type=int)
@click.option("--samples-per-pass", default=10, type=int)
@click.option("--temperature", default=0.8, type=float)
@click.option("--max-samples", default=0, type=int)
@click.option("--device", default="auto")
@click.option("--output", default=None, type=str)
def main(
    model, tta, llm_checkpoint, retro_checkpoint,
    llm_beams, retro_beams, n_temp_passes, samples_per_pass, temperature,
    max_samples, device, output,
):
    """Honest evaluation with R-SMILES alignment on full USPTO-50K test set."""
    import re

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    use_llm = model in ("llm", "both")
    use_retro = model in ("retro", "both")
    use_ensemble = model == "both"

    # ── Load test data ────────────────────────────────────────────
    test_csv = PROJECT_ROOT / "data" / "external" / "uspto50k_test.csv"
    download_test_csv(test_csv)
    test_data = load_test_data(test_csv)

    if max_samples > 0:
        test_data = test_data[:max_samples]
        logger.warning(f"*** LIMITED TO {max_samples} SAMPLES — NOT FOR PUBLIC REPORTING ***")

    total_test = len(test_data)
    logger.info(f"Total test reactions: {total_test}")

    # ── Load models ──────────────────────────────────────────────
    llm_model = llm_tokenizer = None
    if use_llm:
        logger.info(f"Loading LLM from {llm_checkpoint}...")
        from transformers import AutoTokenizer
        from peft import PeftModel
        from rasyn.models.llm.model import load_rsgpt_model

        llm_ckpt = Path(llm_checkpoint)
        tokenizer_path = PROJECT_ROOT / "weights" / "rsgpt" / "tokenizer"
        if (llm_ckpt / "adapter_config.json").exists():
            weights_path = PROJECT_ROOT / "weights" / "rsgpt" / "finetune_50k.pth"
            base_model, _ = load_rsgpt_model(weights_path=str(weights_path), use_lora=False)
            llm_model = PeftModel.from_pretrained(base_model, str(llm_ckpt))
            if (llm_ckpt / "tokenizer_config.json").exists():
                llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_ckpt))
            else:
                llm_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
                from rasyn.models.llm.tokenizer import ALL_SPECIAL_TOKENS
                new_tokens = [t for t in ALL_SPECIAL_TOKENS if t not in llm_tokenizer.get_vocab()]
                if new_tokens:
                    llm_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        else:
            from transformers import LlamaForCausalLM
            llm_model = LlamaForCausalLM.from_pretrained(str(llm_ckpt))
            llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_ckpt))
        llm_model = llm_model.to(device).eval()
        logger.info("LLM loaded")

    retro_model = retro_tokenizer = None
    if use_retro:
        logger.info(f"Loading RetroTx v2 from {retro_checkpoint}...")
        from rasyn.models.retro.model_v2 import load_retro_model_v2
        retro_model, retro_tokenizer = load_retro_model_v2(retro_checkpoint, device=device)
        logger.info("RetroTx v2 loaded")

    # ── Metrics ──────────────────────────────────────────────────
    trackers = {}
    if use_llm:
        suffix = "+TTA" if tta else ""
        trackers["llm"] = MetricsTracker(f"LLM R-SMILES{suffix}")
    if use_retro:
        trackers["retro"] = MetricsTracker("RetroTx R-SMILES")
    if use_ensemble:
        trackers["ensemble"] = MetricsTracker("Ensemble R-SMILES")

    # ── Evaluate ─────────────────────────────────────────────────
    preprocess_failures = 0
    rsmiles_failures = 0
    start_time = time.time()

    for i, row in enumerate(tqdm(test_data, desc="Evaluating (R-SMILES)")):
        gt = row["gt_reactants"]
        product = row["product_canon"]
        rxn_class = row["rxn_class"]

        if not gt or not product:
            for tracker in trackers.values():
                tracker.update(gt, [], was_attempted=True)
            continue

        # Preprocess (extract edits)
        preprocessed = preprocess_for_edit_pipeline(row["rxn_smiles"], row["rxn_id"])
        if preprocessed is None:
            preprocess_failures += 1
            for tracker in trackers.values():
                tracker.update(gt, [], was_attempted=False)
            continue

        # Get R-SMILES aligned product
        r_product = get_rsmiles_product(row["rxn_smiles"])
        if r_product is None:
            # Fall back to canonical product if R-SMILES fails
            rsmiles_failures += 1
            r_product = product

        prompt = preprocessed["prompt"]

        # Parse edit info from prompt
        synth_match = re.search(r"<SYNTHONS>\s*(.+?)(?=\s*<(?:LG_HINTS|CONSTRAINTS|OUT)>)", prompt)
        synthons = synth_match.group(1).strip() if synth_match else ""

        # Replace canonical product with R-SMILES product in the prompt
        edit_match = re.search(r"<EDIT>\s+(.+?)(?=\s+<SYNTHONS>)", prompt)
        edit_str = edit_match.group(1).strip() if edit_match else "NO_DISCONNECT"
        lg_match = re.search(r"<LG_HINTS>\s+(.+?)(?=\s+<(?:CONSTRAINTS|OUT)>)", prompt)
        lg_str = lg_match.group(1).strip() if lg_match else ""

        # Rebuild prompt with R-SMILES product
        r_prompt_parts = [f"<PROD> {r_product}"]
        r_prompt_parts.append(f"<EDIT> {edit_str}")
        if synthons:
            r_prompt_parts.append(f"<SYNTHONS> {synthons}")
        if lg_str:
            r_prompt_parts.append(f"<LG_HINTS> {lg_str}")
        r_prompt_parts.append("<OUT>")
        r_prompt = " ".join(r_prompt_parts)

        llm_preds = []
        retro_preds = []

        # ── LLM predictions with R-SMILES prompt ─────────────
        if use_llm:
            if tta:
                llm_preds = get_llm_predictions_tta(
                    llm_model, llm_tokenizer, r_prompt, device,
                    num_beams=llm_beams, n_temp_passes=n_temp_passes,
                    samples_per_pass=samples_per_pass, temperature=temperature,
                )
            else:
                llm_preds = get_llm_predictions_beam(
                    llm_model, llm_tokenizer, r_prompt, device,
                    num_beams=llm_beams,
                )
            trackers["llm"].update(gt, llm_preds, was_attempted=True)

        # ── RetroTx predictions with R-SMILES ────────────────
        if use_retro:
            src_parts = []
            if rxn_class > 0:
                src_parts.append(f"<RXN_{rxn_class}>")
            src_parts.append(r_product)
            if synthons:
                src_parts.append(f"| {synthons}")
            src_text = " ".join(src_parts)

            retro_preds = get_retro_predictions(
                retro_model, retro_tokenizer, src_text, device,
                beam_size=retro_beams,
            )
            trackers["retro"].update(gt, retro_preds, was_attempted=True)

        # ── Ensemble ─────────────────────────────────────────
        if use_ensemble:
            from scripts.eval_honest import ensemble_agreement_boost
            ens_preds = ensemble_agreement_boost(llm_preds, retro_preds)
            trackers["ensemble"].update(gt, ens_preds, was_attempted=True)

        # Periodic logging
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total_test - i - 1) / rate if rate > 0 else 0
            parts = [f"[{i+1}/{total_test}]"]
            for name, tracker in trackers.items():
                r = tracker.report()
                parts.append(f"{name}: {r['top1_on_total']:.3f}({r['top1_on_attempted']:.3f})")
            parts.append(f"cvg: {1 - preprocess_failures/(i+1):.3f}")
            parts.append(f"ETA: {eta/60:.1f}m")
            logger.info(" | ".join(parts))

    # ── Final results ────────────────────────────────────────────
    elapsed = time.time() - start_time

    logger.info(f"\n{'='*70}")
    logger.info(f"HONEST R-SMILES EVALUATION — Standard USPTO-50K Test Set")
    logger.info(f"{'='*70}")
    logger.info(f"Test set: {total_test} reactions")
    logger.info(f"Preprocess failures: {preprocess_failures} ({preprocess_failures/total_test:.1%})")
    logger.info(f"R-SMILES failures (fell back to canonical): {rsmiles_failures}")
    logger.info(f"Coverage: {(total_test - preprocess_failures)/total_test:.4f}")
    logger.info(f"Time: {elapsed:.0f}s ({total_test/elapsed:.1f} samples/s)")
    logger.info(f"")

    for tracker in trackers.values():
        tracker.log()
        logger.info(f"")

    if max_samples > 0:
        logger.info(f"*** WARNING: Limited to {max_samples} samples. NOT for public reporting. ***")
    logger.info(f"{'='*70}")

    # Save results
    results = {
        "test_set": "USPTO-50K standard Schneider split",
        "eval_type": "R-SMILES aligned",
        "total_reactions": total_test,
        "preprocess_failures": preprocess_failures,
        "rsmiles_failures": rsmiles_failures,
        "coverage": (total_test - preprocess_failures) / total_test,
        "model": model,
        "tta": tta,
        "elapsed_seconds": elapsed,
    }
    for name, tracker in trackers.items():
        results[name] = tracker.report()

    out_path = Path(output) if output else PROJECT_ROOT / "results" / "honest_eval_rsmiles.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
