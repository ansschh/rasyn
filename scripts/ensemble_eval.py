"""Ensemble evaluation: LLM (RSGPT) + RetroTransformer v2.

Combines predictions from both models using multiple strategies:
- LLM-first: LLM candidates ranked first, RetroTx fills remaining slots
- Agreement boost: Predictions from both models get higher rank
- Weighted score: Combine normalized scores from both models

Usage:
    # Quick test
    python -u scripts/ensemble_eval.py --max-samples 200

    # Full run
    python -u scripts/ensemble_eval.py

    # With round-trip re-ranking
    python -u scripts/ensemble_eval.py \
        --forward-checkpoint checkpoints/forward/uspto50k/best_model.pt

    # RunPod
    nohup python -u scripts/ensemble_eval.py > ensemble_eval.log 2>&1 &
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


# ── SMILES utilities ─────────────────────────────────────────────────

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.logger().setLevel(RDLogger.ERROR)
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def canonicalize_smiles(smi: str) -> str:
    if not RDKIT_AVAILABLE:
        return smi.strip()
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return ""


def normalize_reactants(smiles_str: str) -> str:
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    canon_parts = [canonicalize_smiles(p.strip()) for p in parts if p.strip()]
    canon_parts = [c for c in canon_parts if c]
    return ".".join(sorted(canon_parts))


def tanimoto_similarity(smi1: str, smi2: str) -> float:
    if not RDKIT_AVAILABLE:
        return 0.0
    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


# ── Data loading ─────────────────────────────────────────────────────

def download_test_csv(output_path: Path) -> Path:
    if output_path.exists():
        return output_path
    logger.info("Downloading USPTO-50K test.csv...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TEST_CSV_URL, str(output_path))
    return output_path


def load_and_preprocess_test_data(csv_path, training_products=None):
    from rasyn.preprocess.build_edit_dataset import build_training_example
    examples = []
    skipped_overlap = 0
    skipped_preprocess = 0

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc="Preprocessing"):
        rxn_smiles = row.get("rxn_smiles", "")
        rxn_class = int(row.get("class", 0))
        rxn_id = row.get("id", "")
        if not rxn_smiles:
            continue

        if training_products is not None:
            parts = rxn_smiles.split(">>")
            if len(parts) == 2:
                product = canonicalize_smiles(parts[1].strip())
                if product in training_products:
                    skipped_overlap += 1
                    continue

        example = build_training_example(rxn_smiles, rxn_id)
        if example is None:
            skipped_preprocess += 1
            continue
        example["reaction_class"] = rxn_class
        example["rxn_smiles"] = rxn_smiles
        examples.append(example)

    logger.info(f"Preprocessed: {len(examples)} examples "
                f"(skipped: {skipped_overlap} overlap, {skipped_preprocess} preprocess)")
    return examples


def load_training_products(data_path: Path) -> set[str]:
    import re
    products = set()
    if not data_path.exists():
        return products
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            prompt = ex.get("prompt", "")
            m = re.search(r"<PROD>\s+(.+?)\s+<EDIT>", prompt)
            if m:
                c = canonicalize_smiles(m.group(1).strip())
                if c:
                    products.add(c)
    logger.info(f"Loaded {len(products)} unique training products")
    return products


# ── LLM inference ────────────────────────────────────────────────────

def decode_llm_completion(tokenizer, output_ids) -> str:
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    if "<OUT>" in text:
        completion = text.split("<OUT>")[-1]
        for stop in ["</s>", "<EOS>", "<pad>", "<PAD>"]:
            completion = completion.split(stop)[0]
        return completion.replace("<unk>", "").strip()
    return ""


def get_llm_predictions(
    model, tokenizer, prompt: str, device: str,
    num_beams: int = 10, max_new_tokens: int = 256,
    n_temp_passes: int = 10, samples_per_pass: int = 10,
    temperature: float = 0.8,
) -> list[tuple[str, float]]:
    """Get LLM predictions with TTA (beam + temperature sampling).

    Returns list of (canonical_reactants, score) sorted by frequency.
    """
    from rasyn.models.llm.generate import tokenize_prompt_for_inference

    inputs = tokenize_prompt_for_inference(prompt, tokenizer, max_length=512, device=device)

    # Deterministic beam search
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

    candidates: dict[str, int] = defaultdict(int)
    for seq in base_out:
        comp = decode_llm_completion(tokenizer, seq)
        if comp:
            norm = normalize_reactants(comp)
            if norm:
                candidates[norm] += num_beams  # Weight beam predictions

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

    # Sort by frequency
    max_count = n_temp_passes * samples_per_pass + num_beams
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [(r, count / max_count) for r, count in ranked]


# ── RetroTx v2 inference ─────────────────────────────────────────────

def get_retro_predictions(
    model, tokenizer, src_text: str, device: str,
    beam_size: int = 10, max_len: int = 128,
) -> list[tuple[str, float]]:
    """Get RetroTx v2 beam search predictions.

    Returns list of (canonical_reactants, normalized_score).
    """
    src_ids = torch.tensor(
        [tokenizer.encode(src_text, max_len=256, add_bos=True, add_eos=True)],
        dtype=torch.long, device=device,
    )
    seg_ids = torch.tensor(
        [tokenizer.get_segment_ids(tokenizer.encode(src_text, max_len=256, add_bos=True, add_eos=True))],
        dtype=torch.long, device=device,
    )

    beam_results = model.generate_beam(
        src_ids, tokenizer.bos_token_id, tokenizer.eos_token_id,
        beam_size=beam_size, max_len=max_len, segment_ids=seg_ids,
    )[0]  # First (only) batch element

    predictions = []
    seen = set()
    for token_ids, score in beam_results:
        pred_str = tokenizer.decode(token_ids, skip_special=True)
        norm = normalize_reactants(pred_str)
        if norm and norm not in seen:
            seen.add(norm)
            predictions.append((norm, score))

    # Normalize scores to [0, 1]
    if predictions:
        max_score = predictions[0][1]
        min_score = predictions[-1][1] if len(predictions) > 1 else max_score - 1
        score_range = max_score - min_score if max_score != min_score else 1.0
        predictions = [(r, (s - min_score) / score_range) for r, s in predictions]

    return predictions


# ── Ensemble strategies ──────────────────────────────────────────────

def ensemble_llm_first(
    llm_preds: list[tuple[str, float]],
    retro_preds: list[tuple[str, float]],
    max_k: int = 10,
) -> list[str]:
    """LLM-first: LLM candidates ranked first, RetroTx fills gaps."""
    seen = set()
    result = []
    for r, _ in llm_preds:
        if r not in seen:
            seen.add(r)
            result.append(r)
        if len(result) >= max_k:
            break
    for r, _ in retro_preds:
        if r not in seen:
            seen.add(r)
            result.append(r)
        if len(result) >= max_k:
            break
    return result


def ensemble_agreement_boost(
    llm_preds: list[tuple[str, float]],
    retro_preds: list[tuple[str, float]],
    boost: float = 2.0,
    max_k: int = 10,
) -> list[str]:
    """Agreement boost: predictions from both models get higher score."""
    scores: dict[str, float] = {}

    # LLM scores (rank-based: top-1 gets 1.0, top-2 gets 0.9, etc.)
    for rank, (r, s) in enumerate(llm_preds[:max_k]):
        scores[r] = scores.get(r, 0) + max(0, 1.0 - rank * 0.1)

    # RetroTx scores
    retro_set = set()
    for rank, (r, s) in enumerate(retro_preds[:max_k]):
        retro_set.add(r)
        scores[r] = scores.get(r, 0) + max(0, 1.0 - rank * 0.1)

    # Boost agreement
    for r in scores:
        if r in retro_set and any(r == llm_r for llm_r, _ in llm_preds[:max_k]):
            scores[r] *= boost

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [r for r, _ in ranked[:max_k]]


def ensemble_weighted(
    llm_preds: list[tuple[str, float]],
    retro_preds: list[tuple[str, float]],
    llm_weight: float = 0.7,
    retro_weight: float = 0.3,
    max_k: int = 10,
) -> list[str]:
    """Weighted combination of model scores."""
    scores: dict[str, float] = {}

    for r, s in llm_preds[:20]:
        scores[r] = scores.get(r, 0) + llm_weight * s

    for r, s in retro_preds[:20]:
        scores[r] = scores.get(r, 0) + retro_weight * s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [r for r, _ in ranked[:max_k]]


# ── Round-trip re-ranking ────────────────────────────────────────────

def round_trip_verify(
    predictions: list[str],
    product_smiles: str,
    fwd_model, fwd_tokenizer, device: str,
    max_len: int = 256,
) -> list[tuple[str, float]]:
    """Score predictions with forward model round-trip verification."""
    product_canon = canonicalize_smiles(product_smiles)
    results = []

    for reactants_str in predictions:
        src_ids = torch.tensor(
            [fwd_tokenizer.encode(reactants_str, max_len=max_len)],
            dtype=torch.long, device=device,
        )
        pred_ids = fwd_model.generate_greedy(
            src_ids, fwd_tokenizer.bos_token_id, fwd_tokenizer.eos_token_id,
            max_len=max_len,
        )[0]
        pred_product = fwd_tokenizer.decode(pred_ids)
        pred_canon = canonicalize_smiles(pred_product)

        if pred_canon and pred_canon == product_canon:
            rt_score = 1.0
        elif pred_canon:
            rt_score = tanimoto_similarity(pred_product, product_smiles)
        else:
            rt_score = 0.0

        results.append((reactants_str, rt_score))

    return results


# ── Main ─────────────────────────────────────────────────────────────

@click.command()
@click.option("--llm-checkpoint", default="checkpoints/llm/uspto50k_v6/final")
@click.option("--retro-checkpoint", default="checkpoints/retro_v2/uspto50k/best/model.pt")
@click.option("--forward-checkpoint", default=None, type=str)
@click.option("--max-samples", default=5000, type=int)
@click.option("--llm-beams", default=10, type=int)
@click.option("--retro-beams", default=10, type=int)
@click.option("--n-temp-passes", default=10, type=int, help="Temperature sampling passes for LLM TTA")
@click.option("--temperature", default=0.8, type=float)
@click.option("--check-overlap/--no-check-overlap", default=True)
@click.option("--device", default="auto")
@click.option("--output", default=None, type=str)
def main(
    llm_checkpoint, retro_checkpoint, forward_checkpoint,
    max_samples, llm_beams, retro_beams, n_temp_passes, temperature,
    check_overlap, device, output,
):
    """Ensemble evaluation: LLM + RetroTransformer v2."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load test data ────────────────────────────────────────────
    test_csv = PROJECT_ROOT / "data" / "external" / "uspto50k_test.csv"
    download_test_csv(test_csv)

    training_products = None
    if check_overlap:
        train_data = PROJECT_ROOT / "data" / "processed" / "uspto50k" / "edit_conditioned_train.jsonl"
        training_products = load_training_products(train_data)

    test_examples = load_and_preprocess_test_data(test_csv, training_products)
    if max_samples < len(test_examples):
        test_examples = test_examples[:max_samples]
    logger.info(f"Evaluating on {len(test_examples)} truly unseen examples")

    # ── Load LLM ──────────────────────────────────────────────────
    logger.info(f"Loading LLM from {llm_checkpoint}...")
    from transformers import AutoTokenizer
    from peft import PeftModel
    from rasyn.models.llm.model import load_rsgpt_model

    llm_ckpt = Path(llm_checkpoint)
    if (llm_ckpt / "adapter_config.json").exists():
        weights_path = PROJECT_ROOT / "weights" / "rsgpt" / "finetune_50k.pth"
        base_model, _ = load_rsgpt_model(weights_path=str(weights_path), use_lora=False)
        llm_model = PeftModel.from_pretrained(base_model, str(llm_ckpt))
        llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_ckpt))
    else:
        from transformers import LlamaForCausalLM
        llm_model = LlamaForCausalLM.from_pretrained(str(llm_ckpt))
        llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_ckpt))
    llm_model = llm_model.to(device).eval()
    logger.info("LLM loaded")

    # ── Load RetroTx v2 ──────────────────────────────────────────
    logger.info(f"Loading RetroTx v2 from {retro_checkpoint}...")
    from rasyn.models.retro.model_v2 import load_retro_model_v2
    retro_model, retro_tokenizer = load_retro_model_v2(retro_checkpoint, device=device)
    logger.info("RetroTx v2 loaded")

    # ── Load forward model (optional) ────────────────────────────
    fwd_model = fwd_tokenizer = None
    if forward_checkpoint:
        from rasyn.models.forward.model import load_forward_model
        fwd_model, fwd_tokenizer = load_forward_model(forward_checkpoint, device=device)
        logger.info("Forward model loaded")

    # ── Metrics ───────────────────────────────────────────────────
    metrics = {
        name: {"top1": 0, "top3": 0, "top5": 0, "top10": 0}
        for name in ["llm_only", "retro_only", "llm_first", "agreement", "weighted", "rt_rerank"]
    }
    total = 0
    start_time = time.time()

    for i, ex in enumerate(tqdm(test_examples, desc="Evaluating")):
        prompt = ex["prompt"]
        gt = normalize_reactants(ex["completion"])
        if not gt:
            continue

        # Parse product
        from rasyn.models.llm.tokenizer import parse_edit_prompt
        parsed = parse_edit_prompt(prompt)
        product = parsed["product"]
        if not product:
            continue

        rxn_class = ex.get("reaction_class", 0)

        # ── Get LLM predictions ──────────────────────────────
        llm_preds = get_llm_predictions(
            llm_model, llm_tokenizer, prompt, device,
            num_beams=llm_beams, n_temp_passes=n_temp_passes,
            temperature=temperature,
        )

        # ── Get RetroTx v2 predictions ───────────────────────
        # Build source text: <RXN_X> product | synthons
        src_parts = []
        if rxn_class > 0:
            src_parts.append(f"<RXN_{rxn_class}>")
        src_parts.append(product)
        if parsed.get("synthons"):
            src_parts.append(f"| {parsed['synthons']}")
        src_text = " ".join(src_parts)

        retro_preds = get_retro_predictions(
            retro_model, retro_tokenizer, src_text, device,
            beam_size=retro_beams,
        )

        # ── Ensemble predictions ─────────────────────────────
        total += 1

        llm_only = [r for r, _ in llm_preds[:10]]
        retro_only = [r for r, _ in retro_preds[:10]]
        ens_llm_first = ensemble_llm_first(llm_preds, retro_preds)
        ens_agreement = ensemble_agreement_boost(llm_preds, retro_preds)
        ens_weighted = ensemble_weighted(llm_preds, retro_preds)

        # Score each strategy
        strategy_preds = {
            "llm_only": llm_only,
            "retro_only": retro_only,
            "llm_first": ens_llm_first,
            "agreement": ens_agreement,
            "weighted": ens_weighted,
        }

        for name, preds in strategy_preds.items():
            if preds and preds[0] == gt:
                metrics[name]["top1"] += 1
            if gt in preds[:3]:
                metrics[name]["top3"] += 1
            if gt in preds[:5]:
                metrics[name]["top5"] += 1
            if gt in preds[:10]:
                metrics[name]["top10"] += 1

        # Round-trip re-ranking (on agreement ensemble)
        if fwd_model is not None:
            rt_results = round_trip_verify(
                ens_agreement[:10], product, fwd_model, fwd_tokenizer, device
            )
            # Re-rank: keep original rank but boost RT-verified ones
            rt_preds = sorted(
                [(r, i, rt_s) for i, (r, rt_s) in enumerate(rt_results)],
                key=lambda x: (-x[2], x[1]),  # RT score desc, then original rank
            )
            rt_predictions = [r for r, _, _ in rt_preds]

            if rt_predictions and rt_predictions[0] == gt:
                metrics["rt_rerank"]["top1"] += 1
            if gt in rt_predictions[:3]:
                metrics["rt_rerank"]["top3"] += 1
            if gt in rt_predictions[:5]:
                metrics["rt_rerank"]["top5"] += 1
            if gt in rt_predictions[:10]:
                metrics["rt_rerank"]["top10"] += 1

        # Progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(test_examples) - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"[{i+1}/{len(test_examples)}] "
                f"LLM: {metrics['llm_only']['top1']/total:.3f} | "
                f"Retro: {metrics['retro_only']['top1']/total:.3f} | "
                f"Agree: {metrics['agreement']['top1']/total:.3f} | "
                + (f"RT: {metrics['rt_rerank']['top1']/total:.3f} | " if fwd_model else "")
                + f"Rate: {rate:.2f} s/s | ETA: {eta/3600:.1f}h"
            )

    # ── Final results ─────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"ENSEMBLE EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Samples: {total}, Time: {elapsed:.0f}s")
    logger.info(f"")

    for name in ["llm_only", "retro_only", "llm_first", "agreement", "weighted"]:
        m = metrics[name]
        logger.info(f"  {name.upper()}:")
        logger.info(f"    Top-1: {m['top1']/max(total,1):.4f} ({m['top1']}/{total})")
        logger.info(f"    Top-3: {m['top3']/max(total,1):.4f}")
        logger.info(f"    Top-5: {m['top5']/max(total,1):.4f}")
        logger.info(f"    Top-10: {m['top10']/max(total,1):.4f}")
        logger.info(f"")

    if fwd_model is not None:
        m = metrics["rt_rerank"]
        logger.info(f"  RT_RERANK (agreement + round-trip):")
        logger.info(f"    Top-1: {m['top1']/max(total,1):.4f} ({m['top1']}/{total})")
        logger.info(f"    Top-3: {m['top3']/max(total,1):.4f}")
        logger.info(f"    Top-5: {m['top5']/max(total,1):.4f}")
        logger.info(f"    Top-10: {m['top10']/max(total,1):.4f}")

    logger.info(f"{'='*60}")

    # Save
    results = {"total": total, "elapsed": elapsed}
    for name, m in metrics.items():
        if name == "rt_rerank" and fwd_model is None:
            continue
        results[name] = {k: v / max(total, 1) for k, v in m.items()}

    out_path = Path(output) if output else PROJECT_ROOT / "checkpoints" / "ensemble_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
