"""Evaluate LLM with Test-Time Augmentation (TTA) via temperature sampling.

Uses self-consistency: run multiple stochastic inference passes on the same
canonical prompt, aggregate predictions by canonical form, rank by frequency.
Optionally re-ranks using the forward model (round-trip verification).

Strategy: Temperature sampling provides diversity (the model was trained on
canonical SMILES only, so SMILES augmentation doesn't work). Correct predictions
are high-probability and appear consistently across independent samples.

Usage:
    # Quick validation (~15 min)
    python -u scripts/eval_llm_tta.py --max-samples 200 --n-samples 100

    # Full run (overnight)
    python -u scripts/eval_llm_tta.py --n-samples 200

    # With round-trip re-ranking
    python -u scripts/eval_llm_tta.py --n-samples 200 \
        --forward-checkpoint checkpoints/forward/uspto50k/best_model.pt

    # RunPod background
    nohup python -u scripts/eval_llm_tta.py --n-samples 200 > eval_tta.log 2>&1 &
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
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.logger().setLevel(RDLogger.ERROR)
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
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
    canon_parts = []
    for p in parts:
        c = canonicalize_smiles(p.strip())
        if c:
            canon_parts.append(c)
    return ".".join(sorted(canon_parts))


def tanimoto_similarity(smi1: str, smi2: str) -> float:
    """Tanimoto similarity between two SMILES using Morgan fingerprints."""
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
        logger.info(f"Using cached test CSV: {output_path}")
        return output_path
    logger.info(f"Downloading USPTO-50K test.csv...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TEST_CSV_URL, str(output_path))
    logger.info(f"Downloaded to {output_path}")
    return output_path


def load_and_preprocess_test_data(
    csv_path: Path,
    training_products: set[str] | None = None,
) -> list[dict]:
    """Load test.csv, run edit extraction, filter overlap."""
    from rasyn.preprocess.build_edit_dataset import build_training_example

    examples = []
    skipped_overlap = 0
    skipped_preprocess = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info(f"Processing {len(rows)} test reactions...")

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
        examples.append(example)

    logger.info(
        f"Preprocessed: {len(examples)} examples "
        f"(skipped: {skipped_overlap} overlap, {skipped_preprocess} preprocess failures)"
    )
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
            prod_match = re.search(r"<PROD>\s+(.+?)\s+<EDIT>", prompt)
            if prod_match:
                canon = canonicalize_smiles(prod_match.group(1).strip())
                if canon:
                    products.add(canon)
    logger.info(f"Loaded {len(products)} unique training products")
    return products


# ── Round-trip re-ranking ────────────────────────────────────────────

def round_trip_rerank(
    candidates: list[tuple[str, float]],
    product_smiles: str,
    fwd_model,
    fwd_tokenizer,
    device: str,
    alpha: float = 0.7,
    beta: float = 0.3,
    max_len: int = 256,
) -> list[tuple[str, float, float]]:
    """Re-rank candidates using forward model verification.

    Args:
        candidates: List of (canonical_reactants, tta_score).
        product_smiles: Original product SMILES.
        fwd_model: ForwardTransformer model.
        fwd_tokenizer: ForwardTokenizer.
        device: Device string.
        alpha: Weight for TTA score.
        beta: Weight for round-trip score.

    Returns:
        List of (canonical_reactants, combined_score, rt_score), sorted desc.
    """
    product_canon = canonicalize_smiles(product_smiles)
    reranked = []

    for reactants_str, tta_score in candidates:
        src_ids = torch.tensor(
            [fwd_tokenizer.encode(reactants_str, max_len=max_len)],
            dtype=torch.long,
            device=device,
        )

        pred_ids = fwd_model.generate_greedy(
            src_ids,
            fwd_tokenizer.bos_token_id,
            fwd_tokenizer.eos_token_id,
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

        combined = alpha * tta_score + beta * rt_score
        reranked.append((reactants_str, combined, rt_score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


# ── Decoding helpers ─────────────────────────────────────────────────

def decode_completion(tokenizer, output_ids) -> str:
    """Decode model output and extract the completion after <OUT>."""
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    if "<OUT>" in text:
        completion = text.split("<OUT>")[-1]
        for stop in ["</s>", "<EOS>", "<pad>", "<PAD>"]:
            completion = completion.split(stop)[0]
        completion = completion.replace("<unk>", "").strip()
        return completion
    return ""


def check_topk(gt: str, predictions: list[str], k: int) -> bool:
    """Check if ground truth is in top-k predictions."""
    return gt in predictions[:k]


# ── Main evaluation ──────────────────────────────────────────────────

@click.command()
@click.option("--checkpoint", default="checkpoints/llm/uspto50k_v6/final")
@click.option("--forward-checkpoint", default=None, type=str,
              help="Forward model checkpoint for round-trip re-ranking")
@click.option("--n-samples", default=200, type=int,
              help="Total number of stochastic samples per test product (across all passes)")
@click.option("--samples-per-pass", default=10, type=int,
              help="Number of sequences per generation call")
@click.option("--temperature", default=0.8, type=float,
              help="Sampling temperature for TTA")
@click.option("--top-k", default=50, type=int, help="Top-k for sampling")
@click.option("--top-p", default=0.95, type=float, help="Nucleus sampling threshold")
@click.option("--base-beams", default=10, type=int,
              help="Number of beams for deterministic base evaluation")
@click.option("--max-samples", default=5000, type=int, help="Max test samples to evaluate")
@click.option("--max-new-tokens", default=256, type=int)
@click.option("--check-overlap/--no-check-overlap", default=True)
@click.option("--rt-alpha", default=0.7, type=float, help="TTA score weight for re-ranking")
@click.option("--rt-beta", default=0.3, type=float, help="Round-trip score weight")
@click.option("--device", default="auto")
@click.option("--output", default=None, type=str, help="Output JSON file for results")
def main(
    checkpoint, forward_checkpoint, n_samples, samples_per_pass,
    temperature, top_k, top_p, base_beams, max_samples,
    max_new_tokens, check_overlap, rt_alpha, rt_beta, device, output,
):
    """Evaluate LLM with Test-Time Augmentation via temperature sampling."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = Path(checkpoint)
    n_passes = max(1, n_samples // samples_per_pass)

    logger.info(f"TTA config: {n_passes} passes x {samples_per_pass} samples/pass "
                f"= {n_passes * samples_per_pass} total samples, temp={temperature}")

    # ── Download and preprocess test data ─────────────────────────
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
    logger.info(f"Loading LLM from {checkpoint}...")
    from transformers import AutoTokenizer
    from peft import PeftModel
    from rasyn.models.llm.model import load_rsgpt_model
    from rasyn.models.llm.generate import tokenize_prompt_for_inference

    if (checkpoint / "adapter_config.json").exists():
        weights_path = PROJECT_ROOT / "weights" / "rsgpt" / "finetune_50k.pth"
        base_model, _ = load_rsgpt_model(weights_path=str(weights_path), use_lora=False)
        model = PeftModel.from_pretrained(base_model, str(checkpoint))
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    else:
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(str(checkpoint))
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))

    model = model.to(device)
    model.eval()
    logger.info(f"LLM loaded on {device}")

    # ── Load forward model (optional) ────────────────────────────
    fwd_model = None
    fwd_tokenizer = None
    if forward_checkpoint:
        logger.info(f"Loading forward model from {forward_checkpoint}...")
        from rasyn.models.forward.model import load_forward_model
        fwd_model, fwd_tokenizer = load_forward_model(forward_checkpoint, device=device)
        logger.info("Forward model loaded for round-trip re-ranking")

    # ── Evaluate ──────────────────────────────────────────────────
    top1_base = top3_base = top5_base = top10_base = 0
    top1_tta = top3_tta = top5_tta = top10_tta = 0
    top1_rt = top3_rt = top5_rt = top10_rt = 0
    total = 0
    total_unique_candidates = 0
    total_rt_exact = 0
    total_rt_checked = 0
    start_time = time.time()

    for i, ex in enumerate(tqdm(test_examples, desc="Evaluating")):
        prompt = ex["prompt"]
        gt_completion = ex["completion"]
        gt_normalized = normalize_reactants(gt_completion)

        if not gt_normalized:
            continue

        # Parse product for round-trip
        from rasyn.models.llm.tokenizer import parse_edit_prompt
        parsed = parse_edit_prompt(prompt)
        original_product = parsed["product"]
        if not original_product:
            continue

        # Tokenize once (same prompt for all passes)
        inputs = tokenize_prompt_for_inference(
            prompt, tokenizer, max_length=512, device=device
        )

        # ── Step 1: Deterministic beam search (base) ──────────
        with torch.no_grad():
            base_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=base_beams,
                num_return_sequences=min(base_beams, 10),
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        base_predictions = []
        seen_base = set()
        for seq in base_out:
            comp = decode_completion(tokenizer, seq)
            if comp:
                norm = normalize_reactants(comp)
                if norm and norm not in seen_base:
                    seen_base.add(norm)
                    base_predictions.append(norm)

        # ── Step 2: Temperature sampling (TTA) ────────────────
        all_candidates: dict[str, int] = defaultdict(int)  # canonical → count

        for pass_idx in range(n_passes):
            with torch.no_grad():
                sample_out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=samples_per_pass,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for seq in sample_out:
                comp = decode_completion(tokenizer, seq)
                if comp:
                    norm = normalize_reactants(comp)
                    if norm:
                        all_candidates[norm] += 1

        # Also add beam search predictions to candidate pool
        for pred in base_predictions:
            all_candidates[pred] += base_beams  # Weight beam predictions higher

        # ── Step 3: Rank by frequency (self-consistency) ──────
        tta_ranked = sorted(
            all_candidates.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        tta_predictions = [r for r, _ in tta_ranked]
        total_unique_candidates += len(tta_predictions)

        # ── Metrics ───────────────────────────────────────────
        total += 1

        # Base metrics
        if base_predictions and base_predictions[0] == gt_normalized:
            top1_base += 1
        if check_topk(gt_normalized, base_predictions, 3):
            top3_base += 1
        if check_topk(gt_normalized, base_predictions, 5):
            top5_base += 1
        if check_topk(gt_normalized, base_predictions, 10):
            top10_base += 1

        # TTA metrics
        if tta_predictions and tta_predictions[0] == gt_normalized:
            top1_tta += 1
        if check_topk(gt_normalized, tta_predictions, 3):
            top3_tta += 1
        if check_topk(gt_normalized, tta_predictions, 5):
            top5_tta += 1
        if check_topk(gt_normalized, tta_predictions, 10):
            top10_tta += 1

        # Round-trip re-ranking
        if fwd_model is not None:
            max_count = n_passes * samples_per_pass + base_beams
            tta_with_scores = [
                (r, count / max_count) for r, count in tta_ranked[:20]
            ]
            rt_results = round_trip_rerank(
                tta_with_scores, original_product,
                fwd_model, fwd_tokenizer, device,
                alpha=rt_alpha, beta=rt_beta,
            )
            rt_predictions = [r for r, _, _ in rt_results]

            for _, _, rt_score in rt_results:
                total_rt_checked += 1
                if rt_score == 1.0:
                    total_rt_exact += 1

            if rt_predictions and rt_predictions[0] == gt_normalized:
                top1_rt += 1
            if check_topk(gt_normalized, rt_predictions, 3):
                top3_rt += 1
            if check_topk(gt_normalized, rt_predictions, 5):
                top5_rt += 1
            if check_topk(gt_normalized, rt_predictions, 10):
                top10_rt += 1

        # Progress logging
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(test_examples) - i - 1) / rate if rate > 0 else 0

            logger.info(
                f"[{i+1}/{len(test_examples)}] "
                f"Base: {top1_base/total:.4f} | "
                f"TTA: {top1_tta/total:.4f} | "
                + (f"RT: {top1_rt/total:.4f} | " if fwd_model else "")
                + f"Unique: {total_unique_candidates/total:.1f} | "
                f"Rate: {rate:.2f} s/s | "
                f"ETA: {eta/3600:.1f}h"
            )

    # ── Final results ─────────────────────────────────────────────
    elapsed = time.time() - start_time

    logger.info(f"\n{'='*60}")
    logger.info(f"TTA EVALUATION RESULTS (Temperature Sampling)")
    logger.info(f"{'='*60}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Samples: {total}")
    logger.info(f"TTA: {n_passes} passes x {samples_per_pass} = {n_passes * samples_per_pass} samples, temp={temperature}")
    logger.info(f"Base beams: {base_beams}")
    logger.info(f"Avg unique per sample: {total_unique_candidates/max(total,1):.1f}")
    logger.info(f"Time: {elapsed:.0f}s ({total/elapsed:.2f} samples/s)")
    logger.info(f"")

    logger.info(f"  BASE (beam search, deterministic):")
    logger.info(f"    Top-1:  {top1_base/max(total,1):.4f} ({top1_base}/{total})")
    logger.info(f"    Top-3:  {top3_base/max(total,1):.4f} ({top3_base}/{total})")
    logger.info(f"    Top-5:  {top5_base/max(total,1):.4f} ({top5_base}/{total})")
    logger.info(f"    Top-10: {top10_base/max(total,1):.4f} ({top10_base}/{total})")
    logger.info(f"")

    logger.info(f"  TTA (temperature sampling + beam, frequency-ranked):")
    logger.info(f"    Top-1:  {top1_tta/max(total,1):.4f} ({top1_tta}/{total})")
    logger.info(f"    Top-3:  {top3_tta/max(total,1):.4f} ({top3_tta}/{total})")
    logger.info(f"    Top-5:  {top5_tta/max(total,1):.4f} ({top5_tta}/{total})")
    logger.info(f"    Top-10: {top10_tta/max(total,1):.4f} ({top10_tta}/{total})")

    if fwd_model is not None:
        logger.info(f"")
        logger.info(f"  TTA + ROUND-TRIP RE-RANKING (alpha={rt_alpha}, beta={rt_beta}):")
        logger.info(f"    Top-1:  {top1_rt/max(total,1):.4f} ({top1_rt}/{total})")
        logger.info(f"    Top-3:  {top3_rt/max(total,1):.4f} ({top3_rt}/{total})")
        logger.info(f"    Top-5:  {top5_rt/max(total,1):.4f} ({top5_rt}/{total})")
        logger.info(f"    Top-10: {top10_rt/max(total,1):.4f} ({top10_rt}/{total})")
        logger.info(f"    RT exact match rate: {total_rt_exact/max(total_rt_checked,1):.4f}")

    logger.info(f"{'='*60}")

    # ── Save results ──────────────────────────────────────────────
    results = {
        "checkpoint": str(checkpoint),
        "strategy": "temperature_sampling",
        "n_passes": n_passes,
        "samples_per_pass": samples_per_pass,
        "temperature": temperature,
        "base_beams": base_beams,
        "total_samples": total,
        "elapsed_seconds": elapsed,
        "base": {
            "top1": top1_base / max(total, 1),
            "top3": top3_base / max(total, 1),
            "top5": top5_base / max(total, 1),
            "top10": top10_base / max(total, 1),
        },
        "tta": {
            "top1": top1_tta / max(total, 1),
            "top3": top3_tta / max(total, 1),
            "top5": top5_tta / max(total, 1),
            "top10": top10_tta / max(total, 1),
            "avg_unique_candidates": total_unique_candidates / max(total, 1),
        },
    }
    if fwd_model is not None:
        results["tta_roundtrip"] = {
            "top1": top1_rt / max(total, 1),
            "top3": top3_rt / max(total, 1),
            "top5": top5_rt / max(total, 1),
            "top10": top10_rt / max(total, 1),
            "rt_alpha": rt_alpha,
            "rt_beta": rt_beta,
            "rt_exact_match_rate": total_rt_exact / max(total_rt_checked, 1),
        }

    output_path = Path(output) if output else checkpoint / "tta_eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
