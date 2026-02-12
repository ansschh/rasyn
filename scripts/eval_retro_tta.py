"""Evaluate RetroTransformer v2 with Test-Time Augmentation (TTA).

TTA for RetroTx: Randomize the input SMILES N times, run beam search on
each variant, aggregate predictions by canonical form, rank by frequency.

Unlike the LLM TTA (which uses temperature sampling because the LLM was
trained only on canonical SMILES), the RetroTx WAS trained with SMILES
augmentation (source-only randomization). So input randomization should
work well — the model has seen random SMILES during training.

Usage:
    # Quick test
    python -u scripts/eval_retro_tta.py --max-samples 200 --n-augments 10

    # Full run (on standard test set, honest eval)
    python -u scripts/eval_retro_tta.py --n-augments 20

    # RunPod background
    nohup python -u scripts/eval_retro_tta.py --n-augments 20 > eval_retro_tta.log 2>&1 &
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


def randomize_smiles(smiles: str) -> str:
    """Generate a random SMILES representation."""
    if not RDKIT_AVAILABLE:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        pass
    return smiles


def randomize_multi(smiles_str: str, separator: str = " | ") -> str:
    """Randomize each component of a multi-component string."""
    import random
    parts = smiles_str.split(separator)
    randomized = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Randomize each dot-separated sub-component
        sub_parts = p.split(".")
        rand_subs = [randomize_smiles(s.strip()) for s in sub_parts if s.strip()]
        if rand_subs:
            random.shuffle(rand_subs)
            randomized.append(".".join(rand_subs))
    return separator.join(randomized)


def normalize_reactants(smiles_str: str, remove_mapping: bool = False) -> str:
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    canon_parts = [canonicalize_smiles(p.strip(), remove_mapping=remove_mapping) for p in parts if p.strip()]
    canon_parts = [c for c in canon_parts if c]
    return ".".join(sorted(canon_parts))


# ── Data loading ─────────────────────────────────────────────────────

def download_test_csv(output_path: Path) -> Path:
    if output_path.exists():
        return output_path
    logger.info("Downloading USPTO-50K test.csv...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TEST_CSV_URL, str(output_path))
    return output_path


def load_test_data(csv_path: Path) -> list[dict]:
    """Load ALL test reactions."""
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
            rows.append({
                "rxn_smiles": rxn_smiles,
                "rxn_class": int(row.get("class", 0)),
                "rxn_id": row.get("id", ""),
                "product_canon": canonicalize_smiles(parts[1].strip(), remove_mapping=True),
                "gt_reactants": normalize_reactants(parts[0].strip(), remove_mapping=True),
            })
    logger.info(f"Loaded {len(rows)} test reactions")
    return rows


def preprocess_for_edit_pipeline(rxn_smiles: str, rxn_id: str) -> dict | None:
    from rasyn.preprocess.build_edit_dataset import build_training_example
    return build_training_example(rxn_smiles, rxn_id)


# ── Main ─────────────────────────────────────────────────────────────

@click.command()
@click.option("--checkpoint", default="checkpoints/retro_v2/uspto50k/best/model.pt")
@click.option("--beam-size", default=10, type=int)
@click.option("--max-len", default=128, type=int)
@click.option("--n-augments", default=20, type=int,
              help="Number of SMILES randomizations per sample")
@click.option("--max-samples", default=0, type=int, help="Limit samples (0=all)")
@click.option("--device", default="auto")
@click.option("--output", default=None, type=str)
def main(checkpoint, beam_size, max_len, n_augments, max_samples, device, output):
    """Evaluate RetroTx v2 with TTA (SMILES augmentation + beam aggregation)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test data
    test_csv = PROJECT_ROOT / "data" / "external" / "uspto50k_test.csv"
    download_test_csv(test_csv)
    test_data = load_test_data(test_csv)

    if max_samples > 0:
        test_data = test_data[:max_samples]
        logger.warning(f"*** LIMITED TO {max_samples} SAMPLES ***")

    total_test = len(test_data)

    # Load model
    logger.info(f"Loading RetroTx v2 from {checkpoint}...")
    from rasyn.models.retro.model_v2 import load_retro_model_v2
    model, tokenizer = load_retro_model_v2(str(PROJECT_ROOT / checkpoint), device=device)
    logger.info("Model loaded")

    # Metrics
    # Base = single canonical beam search
    # TTA = N augmented + canonical, aggregated
    top_k_base = {1: 0, 3: 0, 5: 0, 10: 0}
    top_k_tta = {1: 0, 3: 0, 5: 0, 10: 0}
    total = 0
    attempted = 0
    preprocess_failures = 0
    total_unique_tta = 0
    start = time.time()

    import re

    for i, row in enumerate(tqdm(test_data, desc="Evaluating")):
        gt = row["gt_reactants"]
        product = row["product_canon"]
        rxn_class = row["rxn_class"]

        if not gt or not product:
            total += 1
            continue

        # Preprocess
        preprocessed = preprocess_for_edit_pipeline(row["rxn_smiles"], row["rxn_id"])
        if preprocessed is None:
            preprocess_failures += 1
            total += 1
            continue

        prompt = preprocessed["prompt"]
        synth_match = re.search(r"<SYNTHONS>\s*(.+?)(?=\s*<(?:LG_HINTS|CONSTRAINTS|OUT)>)", prompt)
        synthons = synth_match.group(1).strip() if synth_match else ""

        total += 1
        attempted += 1

        # Build canonical source text
        src_parts = []
        if rxn_class > 0:
            src_parts.append(f"<RXN_{rxn_class}>")
        src_parts.append(product)
        if synthons:
            src_parts.append(f"| {synthons}")
        canonical_src = " ".join(src_parts)

        def run_beam(src_text: str) -> list[tuple[str, float]]:
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
            results = []
            for token_ids, score in beam_results:
                pred_str = tokenizer.decode(token_ids)
                norm = normalize_reactants(pred_str)
                if norm:
                    results.append((norm, score))
            return results

        # ── Base: canonical beam search ──────────────────────
        base_results = run_beam(canonical_src)
        base_preds = []
        seen = set()
        for norm, _ in base_results:
            if norm not in seen:
                seen.add(norm)
                base_preds.append(norm)

        for k in top_k_base:
            if gt in base_preds[:k]:
                top_k_base[k] += 1

        # ── TTA: randomize input N times ─────────────────────
        # Self-consistency: take top-1 from each augmented pass,
        # count how many passes agree on each prediction.
        # Using all beams fragments the vote and hurts accuracy.
        tta_candidates: dict[str, int] = defaultdict(int)

        # Include canonical top-1 (strong weight)
        if base_preds:
            tta_candidates[base_preds[0]] += n_augments  # Weight = number of augments

        # Also include canonical top-2..top-k with decreasing weight
        for rank, pred in enumerate(base_preds[1:], start=1):
            tta_candidates[pred] += max(1, n_augments // (rank + 1))

        # Augmented passes: only top-1 from each
        for aug_idx in range(n_augments):
            rand_product = randomize_smiles(product)
            aug_parts = []
            if rxn_class > 0:
                aug_parts.append(f"<RXN_{rxn_class}>")
            aug_parts.append(rand_product)
            if synthons:
                rand_synthons = ".".join(
                    randomize_smiles(s.strip())
                    for s in synthons.split(".")
                    if s.strip()
                )
                aug_parts.append(f"| {rand_synthons}")
            aug_src = " ".join(aug_parts)

            aug_results = run_beam(aug_src)
            # Only use top-1 (self-consistency voting)
            seen_aug = set()
            for norm, _ in aug_results:
                if norm not in seen_aug:
                    seen_aug.add(norm)
                    tta_candidates[norm] += 1
                    break  # Only top-1

        # Rank by frequency
        tta_ranked = sorted(tta_candidates.items(), key=lambda x: x[1], reverse=True)
        tta_preds = [r for r, _ in tta_ranked]
        total_unique_tta += len(tta_preds)

        for k in top_k_tta:
            if gt in tta_preds[:k]:
                top_k_tta[k] += 1

        # Progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (total_test - i - 1) / rate if rate > 0 else 0
            logger.info(
                f"[{i+1}/{total_test}] "
                f"Base Top-1: {top_k_base[1]/max(attempted,1):.4f}({top_k_base[1]}/{attempted}) | "
                f"TTA Top-1: {top_k_tta[1]/max(attempted,1):.4f}({top_k_tta[1]}/{attempted}) | "
                f"Coverage: {attempted/total:.3f} | "
                f"Unique/sample: {total_unique_tta/max(attempted,1):.1f} | "
                f"ETA: {eta/60:.1f}m"
            )

    elapsed = time.time() - start
    coverage = attempted / max(total, 1)

    print(f"\n{'='*70}")
    print(f"RETROTRANSFORMER V2 TTA EVALUATION — Standard USPTO-50K Test Set")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Total test: {total}")
    print(f"Attempted: {attempted} (coverage: {coverage:.4f})")
    print(f"Preprocess failures: {preprocess_failures}")
    print(f"TTA config: {n_augments} augments + canonical, beam_size={beam_size}")
    print(f"Avg unique per sample: {total_unique_tta/max(attempted,1):.1f}")
    print(f"Time: {elapsed:.0f}s ({total/max(elapsed,1):.1f} samples/s)")
    print()
    print("  BASE (canonical beam search only):")
    for k in [1, 3, 5, 10]:
        on_attempted = top_k_base[k] / max(attempted, 1)
        on_total = top_k_base[k] / max(total, 1)
        print(f"    Top-{k}: {on_total:.4f} (on {total} total) | "
              f"{on_attempted:.4f} (on {attempted} attempted) | "
              f"correct={top_k_base[k]}")
    print()
    print(f"  TTA ({n_augments} augments, frequency-ranked):")
    for k in [1, 3, 5, 10]:
        on_attempted = top_k_tta[k] / max(attempted, 1)
        on_total = top_k_tta[k] / max(total, 1)
        print(f"    Top-{k}: {on_total:.4f} (on {total} total) | "
              f"{on_attempted:.4f} (on {attempted} attempted) | "
              f"correct={top_k_tta[k]}")
    print(f"{'='*70}")

    # Save results
    results = {
        "test_set": "USPTO-50K standard Schneider split",
        "checkpoint": checkpoint,
        "total": total,
        "attempted": attempted,
        "coverage": coverage,
        "preprocess_failures": preprocess_failures,
        "tta_augments": n_augments,
        "beam_size": beam_size,
        "elapsed_seconds": elapsed,
        "base": {
            f"top{k}_on_total": top_k_base[k] / max(total, 1)
            for k in [1, 3, 5, 10]
        } | {
            f"top{k}_on_attempted": top_k_base[k] / max(attempted, 1)
            for k in [1, 3, 5, 10]
        },
        "tta": {
            f"top{k}_on_total": top_k_tta[k] / max(total, 1)
            for k in [1, 3, 5, 10]
        } | {
            f"top{k}_on_attempted": top_k_tta[k] / max(attempted, 1)
            for k in [1, 3, 5, 10]
        } | {
            "avg_unique_per_sample": total_unique_tta / max(attempted, 1),
        },
    }

    out_path = Path(output) if output else PROJECT_ROOT / "results" / "retro_tta_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
