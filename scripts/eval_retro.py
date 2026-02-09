"""Evaluate RetroTransformer on edit-conditioned retrosynthesis.

Measures Top-1/3/5 exact match accuracy with beam search.

Usage:
    python scripts/eval_retro.py --checkpoint checkpoints/retro/uspto50k/best/model.pt
    python scripts/eval_retro.py --checkpoint checkpoints/retro/uspto50k/best/model.pt --beam-size 10
"""

from __future__ import annotations

import json
import logging
import sys
import time
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


def canonicalize_smiles(smi: str) -> str:
    """Canonicalize a SMILES string."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return ""


def is_valid_smiles(smi: str) -> bool:
    """Check if a SMILES string is valid (parseable by RDKit)."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi.strip())
        return mol is not None
    except Exception:
        return False


def normalize_reactants(smiles_str: str) -> str:
    """Canonicalize and sort reactant SMILES for comparison."""
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    canon = [canonicalize_smiles(p) for p in parts if p.strip()]
    canon = [c for c in canon if c]
    return ".".join(sorted(canon))


def check_all_components_valid(smiles_str: str) -> bool:
    """Check if every component in a multi-component SMILES is valid."""
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not is_valid_smiles(p):
            return False
    return len(parts) > 0


@click.command()
@click.option("--checkpoint", required=True, help="Path to model.pt checkpoint")
@click.option("--data", default="data/processed/uspto50k/edit_conditioned_train.jsonl")
@click.option("--max-samples", default=500, type=int)
@click.option("--skip", default=0, type=int)
@click.option("--beam-size", default=5, type=int)
@click.option("--max-len", default=256, type=int)
@click.option("--conditioned/--unconditioned", default=True,
              help="Use synthon conditioning or product-only")
@click.option("--device", default="auto")
def main(checkpoint, data, max_samples, skip, beam_size, max_len, conditioned, device):
    """Evaluate RetroTransformer."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = PROJECT_ROOT / checkpoint
    data_path = PROJECT_ROOT / data

    # Load model
    logger.info(f"Loading model from {checkpoint_path}...")
    from rasyn.models.retro.model import load_retro_model
    model, tokenizer = load_retro_model(str(checkpoint_path), device=device)
    logger.info(f"Model loaded on {device}")

    # Load data
    import re
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    if skip > 0:
        examples = examples[skip:]
    if max_samples and max_samples < len(examples):
        examples = examples[:max_samples]
    logger.info(f"Evaluating on {len(examples)} samples (conditioned={conditioned})")

    # Evaluate
    top1 = top3 = top5 = 0
    total = 0
    invalid = 0
    valid_smiles_count = 0
    all_valid_count = 0  # Predictions where ALL components are valid SMILES
    total_predictions = 0
    start = time.time()

    for i, ex in enumerate(tqdm(examples, desc="Evaluating")):
        # Parse prompt
        prompt = ex["prompt"]
        prod_match = re.search(r"<PROD>\s+(.+?)\s+<EDIT>", prompt)
        product = prod_match.group(1).strip() if prod_match else ""
        synth_match = re.search(r"<SYNTHONS>\s+(.+?)\s+<LG_HINTS>", prompt)
        synthons = synth_match.group(1).strip() if synth_match else ""

        gt = normalize_reactants(ex["completion"])
        if not gt or not product:
            continue

        # Build input
        if conditioned and synthons:
            src_text = f"{product}|{synthons}"
        else:
            src_text = product

        src_ids = torch.tensor(
            [tokenizer.encode(src_text, max_len=512)],
            dtype=torch.long, device=device,
        )

        # Beam search
        beam_results = model.generate_beam(
            src_ids,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            beam_size=beam_size,
            max_len=max_len,
        )[0]  # First (only) batch element

        # Extract and normalize predictions
        predictions = []
        for token_ids, score in beam_results:
            pred_str = tokenizer.decode(token_ids)
            total_predictions += 1

            # Check if ALL components are valid SMILES
            if check_all_components_valid(pred_str):
                all_valid_count += 1

            norm = normalize_reactants(pred_str)
            if norm:
                valid_smiles_count += 1
                if norm not in predictions:
                    predictions.append(norm)
            else:
                invalid += 1

        total += 1

        # Check accuracy
        if predictions and predictions[0] == gt:
            top1 += 1
        if gt in predictions[:3]:
            top3 += 1
        if gt in predictions[:5]:
            top5 += 1

        # Periodic logging
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            validity_rate = all_valid_count / max(total_predictions, 1) * 100
            logger.info(
                f"Step {i+1}/{len(examples)} | "
                f"Top-1: {top1/total:.4f} | Top-3: {top3/total:.4f} | "
                f"Top-5: {top5/total:.4f} | "
                f"Valid SMILES: {validity_rate:.1f}% | "
                f"Canon rate: {valid_smiles_count}/{valid_smiles_count+invalid} | "
                f"{(i+1)/elapsed:.1f} samples/s"
            )

    elapsed = time.time() - start

    # Results
    validity_rate = all_valid_count / max(total_predictions, 1)
    canon_rate = valid_smiles_count / max(valid_smiles_count + invalid, 1)

    print("\n" + "=" * 60)
    print("RETROTRANSFORMER EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {total}")
    print(f"Conditioned: {conditioned}")
    print(f"Beam size: {beam_size}")
    print(f"Time: {elapsed:.1f}s ({total/elapsed:.1f} samples/s)")
    print()
    print("--- Validity Metrics ---")
    print(f"  SMILES validity rate:  {validity_rate*100:.1f}% "
          f"({all_valid_count}/{total_predictions} predictions have all components valid)")
    print(f"  Canonicalization rate: {canon_rate*100:.1f}% "
          f"({valid_smiles_count}/{valid_smiles_count+invalid} predictions canonicalize)")
    print()
    print("--- Accuracy Metrics ---")
    print(f"  Top-1 accuracy: {top1/max(total,1):.4f} ({top1}/{total})")
    print(f"  Top-3 accuracy: {top3/max(total,1):.4f} ({top3}/{total})")
    print(f"  Top-5 accuracy: {top5/max(total,1):.4f} ({top5}/{total})")
    print("=" * 60)

    # Save
    results = {
        "checkpoint": str(checkpoint_path),
        "total": total,
        "top1": top1 / max(total, 1),
        "top3": top3 / max(total, 1),
        "top5": top5 / max(total, 1),
        "smiles_validity_rate": validity_rate,
        "canonicalization_rate": canon_rate,
        "total_predictions": total_predictions,
        "all_valid_predictions": all_valid_count,
        "conditioned": conditioned,
        "beam_size": beam_size,
    }
    results_file = checkpoint_path.parent / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
