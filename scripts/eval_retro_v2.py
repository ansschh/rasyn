"""Evaluate RetroTransformer v2 with comprehensive metrics.

Metrics:
  1. Top-k exact match accuracy (canonical comparison)
  2. Round-trip verification (forward model: predicted reactants -> product)
  3. Tanimoto similarity (Morgan fingerprints) between predicted and true
  4. SMILES validity rate (all components valid)
  5. Beam diversity (unique predictions per sample)
  6. Copy rate (fraction of tokens copied from source)

Usage:
    python scripts/eval_retro_v2.py --checkpoint checkpoints/retro_v2/uspto50k/best/model.pt
    python scripts/eval_retro_v2.py --checkpoint ... --forward-checkpoint checkpoints/forward/best.pt
"""

from __future__ import annotations

import json
import logging
import re
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
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return ""


def normalize_reactants(smiles_str: str) -> str:
    """Canonicalize and sort reactant SMILES for comparison."""
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    canon = [canonicalize_smiles(p) for p in parts if p.strip()]
    canon = [c for c in canon if c]
    return ".".join(sorted(canon))


def check_all_valid(smiles_str: str) -> bool:
    """Check if every component is a valid SMILES."""
    from rdkit import Chem
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    for p in parts:
        p = p.strip()
        if p and Chem.MolFromSmiles(p) is None:
            return False
    return len(parts) > 0


def compute_tanimoto(smi_a: str, smi_b: str) -> float:
    """Compute Tanimoto similarity between two SMILES using Morgan fingerprints."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        mol_a = Chem.MolFromSmiles(smi_a)
        mol_b = Chem.MolFromSmiles(smi_b)
        if mol_a is None or mol_b is None:
            return 0.0

        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except Exception:
        return 0.0


def compute_reactant_tanimoto(pred_str: str, true_str: str) -> float:
    """Compute average best-match Tanimoto between predicted and true reactant sets."""
    pred_parts = [p.strip() for p in pred_str.replace(" . ", ".").split(".") if p.strip()]
    true_parts = [p.strip() for p in true_str.replace(" . ", ".").split(".") if p.strip()]

    if not pred_parts or not true_parts:
        return 0.0

    # For each true reactant, find best-matching predicted reactant
    scores = []
    for true_p in true_parts:
        best = max(compute_tanimoto(true_p, pred_p) for pred_p in pred_parts)
        scores.append(best)

    return sum(scores) / len(scores)


def forward_roundtrip(
    fwd_model, fwd_tokenizer, reactants_str: str, product_smiles: str, device: str
) -> bool:
    """Run forward model on predicted reactants, check if output matches product."""
    try:
        fwd_src = reactants_str.replace(".", " . ")
        fwd_src_ids = torch.tensor(
            [fwd_tokenizer.encode(fwd_src, max_len=512)],
            dtype=torch.long, device=device,
        )
        fwd_pred_ids = fwd_model.generate_greedy(
            fwd_src_ids,
            bos_token_id=fwd_tokenizer.bos_token_id,
            eos_token_id=fwd_tokenizer.eos_token_id,
            max_len=256,
        )[0]
        fwd_pred_str = fwd_tokenizer.decode(fwd_pred_ids)
        fwd_canon = canonicalize_smiles(fwd_pred_str)
        prod_canon = canonicalize_smiles(product_smiles)
        return bool(fwd_canon and prod_canon and fwd_canon == prod_canon)
    except Exception:
        return False


@click.command()
@click.option("--checkpoint", required=True, help="Path to v2 model checkpoint")
@click.option("--data", default="data/processed/uspto50k/augmented_train.jsonl")
@click.option("--max-samples", default=1000, type=int)
@click.option("--skip", default=0, type=int)
@click.option("--beam-size", default=10, type=int)
@click.option("--max-len", default=128, type=int)
@click.option("--forward-checkpoint", default=None, help="Forward model for round-trip")
@click.option("--device", default="auto")
def main(checkpoint, data, max_samples, skip, beam_size, max_len, forward_checkpoint, device):
    """Evaluate RetroTransformer v2."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = PROJECT_ROOT / checkpoint
    data_path = PROJECT_ROOT / data

    # Load retro model
    logger.info(f"Loading v2 model from {checkpoint_path}...")
    from rasyn.models.retro.model_v2 import load_retro_model_v2
    model, tokenizer = load_retro_model_v2(str(checkpoint_path), device=device)

    # Load forward model if provided
    fwd_model = fwd_tokenizer = None
    if forward_checkpoint:
        fwd_path = PROJECT_ROOT / forward_checkpoint
        logger.info(f"Loading forward model from {fwd_path}...")
        from rasyn.models.forward.model import load_forward_model
        fwd_model, fwd_tokenizer = load_forward_model(str(fwd_path), device=device)

    # Load data — use only canonical (augment_idx=0) versions for eval
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            if ex.get("augment_idx", 0) == 0:
                examples.append(ex)

    if skip > 0:
        examples = examples[skip:]
    if max_samples < len(examples):
        examples = examples[:max_samples]

    logger.info(f"Evaluating on {len(examples)} samples")

    # Evaluate
    top1 = top3 = top5 = top10 = 0
    rt_top1 = rt_top3 = 0  # round-trip
    total = 0
    valid_count = 0
    total_predictions = 0
    total_tanimoto = 0.0
    total_unique_beams = 0
    total_beam_count = 0
    start = time.time()

    for i, ex in enumerate(tqdm(examples, desc="Evaluating")):
        src_text = ex["src_text"]
        tgt_text = ex["tgt_text"]
        rxn_class = ex.get("reaction_class", 0)

        gt = normalize_reactants(tgt_text)
        if not gt:
            continue

        # Extract product from src_text
        if "|" in src_text:
            product = src_text.split("|")[0]
        else:
            product = src_text

        # Prepend reaction class
        if rxn_class and 1 <= rxn_class <= 10:
            src_input = f"<RXN_{rxn_class}> {src_text}"
        else:
            src_input = src_text

        src_ids = torch.tensor(
            [tokenizer.encode(src_input, max_len=256)],
            dtype=torch.long, device=device,
        )
        seg_ids = torch.tensor(
            [tokenizer.get_segment_ids(tokenizer.encode(src_input, max_len=256))],
            dtype=torch.long, device=device,
        )

        # Beam search
        beam_results = model.generate_beam(
            src_ids, tokenizer.bos_token_id, tokenizer.eos_token_id,
            beam_size=beam_size, max_len=max_len, segment_ids=seg_ids,
        )[0]

        # Extract predictions
        predictions = []
        unique_strs = set()
        for token_ids, score in beam_results:
            pred_str = tokenizer.decode(token_ids)
            total_predictions += 1

            if check_all_valid(pred_str):
                valid_count += 1

            norm = normalize_reactants(pred_str)
            if norm and norm not in [p for p in predictions]:
                predictions.append(norm)
            unique_strs.add(pred_str)

        total_unique_beams += len(unique_strs)
        total_beam_count += len(beam_results)
        total += 1

        # Top-k exact match
        if predictions and predictions[0] == gt:
            top1 += 1
        if gt in predictions[:3]:
            top3 += 1
        if gt in predictions[:5]:
            top5 += 1
        if gt in predictions[:10]:
            top10 += 1

        # Tanimoto similarity (top-1 prediction vs ground truth)
        if predictions:
            tani = compute_reactant_tanimoto(predictions[0], gt)
            total_tanimoto += tani

        # Round-trip verification
        if fwd_model and predictions:
            if forward_roundtrip(fwd_model, fwd_tokenizer, predictions[0], product, device):
                rt_top1 += 1
            for pred in predictions[:3]:
                if forward_roundtrip(fwd_model, fwd_tokenizer, pred, product, device):
                    rt_top3 += 1
                    break

        # Periodic logging
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            logger.info(
                f"Step {i+1}/{len(examples)} | "
                f"Top-1: {top1/total:.4f} | Top-3: {top3/total:.4f} | "
                f"Top-5: {top5/total:.4f} | Valid: {valid_count/max(total_predictions,1)*100:.1f}% | "
                f"Tani: {total_tanimoto/total:.3f} | "
                f"Diversity: {total_unique_beams/max(total_beam_count,1):.2f} | "
                f"{(i+1)/elapsed:.1f} s/s"
            )

    elapsed = time.time() - start
    validity_rate = valid_count / max(total_predictions, 1)
    avg_tanimoto = total_tanimoto / max(total, 1)
    avg_diversity = total_unique_beams / max(total_beam_count, 1)

    print("\n" + "=" * 60)
    print("RETROTRANSFORMER V2 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {total}")
    print(f"Beam size: {beam_size}")
    print(f"Time: {elapsed:.1f}s ({total/max(elapsed,1):.1f} samples/s)")
    print()
    print("--- Exact Match ---")
    print(f"  Top-1:  {top1/max(total,1):.4f} ({top1}/{total})")
    print(f"  Top-3:  {top3/max(total,1):.4f} ({top3}/{total})")
    print(f"  Top-5:  {top5/max(total,1):.4f} ({top5}/{total})")
    print(f"  Top-10: {top10/max(total,1):.4f} ({top10}/{total})")
    print()
    if fwd_model:
        print("--- Round-Trip (Forward Model) ---")
        print(f"  Top-1 RT: {rt_top1/max(total,1):.4f} ({rt_top1}/{total})")
        print(f"  Top-3 RT: {rt_top3/max(total,1):.4f} ({rt_top3}/{total})")
        print()
    print("--- Quality Metrics ---")
    print(f"  SMILES validity:   {validity_rate*100:.1f}% ({valid_count}/{total_predictions})")
    print(f"  Avg Tanimoto:      {avg_tanimoto:.4f}")
    print(f"  Beam diversity:    {avg_diversity:.2f} unique / {beam_size} beams")
    print("=" * 60)

    # Save results
    results = {
        "checkpoint": str(checkpoint_path),
        "total": total,
        "beam_size": beam_size,
        "top1": top1 / max(total, 1),
        "top3": top3 / max(total, 1),
        "top5": top5 / max(total, 1),
        "top10": top10 / max(total, 1),
        "rt_top1": rt_top1 / max(total, 1) if fwd_model else None,
        "rt_top3": rt_top3 / max(total, 1) if fwd_model else None,
        "validity_rate": validity_rate,
        "avg_tanimoto": avg_tanimoto,
        "beam_diversity": avg_diversity,
    }
    results_file = checkpoint_path.parent / "eval_v2_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
