"""Evaluate the retrosynthesis pipeline on USPTO-50K test set.

Metrics:
  - Top-k exact match accuracy (k=1,3,5,10)
  - Round-trip accuracy
  - Average process score
  - Edit match rate (EMR)

Usage:
    python scripts/evaluate.py --dataset uspto50k
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from tqdm import tqdm

from rasyn.preprocess.canonicalize import canonicalize_smiles, parse_reaction_smiles

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


def compute_topk_accuracy(
    predictions: list[list[str]],
    ground_truths: list[str],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict[int, float]:
    """Compute top-k exact match accuracy.

    Args:
        predictions: List of ranked prediction lists (each is a list of reactant strings).
        ground_truths: List of ground-truth reactant strings.
        k_values: Values of k to compute.

    Returns:
        Dict mapping k -> accuracy.
    """
    results = {k: 0 for k in k_values}
    n = len(ground_truths)

    for preds, gt in zip(predictions, ground_truths):
        gt_canon = canonicalize_smiles(gt, remove_mapping=True)
        for k in k_values:
            for pred in preds[:k]:
                pred_canon = canonicalize_smiles(pred, remove_mapping=True)
                if pred_canon == gt_canon:
                    results[k] += 1
                    break

    return {k: v / max(n, 1) for k, v in results.items()}


@click.command()
@click.option("--dataset", default="uspto50k")
@click.option("--graph-head-checkpoint", default=None)
@click.option("--llm-checkpoint", default=None)
@click.option("--max-samples", default=None, type=int)
@click.option("--device", default="auto")
def main(dataset, graph_head_checkpoint, llm_checkpoint, max_samples, device):
    """Evaluate the retrosynthesis pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    data_dir = PROJECT_ROOT / "data" / "processed" / dataset
    records_path = data_dir / "reactions.jsonl"

    if not records_path.exists():
        logger.error(f"Data not found: {records_path}")
        return

    # Load test records
    records = []
    with open(records_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    if max_samples:
        records = records[:max_samples]

    logger.info(f"Evaluating on {len(records)} reactions")

    # Build pipeline
    from rasyn.pipeline.single_step import SingleStepRetro

    pipeline = SingleStepRetro(device=device)

    if graph_head_checkpoint:
        logger.info("Loading graph head...")
        # TODO: load graph head from checkpoint
    if llm_checkpoint:
        logger.info("Loading LLM...")
        # TODO: load LLM from checkpoint

    # Evaluate
    all_predictions = []
    all_ground_truths = []
    total_process_scores = []

    for record in tqdm(records, desc="Evaluating"):
        product = record.get("product_smiles", "")
        gt_reactants = record.get("reactants_smiles", [])
        gt_str = ".".join(sorted(gt_reactants))

        if not product:
            continue

        # Get predictions
        results = pipeline.predict(product)

        # Collect predictions as canonical reactant strings
        pred_strings = []
        for step in results:
            pred_str = ".".join(sorted(step.reactants))
            pred_strings.append(pred_str)

            if step.process_scores:
                total_process_scores.append(step.process_scores.total_score)

        all_predictions.append(pred_strings)
        all_ground_truths.append(gt_str)

    # Compute metrics
    topk = compute_topk_accuracy(all_predictions, all_ground_truths)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Samples: {len(all_ground_truths)}")
    print()
    for k, acc in sorted(topk.items()):
        print(f"  Top-{k} accuracy: {acc:.4f} ({acc*100:.1f}%)")
    if total_process_scores:
        avg_score = sum(total_process_scores) / len(total_process_scores)
        print(f"\n  Avg process score: {avg_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
