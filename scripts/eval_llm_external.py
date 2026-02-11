"""Evaluate LLM on external USPTO-50K test split (truly unseen data).

Downloads the standard USPTO-50K test.csv from RetroXpert GitHub,
preprocesses reactions through our edit extraction pipeline,
and evaluates the LLM on reactions it has NEVER seen during training.

Usage:
    python scripts/eval_llm_external.py
    python scripts/eval_llm_external.py --max-samples 500
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

import click
import torch
from tqdm import tqdm

from rasyn.models.llm.generate import tokenize_prompt_for_inference
from rasyn.preprocess.build_edit_dataset import build_training_example

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

TEST_CSV_URL = "https://raw.githubusercontent.com/uta-smile/RetroXpert/main/data/USPTO50K/canonicalized_csv/test.csv"


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
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    canon_parts = []
    for p in parts:
        c = canonicalize_smiles(p.strip())
        if c:
            canon_parts.append(c)
    return ".".join(sorted(canon_parts))


def download_test_csv(output_path: Path) -> Path:
    """Download standard USPTO-50K test.csv if not cached."""
    if output_path.exists():
        logger.info(f"Using cached test CSV: {output_path}")
        return output_path

    logger.info(f"Downloading USPTO-50K test.csv from {TEST_CSV_URL}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TEST_CSV_URL, str(output_path))
    logger.info(f"Downloaded to {output_path}")
    return output_path


def load_and_preprocess_test_data(
    csv_path: Path,
    training_products: set[str] | None = None,
) -> list[dict]:
    """Load test.csv, run edit extraction, and filter out training reactions.

    Args:
        csv_path: Path to test.csv with columns: id, class, rxn_smiles
        training_products: Set of canonical product SMILES from training set
                          to filter out any potential overlap.

    Returns:
        List of dicts with 'prompt', 'completion', 'metadata', 'reaction_class'.
    """
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

        # Check for overlap with training data
        if training_products is not None:
            # Extract product from reaction SMILES
            parts = rxn_smiles.split(">>")
            if len(parts) == 2:
                product = canonicalize_smiles(parts[1].strip())
                if product in training_products:
                    skipped_overlap += 1
                    continue

        # Run through our preprocessing pipeline
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
    """Load canonical product SMILES from training data for overlap detection."""
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
                product = prod_match.group(1).strip()
                canon = canonicalize_smiles(product)
                if canon:
                    products.add(canon)

    logger.info(f"Loaded {len(products)} unique training products for overlap check")
    return products


@click.command()
@click.option("--checkpoint", default="checkpoints/llm/uspto50k_v6/final")
@click.option("--max-samples", default=5000, type=int)
@click.option("--num-beams", default=10, type=int)
@click.option("--num-beam-groups", default=5, type=int)
@click.option("--diversity-penalty", default=1.0, type=float)
@click.option("--max-new-tokens", default=256, type=int)
@click.option("--check-overlap/--no-check-overlap", default=True,
              help="Filter out reactions whose products appear in training data")
@click.option("--device", default="auto")
def main(checkpoint, max_samples, num_beams, num_beam_groups, diversity_penalty, max_new_tokens, check_overlap, device):
    """Evaluate LLM on external USPTO-50K test split."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = Path(checkpoint)

    # Download test data
    test_csv = PROJECT_ROOT / "data" / "external" / "uspto50k_test.csv"
    download_test_csv(test_csv)

    # Load training products for overlap check
    training_products = None
    if check_overlap:
        train_data = PROJECT_ROOT / "data" / "processed" / "uspto50k" / "edit_conditioned_train.jsonl"
        training_products = load_training_products(train_data)

    # Preprocess test data
    test_examples = load_and_preprocess_test_data(test_csv, training_products)

    if max_samples < len(test_examples):
        test_examples = test_examples[:max_samples]
    logger.info(f"Evaluating on {len(test_examples)} truly unseen examples")

    # Load model
    logger.info(f"Loading model from {checkpoint}...")
    from transformers import AutoTokenizer
    from peft import PeftModel
    from rasyn.models.llm.model import load_rsgpt_model

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
    logger.info(f"Model loaded on {device}")

    # Evaluate
    top1 = top3 = top5 = top10 = 0
    total = 0
    invalid_count = 0
    total_unique_preds = 0
    start_time = time.time()

    for i, ex in enumerate(tqdm(test_examples, desc="Evaluating")):
        prompt = ex["prompt"]
        gt_completion = ex["completion"]
        gt_normalized = normalize_reactants(gt_completion)

        if not gt_normalized:
            continue

        inputs = tokenize_prompt_for_inference(prompt, tokenizer, max_length=512, device=device)

        with torch.no_grad():
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=min(num_beams, 10),
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            if num_beam_groups > 1:
                gen_kwargs["num_beam_groups"] = num_beam_groups
                gen_kwargs["diversity_penalty"] = diversity_penalty
                gen_kwargs["trust_remote_code"] = True
            outputs = model.generate(**gen_kwargs)

        predictions = []
        for output_ids in outputs:
            text = tokenizer.decode(output_ids, skip_special_tokens=False)
            if "<OUT>" in text:
                completion = text.split("<OUT>")[-1]
                for stop in ["</s>", "<EOS>", "<pad>", "<PAD>"]:
                    completion = completion.split(stop)[0]
                completion = completion.replace("<unk>", "").strip()
            else:
                completion = ""

            if completion:
                norm = normalize_reactants(completion)
                if norm:
                    predictions.append(norm)
                else:
                    invalid_count += 1
            else:
                invalid_count += 1

        # Deduplicate
        seen = set()
        unique_preds = []
        for p in predictions:
            if p not in seen:
                seen.add(p)
                unique_preds.append(p)

        total += 1
        total_unique_preds += len(unique_preds)

        if len(unique_preds) >= 1 and unique_preds[0] == gt_normalized:
            top1 += 1
        if gt_normalized in unique_preds[:3]:
            top3 += 1
        if gt_normalized in unique_preds[:5]:
            top5 += 1
        if gt_normalized in unique_preds[:10]:
            top10 += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_unique = total_unique_preds / total
            logger.info(
                f"Step {i+1}/{len(test_examples)} | "
                f"Top-1: {top1/total:.3f} | Top-3: {top3/total:.3f} | "
                f"Top-5: {top5/total:.3f} | Top-10: {top10/total:.3f} | "
                f"Diversity: {avg_unique:.1f} | Rate: {(i+1)/elapsed:.1f} samples/s"
            )

    elapsed = time.time() - start_time
    avg_diversity = total_unique_preds / max(total, 1)

    print("\n" + "=" * 60)
    print("LLM EXTERNAL TEST EVALUATION (TRULY UNSEEN DATA)")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Data: Standard USPTO-50K test split (RetroXpert)")
    print(f"Samples evaluated: {total}")
    print(f"Overlap filtered: {check_overlap}")
    print(f"Beam config: {num_beams} beams, {num_beam_groups} groups, penalty={diversity_penalty}")
    print(f"Invalid generations: {invalid_count}")
    print(f"Avg unique predictions per sample: {avg_diversity:.1f}")
    print(f"Time: {elapsed:.1f}s ({total/max(elapsed,1):.1f} samples/s)")
    print()
    print(f"  Top-1 accuracy:  {top1/max(total,1):.4f} ({top1}/{total})")
    print(f"  Top-3 accuracy:  {top3/max(total,1):.4f} ({top3}/{total})")
    print(f"  Top-5 accuracy:  {top5/max(total,1):.4f} ({top5}/{total})")
    print(f"  Top-10 accuracy: {top10/max(total,1):.4f} ({top10}/{total})")
    print("=" * 60)

    results = {
        "checkpoint": str(checkpoint),
        "data": "USPTO-50K standard test split (external)",
        "total": total,
        "overlap_filtered": check_overlap,
        "num_beams": num_beams,
        "num_beam_groups": num_beam_groups,
        "diversity_penalty": diversity_penalty,
        "top1": top1 / max(total, 1),
        "top3": top3 / max(total, 1),
        "top5": top5 / max(total, 1),
        "top10": top10 / max(total, 1),
        "avg_diversity": avg_diversity,
        "invalid_count": invalid_count,
        "elapsed_seconds": elapsed,
    }
    results_path = checkpoint / "eval_external_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
