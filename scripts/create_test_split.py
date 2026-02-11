"""Create proper train/val/test splits for both models.

Splits reactions by rxn_id to prevent data leakage. Produces:
  - edit_conditioned_{train,val,test}.jsonl  (for LLM)
  - augmented_{val,test}.jsonl               (for RetroTransformer v2 eval)

The augmented train data should be rebuilt separately using build_augmented_dataset.py
on the train split only.

Usage:
    python scripts/create_test_split.py
    python scripts/create_test_split.py --val-frac 0.1 --test-frac 0.1
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def canonicalize_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return smiles


def canonicalize_multi(smiles_str: str, separator: str = " . ") -> str:
    parts = smiles_str.split(separator)
    canon_parts = []
    for p in parts:
        p = p.strip()
        if p:
            c = canonicalize_smiles(p)
            if c:
                canon_parts.append(c)
    canon_parts.sort(key=lambda s: (-len(s), s))
    return separator.join(canon_parts)


def edit_conditioned_to_augmented(ex: dict, rxn_class: int = 0) -> dict | None:
    """Convert edit_conditioned format to augmented format (canonical only)."""
    prompt = ex["prompt"]
    completion = ex["completion"]
    rxn_id = ex.get("metadata", {}).get("rxn_id", "")

    prod_match = re.search(r"<PROD>\s+(.+?)\s+<EDIT>", prompt)
    synth_match = re.search(r"<SYNTHONS>\s+(.+?)\s+<LG_HINTS>", prompt)

    product = prod_match.group(1).strip() if prod_match else ""
    synthons = synth_match.group(1).strip() if synth_match else ""

    if not product or not completion:
        return None

    canon_product = canonicalize_smiles(product)
    tgt_text = canonicalize_multi(completion)
    if not tgt_text:
        return None

    if synthons:
        canon_synthons = canonicalize_multi(synthons)
        src_text = f"{canon_product}|{canon_synthons}"
    else:
        src_text = canon_product

    return {
        "src_text": src_text,
        "tgt_text": tgt_text,
        "reaction_class": rxn_class,
        "rxn_id": rxn_id,
        "augment_idx": 0,
    }


@click.command()
@click.option("--edit-data", default="data/processed/uspto50k/edit_conditioned_train.jsonl")
@click.option("--reactions-data", default="data/processed/uspto50k/reactions.jsonl")
@click.option("--output-dir", default="data/processed/uspto50k")
@click.option("--val-frac", default=0.1, type=float)
@click.option("--test-frac", default=0.1, type=float)
@click.option("--seed", default=42, type=int)
def main(edit_data, reactions_data, output_dir, val_frac, test_frac, seed):
    """Create train/val/test splits for both models."""
    edit_data_path = PROJECT_ROOT / edit_data
    reactions_path = PROJECT_ROOT / reactions_data
    output_dir = PROJECT_ROOT / output_dir

    # Load reaction classes
    rxn_classes = {}
    if reactions_path.exists():
        logger.info(f"Loading reaction classes from {reactions_path}...")
        with open(reactions_path) as f:
            for line in f:
                rxn = json.loads(line.strip())
                rxn_id = rxn.get("id", "")
                rxn_class = rxn.get("reaction_class")
                if rxn_id and rxn_class is not None:
                    rxn_classes[rxn_id] = int(rxn_class)
        logger.info(f"  Loaded {len(rxn_classes)} reaction classes")

    # Load edit-conditioned data
    logger.info(f"Loading edit-conditioned data from {edit_data_path}...")
    examples = []
    with open(edit_data_path) as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    logger.info(f"  Loaded {len(examples)} examples")

    # Group by rxn_id
    rxn_groups: dict[str, list[dict]] = {}
    for ex in examples:
        rxn_id = ex.get("metadata", {}).get("rxn_id", str(id(ex)))
        if rxn_id not in rxn_groups:
            rxn_groups[rxn_id] = []
        rxn_groups[rxn_id].append(ex)

    # Split by rxn_id
    unique_rxn_ids = sorted(rxn_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_rxn_ids)

    n_test = int(len(unique_rxn_ids) * test_frac)
    n_val = int(len(unique_rxn_ids) * val_frac)

    test_rxn_ids = set(unique_rxn_ids[:n_test])
    val_rxn_ids = set(unique_rxn_ids[n_test:n_test + n_val])
    train_rxn_ids = set(unique_rxn_ids[n_test + n_val:])

    logger.info(f"Split: {len(train_rxn_ids)} train, {len(val_rxn_ids)} val, {len(test_rxn_ids)} test reactions")

    # Write edit-conditioned splits
    splits = {
        "train": train_rxn_ids,
        "val": val_rxn_ids,
        "test": test_rxn_ids,
    }

    for split_name, split_ids in splits.items():
        # Edit-conditioned (for LLM)
        ec_path = output_dir / f"edit_conditioned_{split_name}.jsonl"
        ec_count = 0
        with open(ec_path, "w") as f:
            for rxn_id in sorted(split_ids):
                for ex in rxn_groups.get(rxn_id, []):
                    f.write(json.dumps(ex) + "\n")
                    ec_count += 1
        logger.info(f"  {split_name}: {ec_count} edit-conditioned examples -> {ec_path}")

        # Augmented format (canonical only, for retro v2 eval on val/test)
        if split_name in ("val", "test"):
            aug_path = output_dir / f"augmented_{split_name}.jsonl"
            aug_count = 0
            with open(aug_path, "w") as f:
                for rxn_id in sorted(split_ids):
                    for ex in rxn_groups.get(rxn_id, []):
                        rxn_class = rxn_classes.get(rxn_id, 0)
                        aug_ex = edit_conditioned_to_augmented(ex, rxn_class)
                        if aug_ex is not None:
                            f.write(json.dumps(aug_ex) + "\n")
                            aug_count += 1
                        break  # Only one canonical version per reaction
            logger.info(f"  {split_name}: {aug_count} augmented examples -> {aug_path}")

    # Summary
    print("\n" + "=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total reactions: {len(unique_rxn_ids)}")
    print(f"Train: {len(train_rxn_ids)} ({len(train_rxn_ids)/len(unique_rxn_ids)*100:.1f}%)")
    print(f"Val:   {len(val_rxn_ids)} ({len(val_rxn_ids)/len(unique_rxn_ids)*100:.1f}%)")
    print(f"Test:  {len(test_rxn_ids)} ({len(test_rxn_ids)/len(unique_rxn_ids)*100:.1f}%)")
    print()
    print("Files created:")
    print(f"  LLM:     edit_conditioned_{{train,val,test}}.jsonl")
    print(f"  Retro:   augmented_{{val,test}}.jsonl (canonical only)")
    print()
    print("Next steps:")
    print("  1. Rebuild augmented train data from the train split:")
    print("     python scripts/build_augmented_dataset.py \\")
    print("       --edit-data data/processed/uspto50k/edit_conditioned_train.jsonl \\")
    print("       --output data/processed/uspto50k/augmented_train.jsonl")
    print("  2. Re-train models on the train split only (for proper held-out eval)")
    print("=" * 60)


if __name__ == "__main__":
    main()
