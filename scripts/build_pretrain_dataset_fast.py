"""Build pretraining dataset from USPTO-FULL without atom mapping.

For pretraining, we don't need edit conditioning (synthons). We just need
product -> reactants pairs. This script reads the raw CSV directly, skips
atom mapping entirely, and creates augmented training data in minutes
instead of hours.

Usage:
    python -u scripts/build_pretrain_dataset_fast.py
    python -u scripts/build_pretrain_dataset_fast.py --n-augments 5
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import click
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def canonicalize(smiles: str) -> str | None:
    """Canonicalize SMILES, return None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None


def canonicalize_multi(smiles: str) -> str | None:
    """Canonicalize multi-component SMILES (dot-separated)."""
    parts = smiles.split(".")
    canon_parts = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        c = canonicalize(p)
        if c is None:
            return None
        canon_parts.append(c)
    if not canon_parts:
        return None
    return ".".join(sorted(canon_parts))


def randomize_smiles(smiles: str) -> str:
    """Generate a random SMILES representation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, doRandom=True)


def parse_reaction(rxn_smiles: str) -> tuple[str, str] | None:
    """Parse reaction SMILES into (product, reactants).

    Handles both >> and > separated formats.
    Returns canonicalized (product, reactants) or None if invalid.
    """
    # Try >> first (most common)
    if ">>" in rxn_smiles:
        parts = rxn_smiles.split(">>")
        if len(parts) == 2:
            reactants_raw, products_raw = parts[0].strip(), parts[1].strip()
        else:
            return None
    elif ">" in rxn_smiles:
        # reactants>reagents>products format
        parts = rxn_smiles.split(">")
        if len(parts) == 3:
            reactants_raw, _, products_raw = parts[0].strip(), parts[1].strip(), parts[2].strip()
        else:
            return None
    else:
        return None

    # Canonicalize
    product = canonicalize(products_raw)
    reactants = canonicalize_multi(reactants_raw)

    if product is None or reactants is None:
        return None

    return product, reactants


@click.command()
@click.option("--input-csv", default="data/raw/uspto_full.csv",
              help="Path to raw USPTO-FULL CSV")
@click.option("--output", default="data/processed/uspto_full/pretrain_3x_train.jsonl")
@click.option("--n-augments", default=3, type=int,
              help="Number of augmented copies per reaction")
@click.option("--max-reactions", default=0, type=int,
              help="Max reactions to process (0 = all)")
def main(input_csv, output, n_augments, max_reactions):
    """Build pretraining dataset without atom mapping."""
    csv_path = PROJECT_ROOT / input_csv
    output_path = PROJECT_ROOT / output

    if not csv_path.exists():
        logger.error(f"Input CSV not found: {csv_path}")
        logger.error("Run: python -u scripts/download_data.py --datasets uspto_full")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read CSV and find the reaction column
    logger.info(f"Reading {csv_path}...")
    reactions = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rxn_col = None
        for col in reader.fieldnames:
            if "reaction" in col.lower() or "rxn" in col.lower() or col.lower() == "reactions":
                rxn_col = col
                break
        if rxn_col is None:
            # Try first column
            rxn_col = reader.fieldnames[0]
        logger.info(f"Using column '{rxn_col}' for reaction SMILES")

        for row in reader:
            rxn = row[rxn_col].strip()
            if rxn:
                reactions.append(rxn)
            if max_reactions > 0 and len(reactions) >= max_reactions:
                break

    logger.info(f"Read {len(reactions)} reactions")

    # Parse and canonicalize
    total_written = 0
    total_failed = 0

    with open(output_path, "w") as fout:
        for rxn_smiles in tqdm(reactions, desc="Building pretrain dataset"):
            parsed = parse_reaction(rxn_smiles)
            if parsed is None:
                total_failed += 1
                continue

            product, reactants = parsed

            # Write canonical version
            example = {
                "src_text": product,
                "tgt_text": reactants,
                "reaction_class": 0,  # No class info for USPTO-FULL
                "augment_idx": 0,
            }
            fout.write(json.dumps(example) + "\n")
            total_written += 1

            # Write augmented copies
            for aug_idx in range(1, n_augments + 1):
                rand_product = randomize_smiles(product)
                aug_example = {
                    "src_text": rand_product,
                    "tgt_text": reactants,  # Always canonical
                    "reaction_class": 0,
                    "augment_idx": aug_idx,
                }
                fout.write(json.dumps(aug_example) + "\n")
                total_written += 1

    logger.info(f"\nPretrain dataset built:")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Total examples: {total_written}")
    logger.info(f"  Valid reactions: {len(reactions) - total_failed}")
    logger.info(f"  Failed: {total_failed} ({total_failed/len(reactions):.1%})")
    logger.info(f"  Augmentation: {n_augments}x + 1 = {n_augments + 1}x")


if __name__ == "__main__":
    main()
