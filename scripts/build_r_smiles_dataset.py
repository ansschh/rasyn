"""Build R-SMILES (root-aligned) augmented dataset for RetroTransformer v2.

Reads atom-mapped reactions, applies R-SMILES alignment (rooting product and
reactant SMILES at the reaction center atom), then generates augmented copies
with randomized source SMILES (keeping canonical R-SMILES targets).

Usage:
    # Standard R-SMILES dataset with 5x augmentation
    python -u scripts/build_r_smiles_dataset.py

    # 20x augmentation
    python -u scripts/build_r_smiles_dataset.py --n-augments 20

    # Custom output
    python -u scripts/build_r_smiles_dataset.py --output data/processed/uspto50k/r_smiles_train.jsonl
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def randomize_smiles(smiles: str) -> str:
    """Generate a random SMILES representation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, doRandom=True)


def randomize_multi(smiles_str: str, shuffle_order: bool = True) -> str:
    """Randomize each component of a multi-component SMILES string."""
    import random
    parts = smiles_str.split(".")
    randomized = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        randomized.append(randomize_smiles(p))

    if shuffle_order and len(randomized) > 1:
        random.shuffle(randomized)

    return ".".join(randomized)


def load_reactions(data_path: Path) -> list[dict]:
    """Load reactions from reactions.jsonl."""
    reactions = []
    with open(data_path) as f:
        for line in f:
            record = json.loads(line.strip())
            if record.get("atom_mapped_rxn_smiles"):
                reactions.append(record)
    logger.info(f"Loaded {len(reactions)} reactions with atom mapping from {data_path}")
    return reactions


def load_edit_data(data_path: Path) -> dict[str, dict]:
    """Load edit-conditioned data to get synthon/edit info per reaction."""
    import re
    edit_data = {}
    if not data_path.exists():
        return edit_data

    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            rxn_id = ex.get("metadata", {}).get("rxn_id", "")
            if rxn_id:
                # Parse prompt to extract synthons
                prompt = ex.get("prompt", "")
                synth_match = re.search(r"<SYNTHONS>\s*(.+?)(?=\s*<(?:LG_HINTS|CONSTRAINTS|OUT)>)", prompt)
                synthons = synth_match.group(1).strip() if synth_match else ""
                edit_data[rxn_id] = {
                    "synthons": synthons,
                    "prompt": prompt,
                }

    logger.info(f"Loaded edit info for {len(edit_data)} reactions")
    return edit_data


@click.command()
@click.option("--reactions", default="data/processed/uspto50k/reactions.jsonl",
              help="Path to reactions.jsonl with atom-mapped SMILES")
@click.option("--edit-data", default="data/processed/uspto50k/edit_conditioned_train.jsonl",
              help="Path to edit-conditioned data for synthon info")
@click.option("--output", default="data/processed/uspto50k/r_smiles_augmented_train.jsonl")
@click.option("--n-augments", default=5, type=int,
              help="Number of augmented copies per reaction (source-only randomization)")
@click.option("--include-non-rsmiles", is_flag=True,
              help="Also include canonical (non-R-SMILES) version for comparison")
def main(reactions, edit_data, output, n_augments, include_non_rsmiles):
    """Build R-SMILES augmented dataset."""
    from rasyn.preprocess.r_smiles import build_r_smiles_example
    from rasyn.preprocess.canonicalize import canonicalize_smiles, remove_atom_mapping

    reactions_path = PROJECT_ROOT / reactions
    edit_data_path = PROJECT_ROOT / edit_data
    output_path = PROJECT_ROOT / output

    rxn_records = load_reactions(reactions_path)
    edit_info = load_edit_data(edit_data_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_failed = 0

    with open(output_path, "w") as fout:
        for record in tqdm(rxn_records, desc="Building R-SMILES dataset"):
            rxn_smiles = record["atom_mapped_rxn_smiles"]
            rxn_id = record.get("id", "")
            rxn_class = record.get("reaction_class", 0)

            # Build R-SMILES example
            r_example = build_r_smiles_example(rxn_smiles, rxn_id, rxn_class)
            if r_example is None:
                total_failed += 1
                continue

            r_product = r_example["product"]
            r_reactants = r_example["reactants"]

            # Get synthon info if available
            info = edit_info.get(rxn_id, {})
            synthons = info.get("synthons", "")

            # Build source text: product | synthons (R-SMILES aligned)
            if synthons:
                src_text = f"{r_product} | {synthons}"
            else:
                src_text = r_product

            # Write canonical (augment_idx=0)
            example = {
                "src_text": src_text,
                "tgt_text": r_reactants,
                "reaction_class": rxn_class,
                "rxn_id": rxn_id,
                "augment_idx": 0,
                "r_smiles": True,
                "root_atom_map": r_example["root_atom_map"],
            }
            fout.write(json.dumps(example) + "\n")
            total_written += 1

            # Write augmented copies (randomize source, keep canonical R-SMILES target)
            for aug_idx in range(1, n_augments + 1):
                rand_product = randomize_smiles(r_product)

                if synthons:
                    rand_synthons = randomize_multi(synthons, shuffle_order=True)
                    rand_src = f"{rand_product} | {rand_synthons}"
                else:
                    rand_src = rand_product

                aug_example = {
                    "src_text": rand_src,
                    "tgt_text": r_reactants,  # Always canonical R-SMILES
                    "reaction_class": rxn_class,
                    "rxn_id": rxn_id,
                    "augment_idx": aug_idx,
                    "r_smiles": True,
                    "root_atom_map": r_example["root_atom_map"],
                }
                fout.write(json.dumps(aug_example) + "\n")
                total_written += 1

    logger.info(f"\nR-SMILES dataset built:")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Total examples: {total_written}")
    logger.info(f"  Reactions processed: {len(rxn_records) - total_failed}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"  Augmentation: {n_augments}x + 1 canonical = {n_augments + 1}x")
    logger.info(f"  Expected: ~{(len(rxn_records) - total_failed) * (n_augments + 1)} examples")


if __name__ == "__main__":
    main()
