"""Run the full preprocessing pipeline on a USPTO dataset.

Usage:
    python scripts/preprocess_all.py --dataset uspto50k
    python scripts/preprocess_all.py --dataset uspto_full
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from rasyn.preprocess.atom_mapping import has_atom_mapping, map_reactions_batch
from rasyn.preprocess.build_edit_dataset import build_edit_dataset
from rasyn.preprocess.build_lg_cog import build_lg_cooccurrence, save_lg_cog
from rasyn.preprocess.build_lg_vocab import (
    build_lg_vocabulary,
    save_lg_counts,
    save_lg_vocab,
)
from rasyn.preprocess.canonicalize import (
    canonicalize_smiles,
    parse_reaction_smiles,
)
from rasyn.preprocess.extract_edits import extract_edits_from_reaction

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VOCAB_DIR = PROJECT_ROOT / "data" / "vocab"


def load_uspto_csv(path: Path) -> pd.DataFrame:
    """Load a USPTO CSV file and standardize column names."""
    df = pd.read_csv(path)

    # Standardize column names (different datasets use different conventions)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if "rxn" in col_lower or "reaction" in col_lower and "smiles" in col_lower:
            col_map[col] = "rxn_smiles"
        elif col_lower in ("reactionsmi", "rxn_smi"):
            col_map[col] = "rxn_smiles"
        elif col_lower == "class":
            col_map[col] = "reaction_class"
        elif col_lower == "id":
            col_map[col] = "id"

    # If no rxn_smiles column found, check for common alternatives
    if "rxn_smiles" not in col_map.values():
        # Some datasets just have columns like 'reactants' and 'products'
        for col in df.columns:
            if "canonical_rxn" in col.lower():
                col_map[col] = "rxn_smiles"
                break

    df = df.rename(columns=col_map)

    # Ensure we have an ID column
    if "id" not in df.columns:
        df["id"] = [f"rxn_{i}" for i in range(len(df))]

    # Ensure we have rxn_smiles column
    if "rxn_smiles" not in df.columns:
        # Try to find it
        for col in df.columns:
            sample = str(df[col].iloc[0])
            if ">>" in sample or (">") in sample:
                df["rxn_smiles"] = df[col]
                logger.info(f"Using column '{col}' as rxn_smiles")
                break

    return df


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["uspto50k", "uspto_full", "uspto_mit"]),
    default="uspto50k",
)
@click.option("--raw-dir", type=click.Path(), default=str(RAW_DIR))
@click.option("--output-dir", type=click.Path(), default=str(PROCESSED_DIR))
@click.option("--vocab-dir", type=click.Path(), default=str(VOCAB_DIR))
@click.option("--build-vocab-from", type=str, default=None,
              help="Dataset to build LG vocab from (default: same as --dataset)")
def main(dataset, raw_dir, output_dir, vocab_dir, build_vocab_from):
    """Run full preprocessing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir) / dataset
    vocab_dir = Path(vocab_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw data
    csv_path = raw_dir / f"{dataset}.csv"
    if not csv_path.exists():
        logger.error(f"Dataset not found: {csv_path}. Run download_data.py first.")
        return

    logger.info(f"Loading {csv_path}...")
    df = load_uspto_csv(csv_path)
    logger.info(f"Loaded {len(df)} reactions")

    # Step 1.5: Add atom mapping if missing
    sample_rxn = str(df["rxn_smiles"].iloc[0])
    if not has_atom_mapping(sample_rxn.split(">>")[0]):
        logger.info("Reactions lack atom mapping. Running atom mapper...")
        all_rxns = df["rxn_smiles"].tolist()
        mapped = map_reactions_batch(all_rxns, batch_size=10)
        mapped_count = sum(1 for m in mapped if m is not None)
        logger.info(f"Atom mapping complete: {mapped_count}/{len(all_rxns)} mapped")
        df["rxn_smiles_original"] = df["rxn_smiles"]
        df["rxn_smiles"] = [m if m else orig for m, orig in zip(mapped, all_rxns)]
    else:
        logger.info("Reactions already have atom mapping")

    # Step 2: Extract edits for all reactions
    logger.info("Extracting edits from all reactions...")
    records = []
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting edits"):
        rxn_smi = row.get("rxn_smiles", "")
        if not rxn_smi or not isinstance(rxn_smi, str):
            failed += 1
            continue

        labels = extract_edits_from_reaction(rxn_smi)
        if labels is None:
            failed += 1
            continue

        reactants_list, reagents_list, products_list = parse_reaction_smiles(rxn_smi)

        record = {
            "id": row.get("id", ""),
            "rxn_smiles": rxn_smi,
            "product_smiles": canonicalize_smiles(".".join(products_list), remove_mapping=True),
            "reactants_smiles": [canonicalize_smiles(r, remove_mapping=True) for r in reactants_list],
            "reagents_smiles": [canonicalize_smiles(r, remove_mapping=True) for r in reagents_list],
            "reaction_class": row.get("reaction_class"),
            "changed_bonds": labels.changed_bonds,
            "synthon_smiles": labels.synthon_smiles,
            "leaving_groups": labels.leaving_groups,
            "edit_tokens": labels.edit_tokens,
        }
        records.append(record)

    logger.info(f"Successfully extracted {len(records)} / {len(df)} reactions ({failed} failed)")

    # Step 3: Save processed records
    records_path = output_dir / "reactions.jsonl"
    with open(records_path, "w") as f:
        for rec in records:
            # Convert tuples to lists for JSON serialization
            rec_copy = dict(rec)
            rec_copy["changed_bonds"] = [list(b) for b in rec_copy["changed_bonds"]]
            f.write(json.dumps(rec_copy) + "\n")
    logger.info(f"Saved {len(records)} records to {records_path}")

    # Step 4: Build LG vocabulary
    vocab_source = build_vocab_from or dataset
    logger.info(f"Building LG vocabulary from {vocab_source}...")

    rxn_smiles_for_vocab = [r["rxn_smiles"] for r in records]
    lg_to_idx, lg_counter = build_lg_vocabulary(rxn_smiles_for_vocab, min_count=1)
    save_lg_vocab(lg_to_idx, vocab_dir / "lg_vocab.json")
    save_lg_counts(lg_counter, vocab_dir / "lg_counts.json")

    # Step 5: Build LG co-occurrence graph
    logger.info("Building LG co-occurrence graph...")
    cog_matrix = build_lg_cooccurrence(rxn_smiles_for_vocab, lg_to_idx)
    save_lg_cog(cog_matrix, vocab_dir / "lg_cog.npy")

    # Step 6: Build edit-conditioned LLM training dataset
    logger.info("Building edit-conditioned LLM training dataset...")
    llm_records = [{"rxn_smiles": r["rxn_smiles"], "id": r["id"]} for r in records]
    n_examples = build_edit_dataset(
        llm_records,
        output_dir / "edit_conditioned_train.jsonl",
    )

    # Step 7: Summary
    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"  Reactions processed: {len(records)}")
    logger.info(f"  LG vocabulary size:  {len(lg_to_idx)}")
    logger.info(f"  LGCoG matrix shape:  {cog_matrix.shape}")
    logger.info(f"  LLM training examples: {n_examples}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Vocab directory:  {vocab_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
