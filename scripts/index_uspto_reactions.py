#!/usr/bin/env python
"""Index preprocessed USPTO-50K reactions into the reaction_index table.

Reads data/processed/reactions.jsonl, computes Morgan difference fingerprints
for each reaction, and inserts them into the PostgreSQL reaction_index table.

Usage:
    python scripts/index_uspto_reactions.py [--batch-size 500] [--input data/processed/reactions.jsonl]

Run this ONCE on the server after setting up the database.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_reaction_fp(product_smiles: str, reactants_smiles: list[str]) -> np.ndarray | None:
    """Compute Morgan difference fingerprint for a reaction."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return None

    product_fp = AllChem.GetMorganFingerprintAsBitVect(product_mol, 2, nBits=2048)
    product_arr = np.zeros(2048, dtype=np.float32)
    for bit in product_fp.GetOnBits():
        product_arr[bit] = 1.0

    reactant_arr = np.zeros(2048, dtype=np.float32)
    for smi in reactants_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        for bit in fp.GetOnBits():
            reactant_arr[bit] = 1.0

    diff = np.abs(product_arr - reactant_arr)
    return diff


def pack_fingerprint(fp_array: np.ndarray) -> bytes:
    """Pack a 2048-element binary array into 256 bytes."""
    bits = fp_array.astype(np.uint8)
    packed = np.packbits(bits)
    return packed.tobytes()


def main():
    parser = argparse.ArgumentParser(description="Index USPTO reactions into reaction_index table")
    parser.add_argument("--input", type=str, default="data/processed/reactions.jsonl",
                        help="Path to reactions.jsonl")
    parser.add_argument("--batch-size", type=int, default=500, help="DB insert batch size")
    parser.add_argument("--source", type=str, default="USPTO-50K", help="Source label")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Run preprocessing first: python scripts/preprocess_all.py")
        sys.exit(1)

    # Suppress RDKit warnings
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    # Set up DB
    from sqlalchemy.orm import Session
    from rasyn.db.engine import sync_engine
    from rasyn.db.models import Base, ReactionIndex

    logger.info("Creating tables if needed...")
    Base.metadata.create_all(sync_engine)

    # Check if already indexed
    session = Session(sync_engine)
    existing_count = session.query(ReactionIndex).count()
    if existing_count > 0:
        logger.warning(f"reaction_index already has {existing_count} rows.")
        response = input("Clear and re-index? (y/N): ").strip().lower()
        if response == "y":
            session.query(ReactionIndex).delete()
            session.commit()
            logger.info("Cleared existing index.")
        else:
            logger.info("Aborting. Use --force or clear the table manually.")
            session.close()
            sys.exit(0)
    session.close()

    # Read reactions
    logger.info(f"Reading reactions from {input_path}...")
    reactions = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reactions.append(json.loads(line))

    logger.info(f"Loaded {len(reactions)} reactions. Computing fingerprints...")

    # Process and insert in batches
    t0 = time.perf_counter()
    batch = []
    indexed = 0
    skipped = 0

    for i, rxn in enumerate(reactions):
        product = rxn.get("product_smiles", "")
        reactants = rxn.get("reactants_smiles", [])
        if isinstance(reactants, str):
            reactants = [r.strip() for r in reactants.split(".") if r.strip()]

        if not product or not reactants:
            skipped += 1
            continue

        fp = compute_reaction_fp(product, reactants)
        if fp is None:
            skipped += 1
            continue

        fp_bytes = pack_fingerprint(fp)

        # Build reaction SMILES
        rxn_smiles = rxn.get("rxn_smiles", "")
        if not rxn_smiles:
            rxn_smiles = ".".join(reactants) + ">>" + product

        batch.append(ReactionIndex(
            reaction_smiles=rxn_smiles,
            product_smiles=product,
            reactants_smiles=".".join(reactants) if isinstance(reactants, list) else reactants,
            rxn_class=rxn.get("reaction_class"),
            source=args.source,
            year=None,
            fingerprint=fp_bytes,
        ))

        if len(batch) >= args.batch_size:
            session = Session(sync_engine)
            session.add_all(batch)
            session.commit()
            session.close()
            indexed += len(batch)
            batch = []
            elapsed = time.perf_counter() - t0
            rate = indexed / elapsed
            logger.info(f"  Indexed {indexed}/{len(reactions)} ({rate:.0f}/s, {skipped} skipped)")

    # Insert remaining
    if batch:
        session = Session(sync_engine)
        session.add_all(batch)
        session.commit()
        session.close()
        indexed += len(batch)

    elapsed = time.perf_counter() - t0
    logger.info(f"Done! Indexed {indexed} reactions in {elapsed:.1f}s ({skipped} skipped)")
    logger.info(f"Average rate: {indexed / elapsed:.0f} reactions/s")

    # Verify
    session = Session(sync_engine)
    count = session.query(ReactionIndex).count()
    session.close()
    logger.info(f"Verification: reaction_index table has {count} rows.")


if __name__ == "__main__":
    main()
