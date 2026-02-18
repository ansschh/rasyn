#!/usr/bin/env python
"""Index USPTO-FULL reactions into the reaction_index table.

Reads the raw USPTO-FULL CSV (PatentNumber,Year,reactions) directly.
Format: reactants>reagents>products  ('>'-separated components)

Usage:
    python scripts/index_uspto_full.py --input data/raw/uspto_full.csv [--clear]

Run this on the EC2 server after SCP-ing the CSV from RunPod.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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
    bits = fp_array.astype(np.uint8)
    packed = np.packbits(bits)
    return packed.tobytes()


def parse_reaction(rxn_str: str) -> tuple[list[str], list[str], list[str]] | None:
    """Parse 'reactants>reagents>products' into (reactants, reagents, products).

    Returns None if the format is invalid.
    """
    parts = rxn_str.split(">")
    if len(parts) != 3:
        return None

    reactants = [s.strip() for s in parts[0].split(".") if s.strip()]
    products = [s.strip() for s in parts[2].split(".") if s.strip()]

    if not reactants or not products:
        return None

    return reactants, parts[1].split("."), products


def main():
    parser = argparse.ArgumentParser(description="Index USPTO-FULL reactions")
    parser.add_argument("--input", type=str, default="data/raw/uspto_full.csv")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--clear", action="store_true", help="Clear existing index first")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to index (0 = all)")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("SCP the file from RunPod: scp -i ~/.ssh/runpod_key beiyo61gthf01w-64411fb0@ssh.runpod.io:/workspace/rasyn/data/raw/uspto_full.csv /opt/rasyn/app/data/raw/")
        sys.exit(1)

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    from sqlalchemy.orm import Session
    from rasyn.db.engine import sync_engine
    from rasyn.db.models import Base, ReactionIndex

    logger.info("Creating tables if needed...")
    Base.metadata.create_all(sync_engine)

    session = Session(sync_engine)
    existing_count = session.query(ReactionIndex).count()

    if args.clear and existing_count > 0:
        logger.info(f"Clearing {existing_count} existing rows...")
        session.query(ReactionIndex).delete()
        session.commit()
        existing_count = 0
    elif existing_count > 0:
        logger.info(f"reaction_index already has {existing_count} rows. Use --clear to replace.")
        logger.info("Appending new reactions (duplicates may occur)...")
    session.close()

    logger.info(f"Reading USPTO-FULL from {input_path}...")
    t0 = time.perf_counter()
    batch = []
    indexed = 0
    skipped = 0
    total_read = 0

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_read += 1
            if args.max_rows > 0 and total_read > args.max_rows:
                break

            patent = row.get("PatentNumber", "")
            year_str = row.get("Year", "")
            rxn_str = row.get("reactions", "")

            if not rxn_str:
                skipped += 1
                continue

            parsed = parse_reaction(rxn_str)
            if parsed is None:
                skipped += 1
                continue

            reactants, _reagents, products = parsed

            # Take the first (major) product
            product = products[0]

            fp = compute_reaction_fp(product, reactants)
            if fp is None:
                skipped += 1
                continue

            fp_bytes = pack_fingerprint(fp)
            rxn_smiles = ".".join(reactants) + ">>" + product

            # Format source with patent number for URL generation
            source = f"USPTO ({patent})" if patent else "USPTO-FULL"

            year = None
            if year_str:
                try:
                    year = int(year_str)
                except ValueError:
                    pass

            batch.append(ReactionIndex(
                reaction_smiles=rxn_smiles,
                product_smiles=product,
                reactants_smiles=".".join(reactants),
                rxn_class=None,
                source=source,
                year=year,
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
                if indexed % 10000 < args.batch_size:
                    logger.info(f"  Indexed {indexed:,}/{total_read:,} read ({rate:.0f}/s, {skipped:,} skipped)")

    if batch:
        session = Session(sync_engine)
        session.add_all(batch)
        session.commit()
        session.close()
        indexed += len(batch)

    elapsed = time.perf_counter() - t0
    logger.info(f"Done! Indexed {indexed:,} reactions in {elapsed:.1f}s ({skipped:,} skipped)")
    logger.info(f"Average rate: {indexed / elapsed:.0f} reactions/s")

    session = Session(sync_engine)
    count = session.query(ReactionIndex).count()
    session.close()
    logger.info(f"Total reaction_index rows: {count:,}")


if __name__ == "__main__":
    main()
