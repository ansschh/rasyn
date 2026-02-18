#!/usr/bin/env python
"""Download USPTO-FULL and index all 1.8M reactions into the database.

Run this on the EC2 server:
    cd /opt/rasyn/app
    source /opt/rasyn/venv/bin/activate
    python -u scripts/download_and_index_full.py [--clear]

Takes ~30 minutes total (download ~2min, indexing ~25min).
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

USPTO_FULL_URL = "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_FULL.csv"


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress reporting."""
    import urllib.request

    if dest.exists():
        size_mb = dest.stat().st_size / 1024 / 1024
        logger.info(f"File already exists: {dest} ({size_mb:.1f} MB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url}...")

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            mb = downloaded / 1024 / 1024
            if block_num % 500 == 0:
                logger.info(f"  {mb:.1f} MB ({pct:.1f}%)")

    urllib.request.urlretrieve(url, str(dest), reporthook=report)
    size_mb = dest.stat().st_size / 1024 / 1024
    logger.info(f"Downloaded {size_mb:.1f} MB to {dest}")


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


def parse_reaction(rxn_str: str) -> tuple[list[str], list[str]] | None:
    """Parse 'reactants>reagents>products' → (reactants, products)."""
    parts = rxn_str.split(">")
    if len(parts) != 3:
        return None
    reactants = [s.strip() for s in parts[0].split(".") if s.strip()]
    products = [s.strip() for s in parts[2].split(".") if s.strip()]
    if not reactants or not products:
        return None
    return reactants, products


def main():
    parser = argparse.ArgumentParser(description="Download USPTO-FULL and index into DB")
    parser.add_argument("--clear", action="store_true", help="Clear existing index first")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / "data" / "raw" / "uspto_full.csv"

    # Step 1: Download
    download_file(USPTO_FULL_URL, csv_path)

    # Step 2: Set up DB
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    from sqlalchemy.orm import Session
    from rasyn.db.engine import sync_engine
    from rasyn.db.models import Base, ReactionIndex

    Base.metadata.create_all(sync_engine)

    session = Session(sync_engine)
    existing_count = session.query(ReactionIndex).count()

    if args.clear and existing_count > 0:
        logger.info(f"Clearing {existing_count:,} existing rows...")
        session.query(ReactionIndex).delete()
        session.commit()
    elif existing_count > 0:
        logger.info(f"reaction_index has {existing_count:,} rows. Use --clear to replace, or we'll append.")
    session.close()

    # Step 3: Index
    logger.info(f"Indexing reactions from {csv_path}...")
    t0 = time.perf_counter()
    batch = []
    indexed = 0
    skipped = 0
    total_read = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_read += 1

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

            reactants, products = parsed
            product = products[0]

            fp = compute_reaction_fp(product, reactants)
            if fp is None:
                skipped += 1
                continue

            fp_bytes = pack_fingerprint(fp)
            rxn_smiles = ".".join(reactants) + ">>" + product

            source = f"USPTO ({patent})" if patent else "USPTO-FULL"
            year = int(year_str) if year_str.isdigit() else None

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
                if indexed % 50000 < args.batch_size:
                    logger.info(f"  Indexed {indexed:,}/{total_read:,} ({rate:.0f}/s, {skipped:,} skipped)")

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
