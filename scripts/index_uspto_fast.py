#!/usr/bin/env python
"""Fast USPTO-50K indexer — downloads and indexes directly, no preprocessing needed.

Downloads the USPTO-50K Schneider dataset (TSV from rxnfp repo), extracts clean
reaction SMILES, computes Morgan difference fingerprints, and inserts into the
reaction_index table.

Usage:
    python scripts/index_uspto_fast.py [--batch-size 500]

This is the recommended way to set up the evidence index on a fresh EC2 server.
The live evidence search (Semantic Scholar + OpenAlex APIs) provides coverage
beyond this local index for patents and papers across all of chemistry.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Schneider USPTO-50K from rxnfp (rxn4chemistry) — reliable, clean format
# Columns: index, original_rxn, rxn_class, source, rxn (clean SMILES), split
SCHNEIDER_50K_URL = "https://raw.githubusercontent.com/rxn4chemistry/rxnfp/master/data/schneider50k.tsv"


def download_tsv(url: str) -> list[dict]:
    """Download a TSV file and parse it into list of dicts."""
    logger.info(f"Downloading {url}...")
    response = urllib.request.urlopen(url, timeout=60)
    content = response.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content), delimiter="\t")
    return list(reader)


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
    parser = argparse.ArgumentParser(description="Download and index USPTO-50K reactions")
    parser.add_argument("--batch-size", type=int, default=500, help="DB insert batch size")
    args = parser.parse_args()

    # Suppress RDKit warnings
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    from sqlalchemy.orm import Session
    from rasyn.db.engine import sync_engine
    from rasyn.db.models import Base, ReactionIndex

    logger.info("Ensuring tables exist...")
    Base.metadata.create_all(sync_engine)

    # Check if already indexed
    session = Session(sync_engine)
    existing_count = session.query(ReactionIndex).count()
    session.close()
    if existing_count > 0:
        logger.info(f"reaction_index already has {existing_count} rows. Skipping.")
        logger.info("To re-index, run: DELETE FROM reaction_index;")
        return

    # Download Schneider 50K (single TSV with all splits)
    all_rows = download_tsv(SCHNEIDER_50K_URL)
    logger.info(f"Downloaded {len(all_rows)} reactions. Computing fingerprints and indexing...")

    # Count splits
    split_counts = {}
    for row in all_rows:
        s = row.get("split", "unknown")
        split_counts[s] = split_counts.get(s, 0) + 1
    logger.info(f"Splits: {split_counts}")

    t0 = time.perf_counter()
    batch = []
    indexed = 0
    skipped = 0

    for i, row in enumerate(all_rows):
        # The 'rxn' column has clean (unmapped) SMILES: "reactants>>product"
        rxn_smiles = row.get("rxn", "")
        rxn_class = row.get("rxn_class", None)
        patent_source = row.get("source", None)

        if not rxn_smiles or ">>" not in rxn_smiles:
            skipped += 1
            continue

        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            skipped += 1
            continue

        reactants_raw = parts[0].strip()
        product = parts[1].strip()

        if not product or not reactants_raw:
            skipped += 1
            continue

        reactant_list = [r.strip() for r in reactants_raw.split(".") if r.strip()]
        if not reactant_list:
            skipped += 1
            continue

        fp = compute_reaction_fp(product, reactant_list)
        if fp is None:
            skipped += 1
            continue

        fp_bytes = pack_fingerprint(fp)

        batch.append(ReactionIndex(
            reaction_smiles=rxn_smiles,
            product_smiles=product,
            reactants_smiles=".".join(reactant_list),
            rxn_class=rxn_class,
            source=f"USPTO ({patent_source})" if patent_source else "USPTO-50K",
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
            rate = indexed / elapsed if elapsed > 0 else 0
            logger.info(f"  Indexed {indexed}/{len(all_rows)} ({rate:.0f}/s, {skipped} skipped)")

    # Insert remaining
    if batch:
        session = Session(sync_engine)
        session.add_all(batch)
        session.commit()
        session.close()
        indexed += len(batch)

    elapsed = time.perf_counter() - t0
    logger.info(f"Done! Indexed {indexed} reactions in {elapsed:.1f}s ({skipped} skipped)")

    # Verify
    session = Session(sync_engine)
    count = session.query(ReactionIndex).count()
    session.close()
    logger.info(f"Verification: reaction_index table has {count} rows.")


if __name__ == "__main__":
    main()
