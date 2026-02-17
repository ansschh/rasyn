#!/usr/bin/env python
"""Fast USPTO-50K indexer — downloads and indexes directly, no preprocessing needed.

Downloads the USPTO-50K Schneider dataset (raw CSV), extracts product/reactant SMILES,
computes Morgan difference fingerprints, and inserts into the reaction_index table.

Usage:
    python scripts/index_uspto_fast.py [--batch-size 500]

This is the recommended way to set up the evidence index on a fresh EC2 server.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# USPTO-50K dataset URL (Schneider split — the standard benchmark)
USPTO_50K_URL = "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/data/USPTO_50k/raw_test.csv"
USPTO_50K_URLS = {
    "train": "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/data/USPTO_50k/raw_train.csv",
    "val": "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/data/USPTO_50k/raw_val.csv",
    "test": "https://raw.githubusercontent.com/Hanjun-Dai/GLN/master/data/USPTO_50k/raw_test.csv",
}


def download_csv(url: str) -> list[dict]:
    """Download a CSV file and parse it into list of dicts."""
    logger.info(f"Downloading {url}...")
    response = urllib.request.urlopen(url)
    content = response.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


def strip_atom_mapping(smiles: str) -> str:
    """Strip atom mapping numbers from SMILES."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        for atom in mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber")
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


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
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Comma-separated splits to index")
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

    # Download all splits
    splits = [s.strip() for s in args.splits.split(",")]
    all_rows = []
    for split in splits:
        url = USPTO_50K_URLS.get(split)
        if not url:
            logger.warning(f"Unknown split: {split}")
            continue
        rows = download_csv(url)
        logger.info(f"  {split}: {len(rows)} reactions")
        all_rows.extend(rows)

    logger.info(f"Total: {len(all_rows)} reactions. Computing fingerprints and indexing...")

    t0 = time.perf_counter()
    batch = []
    indexed = 0
    skipped = 0

    for i, row in enumerate(all_rows):
        # Parse reaction SMILES: "reactants>>product" format
        rxn_smiles = row.get("reactants>reagents>production", "") or row.get("rxn_smiles", "")
        rxn_class = row.get("class", None) or row.get("reaction_class", None)

        if not rxn_smiles or ">>" not in rxn_smiles:
            skipped += 1
            continue

        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            skipped += 1
            continue

        reactants_raw = parts[0]
        product_raw = parts[1]

        # Strip atom mapping
        product = strip_atom_mapping(product_raw.strip())
        reactant_list = [strip_atom_mapping(r.strip()) for r in reactants_raw.split(".") if r.strip()]

        if not product or not reactant_list:
            skipped += 1
            continue

        fp = compute_reaction_fp(product, reactant_list)
        if fp is None:
            skipped += 1
            continue

        fp_bytes = pack_fingerprint(fp)
        clean_rxn = ".".join(reactant_list) + ">>" + product

        # Convert class to string if it's a number
        if rxn_class is not None:
            try:
                rxn_class = str(int(float(rxn_class)))
            except (ValueError, TypeError):
                rxn_class = str(rxn_class) if rxn_class else None

        batch.append(ReactionIndex(
            reaction_smiles=clean_rxn,
            product_smiles=product,
            reactants_smiles=".".join(reactant_list),
            rxn_class=rxn_class,
            source="USPTO-50K",
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
