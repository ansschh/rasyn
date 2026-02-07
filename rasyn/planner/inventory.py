"""Purchasable molecule inventory for route planning.

Provides lookup for whether a molecule is commercially available
(i.e., a valid starting material / terminal node in route search).
"""

from __future__ import annotations

import logging
from pathlib import Path

from rdkit import Chem

from rasyn.preprocess.canonicalize import canonicalize_smiles

logger = logging.getLogger(__name__)


class MoleculeInventory:
    """Lookup table for purchasable / available starting materials."""

    def __init__(self):
        self._smiles_set: set[str] = set()

    def load_from_file(self, path: str | Path) -> None:
        """Load inventory from a file (one SMILES per line)."""
        path = Path(path)
        count = 0
        with open(path) as f:
            for line in f:
                smi = line.strip()
                if smi and not smi.startswith("#"):
                    canon = canonicalize_smiles(smi)
                    if canon:
                        self._smiles_set.add(canon)
                        count += 1
        logger.info(f"Loaded {count} molecules from {path}")

    def load_from_list(self, smiles_list: list[str]) -> None:
        """Load inventory from a list of SMILES."""
        for smi in smiles_list:
            canon = canonicalize_smiles(smi)
            if canon:
                self._smiles_set.add(canon)

    def is_purchasable(self, smiles: str) -> bool:
        """Check if a molecule is in the inventory."""
        canon = canonicalize_smiles(smiles)
        return canon in self._smiles_set

    def __len__(self) -> int:
        return len(self._smiles_set)

    def __contains__(self, smiles: str) -> bool:
        return self.is_purchasable(smiles)


# Common building blocks (small set for MVP / testing)
COMMON_BUILDING_BLOCKS = [
    # Simple alcohols
    "CO", "CCO", "CC(C)O", "CCCO",
    # Amines
    "N", "CN", "CCN", "CC(C)N",
    # Acids
    "OC=O", "CC(O)=O", "OC(=O)CC(O)=O",
    # Halides
    "CCl", "CBr", "CI", "ClCCl",
    # Aromatics
    "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1",
    "c1ccc(Br)cc1", "c1ccc(Cl)cc1", "c1ccc(F)cc1",
    "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(C=O)cc1",
    # Boronic acids
    "OB(O)c1ccccc1", "OB(O)c1ccc(C)cc1",
    # Common reagents
    "O=C=O", "N#N", "O", "[Na+].[OH-]", "[K+].[OH-]",
    # Amino acids, etc.
    "NCC(O)=O",
]


def get_default_inventory() -> MoleculeInventory:
    """Get a default inventory with common building blocks."""
    inv = MoleculeInventory()
    inv.load_from_list(COMMON_BUILDING_BLOCKS)
    return inv
