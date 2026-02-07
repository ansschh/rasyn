"""RDKit sanity checks for predicted reactants.

Fast, always-on verification:
  - Parse all predicted SMILES
  - Check atom balance
  - Deduplicate after canonicalization
  - Validate molecular properties
"""

from __future__ import annotations

import logging
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors

from rasyn.preprocess.canonicalize import canonicalize_smiles

logger = logging.getLogger(__name__)


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string can be parsed by RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def sanitize_and_canonicalize(smiles: str) -> str | None:
    """Parse, sanitize, and canonicalize a SMILES string.

    Returns canonical SMILES or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def check_atom_balance(
    product_smiles: str,
    reactant_smiles_list: list[str],
    tolerance: int = 10,
) -> tuple[bool, str]:
    """Check if atoms in reactants account for atoms in the product.

    For retrosynthesis, reactants should contain at least all atoms in the product
    (plus possible leaving group atoms). This is a soft check — some discrepancy
    is expected due to implicit hydrogens and reagents.

    Args:
        product_smiles: Product SMILES.
        reactant_smiles_list: List of reactant SMILES.
        tolerance: Max allowed extra heavy atoms in reactants vs product.

    Returns:
        Tuple of (is_balanced, reason_string).
    """
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return False, "Cannot parse product"

    product_atoms = Counter()
    for atom in product_mol.GetAtoms():
        product_atoms[atom.GetSymbol()] += 1

    reactant_atoms = Counter()
    for r_smi in reactant_smiles_list:
        r_mol = Chem.MolFromSmiles(r_smi)
        if r_mol is None:
            return False, f"Cannot parse reactant: {r_smi}"
        for atom in r_mol.GetAtoms():
            reactant_atoms[atom.GetSymbol()] += 1

    # Check that product atoms are a subset of reactant atoms
    for element, count in product_atoms.items():
        if element == "H":
            continue  # Skip hydrogen (implicit H handling is unreliable)
        if reactant_atoms.get(element, 0) < count:
            return False, f"Missing {element}: need {count}, have {reactant_atoms.get(element, 0)}"

    # Check that reactants don't have too many extra atoms (suggests garbage)
    total_product_heavy = sum(c for e, c in product_atoms.items() if e != "H")
    total_reactant_heavy = sum(c for e, c in reactant_atoms.items() if e != "H")
    if total_reactant_heavy > total_product_heavy + tolerance:
        return False, f"Too many extra atoms: {total_reactant_heavy} vs {total_product_heavy}"

    return True, "OK"


def deduplicate_candidates(
    candidates: list[list[str]],
) -> list[list[str]]:
    """Remove duplicate candidate reactant sets after canonicalization.

    Args:
        candidates: List of reactant SMILES lists.

    Returns:
        Deduplicated list.
    """
    seen = set()
    unique = []
    for reactants in candidates:
        canon = tuple(sorted(canonicalize_smiles(s) for s in reactants))
        if canon not in seen:
            seen.add(canon)
            unique.append(reactants)
    return unique


def check_molecular_weight(
    smiles: str,
    max_mw: float = 1500.0,
) -> bool:
    """Check if a molecule's MW is within a reasonable range."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    return mw <= max_mw


def rdkit_verify(
    product_smiles: str,
    reactant_smiles_list: list[str],
) -> dict:
    """Run all RDKit sanity checks on a candidate.

    Returns:
        Dict with check results and overall pass/fail.
    """
    results = {
        "all_valid_smiles": True,
        "atom_balanced": True,
        "atom_balance_reason": "OK",
        "reasonable_mw": True,
        "overall_pass": True,
    }

    # Check each reactant is valid
    for r_smi in reactant_smiles_list:
        if not is_valid_smiles(r_smi):
            results["all_valid_smiles"] = False
            results["overall_pass"] = False
            return results

    # Check product is valid
    if not is_valid_smiles(product_smiles):
        results["overall_pass"] = False
        return results

    # Atom balance
    balanced, reason = check_atom_balance(product_smiles, reactant_smiles_list)
    results["atom_balanced"] = balanced
    results["atom_balance_reason"] = reason

    # MW check
    for r_smi in reactant_smiles_list:
        if not check_molecular_weight(r_smi):
            results["reasonable_mw"] = False

    results["overall_pass"] = results["all_valid_smiles"] and results["atom_balanced"]

    return results
