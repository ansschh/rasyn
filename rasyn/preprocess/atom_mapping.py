"""Add atom mapping to unmapped reaction SMILES.

Uses EPAM Indigo's automap algorithm, which handles complex organic
reactions reliably (unlike RDKit's rdChemReactions.MapReaction).
"""

from __future__ import annotations

import logging

from rdkit import Chem

logger = logging.getLogger(__name__)


def has_atom_mapping(smiles: str) -> bool:
    """Check if a SMILES string contains atom mapping numbers."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(atom.GetAtomMapNum() > 0 for atom in mol.GetAtoms())


def map_single_reaction(rxn_smiles: str) -> str | None:
    """Map a single reaction using Indigo's automap.

    Args:
        rxn_smiles: Unmapped reaction SMILES (reactants>>product).

    Returns:
        Atom-mapped reaction SMILES, or None on failure.
    """
    try:
        from indigo import Indigo
        indigo = Indigo()
        rxn = indigo.loadReaction(rxn_smiles)
        rxn.automap("discard")
        mapped = rxn.smiles()
        if mapped and ":" in mapped:
            return mapped
    except Exception:
        pass
    return None


def map_reactions_batch(
    rxn_smiles_list: list[str],
    batch_size: int = 10,
    n_workers: int = 0,
) -> list[str | None]:
    """Map a batch of reactions with Indigo (single-threaded).

    Args:
        rxn_smiles_list: List of unmapped reaction SMILES.
        batch_size: Not used (kept for API compat).
        n_workers: Ignored — always runs single-threaded.

    Returns:
        List of mapped SMILES (None for failures).
    """
    from tqdm import tqdm
    from indigo import Indigo

    indigo = Indigo()
    n = len(rxn_smiles_list)
    logger.info(f"Atom mapping {n} reactions with Indigo...")

    results = []
    for rxn_smi in tqdm(rxn_smiles_list, desc="Atom mapping"):
        try:
            rxn = indigo.loadReaction(rxn_smi)
            rxn.automap("discard")
            mapped = rxn.smiles()
            if mapped and ":" in mapped:
                results.append(mapped)
            else:
                results.append(None)
        except Exception:
            results.append(None)

    mapped_count = sum(1 for r in results if r is not None)
    logger.info(f"Mapped {mapped_count}/{n} reactions ({mapped_count/n*100:.1f}%)")
    return results
