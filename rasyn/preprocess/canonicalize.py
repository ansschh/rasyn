"""SMILES canonicalization and reaction SMILES parsing utilities.

Key design decisions (from RSGPT paper):
- Remove atom mapping from product inputs to LLM to prevent information leakage.
- Preserve atom mapping only for edit extraction during preprocessing.
- Canonical SMILES via RDKit for all comparisons and deduplication.
"""

from __future__ import annotations

from rdkit import Chem


def canonicalize_smiles(smiles: str, remove_mapping: bool = False) -> str:
    """Canonicalize a SMILES string using RDKit.

    Args:
        smiles: Input SMILES string.
        remove_mapping: If True, strip atom map numbers before canonicalization.

    Returns:
        Canonical SMILES, or empty string if parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    if remove_mapping:
        for atom in mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol)


def canonicalize_mapped_smiles(smiles: str) -> str:
    """Canonicalize while preserving atom map numbers.

    RDKit's canonical ordering is influenced by atom maps, so this
    produces a deterministic but mapping-aware canonical form.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def sort_molecule_list(smiles_list: list[str]) -> list[str]:
    """Sort a list of SMILES by length (longest first), then alphabetically.

    This ensures deterministic ordering of multi-component reaction sides.
    """
    return sorted(smiles_list, key=lambda s: (-len(s), s))


def parse_reaction_smiles(rxn_smiles: str) -> tuple[list[str], list[str], list[str]]:
    """Parse a reaction SMILES into (reactants, reagents, products).

    Handles both formats:
      - 'reactants>>products'        (no reagents)
      - 'reactants>reagents>products' (with reagents)

    Returns:
        Tuple of (reactant_smiles_list, reagent_smiles_list, product_smiles_list).
    """
    parts = rxn_smiles.split(">")
    if len(parts) == 3:
        reactants_str, reagents_str, products_str = parts
    elif len(parts) == 2:
        # Handle '>>' as two empty-string splits
        reactants_str = parts[0]
        reagents_str = ""
        products_str = parts[1]
    else:
        raise ValueError(f"Cannot parse reaction SMILES: {rxn_smiles}")

    reactants = [s for s in reactants_str.split(".") if s]
    reagents = [s for s in reagents_str.split(".") if s]
    products = [s for s in products_str.split(".") if s]

    return reactants, reagents, products


def canonicalize_reaction(
    rxn_smiles: str,
    remove_mapping: bool = False,
) -> str:
    """Canonicalize a full reaction SMILES string.

    Each component molecule is canonicalized individually, then components
    are sorted deterministically.

    Args:
        rxn_smiles: Full reaction SMILES (reactants>>products or reactants>reagents>products).
        remove_mapping: If True, strip atom maps from all molecules.

    Returns:
        Canonicalized reaction SMILES string.
    """
    reactants, reagents, products = parse_reaction_smiles(rxn_smiles)

    canon_reactants = sort_molecule_list(
        [canonicalize_smiles(s, remove_mapping) for s in reactants if canonicalize_smiles(s, remove_mapping)]
    )
    canon_reagents = sort_molecule_list(
        [canonicalize_smiles(s, remove_mapping) for s in reagents if canonicalize_smiles(s, remove_mapping)]
    )
    canon_products = sort_molecule_list(
        [canonicalize_smiles(s, remove_mapping) for s in products if canonicalize_smiles(s, remove_mapping)]
    )

    return (
        ".".join(canon_reactants)
        + ">"
        + ".".join(canon_reagents)
        + ">"
        + ".".join(canon_products)
    )


def remove_atom_mapping(smiles: str) -> str:
    """Remove atom map numbers from a SMILES string and return canonical form."""
    return canonicalize_smiles(smiles, remove_mapping=True)


def get_atom_map_to_idx(mol: Chem.Mol) -> dict[int, int]:
    """Build a mapping from atom map number to atom index for a molecule.

    Args:
        mol: RDKit molecule (must have atom map numbers set).

    Returns:
        Dict mapping atom_map_num -> atom_idx. Atoms without mapping are skipped.
    """
    mapping = {}
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num > 0:
            mapping[map_num] = atom.GetIdx()
    return mapping
