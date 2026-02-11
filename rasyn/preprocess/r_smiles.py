"""Root-aligned SMILES (R-SMILES) for retrosynthesis.

Generates SMILES strings rooted at the reaction center atom, reducing edit
distance between product and reactant representations by ~50%. This makes the
seq2seq task fundamentally easier.

Reference: Zhong et al., "Root-aligned SMILES: A Tight Representation for
Chemical Reaction Prediction" (2022).

Usage:
    from rasyn.preprocess.r_smiles import root_aligned_smiles, get_reaction_center_atom

    center = get_reaction_center_atom(rxn_smiles)
    r_product = root_aligned_smiles(product_smiles, center, atom_mapped_product)
"""

from __future__ import annotations

import logging
from collections import Counter

from rdkit import Chem

from rasyn.preprocess.canonicalize import (
    get_atom_map_to_idx,
    parse_reaction_smiles,
)
from rasyn.preprocess.extract_edits import find_changed_bonds

logger = logging.getLogger(__name__)


def get_reaction_center_atoms(rxn_smiles: str) -> list[int]:
    """Find all reaction center atom map numbers, ordered by involvement.

    Returns atom map numbers of atoms involved in changed bonds,
    sorted by frequency (most involved first).
    """
    changed_bonds = find_changed_bonds(rxn_smiles)
    if not changed_bonds:
        return []

    atom_count: Counter[int] = Counter()
    for bc in changed_bonds:
        atom_count[bc.atom_map_1] += 1
        atom_count[bc.atom_map_2] += 1

    return [atom for atom, _ in atom_count.most_common()]


def get_reaction_center_atom(rxn_smiles: str) -> int | None:
    """Find the most central reaction center atom map number."""
    atoms = get_reaction_center_atoms(rxn_smiles)
    return atoms[0] if atoms else None


def root_aligned_smiles(
    smiles: str,
    root_atom_map: int,
    atom_mapped_smiles: str | None = None,
) -> str:
    """Generate SMILES rooted at the specified atom (by atom map number).

    Args:
        smiles: Clean SMILES (without atom mapping).
        root_atom_map: Atom map number to root at.
        atom_mapped_smiles: Atom-mapped version of the same molecule,
            used to find which atom index corresponds to the atom map number.
            If None, assumes smiles itself has atom mapping.

    Returns:
        SMILES string rooted at the specified atom, or canonical SMILES if
        the atom map is not found.
    """
    # Parse the atom-mapped molecule to find the root atom index
    ref_smiles = atom_mapped_smiles or smiles
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    if ref_mol is None:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if Chem.MolFromSmiles(smiles) else smiles

    map_to_idx = get_atom_map_to_idx(ref_mol)
    root_idx_in_ref = map_to_idx.get(root_atom_map)

    if root_idx_in_ref is None:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else smiles

    # If smiles is clean (no atom map), we need to match atoms between
    # the mapped and clean versions. RDKit canonical atom ordering may differ.
    # Strategy: use the atom-mapped mol, remove mapping, generate rooted SMILES.
    mol = Chem.RWMol(ref_mol)
    for atom in mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")

    try:
        rooted = Chem.MolToSmiles(mol, rootedAtAtom=root_idx_in_ref)
        return rooted
    except Exception:
        return Chem.MolToSmiles(mol)


def root_aligned_smiles_multi(
    smiles_str: str,
    root_atom_map: int,
    atom_mapped_smiles_str: str,
    separator: str = ".",
) -> str:
    """Root-align multi-component SMILES (e.g., reactants with multiple molecules).

    The component containing the root atom gets rooted; others stay canonical.
    Components are sorted deterministically (longest first, then alphabetical).
    """
    clean_parts = smiles_str.split(separator)
    mapped_parts = atom_mapped_smiles_str.split(separator)

    aligned = []
    for i, (clean, mapped) in enumerate(zip(clean_parts, mapped_parts)):
        clean = clean.strip()
        mapped = mapped.strip()
        if not clean:
            continue

        mapped_mol = Chem.MolFromSmiles(mapped)
        if mapped_mol is None:
            aligned.append(clean)
            continue

        map_to_idx = get_atom_map_to_idx(mapped_mol)
        if root_atom_map in map_to_idx:
            aligned.append(root_aligned_smiles(clean, root_atom_map, mapped))
        else:
            mol = Chem.MolFromSmiles(clean)
            aligned.append(Chem.MolToSmiles(mol) if mol else clean)

    # Sort: longest first, then alphabetical
    aligned.sort(key=lambda s: (-len(s), s))
    return separator.join(aligned)


def build_r_smiles_example(
    rxn_smiles: str,
    rxn_id: str = "",
    reaction_class: int = 0,
) -> dict | None:
    """Build an R-SMILES aligned training example from an atom-mapped reaction.

    Args:
        rxn_smiles: Atom-mapped reaction SMILES (reactants>>product).
        rxn_id: Reaction identifier.
        reaction_class: USPTO reaction class (1-10).

    Returns:
        Dict with keys: src_text, tgt_text, rxn_id, reaction_class, root_atom_map.
        Or None if processing fails.
    """
    reactants_list, _, products_list = parse_reaction_smiles(rxn_smiles)
    if not reactants_list or not products_list:
        return None

    product_mapped = ".".join(products_list)
    reactants_mapped = ".".join(reactants_list)

    center_atom = get_reaction_center_atom(rxn_smiles)
    if center_atom is None:
        return None

    # Root-align product (clean, no atom mapping)
    product_mol = Chem.MolFromSmiles(product_mapped)
    if product_mol is None:
        return None

    map_to_idx = get_atom_map_to_idx(product_mol)
    root_idx = map_to_idx.get(center_atom)
    if root_idx is None:
        return None

    # Generate clean product rooted at reaction center
    clean_product_mol = Chem.RWMol(product_mol)
    for atom in clean_product_mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")

    try:
        r_product = Chem.MolToSmiles(clean_product_mol, rootedAtAtom=root_idx)
    except Exception:
        return None

    # Root-align each reactant
    r_reactants = []
    for r_mapped in reactants_list:
        r_mol = Chem.MolFromSmiles(r_mapped)
        if r_mol is None:
            continue

        r_map = get_atom_map_to_idx(r_mol)
        clean_r_mol = Chem.RWMol(r_mol)
        for atom in clean_r_mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber")

        if center_atom in r_map:
            r_root_idx = r_map[center_atom]
            try:
                r_reactants.append(Chem.MolToSmiles(clean_r_mol, rootedAtAtom=r_root_idx))
            except Exception:
                r_reactants.append(Chem.MolToSmiles(clean_r_mol))
        else:
            r_reactants.append(Chem.MolToSmiles(clean_r_mol))

    if not r_reactants:
        return None

    # Sort reactants deterministically
    r_reactants.sort(key=lambda s: (-len(s), s))

    return {
        "product": r_product,
        "reactants": ".".join(r_reactants),
        "rxn_id": rxn_id,
        "reaction_class": reaction_class,
        "root_atom_map": center_atom,
    }
