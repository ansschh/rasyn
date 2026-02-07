"""Extract reaction edits from atom-mapped reaction SMILES.

This is the core preprocessing script — the foundation of the entire pipeline.
Given an atom-mapped reaction SMILES, it extracts:
  1. Changed bonds (reaction center) by comparing bond connectivity
  2. Synthons (product fragments after breaking reaction-center bonds)
  3. Leaving groups (substructure in reactant not in corresponding synthon)
  4. Edit token string for LLM conditioning

Algorithm:
  1. Parse reactants>>product atom-mapped SMILES
  2. Build bond-info dicts for product and reactants (atom_map_pair -> bond_order)
  3. Diff to find changed bonds
  4. Fragment product at broken bonds -> synthons
  5. Match synthons to reactants via atom maps -> extract leaving groups
  6. Generate edit token string
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem

from rasyn.preprocess.canonicalize import (
    canonicalize_smiles,
    parse_reaction_smiles,
    get_atom_map_to_idx,
)
from rasyn.schema import EditLabels

logger = logging.getLogger(__name__)


@dataclass
class BondChange:
    """A single bond change between reactants and product."""

    atom_map_1: int
    atom_map_2: int
    reactant_order: float  # 0.0 if bond doesn't exist in reactants
    product_order: float   # 0.0 if bond doesn't exist in product
    change_type: str       # 'broken', 'formed', 'order_changed'


def _get_bond_info(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Build mapping from (atom_map_1, atom_map_2) -> bond_order for a molecule.

    Args:
        mol: RDKit molecule with atom map numbers.

    Returns:
        Dict with sorted (map_1, map_2) tuples as keys and bond order as values.
    """
    bond_info = {}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetAtomMapNum()
        a2 = bond.GetEndAtom().GetAtomMapNum()
        if a1 == 0 or a2 == 0:
            continue  # Skip unmapped atoms
        if a1 > a2:
            a1, a2 = a2, a1
        bond_info[(a1, a2)] = bond.GetBondTypeAsDouble()
    return bond_info


def find_changed_bonds(rxn_smiles: str) -> list[BondChange]:
    """Identify bonds that changed between reactants and product.

    Args:
        rxn_smiles: Atom-mapped reaction SMILES (reactants>>product).

    Returns:
        List of BondChange objects describing each changed bond.
    """
    reactants_str, _, products_str = parse_reaction_smiles(rxn_smiles)

    # Parse product (take first/main product)
    product_smi = ".".join(products_str)
    product_mol = Chem.MolFromSmiles(product_smi)
    if product_mol is None:
        logger.warning(f"Cannot parse product: {product_smi}")
        return []

    # Parse all reactants
    reactants_smi = ".".join(reactants_str)
    reactant_mol = Chem.MolFromSmiles(reactants_smi)
    if reactant_mol is None:
        logger.warning(f"Cannot parse reactants: {reactants_smi}")
        return []

    # Get bond info for product and reactants
    prod_bonds = _get_bond_info(product_mol)
    react_bonds = _get_bond_info(reactant_mol)

    # Find all changed bonds
    all_pairs = set(list(prod_bonds.keys()) + list(react_bonds.keys()))
    changes = []

    for (a1, a2) in all_pairs:
        prod_order = prod_bonds.get((a1, a2), 0.0)
        react_order = react_bonds.get((a1, a2), 0.0)

        if prod_order != react_order:
            if react_order == 0.0 and prod_order > 0.0:
                change_type = "formed"  # Bond exists in product but not reactants
            elif react_order > 0.0 and prod_order == 0.0:
                change_type = "broken"  # Bond exists in reactants but not product
            else:
                change_type = "order_changed"  # Bond order changed

            changes.append(BondChange(
                atom_map_1=a1,
                atom_map_2=a2,
                reactant_order=react_order,
                product_order=prod_order,
                change_type=change_type,
            ))

    return changes


def extract_synthons(
    product_smiles: str,
    changed_bonds: list[BondChange],
) -> list[str]:
    """Break product at reaction-center bonds to get synthons.

    For retrosynthesis, we break bonds that were *formed* in the forward
    direction (i.e., bonds present in product but not in reactants).

    Args:
        product_smiles: Atom-mapped product SMILES.
        changed_bonds: List of BondChange from find_changed_bonds.

    Returns:
        List of synthon SMILES (with dummy atoms at cut points).
    """
    mol = Chem.MolFromSmiles(product_smiles)
    if mol is None:
        return []

    # Map atom_map_num -> atom_idx
    map_to_idx = get_atom_map_to_idx(mol)

    # Find bond indices to break: bonds that were FORMED (exist in product, not reactants)
    bond_indices_to_break = []
    for bc in changed_bonds:
        if bc.change_type == "formed":
            idx_1 = map_to_idx.get(bc.atom_map_1)
            idx_2 = map_to_idx.get(bc.atom_map_2)
            if idx_1 is not None and idx_2 is not None:
                bond = mol.GetBondBetweenAtoms(idx_1, idx_2)
                if bond is not None:
                    bond_indices_to_break.append(bond.GetIdx())

    if not bond_indices_to_break:
        # No bonds to break -> whole product is the synthon
        # This happens for bond-order-change-only reactions (e.g., reduction)
        return [Chem.MolToSmiles(mol)]

    # Fragment at the bonds
    try:
        fragmented = Chem.FragmentOnBonds(mol, bond_indices_to_break, addDummies=True)
        frag_smiles = Chem.MolToSmiles(fragmented).split(".")
        return [s for s in frag_smiles if s]
    except Exception as e:
        logger.warning(f"Fragmentation failed: {e}")
        return [Chem.MolToSmiles(mol)]


def _match_synthon_to_reactant(
    synthon_smiles: str,
    reactant_smiles_list: list[str],
) -> tuple[str | None, str]:
    """Match a synthon to its corresponding reactant via atom map numbers.

    Args:
        synthon_smiles: SMILES of a single synthon (from product fragmentation).
        reactant_smiles_list: List of all reactant SMILES.

    Returns:
        Tuple of (matched_reactant_smiles, leaving_group_smiles).
        If no match found, returns (None, "").
    """
    synthon_mol = Chem.MolFromSmiles(synthon_smiles)
    if synthon_mol is None:
        return None, ""

    # Get atom map numbers in synthon (excluding dummy atoms)
    synthon_maps = set()
    for atom in synthon_mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num > 0:
            synthon_maps.add(map_num)

    if not synthon_maps:
        return None, ""

    # Find the reactant that shares the most atom maps with this synthon
    best_reactant = None
    best_overlap = 0

    for r_smi in reactant_smiles_list:
        r_mol = Chem.MolFromSmiles(r_smi)
        if r_mol is None:
            continue
        r_maps = {a.GetAtomMapNum() for a in r_mol.GetAtoms() if a.GetAtomMapNum() > 0}
        overlap = len(synthon_maps & r_maps)
        if overlap > best_overlap:
            best_overlap = overlap
            best_reactant = r_smi

    if best_reactant is None:
        return None, ""

    # Extract leaving group: atoms in reactant not in synthon
    r_mol = Chem.MolFromSmiles(best_reactant)
    r_maps = {a.GetAtomMapNum() for a in r_mol.GetAtoms() if a.GetAtomMapNum() > 0}
    lg_maps = r_maps - synthon_maps

    if not lg_maps:
        return best_reactant, "H"  # Only hydrogen difference

    # Extract LG substructure
    lg_atom_indices = []
    for atom in r_mol.GetAtoms():
        if atom.GetAtomMapNum() in lg_maps:
            lg_atom_indices.append(atom.GetIdx())

    if not lg_atom_indices:
        return best_reactant, "H"

    # Build LG SMILES from the leaving group atoms
    try:
        # Create editable molecule, remove non-LG atoms
        rw_mol = Chem.RWMol(r_mol)
        # Clear atom maps for clean LG SMILES
        for atom in rw_mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber")

        # Get the fragment containing only LG atoms
        all_indices = set(range(rw_mol.GetNumAtoms()))
        keep_indices = set(lg_atom_indices)
        remove_indices = sorted(all_indices - keep_indices, reverse=True)

        for idx in remove_indices:
            rw_mol.RemoveAtom(idx)

        lg_smiles = Chem.MolToSmiles(rw_mol.GetMol())
        if not lg_smiles:
            lg_smiles = "H"
        return best_reactant, lg_smiles
    except Exception:
        return best_reactant, "H"


def extract_leaving_groups(
    synthon_smiles_list: list[str],
    reactant_smiles_list: list[str],
) -> list[str]:
    """Extract leaving groups by matching each synthon to its reactant.

    Args:
        synthon_smiles_list: Synthon SMILES from product fragmentation.
        reactant_smiles_list: All reactant SMILES.

    Returns:
        List of leaving group SMILES (one per synthon).
    """
    leaving_groups = []
    for synthon in synthon_smiles_list:
        _, lg = _match_synthon_to_reactant(synthon, reactant_smiles_list)
        leaving_groups.append(lg if lg else "H")
    return leaving_groups


def format_edit_tokens(
    product_smiles: str,
    changed_bonds: list[BondChange],
    synthon_smiles_list: list[str],
    leaving_groups: list[str],
    product_mol: Chem.Mol | None = None,
) -> str:
    """Format edit information into the Edit Token Language for LLM conditioning.

    Format:
        <PROD> {product_smiles}
        <EDIT> DISCONNECT {atom_idx_i}-{atom_idx_j} [...]
        <SYNTHONS> {synthon_1} . {synthon_2}
        <LG_HINTS> [{lg_1}] [{lg_2}]
        <OUT>

    Args:
        product_smiles: Canonical product SMILES (without atom mapping).
        changed_bonds: Extracted bond changes.
        synthon_smiles_list: Synthon SMILES.
        leaving_groups: Leaving group SMILES.
        product_mol: Optional pre-parsed product molecule for atom index lookup.

    Returns:
        Formatted edit token string.
    """
    # Build DISCONNECT tokens
    disconnect_parts = []
    for bc in changed_bonds:
        if bc.change_type == "formed":
            # For LLM conditioning, use atom map numbers (stable across canonicalizations)
            disconnect_parts.append(f"DISCONNECT {bc.atom_map_1}-{bc.atom_map_2}")

    edit_str = " ".join(disconnect_parts) if disconnect_parts else "NO_DISCONNECT"

    # Clean synthon SMILES (remove atom mapping for LLM input)
    clean_synthons = []
    for s in synthon_smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            for atom in mol.GetAtoms():
                atom.ClearProp("molAtomMapNumber")
            clean_synthons.append(Chem.MolToSmiles(mol))
        else:
            clean_synthons.append(s)

    synthons_str = " . ".join(clean_synthons)

    # LG hints
    lg_parts = [f"[{lg}]" for lg in leaving_groups]
    lg_str = " ".join(lg_parts)

    # Remove atom mapping from product for LLM input
    clean_product = canonicalize_smiles(product_smiles, remove_mapping=True)

    return f"<PROD> {clean_product} <EDIT> {edit_str} <SYNTHONS> {synthons_str} <LG_HINTS> {lg_str} <OUT>"


def extract_edits_from_reaction(rxn_smiles: str) -> EditLabels | None:
    """Full extraction pipeline: reaction SMILES -> EditLabels.

    This is the main entry point for preprocessing.

    Args:
        rxn_smiles: Atom-mapped reaction SMILES (reactants>>product or reactants>reagents>product).

    Returns:
        EditLabels dataclass with all extracted information, or None if extraction fails.
    """
    # Parse reaction
    reactants_list, _, products_list = parse_reaction_smiles(rxn_smiles)
    if not reactants_list or not products_list:
        logger.warning(f"Empty reactants or products: {rxn_smiles}")
        return None

    product_smi = ".".join(products_list)

    # Step 1: Find changed bonds
    changed_bonds = find_changed_bonds(rxn_smiles)
    if not changed_bonds:
        logger.debug(f"No changed bonds found: {rxn_smiles}")
        return None

    # Step 2: Extract synthons
    synthon_smiles = extract_synthons(product_smi, changed_bonds)

    # Step 3: Extract leaving groups
    leaving_groups = extract_leaving_groups(synthon_smiles, reactants_list)

    # Step 4: Build edit token string
    edit_tokens = format_edit_tokens(
        product_smiles=product_smi,
        changed_bonds=changed_bonds,
        synthon_smiles_list=synthon_smiles,
        leaving_groups=leaving_groups,
    )

    # Convert changed bonds to atom-index pairs on the product
    product_mol = Chem.MolFromSmiles(product_smi)
    map_to_idx = get_atom_map_to_idx(product_mol) if product_mol else {}
    changed_bond_indices = []
    for bc in changed_bonds:
        idx_1 = map_to_idx.get(bc.atom_map_1)
        idx_2 = map_to_idx.get(bc.atom_map_2)
        if idx_1 is not None and idx_2 is not None:
            changed_bond_indices.append((min(idx_1, idx_2), max(idx_1, idx_2)))

    return EditLabels(
        changed_bonds=changed_bond_indices,
        synthon_smiles=synthon_smiles,
        leaving_groups=leaving_groups,
        edit_tokens=edit_tokens,
    )


def process_dataset(
    reactions: list[dict],
    id_key: str = "id",
    rxn_key: str = "rxn_smiles",
) -> list[dict]:
    """Process a list of reaction dicts, extracting edits for each.

    Args:
        reactions: List of dicts with at least rxn_smiles field.
        id_key: Key for reaction ID.
        rxn_key: Key for reaction SMILES.

    Returns:
        List of dicts with added 'edit_labels' field.
    """
    results = []
    failed = 0
    for rxn in reactions:
        rxn_smi = rxn.get(rxn_key, "")
        labels = extract_edits_from_reaction(rxn_smi)
        if labels is None:
            failed += 1
            continue

        result = dict(rxn)
        result["edit_labels"] = labels
        results.append(result)

    logger.info(f"Processed {len(results)} reactions, {failed} failed")
    return results
