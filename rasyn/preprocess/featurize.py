"""Molecular featurization: SMILES -> PyTorch Geometric Data objects.

Follows Retro-MTGR conventions:
- 28-dim atom features (24 atom-type one-hot + H count + degree + aromaticity + formal charge)
- 5-dim bond features (4 bond-type one-hot + normalized bond energy)
- Bond energy lookup from Handbook of Chemistry and Physics
"""

from __future__ import annotations

import torch
from rdkit import Chem

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None  # Graceful fallback if PyG not installed


# ---------------------------------------------------------------------------
# Atom type vocabulary (24 types, matching Retro-MTGR)
# ---------------------------------------------------------------------------

ATOM_LIST = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg",
    "Na", "Ca", "Fe", "Al", "I", "B", "K", "Se", "Zn", "H",
    "Cu", "Mn", "*", "unknown",
]
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_LIST)}
NUM_ATOM_TYPES = len(ATOM_LIST)

# ---------------------------------------------------------------------------
# Bond energy lookup table (kJ/mol) — from CRC Handbook
# Key format: "SYMBOL-BONDTYPE-SYMBOL" where bondtype is '-', '=', '#', '~'
# ---------------------------------------------------------------------------

BOND_ENERGY_TABLE: dict[str, float] = {
    # Single bonds
    "C-C": 346, "C-N": 305, "C-O": 358, "C-S": 272, "C-F": 485,
    "C-Cl": 327, "C-Br": 285, "C-I": 213, "C-H": 411, "C-P": 264,
    "C-Si": 318, "C-B": 356,
    "N-N": 160, "N-O": 201, "N-H": 386, "N-F": 272, "N-Cl": 200,
    "N-Br": 243, "N-S": 467,
    "O-O": 142, "O-H": 459, "O-S": 265, "O-P": 335, "O-Si": 452,
    "S-S": 266, "S-H": 363,
    "F-F": 155, "Cl-Cl": 242, "Br-Br": 190, "I-I": 149,
    "H-H": 432, "H-F": 568, "H-Cl": 427, "H-Br": 363, "H-I": 295,
    "P-P": 201, "Si-Si": 222, "Si-O": 452, "Si-N": 355,
    # Double bonds
    "C=C": 614, "C=N": 615, "C=O": 799, "C=S": 536,
    "N=N": 418, "N=O": 607, "O=O": 494, "S=O": 522,
    "P=O": 544, "P=S": 335,
    # Triple bonds
    "C#C": 839, "C#N": 891, "N#N": 945, "C#O": 1077,
    # Aromatic (approximate as 1.5x single)
    "C~C": 518, "C~N": 458, "C~O": 537, "N~N": 289, "C~S": 408,
}

# Maximum bond energy for normalization
_MAX_BOND_ENERGY = max(BOND_ENERGY_TABLE.values()) if BOND_ENERGY_TABLE else 1077.0

BOND_TYPE_SYMBOL = {
    Chem.rdchem.BondType.SINGLE: "-",
    Chem.rdchem.BondType.DOUBLE: "=",
    Chem.rdchem.BondType.TRIPLE: "#",
    Chem.rdchem.BondType.AROMATIC: "~",
}

# Atomic mass table for the atom feature vector
ATOMIC_MASS = {
    "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.065, "F": 18.998,
    "Si": 28.086, "P": 30.974, "Cl": 35.453, "Br": 79.904, "Mg": 24.305,
    "Na": 22.990, "Ca": 40.078, "Fe": 55.845, "Al": 26.982, "I": 126.904,
    "B": 10.811, "K": 39.098, "Se": 78.960, "Zn": 65.380, "H": 1.008,
    "Cu": 63.546, "Mn": 54.938, "*": 0.0, "unknown": 0.0,
}


def get_atom_features(atom: Chem.Atom) -> list[float]:
    """Compute the 28-dimensional atom feature vector (Retro-MTGR style).

    Features:
        [0:24]  atom type one-hot (24 types)
        [24]    total number of hydrogens
        [25]    degree (number of heavy-atom neighbors)
        [26]    is aromatic (binary)
        [27]    formal charge
    """
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TO_IDX:
        symbol = "unknown"

    # 24-dim one-hot
    one_hot = [0.0] * NUM_ATOM_TYPES
    one_hot[ATOM_TO_IDX[symbol]] = 1.0

    features = one_hot + [
        float(atom.GetTotalNumHs()),
        float(atom.GetDegree()),
        1.0 if atom.GetIsAromatic() else 0.0,
        float(atom.GetFormalCharge()),
    ]
    return features


def get_bond_energy(mol: Chem.Mol, idx_i: int, idx_j: int) -> float:
    """Look up the theoretical bond energy for a bond between atoms i and j.

    Returns normalized bond energy (0-1 scale). Returns 0.5 if not found.
    """
    bond = mol.GetBondBetweenAtoms(idx_i, idx_j)
    if bond is None:
        return 0.0

    sym_i = mol.GetAtomWithIdx(idx_i).GetSymbol()
    sym_j = mol.GetAtomWithIdx(idx_j).GetSymbol()
    bt_sym = BOND_TYPE_SYMBOL.get(bond.GetBondType(), "-")

    # Try both orderings
    key1 = f"{sym_i}{bt_sym}{sym_j}"
    key2 = f"{sym_j}{bt_sym}{sym_i}"

    energy = BOND_ENERGY_TABLE.get(key1, BOND_ENERGY_TABLE.get(key2, _MAX_BOND_ENERGY * 0.5))
    return energy / _MAX_BOND_ENERGY


def get_bond_features(bond: Chem.Bond, mol: Chem.Mol) -> list[float]:
    """Compute the 5-dimensional bond feature vector.

    Features:
        [0:4]  bond type one-hot (single, double, triple, aromatic)
        [4]    normalized bond energy
    """
    bt = bond.GetBondType()
    bond_type_one_hot = [
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
    ]

    energy = get_bond_energy(mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return bond_type_one_hot + [energy]


def mol_to_pyg_data(
    smiles: str,
    atom_map_labels: dict[int, int] | None = None,
) -> Data | None:
    """Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles: Molecular SMILES (may or may not have atom mapping).
        atom_map_labels: Optional dict mapping atom_idx -> label for supervised tasks.

    Returns:
        PyG Data object with x (atom features), edge_index, edge_attr (bond features),
        and optional y (labels). Returns None if SMILES cannot be parsed.
    """
    if Data is None:
        raise ImportError("torch_geometric is required for mol_to_pyg_data")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge index and edge features (bidirectional)
    edge_index_list = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond, mol)
        # Add both directions
        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        edge_attr_list.append(bf)
        edge_attr_list.append(bf)

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    data.num_atoms = mol.GetNumAtoms()

    if atom_map_labels is not None:
        y = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
        for idx, label in atom_map_labels.items():
            y[idx] = label
        data.y = y

    return data


def mol_to_adjacency_with_energy(smiles: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Convert SMILES to adjacency matrix with bond energies (Retro-MTGR style).

    Returns:
        Tuple of (adjacency_matrix, atom_features) or None if parse fails.
        adjacency_matrix: [N, N] float tensor with normalized bond energies.
        atom_features: [N, 28] float tensor.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n = mol.GetNumAtoms()

    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # Adjacency with bond energies
    adj = torch.zeros((n, n), dtype=torch.float)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        energy = get_bond_energy(mol, i, j)
        adj[i, j] = energy
        adj[j, i] = energy

    return adj, x
