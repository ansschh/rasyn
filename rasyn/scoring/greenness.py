"""Greenness scoring: environmental and sustainability proxies.

Evaluates green chemistry principles:
  - Solvent choice (penalize harmful solvents, reward green ones)
  - Atom economy estimate
  - Component count as PMI proxy
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors


# Solvent greenness scores (0 = bad, 1 = good)
# Based on CHEM21 solvent guide and GSK solvent selection guide
SOLVENT_SCORES = {
    # Green solvents
    "O": {"name": "water", "score": 1.0},
    "CCO": {"name": "ethanol", "score": 0.9},
    "CC(C)O": {"name": "isopropanol", "score": 0.85},
    "CC(=O)C": {"name": "acetone", "score": 0.8},
    "CC(=O)OCC": {"name": "ethyl acetate", "score": 0.8},
    "COC(C)(C)C": {"name": "MTBE", "score": 0.75},
    "CO": {"name": "methanol", "score": 0.7},
    # Amber solvents
    "CCCCCC": {"name": "hexane", "score": 0.5},
    "c1ccccc1": {"name": "benzene", "score": 0.2},
    "Cc1ccccc1": {"name": "toluene", "score": 0.55},
    "C1CCCC1": {"name": "cyclopentane", "score": 0.5},
    "CC#N": {"name": "MeCN", "score": 0.6},
    "C1CCOC1": {"name": "THF", "score": 0.5},
    "CCCC#N": {"name": "butyronitrile", "score": 0.45},
    # Red solvents
    "ClCCl": {"name": "DCM", "score": 0.2},
    "ClC(Cl)Cl": {"name": "chloroform", "score": 0.15},
    "CN(C)C=O": {"name": "DMF", "score": 0.25},
    "CN1CCCC1=O": {"name": "NMP", "score": 0.25},
    "C1COCCO1": {"name": "1,4-dioxane", "score": 0.1},
    "ClCCCl": {"name": "DCE", "score": 0.1},
    "CCOCC": {"name": "diethyl ether", "score": 0.35},
    "CS(C)=O": {"name": "DMSO", "score": 0.4},
    "CC(C)=O": {"name": "acetone", "score": 0.8},
    "CCCCCCCC": {"name": "octane", "score": 0.45},
}


def compute_atom_economy(
    product_smiles: str,
    reactant_smiles_list: list[str],
) -> float:
    """Estimate atom economy: MW(product) / sum(MW(reactants)).

    Perfect atom economy = 1.0 (all atoms end up in product).
    """
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return 0.0

    product_mw = Descriptors.MolWt(product_mol)

    total_reactant_mw = 0.0
    for r_smi in reactant_smiles_list:
        r_mol = Chem.MolFromSmiles(r_smi)
        if r_mol is not None:
            total_reactant_mw += Descriptors.MolWt(r_mol)

    if total_reactant_mw == 0:
        return 0.0

    return min(1.0, product_mw / total_reactant_mw)


def score_solvents(solvent_smiles_list: list[str]) -> tuple[float, list[dict]]:
    """Score the solvent choices for greenness.

    Returns:
        Tuple of (average_solvent_score, details_list).
    """
    if not solvent_smiles_list:
        return 0.7, []  # No solvents specified = neutral

    details = []
    total_score = 0.0

    for smi in solvent_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        canon = Chem.MolToSmiles(mol) if mol else smi
        solvent_info = SOLVENT_SCORES.get(canon)

        if solvent_info:
            total_score += solvent_info["score"]
            details.append({
                "solvent": solvent_info["name"],
                "smiles": canon,
                "score": solvent_info["score"],
            })
        else:
            total_score += 0.5  # Unknown solvent = neutral
            details.append({
                "solvent": "unknown",
                "smiles": canon,
                "score": 0.5,
            })

    avg_score = total_score / len(solvent_smiles_list)
    return avg_score, details


def compute_greenness_score(
    product_smiles: str,
    reactant_smiles_list: list[str],
    reagent_smiles_list: list[str] | None = None,
    solvent_smiles_list: list[str] | None = None,
) -> tuple[float, dict]:
    """Compute a greenness score for a retrosynthetic step.

    Score range: 0.0 (not green) to 1.0 (very green).

    Returns:
        Tuple of (score, details_dict).
    """
    if reagent_smiles_list is None:
        reagent_smiles_list = []
    if solvent_smiles_list is None:
        solvent_smiles_list = []

    details = {}

    # Atom economy
    ae = compute_atom_economy(product_smiles, reactant_smiles_list)
    details["atom_economy"] = ae

    # Solvent score
    solvent_score, solvent_details = score_solvents(solvent_smiles_list)
    details["solvent_score"] = solvent_score
    details["solvent_details"] = solvent_details

    # Component count penalty (proxy for PMI)
    n_components = len(reactant_smiles_list) + len(reagent_smiles_list) + len(solvent_smiles_list)
    component_penalty = max(0.0, (n_components - 3) * 0.05)
    details["component_count"] = n_components
    details["component_penalty"] = component_penalty

    # Combined score (weighted)
    score = (
        0.4 * ae
        + 0.4 * solvent_score
        + 0.2 * max(0.0, 1.0 - component_penalty)
    )

    return score, details
