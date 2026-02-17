"""Green chemistry scoring module — atom economy, E-factor, solvent scoring.

All calculations are pure RDKit math — no external dependencies.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# CHEM21 solvent scoring (0 = hazardous, 1 = green)
# Simplified scoring based on CHEM21 solvent selection guide
SOLVENT_SCORES = {
    "water": 1.0,
    "ethanol": 0.9,
    "isopropanol": 0.85,
    "ethyl acetate": 0.8,
    "acetone": 0.75,
    "methanol": 0.7,
    "toluene": 0.4,
    "thf": 0.35,
    "dcm": 0.2,
    "dichloromethane": 0.2,
    "dmf": 0.25,
    "dmso": 0.5,
    "chloroform": 0.15,
    "dioxane": 0.1,
    "diethyl ether": 0.3,
    "hexane": 0.35,
    "acetonitrile": 0.45,
    "pyridine": 0.3,
}


def _mol_weight(smiles: str) -> float | None:
    """Get molecular weight from SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.ExactMolWt(mol)


def atom_economy(product_smiles: str, reactant_smiles_list: list[str]) -> float | None:
    """Calculate atom economy: MW(desired product) / MW(all reactants) * 100.

    Returns percentage (0-100) or None if calculation fails.
    """
    product_mw = _mol_weight(product_smiles)
    if product_mw is None or product_mw == 0:
        return None

    total_reactant_mw = 0.0
    for smi in reactant_smiles_list:
        mw = _mol_weight(smi)
        if mw is None:
            return None
        total_reactant_mw += mw

    if total_reactant_mw == 0:
        return None

    return round((product_mw / total_reactant_mw) * 100, 1)


def e_factor_estimate(product_smiles: str, reactant_smiles_list: list[str]) -> float | None:
    """Estimate E-factor: (MW waste) / (MW product).

    Lower is better. Ideal = 0 (all atoms incorporated).
    This is a simplified estimate — real E-factor needs actual mass of waste.
    """
    product_mw = _mol_weight(product_smiles)
    if product_mw is None or product_mw == 0:
        return None

    total_reactant_mw = sum(
        _mol_weight(s) or 0 for s in reactant_smiles_list
    )
    if total_reactant_mw == 0:
        return None

    waste_mw = total_reactant_mw - product_mw
    return round(max(0, waste_mw / product_mw), 2)


def solvent_score(solvent_name: str | None = None) -> float:
    """Score a solvent on the CHEM21 green scale (0-1, 1 = greenest).

    Returns 0.5 (neutral) if solvent unknown.
    """
    if not solvent_name:
        return 0.5
    return SOLVENT_SCORES.get(solvent_name.lower().strip(), 0.5)


def score_routes(target_smiles: str, routes: list[dict]) -> dict:
    """Score all routes for green chemistry metrics.

    Returns dict matching GreenChemResult schema.
    """
    if not routes:
        return {"atom_economy": None, "e_factor": None, "solvent_score": None, "details": {}}

    # Score the best route (rank 1)
    best_route = routes[0]
    steps = best_route.get("steps", [])

    ae_values = []
    ef_values = []

    for step in steps:
        product = step.get("product", "")
        reactants = step.get("reactants", [])
        if product and reactants:
            ae = atom_economy(product, reactants)
            ef = e_factor_estimate(product, reactants)
            if ae is not None:
                ae_values.append(ae)
            if ef is not None:
                ef_values.append(ef)

    avg_ae = round(sum(ae_values) / len(ae_values), 1) if ae_values else None
    avg_ef = round(sum(ef_values) / len(ef_values), 2) if ef_values else None

    return {
        "atom_economy": avg_ae,
        "e_factor": avg_ef,
        "solvent_score": 0.5,  # Default — need reaction conditions for real score
        "details": {
            "per_step_atom_economy": ae_values,
            "per_step_e_factor": ef_values,
            "num_steps_scored": len(ae_values),
        },
    }
