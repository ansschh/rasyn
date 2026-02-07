"""Scalability scoring: operational feasibility at scale.

Penalizes conditions and reagents that are problematic at larger scale:
  - Cryogenic temperatures
  - High pressure
  - Sensitive reagents (air/moisture)
  - Excessive protecting group usage
  - Rare/exotic catalysts
  - High component count
"""

from __future__ import annotations

from rdkit import Chem


# Sensitive reagents that are problematic at scale
SENSITIVE_REAGENTS = {
    "CCCC[Li]": {"name": "n-BuLi", "penalty": 0.25, "reason": "pyrophoric, cryogenic"},
    "CC(C)[Li]": {"name": "t-BuLi", "penalty": 0.35, "reason": "pyrophoric, extreme cryo"},
    "[Na][H]": {"name": "NaH", "penalty": 0.10, "reason": "moisture sensitive"},
    "[Li][AlH4]": {"name": "LiAlH4", "penalty": 0.15, "reason": "moisture sensitive, fire risk"},
    "O=S(=O)(O)O": {"name": "H2SO4", "penalty": 0.05, "reason": "corrosive"},
    "[Na]OCC": {"name": "NaOEt", "penalty": 0.05, "reason": "moisture sensitive"},
}

# Protecting group SMARTS patterns (penalize excessive usage)
PROTECTING_GROUP_PATTERNS = {
    "Boc": "OC(=O)C(C)(C)C",
    "Fmoc": "OC(=O)OCC1c2ccccc2-c2ccccc21",
    "Cbz": "OC(=O)OCc1ccccc1",
    "TBS": "[Si](C)(C)C(C)(C)C",
    "TMS": "[Si](C)(C)C",
    "Bn": "[CH2]c1ccccc1",
    "PMB": "[CH2]c1ccc(OC)cc1",
    "TIPS": "[Si](C(C)C)(C(C)C)C(C)C",
}


def compute_scalability_score(
    reactant_smiles_list: list[str],
    conditions: dict | None = None,
    reagent_smiles_list: list[str] | None = None,
) -> tuple[float, dict]:
    """Compute a scalability score for a retrosynthetic step.

    Score range: 0.0 (very hard to scale) to 1.0 (easy to scale).

    Args:
        reactant_smiles_list: Reactant SMILES.
        conditions: Optional dict with temp_c, pressure_bar, etc.
        reagent_smiles_list: Optional reagent SMILES.

    Returns:
        Tuple of (score, details_dict).
    """
    if conditions is None:
        conditions = {}
    if reagent_smiles_list is None:
        reagent_smiles_list = []

    details = {
        "penalties": [],
        "total_penalty": 0.0,
    }
    penalty = 0.0

    # Temperature penalty
    temp_c = conditions.get("temp_c")
    if temp_c is not None:
        if temp_c < -78:
            penalty += 0.3
            details["penalties"].append(f"extreme cryo ({temp_c}C)")
        elif temp_c < -40:
            penalty += 0.2
            details["penalties"].append(f"cryogenic ({temp_c}C)")
        elif temp_c < 0:
            penalty += 0.1
            details["penalties"].append(f"sub-zero ({temp_c}C)")
        elif temp_c > 200:
            penalty += 0.15
            details["penalties"].append(f"high temp ({temp_c}C)")

    # Pressure penalty
    pressure = conditions.get("pressure_bar")
    if pressure is not None:
        if pressure > 50:
            penalty += 0.25
            details["penalties"].append(f"high pressure ({pressure} bar)")
        elif pressure > 10:
            penalty += 0.1
            details["penalties"].append(f"elevated pressure ({pressure} bar)")

    # Sensitive reagent penalties
    all_molecules = reactant_smiles_list + reagent_smiles_list
    for smi in all_molecules:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in SENSITIVE_REAGENTS:
            info = SENSITIVE_REAGENTS[canon]
            penalty += info["penalty"]
            details["penalties"].append(f"sensitive reagent: {info['name']} ({info['reason']})")

    # Component count penalty (more components = harder workup)
    n_components = len(reactant_smiles_list) + len(reagent_smiles_list)
    if n_components > 4:
        penalty += 0.15
        details["penalties"].append(f"high component count ({n_components})")
    elif n_components > 3:
        penalty += 0.05
        details["penalties"].append(f"moderate component count ({n_components})")

    # Protecting group detection (penalize PG-heavy chemistry)
    pg_count = 0
    for smi in all_molecules:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for pg_name, pg_smarts in PROTECTING_GROUP_PATTERNS.items():
            pattern = Chem.MolFromSmarts(pg_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                pg_count += 1

    if pg_count > 2:
        penalty += 0.15
        details["penalties"].append(f"heavy PG usage ({pg_count} groups)")
    elif pg_count > 0:
        penalty += 0.05 * pg_count
        details["penalties"].append(f"PG usage ({pg_count} groups)")

    details["total_penalty"] = penalty
    score = max(0.0, 1.0 - penalty)
    return score, details
