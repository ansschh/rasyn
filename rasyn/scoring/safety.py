"""Safety scoring: reagent hazard flags + dangerous motif detection.

Identifies safety risks in proposed retrosynthetic steps through:
  1. Reagent/reactant hazard lookup (known dangerous chemicals)
  2. SMARTS-based dangerous motif detection (azides, peroxides, etc.)
  3. Binary safety flags per risk category
"""

from __future__ import annotations

from rdkit import Chem


# SMARTS patterns for dangerous functional groups / motifs
HAZARDOUS_MOTIFS = {
    "azide": "[N-]=[N+]=[N-]",
    "organic_azide": "[C,c][N]=[N+]=[N-]",
    "diazo": "[C]=[N+]=[N-]",
    "peroxide": "[O][O]",
    "acyl_chloride": "[C](=O)[Cl]",
    "acid_fluoride": "[C](=O)[F]",
    "isocyanate": "[N]=[C]=[O]",
    "acyl_azide": "[C](=O)[N]=[N+]=[N-]",
    "nitro_aromatic": "[c][N+](=O)[O-]",
    "nitroso": "[N]=O",
    "epoxide": "C1OC1",
    "aziridine": "C1NC1",
    "thiol": "[SH]",
    "phosphine": "[P]([C,c])([C,c])[C,c]",
    "hydrazine": "[NH][NH]",
    "hydroxylamine": "[NH][OH]",
}

# Compiled SMARTS patterns
_COMPILED_MOTIFS: dict[str, Chem.Mol] = {}


def _get_compiled_motifs() -> dict[str, Chem.Mol]:
    """Lazily compile SMARTS patterns."""
    global _COMPILED_MOTIFS
    if not _COMPILED_MOTIFS:
        for name, smarts in HAZARDOUS_MOTIFS.items():
            mol = Chem.MolFromSmarts(smarts)
            if mol is not None:
                _COMPILED_MOTIFS[name] = mol
    return _COMPILED_MOTIFS


# Known hazardous reagents (canonical SMILES -> hazard info)
HAZARDOUS_REAGENTS = {
    # Pyrophorics
    "CCCC[Li]": {"category": "pyrophoric", "name": "n-BuLi", "severity": "high"},
    "CC(C)[Li]": {"category": "pyrophoric", "name": "t-BuLi", "severity": "critical"},
    "[Li]CC": {"category": "pyrophoric", "name": "EtLi", "severity": "high"},
    # Strong reductants
    "CC(C)[Al](OC(C)C)CC": {"category": "pyrophoric", "name": "DIBAL-H", "severity": "high"},
    "[Na][H]": {"category": "reactive", "name": "NaH", "severity": "medium"},
    "[Li][AlH4]": {"category": "reactive", "name": "LiAlH4", "severity": "medium"},
    # Toxic
    "ClC(Cl)Cl": {"category": "toxic", "name": "chloroform", "severity": "medium"},
    "C(=O)(Cl)Cl": {"category": "toxic", "name": "phosgene", "severity": "critical"},
    "[C-]#[O+]": {"category": "toxic", "name": "CO", "severity": "high"},
    # Explosive precursors
    "O=[N+]([O-])c1ccccc1": {"category": "explosive_precursor", "name": "nitrobenzene", "severity": "medium"},
}

# Hazardous solvent list
HAZARDOUS_SOLVENTS = {
    "ClCCl": {"name": "DCM", "severity": "medium", "reason": "carcinogen suspect"},
    "ClC(Cl)Cl": {"name": "chloroform", "severity": "medium", "reason": "hepatotoxic"},
    "C1CCOC1": {"name": "THF", "severity": "low", "reason": "peroxide former"},
    "CCOCC": {"name": "diethyl ether", "severity": "medium", "reason": "peroxide former, highly flammable"},
    "CS(C)=O": {"name": "DMSO", "severity": "low", "reason": "skin penetrant"},
    "CN(C)C=O": {"name": "DMF", "severity": "medium", "reason": "reproductive toxin"},
    "CN1CCCC1=O": {"name": "NMP", "severity": "medium", "reason": "reproductive toxin"},
    "CC#N": {"name": "MeCN", "severity": "low", "reason": "flammable"},
    "C1COCCO1": {"name": "1,4-dioxane", "severity": "high", "reason": "carcinogen"},
    "ClCCCl": {"name": "DCE", "severity": "high", "reason": "carcinogen"},
}


def detect_hazardous_motifs(smiles: str) -> list[dict]:
    """Detect hazardous functional groups in a molecule.

    Returns:
        List of dicts with 'motif_name' and 'count' for each match.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    motifs = _get_compiled_motifs()
    matches = []

    for name, pattern in motifs.items():
        hits = mol.GetSubstructMatches(pattern)
        if hits:
            matches.append({"motif_name": name, "count": len(hits)})

    return matches


def check_reagent_hazards(smiles: str) -> dict | None:
    """Check if a SMILES matches a known hazardous reagent."""
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if Chem.MolFromSmiles(smiles) else None
    if canon and canon in HAZARDOUS_REAGENTS:
        return HAZARDOUS_REAGENTS[canon]
    return None


def compute_safety_score(
    reactant_smiles_list: list[str],
    reagent_smiles_list: list[str] | None = None,
) -> tuple[float, dict]:
    """Compute a safety score for a retrosynthetic step.

    Score range: 0.0 (very dangerous) to 1.0 (safe).

    Returns:
        Tuple of (score, details_dict).
    """
    if reagent_smiles_list is None:
        reagent_smiles_list = []

    all_molecules = reactant_smiles_list + reagent_smiles_list
    details = {
        "hazardous_motifs": [],
        "hazardous_reagents": [],
        "risk_tags": [],
        "num_issues": 0,
    }

    penalty = 0.0

    for smi in all_molecules:
        # Check motifs
        motifs = detect_hazardous_motifs(smi)
        for motif in motifs:
            details["hazardous_motifs"].append(motif)
            if motif["motif_name"] in ("azide", "diazo", "organic_azide", "acyl_azide"):
                penalty += 0.3
                details["risk_tags"].append("explosive_risk")
            elif motif["motif_name"] in ("peroxide",):
                penalty += 0.2
                details["risk_tags"].append("peroxide_risk")
            elif motif["motif_name"] in ("isocyanate", "acyl_chloride"):
                penalty += 0.1
                details["risk_tags"].append("toxic_gas_risk")
            else:
                penalty += 0.05

        # Check known hazardous reagents
        hazard = check_reagent_hazards(smi)
        if hazard:
            details["hazardous_reagents"].append(hazard)
            severity_penalty = {"low": 0.05, "medium": 0.15, "high": 0.3, "critical": 0.5}
            penalty += severity_penalty.get(hazard["severity"], 0.1)
            details["risk_tags"].append(hazard["category"])

    details["num_issues"] = len(details["hazardous_motifs"]) + len(details["hazardous_reagents"])
    details["risk_tags"] = list(set(details["risk_tags"]))

    score = max(0.0, 1.0 - penalty)
    return score, details
