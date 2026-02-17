"""Safety screening module — PAINS/BRENK alerts + Lipinski druglikeness.

Uses RDKit FilterCatalog for structural alerts and Descriptors for properties.
All CPU-only, no GPU needed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Lazy-loaded filter catalogs
_pains_catalog = None
_brenk_catalog = None


def _get_catalogs():
    """Lazy-load RDKit filter catalogs."""
    global _pains_catalog, _brenk_catalog
    if _pains_catalog is None:
        from rdkit.Chem.FilterCatalog import (
            FilterCatalog,
            FilterCatalogParams,
        )

        pains_params = FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _pains_catalog = FilterCatalog(pains_params)

        brenk_params = FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        _brenk_catalog = FilterCatalog(brenk_params)

    return _pains_catalog, _brenk_catalog


def screen_molecule(smiles: str) -> dict:
    """Screen a molecule for structural alerts and druglikeness.

    Args:
        smiles: SMILES string of the molecule to screen.

    Returns:
        Dict matching SafetyResult schema.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"alerts": [], "druglikeness": None, "tox_flags": []}

    # --- Structural alerts ---
    alerts = []
    pains_cat, brenk_cat = _get_catalogs()

    for catalog, source in [(pains_cat, "PAINS"), (brenk_cat, "BRENK")]:
        entries = catalog.GetMatches(mol)
        for entry in entries:
            alerts.append({
                "name": entry.GetDescription(),
                "smarts": None,
                "severity": "critical" if source == "PAINS" else "warning",
                "description": f"{source} filter match",
            })

    # --- Lipinski druglikeness ---
    mw = round(Descriptors.ExactMolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    violations = []
    if mw > 500:
        violations.append("MW > 500")
    if logp > 5:
        violations.append("LogP > 5")
    if hbd > 5:
        violations.append("HBD > 5")
    if hba > 10:
        violations.append("HBA > 10")

    druglikeness = {
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "passes_lipinski": len(violations) <= 1,  # Lipinski allows 1 violation
        "violations": violations,
    }

    # --- Toxicity flags (simple heuristic) ---
    tox_flags = []
    # Check for reactive functional groups
    reactive_patterns = {
        "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
        "epoxide": "C1OC1",
        "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
        "aldehyde": "[CX3H1](=O)",
    }
    for name, smarts in reactive_patterns.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            tox_flags.append(name)

    return {
        "alerts": alerts,
        "druglikeness": druglikeness,
        "tox_flags": tox_flags,
    }
