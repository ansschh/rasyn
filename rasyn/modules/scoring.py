"""Honest route metrics — raw values only, no fabricated composite scores.

Every metric is either:
- Computed from real chemistry (RDKit molecular weights, fingerprint similarity)
- Directly from model output (beam search confidence)
- A factual count (number of alerts, number of steps)

Nothing is normalized into a fake 0-1 range with arbitrary weights.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint checking (RDKit substructure matching)
# ---------------------------------------------------------------------------

CONSTRAINT_SMARTS = {
    "no_pd": ("[#46]", "Contains palladium"),
    "no_azide": ("[N-]=[N+]=[N-]", "Contains azide group"),
}

# Protecting group SMARTS for min_pg constraint
PG_SMARTS = [
    "OC(=O)OC(C)(C)C",          # Boc
    "OC(=O)OCC1c2ccccc2-c2ccccc21",  # Fmoc
    "OC(=O)OCc1ccccc1",          # Cbz
    "[Si](C)(C)C(C)(C)C",       # TBDMS
]


def _check_constraint_violations(route: dict, constraints: dict) -> list[str]:
    """Check a route against user-specified constraints.

    Returns a list of violated constraint names.
    """
    if not constraints:
        return []

    violations = []

    try:
        from rdkit import Chem
    except ImportError:
        return []

    # Collect all SMILES in the route
    all_smiles = set()
    for step in route.get("steps", []):
        all_smiles.add(step.get("product", ""))
        all_smiles.update(step.get("reactants", []))
    all_smiles.update(route.get("starting_materials", []))
    all_smiles.discard("")

    # Parse all molecules once
    mols = []
    for smi in all_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)

    # Check SMARTS-based constraints
    for constraint_name, (smarts_str, _desc) in CONSTRAINT_SMARTS.items():
        if not constraints.get(constraint_name):
            continue
        pattern = Chem.MolFromSmarts(smarts_str)
        if pattern is None:
            continue
        for mol in mols:
            if mol.HasSubstructMatch(pattern):
                violations.append(constraint_name)
                break

    # min_pg: check if route uses protecting groups (flag if <2 PG steps found)
    if constraints.get("min_pg"):
        pg_count = 0
        for pg_smarts in PG_SMARTS:
            pattern = Chem.MolFromSmarts(pg_smarts)
            if pattern is None:
                continue
            for mol in mols:
                if mol.HasSubstructMatch(pattern):
                    pg_count += 1
                    break
        if pg_count < 2:
            violations.append("min_pg")

    # stock_prefer: not a hard constraint, handled in ranking
    # no_cryo: advisory only, passed to copilot context

    return violations


def compute_route_metrics(
    route: dict,
    safety: dict | None = None,
    green: dict | None = None,
    evidence: list[dict] | None = None,
) -> dict:
    """Compute honest, raw metrics for a retrosynthetic route.

    Returns a dict of real values — no composite score, no arbitrary weights.
    """
    steps = route.get("steps", [])
    if not steps:
        return {"metrics": {}, "rank_key": 0.0}

    # --- Model confidence (REAL: from beam search / model logits) ---
    confidences = [s.get("score") for s in steps if s.get("score") is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else None

    # --- Step count (FACT: just a count) ---
    num_steps = len(steps)

    # --- Atom economy % (REAL: from RDKit molecular weights) ---
    atom_economy_pct = None
    e_factor = None
    if green:
        atom_economy_pct = green.get("atom_economy")  # already a real % from RDKit
        e_factor = green.get("e_factor")  # real waste/product MW ratio

    # --- Safety (REAL: from RDKit FilterCatalog PAINS/BRENK screening) ---
    alert_count = 0
    alert_names: list[str] = []
    tox_flag_count = 0
    tox_flags: list[str] = []
    if safety:
        alerts = safety.get("alerts", [])
        alert_count = len(alerts)
        alert_names = [a.get("name", "unknown") if isinstance(a, dict) else str(a) for a in alerts[:5]]
        tox_flags_raw = safety.get("tox_flags", [])
        tox_flag_count = len(tox_flags_raw)
        tox_flags = [str(f) for f in tox_flags_raw[:5]]

    # --- Evidence (REAL: from fingerprint search + live API) ---
    evidence_list = evidence or []
    evidence_count = len(evidence_list)
    local_hits = [e for e in evidence_list if e.get("similarity", 0) > 0]
    live_hits = [e for e in evidence_list if e.get("similarity", 0) == 0]
    top_similarity = max((e["similarity"] for e in local_hits), default=None)

    # --- Starting material availability (FACT: from route data) ---
    sm = route.get("starting_materials", [])
    sm_total = len(sm)
    all_purchasable = route.get("all_purchasable", False)

    metrics = {
        # Model output (real)
        "model_confidence": round(avg_confidence, 3) if avg_confidence is not None else None,

        # Route structure (fact)
        "num_steps": num_steps,

        # Green chemistry (real RDKit calculations)
        "atom_economy_pct": atom_economy_pct,
        "e_factor": e_factor,

        # Safety (real RDKit screening)
        "safety_alert_count": alert_count,
        "safety_alerts": alert_names,
        "tox_flag_count": tox_flag_count,
        "tox_flags": tox_flags,

        # Evidence (real fingerprint + API search)
        "evidence_count": evidence_count,
        "evidence_local_hits": len(local_hits),
        "evidence_live_hits": len(live_hits),
        "evidence_top_similarity": round(top_similarity, 3) if top_similarity is not None else None,

        # Starting materials (fact)
        "starting_materials_total": sm_total,
        "all_purchasable": all_purchasable,
    }

    # Rank key: model confidence is the only genuinely predictive metric
    rank_key = avg_confidence if avg_confidence is not None else 0.0

    return {"metrics": metrics, "rank_key": rank_key}


def _objective_sort_key(route: dict, objective: str):
    """Return a sort key tuple for objective-based ranking.

    Higher values are better (sorted descending).
    """
    metrics = route.get("metrics", route.get("score_breakdown", {}))
    confidence = metrics.get("model_confidence") or route.get("overall_score", 0)
    num_steps = metrics.get("num_steps", route.get("num_steps", 99))
    alert_count = metrics.get("safety_alert_count", 0)
    atom_economy = metrics.get("atom_economy_pct") or 0
    all_purchasable = 1 if route.get("all_purchasable") else 0
    sm_total = metrics.get("starting_materials_total", route.get("num_steps", 99))
    violated = 1 if route.get("constraint_violations") else 0

    if objective == "fastest":
        # Fewer steps better, then confidence
        return (-violated, -num_steps, confidence)
    elif objective == "cheapest":
        # All purchasable first, then fewer starting materials, then confidence
        return (-violated, all_purchasable, -sm_total, confidence)
    elif objective == "safest":
        # Fewer safety alerts, then confidence
        return (-violated, -alert_count, confidence)
    elif objective == "greenest":
        # Higher atom economy, then confidence
        return (-violated, atom_economy, confidence)
    else:  # "default"
        # Model confidence descending (violated routes pushed to bottom)
        return (-violated, confidence)


def score_and_rank_routes(
    routes: list[dict],
    safety: dict | None = None,
    green: dict | None = None,
    evidence: list[dict] | None = None,
    constraints: dict | None = None,
    objective: str = "default",
) -> list[dict]:
    """Compute metrics, check constraints, and rank routes.

    Updates routes in-place with raw metrics and constraint violations.
    Violating routes are pushed to bottom but kept visible (flagged).
    """
    for route in routes:
        result = compute_route_metrics(route, safety=safety, green=green, evidence=evidence)
        route["metrics"] = result["metrics"]
        # Keep overall_score as model confidence for backward compat with frontend sorting
        route["overall_score"] = result["rank_key"]
        # Map to score_breakdown for backward compat (frontend reads this)
        route["score_breakdown"] = result["metrics"]

        # Check constraint violations
        violations = _check_constraint_violations(route, constraints)
        if violations:
            route["constraint_violations"] = violations

    # Rank by objective (violated routes always at bottom)
    routes.sort(key=lambda r: _objective_sort_key(r, objective), reverse=True)
    for i, route in enumerate(routes):
        route["rank"] = i + 1

    return routes
