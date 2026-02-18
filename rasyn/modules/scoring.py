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


def score_and_rank_routes(
    routes: list[dict],
    safety: dict | None = None,
    green: dict | None = None,
    evidence: list[dict] | None = None,
) -> list[dict]:
    """Compute metrics and rank routes by model confidence.

    Updates routes in-place with raw metrics. Ranks by the only real
    predictive signal: average model confidence across steps.
    """
    for route in routes:
        result = compute_route_metrics(route, safety=safety, green=green, evidence=evidence)
        route["metrics"] = result["metrics"]
        # Keep overall_score as model confidence for backward compat with frontend sorting
        route["overall_score"] = result["rank_key"]
        # Map to score_breakdown for backward compat (frontend reads this)
        route["score_breakdown"] = result["metrics"]

    # Rank by model confidence (descending)
    routes.sort(key=lambda r: r.get("overall_score", 0), reverse=True)
    for i, route in enumerate(routes):
        route["rank"] = i + 1

    return routes
