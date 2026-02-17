"""Composite route scoring and ranking.

Produces a single overall score per route from multiple dimensions:
- Round-trip confidence (model + forward validation)
- Step count penalty
- Commercial availability of starting materials
- Safety (from PAINS/BRENK screening)
- Green chemistry (atom economy + E-factor + solvent)
- Literature precedent
"""

from __future__ import annotations

import logging
from statistics import mean

logger = logging.getLogger(__name__)


def score_route(route: dict, safety: dict | None = None, green: dict | None = None) -> dict:
    """Compute composite scores for a retrosynthetic route.

    Args:
        route: Route dict with steps, starting_materials, etc.
        safety: Safety screening result from safety module.
        green: Green chemistry scoring result from green module.

    Returns:
        Dict with per-dimension scores and overall composite score.
    """
    steps = route.get("steps", [])
    if not steps:
        return {"overall": 0.0, "breakdown": {}}

    # 1. Round-trip confidence
    confidences = [s.get("score", 0.5) for s in steps]
    rt_score = mean(confidences) if confidences else 0.5

    # 2. Step count penalty — fewer steps is better
    n = len(steps)
    step_penalty = 1.0 / (1 + 0.15 * n)

    # 3. Commercial availability of starting materials
    sm = route.get("starting_materials", [])
    all_purchasable = route.get("all_purchasable", False)
    if all_purchasable:
        avail_score = 1.0
    elif sm:
        # Estimate: check inventory coverage (default assumes partial)
        avail_score = 0.5
    else:
        avail_score = 0.3

    # 4. Safety score (from PAINS/BRENK alerts)
    safety_score = 1.0
    if safety:
        n_alerts = len(safety.get("alerts", []))
        n_tox = len(safety.get("tox_flags", []))
        safety_score = max(0.0, 1.0 - n_alerts * 0.15 - n_tox * 0.1)

    # 5. Green chemistry score
    green_score = 0.5  # default
    if green:
        ae = green.get("atom_economy")
        ef = green.get("e_factor")
        sol = green.get("solvent_score")
        components = []
        if ae is not None:
            components.append(min(ae / 100.0, 1.0))
        if ef is not None:
            components.append(max(0.0, 1.0 - ef * 0.1))
        if sol is not None:
            components.append(sol)
        if components:
            green_score = mean(components)

    # 6. Precedent score (stub — will be populated from evidence module)
    precedent_score = 0.3

    # Weighted composite
    weights = {
        "roundtrip_confidence": 0.30,
        "step_efficiency": 0.10,
        "availability": 0.20,
        "safety": 0.15,
        "green_chemistry": 0.10,
        "precedent": 0.15,
    }
    breakdown = {
        "roundtrip_confidence": round(rt_score, 3),
        "step_efficiency": round(step_penalty, 3),
        "availability": round(avail_score, 3),
        "safety": round(safety_score, 3),
        "green_chemistry": round(green_score, 3),
        "precedent": round(precedent_score, 3),
    }

    overall = sum(weights[k] * breakdown[k] for k in weights)

    return {
        "overall": round(overall, 3),
        "breakdown": breakdown,
        "weights": weights,
    }


def score_and_rank_routes(
    routes: list[dict],
    safety: dict | None = None,
    green: dict | None = None,
) -> list[dict]:
    """Score and re-rank all routes by composite score.

    Updates routes in-place with score_breakdown and re-ranks them.
    Returns the sorted route list.
    """
    for route in routes:
        scores = score_route(route, safety=safety, green=green)
        route["overall_score"] = scores["overall"]
        route["score_breakdown"] = scores["breakdown"]

    # Re-rank by overall score (descending)
    routes.sort(key=lambda r: r.get("overall_score", 0), reverse=True)
    for i, route in enumerate(routes):
        route["rank"] = i + 1

    return routes


def explain_ranking(route: dict) -> list[dict]:
    """Generate human-readable explanations for why a route ranks where it does.

    Returns list of {factor, value, impact, description} dicts.
    """
    breakdown = route.get("score_breakdown", {})
    explanations = []

    factor_labels = {
        "roundtrip_confidence": ("Model Confidence", "How well the forward model validates predicted reactions"),
        "step_efficiency": ("Step Count", "Fewer synthesis steps means simpler execution"),
        "availability": ("Starting Material Availability", "Whether starting materials are commercially purchasable"),
        "safety": ("Safety Profile", "Absence of structural alerts (PAINS/BRENK) and hazardous groups"),
        "green_chemistry": ("Green Chemistry", "Atom economy, E-factor, and solvent sustainability"),
        "precedent": ("Literature Precedent", "Similarity to known reactions in the literature"),
    }

    for key, (label, description) in factor_labels.items():
        value = breakdown.get(key, 0.5)
        if value >= 0.8:
            impact = "positive"
        elif value >= 0.5:
            impact = "neutral"
        else:
            impact = "negative"

        explanations.append({
            "factor": label,
            "value": f"{value:.0%}",
            "raw_value": value,
            "impact": impact,
            "description": description,
        })

    return explanations
