"""Learn module — institutional memory for the Chemistry OS.

Slice 9: Outcome recording, similarity search, insight generation,
route ranking explainer, and feedback loop.

Uses Morgan fingerprints for reaction similarity (no DRFP dependency).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Morgan fingerprint-based reaction similarity
# ---------------------------------------------------------------------------

def _reaction_fingerprint(reaction_smiles: str) -> Optional[list[float]]:
    """Compute a difference Morgan fingerprint for a reaction.

    Takes reaction SMILES in format 'reactants>>product' and returns a
    2048-bit fingerprint as a float list (product_fp - reactants_fp).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import numpy as np

        parts = reaction_smiles.split(">>")
        if len(parts) != 2:
            return None

        reactants_smi, product_smi = parts[0].strip(), parts[1].strip()

        # Parse product
        prod_mol = Chem.MolFromSmiles(product_smi)
        if prod_mol is None:
            return None
        prod_fp = AllChem.GetMorganFingerprintAsBitVect(prod_mol, 2, nBits=2048)
        prod_arr = np.zeros(2048)
        for bit in prod_fp.GetOnBits():
            prod_arr[bit] = 1.0

        # Parse reactants (may be dot-separated)
        react_fp_combined = np.zeros(2048)
        for rsmi in reactants_smi.split("."):
            rmol = Chem.MolFromSmiles(rsmi.strip())
            if rmol is None:
                continue
            rfp = AllChem.GetMorganFingerprintAsBitVect(rmol, 2, nBits=2048)
            for bit in rfp.GetOnBits():
                react_fp_combined[bit] = 1.0

        # Difference fingerprint
        diff = prod_arr - react_fp_combined
        return diff.tolist()

    except Exception as e:
        logger.warning(f"Fingerprint computation failed: {e}")
        return None


def _tanimoto_similarity(fp1: list[float], fp2: list[float]) -> float:
    """Compute Tanimoto similarity between two fingerprint vectors."""
    try:
        import numpy as np
        a = np.array(fp1)
        b = np.array(fp2)
        # Convert to binary for Tanimoto
        a_bin = (a != 0).astype(float)
        b_bin = (b != 0).astype(float)
        intersection = np.sum(a_bin * b_bin)
        union = np.sum(a_bin) + np.sum(b_bin) - intersection
        if union == 0:
            return 0.0
        return float(intersection / union)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Outcome recording
# ---------------------------------------------------------------------------

def record_outcome(
    outcome: str,
    actual_yield: float | None = None,
    failure_reason: str | None = None,
    conditions: dict | None = None,
    notes: str | None = None,
    reaction_id: int | None = None,
    experiment_id: str | None = None,
    reaction_smiles: str | None = None,
) -> dict:
    """Record the outcome of an experiment or reaction.

    Updates the Reaction row if reaction_id is provided, or finds matching
    reactions by experiment_id or reaction_smiles.

    Returns dict with reaction_id and count of insights generated.
    """
    from rasyn.db.engine import sync_engine
    from rasyn.db.models import Reaction, Experiment
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select

    SessionLocal = sessionmaker(bind=sync_engine)
    session = SessionLocal()

    try:
        rxn = None

        if reaction_id:
            rxn = session.get(Reaction, reaction_id)
        elif experiment_id:
            # Find reactions linked to this experiment via route
            exp = session.execute(
                select(Experiment).where(Experiment.id == experiment_id)
            ).scalar_one_or_none()
            if exp and exp.product_smiles:
                rxn = session.execute(
                    select(Reaction).where(
                        Reaction.product_smiles == exp.product_smiles
                    ).order_by(Reaction.created_at.desc())
                ).scalars().first()
        elif reaction_smiles:
            # Find by reaction SMILES
            rxn = session.execute(
                select(Reaction).where(
                    Reaction.reaction_smiles == reaction_smiles
                ).order_by(Reaction.created_at.desc())
            ).scalars().first()

        if rxn is None:
            # Create a new reaction record if none found
            parts = (reaction_smiles or "").split(">>")
            product = parts[-1].strip() if parts else ""
            reactants = parts[0].strip() if len(parts) > 1 else ""

            rxn = Reaction(
                product_smiles=product,
                reactants_smiles=reactants,
                reaction_smiles=reaction_smiles or "",
                conditions=conditions,
            )
            session.add(rxn)
            session.flush()

        # Update outcome
        rxn.outcome = outcome
        if actual_yield is not None:
            rxn.actual_yield = actual_yield
        if failure_reason:
            rxn.failure_reason = failure_reason
        if conditions:
            rxn.conditions = conditions

        session.commit()

        # Generate insights from this and similar reactions
        insights_count = _regenerate_insights_for_similar(rxn.id, session)

        return {
            "reaction_id": rxn.id,
            "outcome": outcome,
            "insights_generated": insights_count,
            "message": f"Outcome '{outcome}' recorded for reaction {rxn.id}",
        }

    except Exception as e:
        session.rollback()
        logger.exception(f"Failed to record outcome: {e}")
        raise
    finally:
        session.close()


def _regenerate_insights_for_similar(reaction_id: int, session) -> int:
    """After recording an outcome, check if this creates new insights."""
    # For now, return 0 — insights are generated on-demand via generate_insights()
    return 0


# ---------------------------------------------------------------------------
# Insight generation
# ---------------------------------------------------------------------------

def generate_insights(
    reaction_smiles: str | None = None,
    target_smiles: str | None = None,
    top_k: int = 20,
) -> dict:
    """Generate actionable insights from past experiment outcomes.

    Searches for similar reactions with recorded outcomes and produces
    failure avoidance warnings, optimization suggestions, and preferences.
    """
    from rasyn.db.engine import sync_engine
    from rasyn.db.models import Reaction
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select

    SessionLocal = sessionmaker(bind=sync_engine)
    session = SessionLocal()

    try:
        # Get all reactions with outcomes
        stmt = select(Reaction).where(Reaction.outcome.isnot(None))
        results = session.execute(stmt).scalars().all()

        insights = []
        total_experiments = len(results)

        if reaction_smiles:
            query_fp = _reaction_fingerprint(reaction_smiles)
        elif target_smiles:
            # Construct a dummy reaction for similarity
            query_fp = _reaction_fingerprint(f">>{ target_smiles}")
        else:
            query_fp = None

        # Compute similarities and generate insights
        similar_reactions = []
        for rxn in results:
            sim = 0.5  # default similarity if we can't compute fingerprints
            if query_fp and rxn.reaction_smiles:
                rxn_fp = _reaction_fingerprint(rxn.reaction_smiles)
                if rxn_fp:
                    sim = _tanimoto_similarity(query_fp, rxn_fp)

            similar_reactions.append({
                "id": rxn.id,
                "reaction_smiles": rxn.reaction_smiles or "",
                "outcome": rxn.outcome,
                "actual_yield": rxn.actual_yield,
                "failure_reason": rxn.failure_reason,
                "conditions": rxn.conditions or {},
                "similarity": sim,
            })

        # Sort by similarity
        similar_reactions.sort(key=lambda x: x["similarity"], reverse=True)
        similar_reactions = similar_reactions[:top_k]

        # Generate failure avoidance insights
        for r in similar_reactions:
            if r["outcome"] == "failure" and r["similarity"] > 0.3:
                reason = r.get("failure_reason") or "Unknown reason"
                insights.append({
                    "id": f"ins_fail_{r['id']}",
                    "type": "failure_avoidance",
                    "rule": f"Similar reaction failed: {reason}",
                    "source": f"Reaction #{r['id']} (similarity: {r['similarity']:.0%})",
                    "confidence": r["similarity"],
                    "timesApplied": 0,
                })

        # Generate optimization insights from successes
        successes = [r for r in similar_reactions if r["outcome"] == "success" and r.get("actual_yield")]
        if successes:
            best = max(successes, key=lambda x: x["actual_yield"] or 0)
            conds = best.get("conditions", {})
            cond_str = ", ".join(f"{k}: {v}" for k, v in conds.items()) if conds else "N/A"
            insights.append({
                "id": f"ins_opt_{best['id']}",
                "type": "optimization",
                "rule": f"Best known yield: {best['actual_yield']:.0f}% with conditions: {cond_str}",
                "source": f"Reaction #{best['id']} (similarity: {best['similarity']:.0%})",
                "confidence": best["similarity"],
                "timesApplied": 0,
            })

        # Generate preference insights (most common successful conditions)
        if len(successes) >= 2:
            # Find common solvents/catalysts
            conditions_freq: dict[str, int] = {}
            for r in successes:
                conds = r.get("conditions", {})
                for k, v in conds.items():
                    key = f"{k}={v}"
                    conditions_freq[key] = conditions_freq.get(key, 0) + 1

            for cond, count in sorted(conditions_freq.items(), key=lambda x: -x[1]):
                if count >= 2:
                    insights.append({
                        "id": f"ins_pref_{hash(cond) % 10000}",
                        "type": "preference",
                        "rule": f"Preferred condition: {cond} (used in {count} successful reactions)",
                        "source": f"{count} past experiments",
                        "confidence": min(count / len(successes), 1.0),
                        "timesApplied": count,
                    })
                    break  # Just the top preference

        # Compute total applications
        total_applications = sum(i.get("timesApplied", 0) for i in insights)

        return {
            "insights": insights,
            "total_experiments": total_experiments,
            "total_applications": total_applications,
        }

    except Exception as e:
        logger.exception(f"Insight generation failed: {e}")
        return {"insights": [], "total_experiments": 0, "total_applications": 0}
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Route ranking explainer
# ---------------------------------------------------------------------------

def explain_ranking(route: dict, comparison_route: dict | None = None) -> dict:
    """Explain why a route was ranked as it was.

    Uses honest raw metrics — every value cited is real, computed data.
    """
    m = route.get("score_breakdown") or route.get("metrics") or {}
    num_steps = m.get("num_steps", len(route.get("steps", [])))
    purchasable = m.get("all_purchasable", route.get("all_purchasable", False))
    confidence = m.get("model_confidence", route.get("overall_score"))

    factors = []

    # 1. Model confidence (REAL: from beam search)
    if confidence is not None:
        impact = "positive" if confidence > 0.7 else "neutral" if confidence > 0.4 else "negative"
        factors.append({
            "factor": "Model Confidence",
            "value": f"{confidence:.1%}",
            "impact": impact,
            "detail": f"Average beam search confidence across {num_steps} step(s). Routes are ranked by this metric.",
        })

    # 2. Step count (FACT)
    factors.append({
        "factor": "Step Count",
        "value": str(num_steps),
        "impact": "positive" if num_steps <= 3 else "neutral" if num_steps <= 5 else "negative",
        "detail": f"Route has {num_steps} synthetic step(s).",
    })

    # 3. Atom economy (REAL: RDKit MW calculation)
    ae = m.get("atom_economy_pct")
    if ae is not None:
        impact = "positive" if ae > 70 else "neutral" if ae > 40 else "negative"
        factors.append({
            "factor": "Atom Economy",
            "value": f"{ae:.1f}%",
            "impact": impact,
            "detail": f"Calculated from RDKit molecular weights: {ae:.1f}% of reactant atoms incorporated into product.",
        })

    # 4. E-factor (REAL: RDKit MW calculation)
    ef = m.get("e_factor")
    if ef is not None:
        impact = "positive" if ef < 1.0 else "neutral" if ef < 5.0 else "negative"
        factors.append({
            "factor": "E-Factor",
            "value": f"{ef:.2f}",
            "impact": impact,
            "detail": f"Waste-to-product ratio by molecular weight. Ideal = 0, pharma typical = 25-100.",
        })

    # 5. Safety alerts (REAL: RDKit PAINS/BRENK)
    alert_count = m.get("safety_alert_count", 0)
    alert_names = m.get("safety_alerts", [])
    impact = "positive" if alert_count == 0 else "negative" if alert_count > 2 else "neutral"
    detail = "No structural alerts from PAINS/BRENK screening." if alert_count == 0 else f"Alerts: {', '.join(alert_names[:3])}"
    factors.append({
        "factor": "Safety Alerts",
        "value": f"{alert_count} alert(s)",
        "impact": impact,
        "detail": detail,
    })

    # 6. Evidence (REAL: fingerprint search + live APIs)
    ev_count = m.get("evidence_count", 0)
    top_sim = m.get("evidence_top_similarity")
    local = m.get("evidence_local_hits", 0)
    live = m.get("evidence_live_hits", 0)
    if ev_count > 0:
        sim_str = f", top Tanimoto: {top_sim:.2f}" if top_sim else ""
        factors.append({
            "factor": "Literature Evidence",
            "value": f"{ev_count} hit(s)",
            "impact": "positive" if ev_count >= 3 else "neutral",
            "detail": f"{local} local reaction match(es), {live} published paper(s){sim_str}.",
        })
    else:
        factors.append({
            "factor": "Literature Evidence",
            "value": "None found",
            "impact": "neutral",
            "detail": "No similar reactions found in USPTO index or literature APIs.",
        })

    # 7. Starting material availability (FACT)
    sm_total = m.get("starting_materials_total", 0)
    factors.append({
        "factor": "Starting Materials",
        "value": f"{'All purchasable' if purchasable else f'{sm_total} identified'}",
        "impact": "positive" if purchasable else "neutral",
        "detail": "All starting materials are commercially available." if purchasable else "Availability not yet confirmed for all materials.",
    })

    # Comparison
    if comparison_route:
        comp_conf = (comparison_route.get("score_breakdown") or {}).get("model_confidence", comparison_route.get("overall_score", 0))
        question = f"Why was this route (confidence {confidence:.1%}) ranked above the alternative (confidence {comp_conf:.1%})?"
    else:
        question = f"Why was this route ranked #{route.get('rank', '?')}?"

    return {
        "question": question,
        "factors": factors,
    }


# ---------------------------------------------------------------------------
# Past experiments retrieval
# ---------------------------------------------------------------------------

def get_past_experiments(limit: int = 50) -> dict:
    """Retrieve past experiments with outcomes for the Learn view."""
    from rasyn.db.engine import sync_engine
    from rasyn.db.models import Reaction
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select, func

    SessionLocal = sessionmaker(bind=sync_engine)
    session = SessionLocal()

    try:
        # Get reactions with outcomes
        stmt = (
            select(Reaction)
            .where(Reaction.outcome.isnot(None))
            .order_by(Reaction.created_at.desc())
            .limit(limit)
        )
        reactions = session.execute(stmt).scalars().all()

        # Count totals
        total_stmt = select(func.count(Reaction.id)).where(Reaction.outcome.isnot(None))
        total = session.execute(total_stmt).scalar() or 0

        success_count = session.execute(
            select(func.count(Reaction.id)).where(Reaction.outcome == "success")
        ).scalar() or 0

        avg_yield = session.execute(
            select(func.avg(Reaction.actual_yield)).where(
                Reaction.outcome == "success",
                Reaction.actual_yield.isnot(None),
            )
        ).scalar()

        experiments = []
        for rxn in reactions:
            experiments.append({
                "id": f"RXN-{rxn.id:04d}",
                "date": rxn.created_at.strftime("%Y-%m-%d") if rxn.created_at else "",
                "target": rxn.product_smiles or "",
                "reaction": rxn.reaction_smiles or f"{rxn.reactants_smiles}>>{rxn.product_smiles}",
                "conditions": str(rxn.conditions or {}),
                "outcome": rxn.outcome or "unknown",
                "yield": f"{rxn.actual_yield:.0f}%" if rxn.actual_yield else None,
                "notes": rxn.failure_reason or "",
                "scaffold": rxn.product_smiles[:30] if rxn.product_smiles else "",
                "impactOnPlanning": (
                    f"Failure recorded — avoid similar conditions"
                    if rxn.outcome == "failure"
                    else f"Success — {rxn.actual_yield:.0f}% yield achievable"
                    if rxn.actual_yield
                    else "Outcome recorded for future reference"
                ),
            })

        success_rate = success_count / total if total > 0 else 0

        return {
            "total_experiments": total,
            "total_reactions": total,
            "success_rate": round(success_rate, 3),
            "avg_yield": round(avg_yield, 1) if avg_yield else None,
            "total_insights": 0,  # Will be populated by generate_insights
            "past_experiments": experiments,
        }

    except Exception as e:
        logger.exception(f"Failed to get past experiments: {e}")
        return {
            "total_experiments": 0,
            "total_reactions": 0,
            "success_rate": 0,
            "avg_yield": None,
            "total_insights": 0,
            "past_experiments": [],
        }
    finally:
        session.close()
