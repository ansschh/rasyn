"""Learn API routes — outcome recording, insights, ranking explainer.

Slice 9: POST /learn/record-outcome, GET /learn/insights,
         GET /learn/explain-ranking/{route_id}, GET /learn/experiments
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query

from rasyn.api.schemas_v2 import (
    OutcomeRequest,
    OutcomeResponse,
    InsightsResponse,
    RankingExplanation,
    LearnStatsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["learn"])


@router.post("/learn/record-outcome", response_model=OutcomeResponse)
async def record_outcome(req: OutcomeRequest):
    """Record the outcome of an experiment or reaction.

    This creates a feedback loop: outcomes are used to generate insights
    that improve future route planning and ranking.
    """
    try:
        from rasyn.modules.learn import record_outcome as _record
        from rasyn.modules.admin import log_action

        result = _record(
            outcome=req.outcome,
            actual_yield=req.actual_yield,
            failure_reason=req.failure_reason,
            conditions=req.conditions,
            notes=req.notes,
            reaction_id=req.reaction_id,
            experiment_id=req.experiment_id,
            reaction_smiles=req.reaction_smiles,
        )

        # Audit log
        log_action(
            action="Record outcome",
            resource=f"reaction:{result.get('reaction_id')}",
            details=f"Outcome: {req.outcome}, Yield: {req.actual_yield}%",
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Outcome recording failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record outcome: {str(e)[:200]}")


@router.get("/learn/insights", response_model=InsightsResponse)
async def get_insights(
    reaction_smiles: str | None = Query(None, description="Reaction SMILES for targeted insights"),
    target_smiles: str | None = Query(None, description="Target molecule SMILES"),
    top_k: int = Query(20, ge=1, le=100),
):
    """Get actionable insights from institutional memory.

    Returns failure avoidance warnings, optimization suggestions,
    and condition preferences based on past experiment outcomes.
    """
    try:
        from rasyn.modules.learn import generate_insights

        result = generate_insights(
            reaction_smiles=reaction_smiles,
            target_smiles=target_smiles,
            top_k=top_k,
        )

        return result

    except Exception as e:
        logger.exception(f"Insight generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)[:200]}")


@router.get("/learn/explain-ranking/{route_id}")
async def explain_ranking(route_id: str):
    """Explain why a specific route was ranked as it was.

    Fetches the route from the database and generates a human-readable
    explanation of each ranking factor.
    """
    try:
        from rasyn.db.engine import async_session
        from rasyn.db.models import Route
        from rasyn.modules.learn import explain_ranking as _explain
        from sqlalchemy import select

        async with async_session() as session:
            # Try by route_id string first
            result = await session.execute(
                select(Route).where(Route.route_id == route_id)
            )
            route_row = result.scalar_first()

            # Try by DB id
            if not route_row:
                try:
                    result = await session.execute(
                        select(Route).where(Route.id == int(route_id))
                    )
                    route_row = result.scalar_first()
                except (ValueError, TypeError):
                    pass

        if not route_row:
            raise HTTPException(status_code=404, detail=f"Route '{route_id}' not found")

        route_data = route_row.tree or {}
        route_data["rank"] = route_row.rank
        route_data["overall_score"] = route_row.score or 0
        route_data["score_breakdown"] = route_row.score_breakdown
        route_data["num_steps"] = route_row.num_steps or 0
        route_data["all_purchasable"] = route_row.all_purchasable

        explanation = _explain(route_data)
        return explanation

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Ranking explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/learn/explain-ranking")
async def explain_ranking_from_job(
    job_id: str = Query(..., description="Job ID"),
    route_index: int = Query(0, ge=0, description="Route index (0 = best)"),
):
    """Explain ranking for a route within a job result."""
    try:
        from rasyn.db.engine import async_session
        from rasyn.db.models import Job
        from rasyn.modules.learn import explain_ranking as _explain
        from sqlalchemy import select

        async with async_session() as session:
            result = await session.execute(
                select(Job).where(Job.id == uuid.UUID(job_id))
            )
            job = result.scalar_one_or_none()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.result or "routes" not in job.result:
            raise HTTPException(status_code=400, detail="Job has no routes")

        routes = job.result["routes"]
        if route_index >= len(routes):
            raise HTTPException(status_code=400, detail=f"Route index {route_index} out of range")

        route = routes[route_index]
        comparison = routes[route_index + 1] if route_index + 1 < len(routes) else None

        explanation = _explain(route, comparison)
        return explanation

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Ranking explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/learn/experiments", response_model=LearnStatsResponse)
async def get_experiments(
    limit: int = Query(50, ge=1, le=200),
):
    """Get past experiments with outcomes and stats."""
    try:
        from rasyn.modules.learn import get_past_experiments

        result = get_past_experiments(limit=limit)

        # Also get insight count
        from rasyn.modules.learn import generate_insights
        insights = generate_insights()
        result["total_insights"] = len(insights.get("insights", []))

        return result

    except Exception as e:
        logger.exception(f"Experiments retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])
