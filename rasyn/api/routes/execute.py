"""Execute API routes — protocol generation, PDF export, ELN push.

Slice 7: POST /execute/generate-protocol, POST /execute/export-pdf
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from rasyn.api.schemas_v2 import ProtocolRequest, ExperimentResult

logger = logging.getLogger(__name__)
router = APIRouter(tags=["execute"])


@router.post("/execute/generate-protocol", response_model=ExperimentResult)
async def generate_protocol(req: ProtocolRequest):
    """Generate a lab-ready experiment protocol from a route step.

    Takes a route (from PlanResult) and step index, returns a complete
    ExperimentTemplate with protocol steps, reagent table, samples, and workup.
    """
    try:
        from rasyn.modules.execute import generate_experiment

        result = generate_experiment(
            route=req.route,
            step_index=req.step_index,
            scale=req.scale,
        )

        # Save to database
        try:
            _save_experiment(result)
        except Exception as e:
            logger.warning(f"Failed to save experiment to DB: {e}")

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Protocol generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Protocol generation failed: {str(e)[:200]}")


@router.post("/execute/export-pdf")
async def export_pdf(req: ProtocolRequest):
    """Generate and return a PDF of the experiment protocol."""
    try:
        from rasyn.modules.execute import generate_experiment, export_protocol_pdf

        experiment = generate_experiment(
            route=req.route,
            step_index=req.step_index,
            scale=req.scale,
        )
        pdf_bytes = export_protocol_pdf(experiment)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{experiment["id"]}_protocol.pdf"',
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"PDF export failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)[:200]}")


@router.post("/execute/generate-from-job/{job_id}")
async def generate_from_job(job_id: str, step_index: int = 0, scale: str = "0.5 mmol"):
    """Generate protocol from a completed retrosynthesis job.

    Fetches the job result, picks the top route, and generates a protocol.
    """
    try:
        from rasyn.db.engine import async_session
        from rasyn.db.models import Job
        from rasyn.modules.execute import generate_experiment
        from sqlalchemy import select

        async with async_session() as session:
            result = await session.execute(select(Job).where(Job.id == uuid.UUID(job_id)))
            job = result.scalar_one_or_none()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed":
            raise HTTPException(status_code=400, detail=f"Job not completed (status: {job.status})")
        if not job.result or "routes" not in job.result:
            raise HTTPException(status_code=400, detail="Job has no routes")

        routes = job.result["routes"]
        if not routes:
            raise HTTPException(status_code=400, detail="No routes available")

        # Use best route (rank 1)
        route = routes[0]
        experiment = generate_experiment(route=route, step_index=step_index, scale=scale)

        try:
            _save_experiment(experiment)
        except Exception as e:
            logger.warning(f"Failed to save experiment to DB: {e}")

        return experiment

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Protocol from job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])


def _save_experiment(exp_data: dict) -> None:
    """Save experiment and samples to database."""
    from sqlalchemy.orm import Session as SyncSession
    from rasyn.db.engine import sync_engine
    from rasyn.db.models import Experiment, Sample
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=sync_engine)
    session = SessionLocal()
    try:
        experiment = Experiment(
            id=exp_data["id"],
            route_id=exp_data.get("route_id"),
            step_number=exp_data.get("stepNumber"),
            product_smiles=exp_data.get("product_smiles"),
            reaction_name=exp_data.get("reactionName"),
            scale=exp_data.get("scale", "0.5 mmol"),
            protocol=exp_data.get("protocol"),
            reagents=exp_data.get("reagents"),
            workup=exp_data.get("workupChecklist"),
        )
        session.add(experiment)

        for s in exp_data.get("samples", []):
            sample = Sample(
                id=s["id"],
                experiment_id=exp_data["id"],
                label=s.get("label"),
                sample_type=s.get("type"),
                planned_analysis=s.get("plannedAnalysis"),
            )
            session.add(sample)

        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()
