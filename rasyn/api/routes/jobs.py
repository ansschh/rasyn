"""API v2 routes — job-based retrosynthesis with SSE streaming."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from rasyn.api.schemas_v2 import (
    JobStatus,
    PlanRequest,
    PlanResult,
    PlanStartResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["jobs"])


@router.post("/plan", response_model=PlanStartResponse)
async def create_plan(req: PlanRequest, request: Request):
    """Submit a retrosynthesis planning job.

    Returns a job_id. Use GET /jobs/{job_id} for the result,
    or GET /jobs/{job_id}/stream for real-time SSE events.
    """
    from rasyn.db.engine import async_session
    from rasyn.db.models import Job
    from rasyn.worker.tasks import run_retrosynthesis

    # Validate SMILES
    from rasyn.preprocess.canonicalize import canonicalize_smiles
    canon = canonicalize_smiles(req.smiles)
    if not canon:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {req.smiles}")

    # Create job in DB
    job_id = uuid.uuid4()
    async with async_session() as session:
        job = Job(
            id=job_id,
            smiles=canon,
            status="queued",
            config={
                "top_k": req.top_k,
                "models": req.models,
                "constraints": req.constraints,
                "novelty_mode": req.novelty_mode.value if req.novelty_mode else "balanced",
                "objective": req.objective.value if req.objective else "default",
            },
        )
        session.add(job)
        await session.commit()

    # Dispatch to Celery worker
    run_retrosynthesis.delay(
        job_id=str(job_id),
        smiles=canon,
        top_k=req.top_k,
        models=req.models,
        constraints=req.constraints,
        novelty_mode=req.novelty_mode.value if req.novelty_mode else "balanced",
        objective=req.objective.value if req.objective else "default",
    )

    logger.info(f"Job {job_id} created for {canon[:50]}...")
    return PlanStartResponse(job_id=job_id, status=JobStatus.queued)


@router.get("/jobs/{job_id}")
async def get_job(job_id: uuid.UUID):
    """Get job status and result."""
    from sqlalchemy import select

    from rasyn.db.engine import async_session
    from rasyn.db.models import Job

    async with async_session() as session:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # If completed, return the full PlanResult from JSONB
    if job.result:
        return job.result

    return {
        "job_id": str(job.id),
        "smiles": job.smiles,
        "status": job.status,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: uuid.UUID):
    """SSE stream of job events.

    Connect with EventSource:
        const es = new EventSource('/api/v2/jobs/{job_id}/stream');
        es.addEventListener('step_complete', (e) => { ... });
    """
    from rasyn.events import subscribe_events

    # Verify job exists
    from sqlalchemy import select

    from rasyn.db.engine import async_session
    from rasyn.db.models import Job

    async with async_session() as session:
        result = await session.execute(select(Job.id).where(Job.id == job_id))
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        # Send initial keepalive
        yield ": connected\n\n"
        async for event_str in subscribe_events(str(job_id)):
            yield event_str

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
