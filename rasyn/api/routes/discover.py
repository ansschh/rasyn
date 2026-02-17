"""API v2 routes — literature discovery and evidence search."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["discover"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DiscoverRequest(BaseModel):
    query: str = Field(..., description="Search query (e.g., 'C-N bond formation without Pd')")
    smiles: str | None = Field(None, description="Optional target molecule SMILES for structure search")
    max_results: int = Field(20, ge=1, le=50)


class DiscoverStartResponse(BaseModel):
    job_id: uuid.UUID
    status: str = "queued"


class PaperResult(BaseModel):
    title: str
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    citation_count: int = 0
    source: str = "unknown"
    journal: str | None = None
    abstract: str | None = None
    url: str | None = None
    source_type: str = "paper"


class DiscoverResult(BaseModel):
    job_id: str | None = None
    query: str
    smiles: str | None = None
    papers: list[PaperResult] = Field(default_factory=list)
    compound_info: dict = Field(default_factory=dict)
    sources_queried: list[str] = Field(default_factory=list)
    total_results: int = 0
    compute_time_ms: float | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/discover/search", response_model=DiscoverStartResponse)
async def start_discovery(req: DiscoverRequest):
    """Submit a literature discovery search job.

    Returns a job_id. Use GET /discover/{job_id}/results for results,
    or GET /jobs/{job_id}/stream for SSE events.
    """
    from rasyn.db.engine import async_session
    from rasyn.db.models import Job
    from rasyn.worker.tasks import run_discovery

    job_id = uuid.uuid4()
    async with async_session() as session:
        job = Job(
            id=job_id,
            smiles=req.smiles or req.query,
            job_type="DISCOVER",
            status="queued",
            config={
                "query": req.query,
                "smiles": req.smiles,
                "max_results": req.max_results,
            },
        )
        session.add(job)
        await session.commit()

    run_discovery.delay(
        job_id=str(job_id),
        query=req.query,
        smiles=req.smiles,
    )

    logger.info(f"Discovery job {job_id} created for query: {req.query[:60]}")
    return DiscoverStartResponse(job_id=job_id, status="queued")


@router.get("/discover/{job_id}/results")
async def get_discovery_results(job_id: uuid.UUID):
    """Get discovery search results."""
    from sqlalchemy import select

    from rasyn.db.engine import async_session
    from rasyn.db.models import Job

    async with async_session() as session:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.result:
        return job.result

    return {
        "job_id": str(job.id),
        "status": job.status,
        "error": job.error,
    }


@router.get("/discover/quick")
async def quick_discovery(query: str, smiles: str | None = None, max_results: int = 10):
    """Synchronous quick literature search (no job queue).

    For small/fast queries where you don't need async processing.
    """
    from rasyn.modules.discover import search_literature_sync

    result = search_literature_sync(query, smiles=smiles, max_results=max_results, timeout=15.0)
    return result
