"""Celery tasks for retrosynthesis jobs.

Slice 1: Full infrastructure with mock chemistry to validate the pipeline.
Slice 2: Real model inference via PipelineService + enrichment modules.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from sqlalchemy import update
from sqlalchemy.orm import Session

from rasyn.worker.app import app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded heavy dependencies (loaded once per worker process)
# ---------------------------------------------------------------------------

_model_manager = None
_pipeline_service = None
_sync_session_factory = None


def _get_db_session() -> Session:
    """Get a sync SQLAlchemy session for the worker."""
    global _sync_session_factory
    if _sync_session_factory is None:
        from sqlalchemy.orm import sessionmaker
        from rasyn.db.engine import sync_engine
        from rasyn.db.models import Base

        # Ensure tables exist
        Base.metadata.create_all(sync_engine)
        _sync_session_factory = sessionmaker(bind=sync_engine)
    return _sync_session_factory()


def _get_pipeline():
    """Lazy-initialize ModelManager + PipelineService on first task."""
    global _model_manager, _pipeline_service
    if _pipeline_service is None:
        from pathlib import Path

        import yaml

        from rasyn.service.model_manager import ModelManager
        from rasyn.service.pipeline_service import PipelineService

        config_path = Path(__file__).parent.parent.parent / "configs" / "serve.yaml"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        _model_manager = ModelManager(config)
        _pipeline_service = PipelineService(_model_manager)
        logger.info("Worker: ModelManager + PipelineService initialized.")
    return _pipeline_service


def _update_job(job_id: str, **kwargs) -> None:
    """Update a job row in the database."""
    from rasyn.db.models import Job

    session = _get_db_session()
    try:
        session.execute(
            update(Job).where(Job.id == uuid.UUID(job_id)).values(
                updated_at=datetime.now(timezone.utc), **kwargs
            )
        )
        session.commit()
    finally:
        session.close()


def _emit(job_id: str, kind: str, message: str = "", data: dict | None = None) -> None:
    """Publish an SSE event via Redis."""
    from rasyn.events import publish_event_sync

    publish_event_sync(job_id, kind, message, data)


# ---------------------------------------------------------------------------
# Main retrosynthesis task
# ---------------------------------------------------------------------------

@app.task(bind=True, name="rasyn.retrosynthesis", max_retries=1)
def run_retrosynthesis(self, job_id: str, smiles: str, top_k: int = 5, models: list | None = None):
    """Run retrosynthesis for a given target molecule.

    Produces SSE events throughout execution and saves PlanResult to the DB.
    """
    models = models or ["retro_v2", "llm"]
    t0 = time.perf_counter()

    try:
        # Mark as running
        _update_job(job_id, status="running")
        _emit(job_id, "planning_started", f"Starting retrosynthesis for {smiles[:50]}...")

        # --- Run models ---
        pipeline = _get_pipeline()
        all_predictions = []

        if "llm" in models:
            _emit(job_id, "model_running", "Running RSGPT-3.2B (LLM) model...")
            try:
                llm_results = pipeline._run_llm_pipeline(smiles, top_k, use_verification=True)
                all_predictions.extend(llm_results)
                _emit(job_id, "step_complete", f"LLM produced {len(llm_results)} predictions", {"model": "llm", "count": len(llm_results)})
            except Exception as e:
                logger.warning(f"LLM pipeline failed: {e}")
                _emit(job_id, "warning", f"LLM model failed: {str(e)[:100]}")

        if "retro_v2" in models:
            _emit(job_id, "model_running", "Running RetroTransformer v2 (69.7% top-1)...")
            try:
                retro_results = pipeline._run_retro_pipeline(smiles, top_k)
                all_predictions.extend(retro_results)
                _emit(job_id, "step_complete", f"RetroTx v2 produced {len(retro_results)} predictions", {"model": "retro_v2", "count": len(retro_results)})
            except Exception as e:
                logger.warning(f"RetroTx v2 pipeline failed: {e}")
                _emit(job_id, "warning", f"RetroTx v2 model failed: {str(e)[:100]}")

        # --- Deduplicate & rank ---
        _emit(job_id, "enriching", "Deduplicating and ranking predictions...")
        seen = set()
        deduped = []
        for pred in all_predictions:
            key = ".".join(sorted(pred.get("reactants_smiles", [])))
            if key and key not in seen:
                seen.add(key)
                deduped.append(pred)
        deduped = deduped[:top_k]

        # Build routes (each single-step prediction = 1 route for now)
        routes = []
        for i, pred in enumerate(deduped):
            routes.append({
                "route_id": f"route_{i+1}",
                "rank": i + 1,
                "steps": [{
                    "product": smiles,
                    "reactants": pred.get("reactants_smiles", []),
                    "model": pred.get("model_source", "unknown"),
                    "score": pred.get("confidence", 0.5),
                    "rxn_class": None,
                    "conditions": None,
                }],
                "overall_score": pred.get("confidence", 0.5),
                "num_steps": 1,
                "starting_materials": pred.get("reactants_smiles", []),
                "all_purchasable": False,
            })

        # --- Enrichment ---
        _emit(job_id, "enriching", "Running safety screening (PAINS/BRENK)...")
        safety = _run_safety(smiles)

        _emit(job_id, "enriching", "Computing green chemistry metrics...")
        green_chem = _run_green_chem(smiles, routes)

        _emit(job_id, "enriching", "Searching literature evidence...")
        evidence = _run_evidence(smiles, routes)

        # --- Build final PlanResult ---
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        result = {
            "job_id": job_id,
            "smiles": smiles,
            "status": "completed",
            "routes": routes,
            "safety": safety,
            "evidence": evidence,
            "green_chem": green_chem,
            "sourcing": None,  # TODO: Slice 3
            "compute_time_ms": elapsed_ms,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        _update_job(job_id, status="completed", result=result)
        _emit(job_id, "completed", f"Retrosynthesis complete — {len(routes)} routes in {elapsed_ms:.0f}ms", {"routes_count": len(routes)})

        return result

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        error_msg = str(exc)[:500]
        logger.exception(f"Job {job_id} failed: {error_msg}")
        _update_job(job_id, status="failed", error=error_msg)
        _emit(job_id, "failed", f"Job failed: {error_msg}")
        raise self.retry(exc=exc, countdown=5) if self.request.retries < self.max_retries else None


# ---------------------------------------------------------------------------
# Enrichment helpers (Slice 2)
# ---------------------------------------------------------------------------

def _run_safety(smiles: str) -> dict | None:
    """Run safety screening on target molecule."""
    try:
        from rasyn.modules.safety import screen_molecule
        return screen_molecule(smiles)
    except Exception as e:
        logger.warning(f"Safety screening failed: {e}")
        return None


def _run_green_chem(smiles: str, routes: list[dict]) -> dict | None:
    """Compute green chemistry metrics."""
    try:
        from rasyn.modules.green import score_routes
        return score_routes(smiles, routes)
    except Exception as e:
        logger.warning(f"Green chemistry scoring failed: {e}")
        return None


def _run_evidence(smiles: str, routes: list[dict]) -> list[dict]:
    """Search for literature evidence."""
    try:
        from rasyn.modules.evidence import find_evidence
        return find_evidence(smiles, routes)
    except Exception as e:
        logger.warning(f"Evidence search failed: {e}")
        return []
