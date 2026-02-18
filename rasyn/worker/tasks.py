"""Celery tasks for retrosynthesis jobs.

Slice 1: Infrastructure + SSE events
Slice 2: Real model inference + safety/green/evidence enrichment
Slice 3: Multi-step planning + sourcing + scoring + discovery
Slice 7-8: Protocol generation + instrument analysis
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
# Main retrosynthesis task (Slice 3: multi-step + full enrichment)
# ---------------------------------------------------------------------------

@app.task(bind=True, name="rasyn.retrosynthesis", max_retries=1)
def run_retrosynthesis(self, job_id: str, smiles: str, top_k: int = 5,
                       models: list | None = None, use_multistep: bool = True,
                       constraints: dict | None = None,
                       novelty_mode: str = "balanced",
                       objective: str = "default"):
    """Run retrosynthesis for a given target molecule.

    Produces SSE events throughout execution and saves PlanResult to the DB.

    Slice 3 upgrade: Uses multi-step A* search with combined model expansion,
    composite scoring, sourcing lookups, and literature discovery.
    """
    models = models or ["retro_v2", "llm"]
    t0 = time.perf_counter()

    try:
        # Mark as running
        _update_job(job_id, status="running")
        _emit(job_id, "planning_started", f"Starting retrosynthesis for {smiles[:50]}...")

        pipeline = _get_pipeline()

        # --- Phase 1: Route Planning ---
        def emit_fn(kind, message="", data=None):
            _emit(job_id, kind, message, data)

        if use_multistep:
            _emit(job_id, "info", "Phase 1/5: Multi-step route planning (A* search)...")
            from rasyn.modules.planner import run_multistep_planning
            routes = run_multistep_planning(
                target_smiles=smiles,
                pipeline_service=pipeline,
                top_k=top_k,
                max_depth=6,
                max_time=90.0,
                emit_fn=emit_fn,
                novelty_mode=novelty_mode,
            )
        else:
            # Fallback: single-step only (as in Slice 2)
            _emit(job_id, "info", "Phase 1/5: Single-step retrosynthesis...")
            from rasyn.modules.planner import _run_single_step_fallback
            routes = _run_single_step_fallback(smiles, pipeline, top_k, emit_fn=emit_fn,
                                                novelty_mode=novelty_mode)

        # --- Phase 2: Enrichment ---
        _emit(job_id, "enriching", "Phase 2/5: Safety screening + green chemistry...")
        safety = _run_safety(smiles)
        green_chem = _run_green_chem(smiles, routes)

        # --- Phase 3: Evidence search (needed for precedent scoring) ---
        _emit(job_id, "enriching", "Phase 3/5: Literature evidence search...")
        evidence = _run_evidence(smiles, routes)

        # --- Phase 4: Scoring & Ranking (uses evidence for precedent score) ---
        _emit(job_id, "enriching", "Phase 4/5: Scoring, constraint filtering, and ranking...")
        routes = _score_and_rank(routes, safety, green_chem, evidence,
                                 constraints=constraints, objective=objective)

        # --- Phase 5: Sourcing + Discovery ---
        _emit(job_id, "enriching", "Phase 5/5: Sourcing lookups + literature search...")
        sourcing = _run_sourcing(routes)
        discovery = _run_discovery(smiles)

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
            "sourcing": sourcing,
            "discovery": discovery,
            "compute_time_ms": elapsed_ms,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        _update_job(job_id, status="completed", result=result)
        _save_routes_to_db(job_id, routes)
        _emit(job_id, "completed",
              f"Retrosynthesis complete — {len(routes)} routes in {elapsed_ms:.0f}ms",
              {"routes_count": len(routes), "compute_time_ms": elapsed_ms})

        return result

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        error_msg = str(exc)[:500]
        logger.exception(f"Job {job_id} failed: {error_msg}")
        _update_job(job_id, status="failed", error=error_msg)
        _emit(job_id, "failed", f"Job failed: {error_msg}")
        raise self.retry(exc=exc, countdown=5) if self.request.retries < self.max_retries else None


# ---------------------------------------------------------------------------
# Enrichment helpers
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


def _score_and_rank(routes: list[dict], safety: dict | None, green: dict | None,
                    evidence: list[dict] | None = None,
                    constraints: dict | None = None,
                    objective: str = "default") -> list[dict]:
    """Score and re-rank routes using composite scoring."""
    try:
        from rasyn.modules.scoring import score_and_rank_routes
        return score_and_rank_routes(routes, safety=safety, green=green, evidence=evidence,
                                     constraints=constraints, objective=objective)
    except Exception as e:
        logger.warning(f"Scoring failed: {e}")
        return routes


def _run_sourcing(routes: list[dict]) -> dict | None:
    """Look up sourcing for starting materials across all routes."""
    try:
        from rasyn.modules.sourcing import search_vendors

        # Collect all unique starting materials across routes
        all_sm = set()
        for route in routes:
            for sm in route.get("starting_materials", []):
                if sm:
                    all_sm.add(sm)

        if not all_sm:
            return None

        result = search_vendors(list(all_sm), timeout=8.0)

        # Mark routes as purchasable based on results
        available_smiles = set()
        for item in result.get("items", []):
            if item.get("in_stock"):
                available_smiles.add(item["smiles"])

        for route in routes:
            sm = set(route.get("starting_materials", []))
            if sm and sm.issubset(available_smiles):
                route["all_purchasable"] = True

        return result
    except Exception as e:
        logger.warning(f"Sourcing lookup failed: {e}")
        return None


def _run_discovery(smiles: str) -> dict | None:
    """Run literature discovery for the target molecule."""
    try:
        from rasyn.modules.discover import search_literature_sync

        # Build a query from the SMILES
        query = f"retrosynthesis {smiles[:30]} organic chemistry synthesis"
        result = search_literature_sync(query, smiles=smiles, max_results=10, timeout=10.0)
        return result
    except Exception as e:
        logger.warning(f"Literature discovery failed: {e}")
        return None


def _save_routes_to_db(job_id: str, routes: list[dict]) -> None:
    """Persist route and reaction data to the database."""
    try:
        from rasyn.db.models import Route as RouteModel, Reaction

        session = _get_db_session()
        try:
            for route in routes:
                route_row = RouteModel(
                    job_id=uuid.UUID(job_id),
                    route_id=route.get("route_id", "route_1"),
                    rank=route.get("rank", 1),
                    tree=route,
                    score=route.get("overall_score"),
                    score_breakdown=route.get("score_breakdown"),
                    num_steps=route.get("num_steps", 0),
                    all_purchasable=route.get("all_purchasable", False),
                )
                session.add(route_row)
                session.flush()

                for j, step in enumerate(route.get("steps", [])):
                    rxn = Reaction(
                        route_id=route_row.id,
                        step_number=j + 1,
                        product_smiles=step.get("product", ""),
                        reactants_smiles=".".join(step.get("reactants", [])),
                        reaction_smiles=".".join(step.get("reactants", [])) + ">>" + step.get("product", ""),
                        model_source=step.get("model"),
                        confidence=step.get("score"),
                    )
                    session.add(rxn)

            session.commit()
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"Failed to save routes to DB: {e}")


# ---------------------------------------------------------------------------
# Discovery search task (standalone, for /api/v2/discover endpoint)
# ---------------------------------------------------------------------------

@app.task(bind=True, name="rasyn.discover", max_retries=1)
def run_discovery(self, job_id: str, query: str, smiles: str | None = None):
    """Run literature discovery as a standalone job."""
    t0 = time.perf_counter()

    try:
        _update_job(job_id, status="running")
        _emit(job_id, "planning_started", f"Searching literature for: {query[:80]}...")

        from rasyn.modules.discover import search_literature_sync
        result = search_literature_sync(query, smiles=smiles, max_results=20, timeout=15.0)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        full_result = {
            "job_id": job_id,
            "query": query,
            "smiles": smiles,
            "status": "completed",
            **result,
            "compute_time_ms": elapsed_ms,
        }

        _update_job(job_id, status="completed", result=full_result)
        _emit(job_id, "completed",
              f"Found {result.get('total_results', 0)} papers in {elapsed_ms:.0f}ms",
              {"total_results": result.get("total_results", 0)})

        return full_result

    except Exception as exc:
        error_msg = str(exc)[:500]
        logger.exception(f"Discovery job {job_id} failed: {error_msg}")
        _update_job(job_id, status="failed", error=error_msg)
        _emit(job_id, "failed", f"Discovery failed: {error_msg}")
        raise self.retry(exc=exc, countdown=5) if self.request.retries < self.max_retries else None


# ---------------------------------------------------------------------------
# Protocol generation task (Slice 7)
# ---------------------------------------------------------------------------

@app.task(bind=True, name="rasyn.generate_protocol", max_retries=1)
def run_protocol_generation(self, job_id: str, route: dict, step_index: int = 0,
                             scale: str = "0.5 mmol"):
    """Generate a lab-ready protocol from a retrosynthetic route step."""
    t0 = time.perf_counter()

    try:
        _update_job(job_id, status="running")
        _emit(job_id, "planning_started", "Generating experiment protocol...")

        from rasyn.modules.execute import generate_experiment

        _emit(job_id, "info", "Calculating stoichiometry and reagent amounts...")
        experiment = generate_experiment(route=route, step_index=step_index, scale=scale)

        _emit(job_id, "info", f"Protocol generated: {len(experiment.get('protocol', []))} steps, "
              f"{len(experiment.get('reagents', []))} reagents")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        result = {
            "job_id": job_id,
            "status": "completed",
            "experiment": experiment,
            "compute_time_ms": elapsed_ms,
        }

        _update_job(job_id, status="completed", result=result)
        _emit(job_id, "completed",
              f"Protocol ready — {experiment.get('id', 'EXP')}",
              {"experiment_id": experiment.get("id")})

        return result

    except Exception as exc:
        error_msg = str(exc)[:500]
        logger.exception(f"Protocol generation {job_id} failed: {error_msg}")
        _update_job(job_id, status="failed", error=error_msg)
        _emit(job_id, "failed", f"Protocol generation failed: {error_msg}")
        raise self.retry(exc=exc, countdown=5) if self.request.retries < self.max_retries else None


# ---------------------------------------------------------------------------
# Analysis task (Slice 8)
# ---------------------------------------------------------------------------

@app.task(bind=True, name="rasyn.analyze", max_retries=1)
def run_analysis(self, job_id: str, file_paths: list[str],
                  expected_product_smiles: str | None = None,
                  expected_mw: float | None = None):
    """Analyze uploaded instrument files."""
    t0 = time.perf_counter()

    try:
        _update_job(job_id, status="running")
        _emit(job_id, "planning_started",
              f"Analyzing {len(file_paths)} instrument file(s)...")

        from rasyn.modules.analyze import analyze_batch

        _emit(job_id, "info", "Parsing and interpreting instrument data...")
        result = analyze_batch(
            file_paths,
            expected_product_smiles=expected_product_smiles,
            expected_mw=expected_mw,
        )

        summary = result.get("summary", {})
        _emit(job_id, "info",
              f"Analysis complete: {summary.get('interpreted', 0)} interpreted, "
              f"{summary.get('anomalies', 0)} anomalies")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        full_result = {
            "job_id": job_id,
            "status": "completed",
            **result,
            "compute_time_ms": elapsed_ms,
        }

        _update_job(job_id, status="completed", result=full_result)
        _emit(job_id, "completed",
              f"Analysis complete — {summary.get('total', 0)} files processed",
              summary)

        return full_result

    except Exception as exc:
        error_msg = str(exc)[:500]
        logger.exception(f"Analysis job {job_id} failed: {error_msg}")
        _update_job(job_id, status="failed", error=error_msg)
        _emit(job_id, "failed", f"Analysis failed: {error_msg}")
        raise self.retry(exc=exc, countdown=5) if self.request.retries < self.max_retries else None
