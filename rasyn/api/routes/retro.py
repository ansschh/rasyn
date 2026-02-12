"""Retrosynthesis API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from rasyn.api.schemas import (
    HealthResponse,
    MultiStepRequest,
    MultiStepResponse,
    Prediction,
    RouteInfo,
    RouteStep,
    SingleStepRequest,
    SingleStepResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["retrosynthesis"])


@router.post("/retro/single-step", response_model=SingleStepResponse)
async def single_step(req: SingleStepRequest, request: Request):
    """Run single-step retrosynthesis on a product molecule."""
    pipeline = request.app.state.pipeline
    result = await pipeline.single_step(
        product_smiles=req.smiles,
        model=req.model,
        top_k=req.top_k,
        use_verification=req.use_verification,
    )
    predictions = [Prediction(**p) for p in result.get("predictions", [])]
    return SingleStepResponse(
        product=result.get("product", req.smiles),
        predictions=predictions,
        compute_time_ms=result.get("compute_time_ms", 0),
        error=result.get("error"),
    )


@router.post("/retro/multi-step", response_model=MultiStepResponse)
async def multi_step(req: MultiStepRequest, request: Request):
    """Run multi-step retrosynthetic route planning."""
    pipeline = request.app.state.pipeline
    result = await pipeline.multi_step(
        target_smiles=req.smiles,
        max_depth=req.max_depth,
        max_routes=req.max_routes,
    )
    routes = []
    for r in result.get("routes", []):
        steps = [RouteStep(**s) for s in r.get("steps", [])]
        routes.append(RouteInfo(
            steps=steps,
            total_score=r.get("total_score", 0),
            num_steps=r.get("num_steps", 0),
            all_available=r.get("all_available", False),
            starting_materials=r.get("starting_materials", []),
        ))
    return MultiStepResponse(
        target=result.get("target", req.smiles),
        routes=routes,
        compute_time_ms=result.get("compute_time_ms", 0),
        error=result.get("error"),
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Health check — returns loaded models and device."""
    mm = request.app.state.model_manager
    return HealthResponse(
        status="ok",
        models_loaded=mm.loaded_models(),
        device=mm.device,
    )
