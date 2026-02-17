"""PlanResult contract — frozen JSON schema for the Chemistry OS.

This is the single source of truth for the frontend-backend contract.
All API v2 endpoints produce / consume these models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PlanRequest(BaseModel):
    smiles: str = Field(..., description="Target molecule SMILES")
    top_k: int = Field(5, ge=1, le=50, description="Max routes to return")
    models: list[str] = Field(
        default=["retro_v2", "llm"],
        description="Models to use: 'retro_v2', 'llm', or both",
    )
    constraints: dict | None = Field(
        None, description="Optional constraints (no_pd, max_steps, etc.)"
    )


# ---------------------------------------------------------------------------
# Route / Step
# ---------------------------------------------------------------------------

class Step(BaseModel):
    product: str = Field(..., description="Product SMILES for this step")
    reactants: list[str] = Field(..., description="Reactant SMILES list")
    model: str = Field(..., description="Model that predicted this: 'retro_v2' or 'llm'")
    score: float = Field(..., description="Confidence score 0-1")
    rxn_class: str | None = Field(None, description="Reaction class label")
    conditions: dict | None = Field(None, description="Predicted conditions")


class Route(BaseModel):
    route_id: str = Field(..., description="Unique route identifier")
    rank: int = Field(..., description="Rank (1 = best)")
    steps: list[Step] = Field(..., description="Retrosynthetic steps")
    overall_score: float = Field(..., description="Combined route score 0-1")
    num_steps: int = Field(..., description="Total number of steps")
    starting_materials: list[str] = Field(
        default_factory=list, description="Terminal starting materials SMILES"
    )
    all_purchasable: bool = Field(False, description="All starting materials purchasable")


# ---------------------------------------------------------------------------
# Enrichment: Safety
# ---------------------------------------------------------------------------

class StructuralAlert(BaseModel):
    name: str
    smarts: str | None = None
    severity: str = "warning"  # "warning" or "critical"
    description: str | None = None


class DrugLikeness(BaseModel):
    mw: float
    logp: float
    hbd: int
    hba: int
    passes_lipinski: bool
    violations: list[str] = Field(default_factory=list)


class SafetyResult(BaseModel):
    alerts: list[StructuralAlert] = Field(default_factory=list)
    druglikeness: DrugLikeness | None = None
    tox_flags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Enrichment: Evidence
# ---------------------------------------------------------------------------

class EvidenceHit(BaseModel):
    rxn_smiles: str
    similarity: float
    source: str  # "USPTO", "Reaxys", "literature"
    year: int | None = None
    title: str | None = None
    doi: str | None = None


# ---------------------------------------------------------------------------
# Enrichment: Green Chemistry
# ---------------------------------------------------------------------------

class GreenChemResult(BaseModel):
    atom_economy: float | None = None
    e_factor: float | None = None
    solvent_score: float | None = None  # 0-1 (1 = greenest)
    details: dict | None = None


# ---------------------------------------------------------------------------
# Enrichment: Sourcing
# ---------------------------------------------------------------------------

class SourcingItem(BaseModel):
    smiles: str
    vendor: str | None = None
    catalog_id: str | None = None
    price_per_gram: float | None = None
    lead_time_days: int | None = None
    in_stock: bool = False
    url: str | None = None


class SourcingResult(BaseModel):
    items: list[SourcingItem] = Field(default_factory=list)
    total_estimated_cost: float | None = None


# ---------------------------------------------------------------------------
# SSE Event
# ---------------------------------------------------------------------------

class JobEvent(BaseModel):
    kind: str = Field(..., description="Event type: planning_started, model_running, step_complete, enriching, completed, failed")
    message: str = Field("", description="Human-readable log message")
    data: dict | None = Field(None, description="Structured event payload")
    ts: datetime | None = None


# ---------------------------------------------------------------------------
# PlanResult — the main contract
# ---------------------------------------------------------------------------

class PlanResult(BaseModel):
    job_id: UUID
    smiles: str
    status: JobStatus
    routes: list[Route] = Field(default_factory=list)
    safety: SafetyResult | None = None
    evidence: list[EvidenceHit] = Field(default_factory=list)
    green_chem: GreenChemResult | None = None
    sourcing: SourcingResult | None = None
    compute_time_ms: float | None = None
    error: str | None = None
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# API responses
# ---------------------------------------------------------------------------

class PlanStartResponse(BaseModel):
    job_id: UUID
    status: JobStatus = JobStatus.queued
