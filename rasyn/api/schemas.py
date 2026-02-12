"""Pydantic request/response models for the Rasyn API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class SingleStepRequest(BaseModel):
    smiles: str = Field(..., description="Product SMILES string")
    model: str = Field("llm", description="Model to use: 'llm', 'retro', or 'both'")
    top_k: int = Field(10, ge=1, le=50, description="Max number of predictions")
    use_verification: bool = Field(True, description="Run round-trip verification")


class MultiStepRequest(BaseModel):
    smiles: str = Field(..., description="Target product SMILES string")
    max_depth: int = Field(10, ge=1, le=20, description="Max retrosynthetic depth")
    max_routes: int = Field(5, ge=1, le=20, description="Max routes to return")


class ValidateRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string to validate")


# ---------------------------------------------------------------------------
# Responses — Single Step
# ---------------------------------------------------------------------------

class VerificationInfo(BaseModel):
    rdkit_valid: bool
    forward_match_score: float
    overall_confidence: float


class EditInfo(BaseModel):
    bonds: list[list[int]]
    synthons: list[str]
    leaving_groups: list[str]


class Prediction(BaseModel):
    rank: int
    reactants_smiles: list[str]
    confidence: float
    model_source: str = Field(description="'llm' or 'retro_v2'")
    verification: VerificationInfo | None = None
    edit_info: EditInfo | None = None


class SingleStepResponse(BaseModel):
    product: str
    predictions: list[Prediction]
    compute_time_ms: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Responses — Multi Step
# ---------------------------------------------------------------------------

class RouteStep(BaseModel):
    product: str
    reactants: list[str]
    confidence: float


class RouteInfo(BaseModel):
    steps: list[RouteStep]
    total_score: float
    num_steps: int
    all_available: bool
    starting_materials: list[str]


class MultiStepResponse(BaseModel):
    target: str
    routes: list[RouteInfo]
    compute_time_ms: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Responses — Molecule
# ---------------------------------------------------------------------------

class ValidateResponse(BaseModel):
    valid: bool
    canonical: str | None = None
    formula: str | None = None
    mol_weight: float | None = None
    svg: str | None = Field(None, description="Base64-encoded SVG")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    device: str
