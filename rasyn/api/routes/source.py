"""API v2 routes — chemical sourcing and procurement."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["source"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SourceQuoteRequest(BaseModel):
    smiles_list: list[str] = Field(..., description="List of SMILES to look up", min_length=1, max_length=50)


class SourcingItem(BaseModel):
    smiles: str
    vendor: str | None = None
    catalog_id: str | None = None
    price_per_gram: float | None = None
    lead_time_days: int | None = None
    in_stock: bool = False
    url: str | None = None
    properties: dict | None = None


class SourceQuoteResponse(BaseModel):
    items: list[SourcingItem] = Field(default_factory=list)
    total_estimated_cost: float | None = None
    summary: dict = Field(default_factory=dict)


class AlternateRequest(BaseModel):
    smiles: str = Field(..., description="SMILES of unavailable compound")
    top_k: int = Field(5, ge=1, le=20)


class AlternateItem(BaseModel):
    smiles: str
    similarity: float
    source: str


class AlternateResponse(BaseModel):
    query_smiles: str
    alternates: list[AlternateItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/source/quote", response_model=SourceQuoteResponse)
async def get_sourcing_quotes(req: SourceQuoteRequest):
    """Look up commercial availability and pricing for compounds.

    Searches PubChem (free), and optionally ChemSpace and MolPort
    if API keys are configured.
    """
    from rasyn.modules.sourcing import search_vendors

    result = search_vendors(req.smiles_list, timeout=10.0)
    return SourceQuoteResponse(
        items=[SourcingItem(**item) for item in result.get("items", [])],
        total_estimated_cost=result.get("total_estimated_cost"),
        summary=result.get("summary", {}),
    )


@router.post("/source/alternates", response_model=AlternateResponse)
async def find_alternate_building_blocks(req: AlternateRequest):
    """Find structurally similar alternate building blocks.

    When a specific starting material is unavailable, this suggests
    similar compounds that might serve the same role.
    """
    from rasyn.modules.sourcing import find_alternates

    alternates = find_alternates(req.smiles, top_k=req.top_k)
    return AlternateResponse(
        query_smiles=req.smiles,
        alternates=[AlternateItem(**a) for a in alternates],
    )


@router.get("/source/check/{smiles}")
async def check_availability(smiles: str):
    """Quick availability check for a single compound."""
    from rasyn.modules.sourcing import search_vendors

    result = search_vendors([smiles], timeout=8.0)
    items = result.get("items", [])
    any_available = any(i.get("in_stock") for i in items)

    return {
        "smiles": smiles,
        "available": any_available,
        "offers": items,
    }
