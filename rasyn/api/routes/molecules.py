"""Molecule utility API routes."""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, Query
from fastapi.responses import Response

from rasyn.api.schemas import ValidateRequest, ValidateResponse
from rasyn.service.molecule_service import MoleculeService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["molecules"])

_mol_svc = MoleculeService()


@router.post("/validate", response_model=ValidateResponse)
async def validate_smiles(req: ValidateRequest):
    """Validate a SMILES string and return molecular info."""
    result = _mol_svc.validate_and_info(req.smiles)
    return ValidateResponse(**result)


@router.get("/image")
async def molecule_image(smiles: str = Query(..., description="SMILES string")):
    """Render a molecule as an SVG image."""
    svg_b64 = _mol_svc.draw_molecule_svg(smiles)
    if svg_b64 is None:
        return Response(content="Invalid SMILES", status_code=400)
    svg_bytes = base64.b64decode(svg_b64)
    return Response(content=svg_bytes, media_type="image/svg+xml")
