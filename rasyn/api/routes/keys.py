"""API key management routes (admin-only)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from rasyn.api.security import get_key_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["api-keys"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable name for this key")
    role: str = Field("user", description="Key role: 'user' or 'admin'")


class CreateKeyResponse(BaseModel):
    id: str
    key: str = Field(description="The API key — save it now, it won't be shown again!")
    name: str
    role: str
    created_at: str


class KeyInfo(BaseModel):
    id: str
    name: str
    role: str
    created_at: str
    last_used_at: str | None
    request_count: int
    is_active: int
    created_by: str | None


class KeyListResponse(BaseModel):
    keys: list[KeyInfo]
    total: int


class RevokeResponse(BaseModel):
    revoked: bool
    key_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/keys", response_model=CreateKeyResponse)
async def create_key(req: CreateKeyRequest, request: Request):
    """Create a new API key. Requires admin key.

    The raw key is returned ONLY in this response — store it securely.
    """
    if req.role not in ("user", "admin"):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_role", "message": "Role must be 'user' or 'admin'."},
        )

    admin_info = getattr(request.state, "api_key_info", {})
    created_by = admin_info.get("name", "unknown")

    store = get_key_store()
    result = store.create_key(name=req.name, role=req.role, created_by=created_by)

    logger.info(f"API key created: id={result['id']} name={req.name} role={req.role} by={created_by}")
    return CreateKeyResponse(**result)


@router.get("/keys", response_model=KeyListResponse)
async def list_keys(request: Request):
    """List all API keys (without raw values). Requires admin key."""
    store = get_key_store()
    keys = store.list_keys()
    return KeyListResponse(keys=[KeyInfo(**k) for k in keys], total=len(keys))


@router.delete("/keys/{key_id}", response_model=RevokeResponse)
async def revoke_key(key_id: str, request: Request):
    """Revoke an API key by ID. Requires admin key."""
    store = get_key_store()
    revoked = store.revoke_key(key_id)

    if revoked:
        logger.info(f"API key revoked: id={key_id}")
    else:
        logger.warning(f"Attempted to revoke non-existent key: id={key_id}")

    return RevokeResponse(revoked=revoked, key_id=key_id)
