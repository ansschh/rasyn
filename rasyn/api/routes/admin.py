"""Admin API routes — audit log, guardrails, permissions.

Slice 10: GET /admin/audit-log, POST /admin/guardrails-check,
          GET /admin/permissions/{role}
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from rasyn.api.schemas_v2 import (
    AuditLogResponse,
    GuardrailCheckRequest,
    GuardrailCheckResponse,
    PermissionInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["admin"])


@router.get("/admin/audit-log", response_model=AuditLogResponse)
async def get_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    user_id: str | None = Query(None, description="Filter by user ID"),
):
    """Get audit log entries.

    Returns a paginated list of all actions recorded by the system.
    """
    try:
        from rasyn.modules.admin import get_audit_log as _get_log

        result = _get_log(limit=limit, offset=offset, user_id=user_id)
        return result

    except Exception as e:
        logger.exception(f"Audit log retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/admin/guardrails-check", response_model=GuardrailCheckResponse)
async def guardrails_check(req: GuardrailCheckRequest):
    """Check a molecule against safety guardrails.

    Screens for explosive precursors, chemical weapon precursors,
    controlled substances, and other dangerous compound classes.
    """
    try:
        from rasyn.modules.admin import check_guardrails, log_action

        result = check_guardrails(req.smiles)

        # Log the check (especially if blocked)
        if result.get("blocked"):
            log_action(
                action="Guardrails BLOCKED",
                resource=req.smiles[:100],
                details=f"Blocked: {[a['name'] for a in result.get('alerts', [])]}",
            )
        else:
            log_action(
                action="Guardrails check",
                resource=req.smiles[:100],
                details="Passed",
            )

        return result

    except Exception as e:
        logger.exception(f"Guardrails check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/admin/permissions/{role}", response_model=PermissionInfo)
async def get_permissions(role: str):
    """Get permissions for a specific role."""
    from rasyn.modules.admin import PERMISSIONS, get_role_permissions

    if role not in PERMISSIONS:
        raise HTTPException(status_code=404, detail=f"Role '{role}' not found. Valid: {list(PERMISSIONS.keys())}")

    return {
        "role": role,
        "permissions": get_role_permissions(role),
    }


@router.get("/admin/roles")
async def list_roles():
    """List all available roles and their permission counts."""
    from rasyn.modules.admin import PERMISSIONS, get_role_permissions

    roles = []
    for role_name in PERMISSIONS:
        perms = get_role_permissions(role_name)
        roles.append({
            "role": role_name,
            "permission_count": len(perms),
            "permissions": perms,
        })
    return {"roles": roles}
