"""Admin module — audit logging, RBAC, and guardrails.

Slice 10: Enterprise controls for the Chemistry OS.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RBAC Permissions
# ---------------------------------------------------------------------------

PERMISSIONS: dict[str, set[str]] = {
    "admin": {"*"},
    "researcher": {
        "plan:create", "plan:read",
        "execute:create", "execute:read",
        "analyze:upload", "analyze:read",
        "source:read", "source:create",
        "learn:read", "learn:write",
        "discover:read", "discover:create",
    },
    "viewer": {
        "plan:read",
        "source:read",
        "learn:read",
        "analyze:read",
        "discover:read",
        "execute:read",
    },
}


def check_permission(role: str, permission: str) -> bool:
    """Check if a role has a specific permission.

    Args:
        role: User role (admin, researcher, viewer)
        permission: Permission string (e.g. 'plan:create')

    Returns:
        True if allowed, False otherwise
    """
    role_perms = PERMISSIONS.get(role, set())
    return "*" in role_perms or permission in role_perms


def get_role_permissions(role: str) -> list[str]:
    """Get all permissions for a role."""
    perms = PERMISSIONS.get(role, set())
    if "*" in perms:
        return ["Full access"]
    return sorted(perms)


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def log_action(
    action: str,
    resource: str | None = None,
    details: str | None = None,
    user_id: str | None = None,
    user_name: str | None = None,
    ip_address: str | None = None,
    status_code: int | None = None,
) -> None:
    """Record an action in the audit log.

    Non-blocking: failures are logged but do not raise.
    """
    try:
        from rasyn.db.engine import sync_engine
        from rasyn.db.models import AuditLog
        from sqlalchemy.orm import sessionmaker

        SessionLocal = sessionmaker(bind=sync_engine)
        session = SessionLocal()
        try:
            entry = AuditLog(
                user_id=user_id or "anonymous",
                user_name=user_name or user_id or "Anonymous",
                action=action,
                resource=resource,
                details=details,
                ip_address=ip_address,
                status_code=status_code,
            )
            session.add(entry)
            session.commit()
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"Audit log failed (non-fatal): {e}")


def get_audit_log(
    limit: int = 100,
    offset: int = 0,
    user_id: str | None = None,
) -> dict:
    """Retrieve audit log entries.

    Args:
        limit: Max entries to return
        offset: Offset for pagination
        user_id: Filter by user_id (optional)

    Returns:
        Dict with entries list and total count
    """
    try:
        from rasyn.db.engine import sync_engine
        from rasyn.db.models import AuditLog
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import select, func

        SessionLocal = sessionmaker(bind=sync_engine)
        session = SessionLocal()
        try:
            stmt = select(AuditLog).order_by(AuditLog.created_at.desc())

            if user_id:
                stmt = stmt.where(AuditLog.user_id == user_id)

            # Count
            count_stmt = select(func.count(AuditLog.id))
            if user_id:
                count_stmt = count_stmt.where(AuditLog.user_id == user_id)
            total = session.execute(count_stmt).scalar() or 0

            # Fetch
            stmt = stmt.offset(offset).limit(limit)
            rows = session.execute(stmt).scalars().all()

            entries = []
            for row in rows:
                entries.append({
                    "id": row.id,
                    "timestamp": row.created_at.strftime("%Y-%m-%d %H:%M") if row.created_at else "",
                    "user": row.user_name or row.user_id or "Unknown",
                    "action": row.action or "",
                    "resource": row.resource or "",
                    "details": row.details,
                    "ip_address": row.ip_address,
                    "status_code": row.status_code,
                })

            return {"entries": entries, "total": total}

        finally:
            session.close()

    except Exception as e:
        logger.exception(f"Audit log retrieval failed: {e}")
        return {"entries": [], "total": 0}


# ---------------------------------------------------------------------------
# Guardrails — blocklists for dangerous compounds
# ---------------------------------------------------------------------------

# SMARTS patterns for blocked compound classes
BLOCKED_PATTERNS: dict[str, dict] = {
    # Explosive precursors
    "trinitro": {
        "smarts": "[N+](=O)[O-].[N+](=O)[O-].[N+](=O)[O-]",
        "category": "explosive_precursor",
        "name": "Trinitro compound",
        "description": "Contains multiple nitro groups — potential explosive precursor",
    },
    "triacetone_triperoxide": {
        "smarts": "C1(C)(C)OOC(C)(C)OOC(C)(C)OO1",
        "category": "explosive_precursor",
        "name": "TATP-like peroxide",
        "description": "Cyclic triperoxide — high explosive",
    },
    "primary_azide": {
        "smarts": "[N-]=[N+]=[N-]",
        "category": "explosive_precursor",
        "name": "Organic azide",
        "description": "Contains azide group — potentially shock-sensitive",
    },
    # Chemical weapon precursors (CWC Schedule 1-3)
    "nerve_agent_p": {
        "smarts": "[P](=O)(F)([O,N])[O,N]",
        "category": "chemical_weapon",
        "name": "Organophosphate nerve agent class",
        "description": "Matches CWC Schedule 1 nerve agent scaffold",
    },
    "mustard_gas": {
        "smarts": "ClCCSCCCl",
        "category": "chemical_weapon",
        "name": "Sulfur mustard",
        "description": "CWC Schedule 1 — chemical warfare agent",
    },
}

# Known controlled substance SMILES (partial list for demo)
CONTROLLED_SMILES: set[str] = set()  # Intentionally empty — real deployment would load from secure DB


def check_guardrails(smiles: str) -> dict:
    """Check a molecule against safety guardrails.

    Args:
        smiles: Target molecule SMILES

    Returns:
        Dict with blocked status and list of alerts
    """
    alerts = []

    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "smiles": smiles,
                "blocked": False,
                "alerts": [{"category": "parse_error", "name": "Invalid SMILES",
                           "severity": "warning", "description": "Could not parse SMILES string"}],
                "requires_review": True,
            }

        # Check SMARTS patterns
        for pattern_name, pattern_info in BLOCKED_PATTERNS.items():
            try:
                smarts_mol = Chem.MolFromSmarts(pattern_info["smarts"])
                if smarts_mol and mol.HasSubstructMatch(smarts_mol):
                    alerts.append({
                        "category": pattern_info["category"],
                        "name": pattern_info["name"],
                        "severity": "critical",
                        "description": pattern_info["description"],
                    })
            except Exception:
                continue

        # Check controlled substances
        canon = Chem.MolToSmiles(mol)
        if canon in CONTROLLED_SMILES:
            alerts.append({
                "category": "controlled_substance",
                "name": "Controlled substance match",
                "severity": "critical",
                "description": "This compound matches a known controlled substance",
            })

        blocked = any(a["severity"] == "critical" for a in alerts)

        return {
            "smiles": smiles,
            "blocked": blocked,
            "alerts": alerts,
            "requires_review": True,  # Always require human review
        }

    except ImportError:
        logger.warning("RDKit not available — guardrails check skipped")
        return {
            "smiles": smiles,
            "blocked": False,
            "alerts": [],
            "requires_review": True,
        }
    except Exception as e:
        logger.exception(f"Guardrails check failed: {e}")
        return {
            "smiles": smiles,
            "blocked": False,
            "alerts": [],
            "requires_review": True,
        }
