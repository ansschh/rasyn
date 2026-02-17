"""Audit middleware — logs all API requests to the audit_log table.

Only logs mutating requests (POST, PUT, DELETE) and specific read paths
to avoid flooding the audit log with health checks and static assets.
"""

from __future__ import annotations

import logging
import threading

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Paths that should be audited (prefix match)
AUDIT_PATHS = {
    "/api/v2/plan",
    "/api/v2/execute",
    "/api/v2/analyze",
    "/api/v2/learn",
    "/api/v2/admin",
    "/api/v2/discover",
    "/api/v2/source",
}

# Paths to skip (exact or prefix)
SKIP_PATHS = {
    "/api/v1/health",
    "/api/v2/jobs",  # SSE streams are too noisy
    "/docs",
    "/openapi.json",
    "/favicon.ico",
}


class AuditMiddleware(BaseHTTPMiddleware):
    """Records API calls to the audit log table."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # Only audit specific paths
        path = request.url.path
        should_audit = False

        for audit_path in AUDIT_PATHS:
            if path.startswith(audit_path):
                should_audit = True
                break

        for skip_path in SKIP_PATHS:
            if path.startswith(skip_path):
                should_audit = False
                break

        # Only audit mutating requests + specific reads
        if should_audit and request.method in ("POST", "PUT", "DELETE"):
            # Fire-and-forget in background thread to not slow down response
            ip = request.client.host if request.client else None
            user_id = getattr(request.state, "user_id", None) if hasattr(request, "state") else None
            thread = threading.Thread(
                target=_log_async,
                args=(request.method, path, user_id, ip, response.status_code),
                daemon=True,
            )
            thread.start()

        return response


def _log_async(method: str, path: str, user_id: str | None, ip: str | None, status_code: int) -> None:
    """Background thread to record audit entry."""
    try:
        from rasyn.modules.admin import log_action

        action = f"{method} {path}"
        log_action(
            action=action,
            resource=path,
            user_id=user_id or "api_user",
            ip_address=ip,
            status_code=status_code,
        )
    except Exception as e:
        logger.debug(f"Audit middleware log failed: {e}")
