"""Security middleware for Rasyn API — API key auth, rate limiting, security headers."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from collections import defaultdict

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

# Paths that do NOT require authentication
PUBLIC_PATHS = {
    "/api/v1/health",      # ALB health check
    "/openapi.json",       # OpenAPI spec (optional, remove in prod)
    "/docs",               # Swagger UI (optional, remove in prod)
    "/redoc",              # ReDoc (optional, remove in prod)
}

# Paths that start with these prefixes are public
PUBLIC_PREFIXES = (
    "/docs",
    "/redoc",
    "/openapi.json",
)


def _constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require a valid API key for all non-public endpoints.

    Keys are checked from:
      1. X-API-Key header
      2. Authorization: Bearer <key> header
      3. ?api_key= query parameter (for browser/Gradio compatibility)

    Configure via:
      - RASYN_API_KEYS env var (comma-separated list of valid keys)
      - Or pass keys directly to constructor
    """

    def __init__(self, app, api_keys: list[str] | None = None):
        super().__init__(app)
        if api_keys:
            self.api_keys = set(api_keys)
        else:
            env_keys = os.environ.get("RASYN_API_KEYS", "")
            self.api_keys = {k.strip() for k in env_keys.split(",") if k.strip()}

        if not self.api_keys:
            logger.warning(
                "No API keys configured! Set RASYN_API_KEYS env var. "
                "All authenticated endpoints will return 401."
            )

    def _extract_key(self, request: Request) -> str | None:
        """Extract API key from request headers or query params."""
        # 1. X-API-Key header
        key = request.headers.get("x-api-key")
        if key:
            return key

        # 2. Authorization: Bearer <key>
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()

        # 3. Query parameter (for Gradio/browser access)
        key = request.query_params.get("api_key")
        if key:
            return key

        return None

    def _is_public(self, path: str) -> bool:
        """Check if path is public (no auth required)."""
        if path in PUBLIC_PATHS:
            return True
        if path.startswith(PUBLIC_PREFIXES):
            return True
        return False

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for public paths
        if self._is_public(path):
            return await call_next(request)

        # Extract and validate key
        key = self._extract_key(request)
        if not key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "authentication_required",
                    "message": "API key required. Pass via X-API-Key header, Authorization: Bearer <key>, or ?api_key= query param.",
                },
            )

        # Check against valid keys (constant-time comparison)
        valid = any(_constant_time_compare(key, valid_key) for valid_key in self.api_keys)
        if not valid:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "invalid_api_key",
                    "message": "The provided API key is not valid.",
                },
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self):
        # {client_ip: [(timestamp, path_group), ...]}
        self._requests: dict[str, list[tuple[float, str]]] = defaultdict(list)

    def _cleanup(self, client_ip: str, window: float):
        """Remove requests older than window."""
        now = time.monotonic()
        self._requests[client_ip] = [
            (ts, pg) for ts, pg in self._requests[client_ip]
            if now - ts < window
        ]

    def is_allowed(self, client_ip: str, path_group: str, limit: int, window: float) -> tuple[bool, dict]:
        """Check if request is within rate limit.

        Returns (allowed, headers_dict).
        """
        now = time.monotonic()
        self._cleanup(client_ip, window)

        # Count requests in this path group
        count = sum(
            1 for ts, pg in self._requests[client_ip]
            if pg == path_group
        )

        remaining = max(0, limit - count)
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(now + window)),
        }

        if count >= limit:
            headers["Retry-After"] = str(int(window))
            return False, headers

        self._requests[client_ip].append((now, path_group))
        headers["X-RateLimit-Remaining"] = str(remaining - 1)
        return True, headers


# Rate limit configuration: path_prefix -> (limit, window_seconds)
RATE_LIMITS = {
    "/api/v1/retro/single-step": (20, 60),   # 20 per minute
    "/api/v1/retro/multi-step": (5, 60),      # 5 per minute (expensive)
    "/api/v1/molecules/": (60, 60),           # 60 per minute
    "/demo": (30, 60),                        # 30 per minute
}

# Global fallback
DEFAULT_RATE_LIMIT = (60, 60)  # 60 per minute

_rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting middleware."""

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path == "/api/v1/health":
            return await call_next(request)

        # Get client IP (check X-Forwarded-For from ALB)
        forwarded = request.headers.get("x-forwarded-for")
        client_ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")

        # Find matching rate limit
        path = request.url.path
        limit, window = DEFAULT_RATE_LIMIT
        path_group = "default"

        for prefix, (lim, win) in RATE_LIMITS.items():
            if path.startswith(prefix):
                limit, window = lim, win
                path_group = prefix
                break

        allowed, headers = _rate_limiter.is_allowed(client_ip, path_group, limit, window)

        if not allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Max {limit} requests per {window}s for this endpoint.",
                    "retry_after": headers.get("Retry-After"),
                },
            )
            for k, v in headers.items():
                response.headers[k] = v
            return response

        response = await call_next(request)
        for k, v in headers.items():
            response.headers[k] = v
        return response


# ---------------------------------------------------------------------------
# Security Headers
# ---------------------------------------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        # HSTS — only add if request came via HTTPS
        if request.headers.get("x-forwarded-proto") == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_api_key() -> str:
    """Generate a secure random API key (48 chars, URL-safe)."""
    return f"rsy_{secrets.token_urlsafe(36)}"
