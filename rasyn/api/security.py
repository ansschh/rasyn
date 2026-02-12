"""Security middleware for Rasyn API — API key auth, rate limiting, security headers."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API Key Store (SQLite-backed)
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = Path(os.environ.get("RASYN_KEYS_DB", "/opt/rasyn/data/api_keys.db"))


def _hash_key(key: str) -> str:
    """SHA-256 hash of an API key (we never store raw keys)."""
    return hashlib.sha256(key.encode()).hexdigest()


class APIKeyStore:
    """SQLite-backed API key store. Stores hashed keys with metadata.

    Keys have roles:
      - 'admin': can create/list/revoke keys + call all API endpoints
      - 'user':  can only call API endpoints

    Thread-safe via per-call connections.
    """

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # Cache of valid key hashes for fast middleware checks
        self._cache: dict[str, dict] = {}
        self._cache_time = 0.0
        self._cache_ttl = 30.0  # refresh cache every 30s

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    key_hash TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    request_count INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    created_by TEXT
                )
            """)

    def _refresh_cache(self):
        now = time.monotonic()
        if now - self._cache_time < self._cache_ttl and self._cache:
            return
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, key_hash, name, role, is_active FROM api_keys WHERE is_active = 1"
            ).fetchall()
        self._cache = {
            row["key_hash"]: {
                "id": row["id"],
                "name": row["name"],
                "role": row["role"],
            }
            for row in rows
        }
        self._cache_time = now

    def validate_key(self, raw_key: str) -> dict | None:
        """Validate an API key. Returns key info dict or None if invalid."""
        self._refresh_cache()
        h = _hash_key(raw_key)
        info = self._cache.get(h)
        if info:
            # Update last_used_at in background (best-effort)
            try:
                with self._conn() as conn:
                    conn.execute(
                        "UPDATE api_keys SET last_used_at = ?, request_count = request_count + 1 WHERE key_hash = ?",
                        (datetime.now(timezone.utc).isoformat(), h),
                    )
            except Exception:
                pass
        return info

    def create_key(self, name: str, role: str = "user", created_by: str | None = None) -> dict:
        """Create a new API key. Returns key info including the raw key (shown only once)."""
        raw_key = f"rsy_{secrets.token_urlsafe(36)}"
        key_hash = _hash_key(raw_key)
        key_id = secrets.token_hex(8)
        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO api_keys (id, key_hash, name, role, created_at, created_by) VALUES (?, ?, ?, ?, ?, ?)",
                (key_id, key_hash, name, role, now, created_by),
            )

        # Invalidate cache
        self._cache_time = 0.0

        return {
            "id": key_id,
            "key": raw_key,  # Only returned at creation time!
            "name": name,
            "role": role,
            "created_at": now,
        }

    def list_keys(self) -> list[dict]:
        """List all keys (without raw key values)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, name, role, created_at, last_used_at, request_count, is_active, created_by FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def revoke_key(self, key_id: str) -> bool:
        """Revoke a key by ID. Returns True if found and revoked."""
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE id = ?", (key_id,)
            )
        self._cache_time = 0.0
        return cursor.rowcount > 0

    def seed_admin_keys(self, raw_keys: list[str]):
        """Seed admin keys from environment variable (idempotent)."""
        for raw_key in raw_keys:
            key_hash = _hash_key(raw_key)
            with self._conn() as conn:
                existing = conn.execute(
                    "SELECT id FROM api_keys WHERE key_hash = ?", (key_hash,)
                ).fetchone()
                if not existing:
                    key_id = secrets.token_hex(8)
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        "INSERT INTO api_keys (id, key_hash, name, role, created_at, created_by) VALUES (?, ?, ?, ?, ?, ?)",
                        (key_id, key_hash, "Admin (env)", "admin", now, "system"),
                    )
                    logger.info(f"Seeded admin key {key_id}")
        self._cache_time = 0.0


# Global key store instance
_key_store: APIKeyStore | None = None


def get_key_store() -> APIKeyStore:
    """Get or create the global key store."""
    global _key_store
    if _key_store is None:
        _key_store = APIKeyStore()
        # Seed admin keys from env
        env_keys = os.environ.get("RASYN_API_KEYS", "")
        admin_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        if admin_keys:
            _key_store.seed_admin_keys(admin_keys)
    return _key_store


# ---------------------------------------------------------------------------
# API Key Authentication Middleware
# ---------------------------------------------------------------------------

# Paths that do NOT require authentication
PUBLIC_PATHS = {
    "/api/v1/health",      # ALB health check
    "/openapi.json",
    "/docs",
    "/redoc",
}

# Path prefixes that are public (no auth at all)
PUBLIC_PREFIXES = (
    "/docs",
    "/redoc",
    "/openapi.json",
)

# ---------------------------------------------------------------------------
# Demo Auth (cookie-based so Gradio internal calls work)
# ---------------------------------------------------------------------------

_DEMO_PASSWORD = os.environ.get("RASYN_DEMO_PASS", "rasyn2026")
_COOKIE_SECRET = os.environ.get("RASYN_COOKIE_SECRET", secrets.token_hex(32))
_DEMO_COOKIE = "rasyn_demo_session"


def _sign_cookie(nonce: str) -> str:
    return hmac.new(_COOKIE_SECRET.encode(), nonce.encode(), hashlib.sha256).hexdigest()[:32]


def _make_cookie() -> str:
    nonce = secrets.token_hex(16)
    return f"{nonce}.{_sign_cookie(nonce)}"


def _verify_cookie(cookie: str) -> bool:
    try:
        nonce, sig = cookie.split(".", 1)
        return hmac.compare_digest(sig, _sign_cookie(nonce))
    except (ValueError, AttributeError):
        return False


def _is_valid_demo_token(token: str) -> bool:
    """Check if a demo token is valid (demo password or API key)."""
    if hmac.compare_digest(token, _DEMO_PASSWORD):
        return True
    try:
        store = get_key_store()
        return store.validate_key(token) is not None
    except Exception:
        return False


_LOGIN_PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rasyn Demo - Login</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;display:flex;align-items:center;justify-content:center;min-height:100vh}
.c{background:#1e293b;border-radius:12px;padding:40px;max-width:420px;width:90%;box-shadow:0 25px 50px rgba(0,0,0,.3)}
h1{font-size:24px;margin-bottom:8px;color:#f8fafc}
p{font-size:14px;color:#94a3b8;margin-bottom:24px}
label{font-size:13px;color:#94a3b8;display:block;margin-bottom:6px}
input{width:100%;padding:10px 14px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#f8fafc;font-size:15px;outline:none}
input:focus{border-color:#3b82f6}
button{width:100%;padding:12px;border-radius:8px;border:none;margin-top:16px;background:#3b82f6;color:#fff;font-size:15px;font-weight:600;cursor:pointer}
button:hover{background:#2563eb}
.err{color:#f87171;font-size:13px;margin-top:12px}
</style></head><body>
<div class="c"><h1>Rasyn Retrosynthesis</h1>
<p>Enter your access token or demo password to continue.</p>
<form method="GET" action="/demo/">
<label for="token">Access Token</label>
<input type="password" name="token" id="token" placeholder="Enter token or password" required autofocus>
<button type="submit">Access Demo</button>
</form>
__ERROR__
</div></body></html>"""


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require a valid API key for all non-public endpoints.

    - /api/v1/* endpoints: checked via X-API-Key header / Bearer / ?api_key=
    - /demo/* paths: checked via session cookie (set with ?token= on first visit)
    - /api/v1/health, /docs, etc.: public
    """

    def __init__(self, app, api_keys: list[str] | None = None):
        super().__init__(app)
        self.store = get_key_store()

    def _extract_key(self, request: Request) -> str | None:
        key = request.headers.get("x-api-key")
        if key:
            return key
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        key = request.query_params.get("api_key")
        if key:
            return key
        return None

    def _is_public(self, path: str) -> bool:
        if path in PUBLIC_PATHS:
            return True
        if path.startswith(PUBLIC_PREFIXES):
            return True
        return False

    def _handle_demo_auth(self, request: Request) -> Response | None:
        """Handle /demo auth via cookies. Returns a Response to send, or None to proceed."""
        # Check for ?token= in query string
        token = request.query_params.get("token")
        if token:
            if _is_valid_demo_token(token):
                # Valid → set cookie and redirect (strip token from URL)
                resp = Response(status_code=302)
                resp.headers["location"] = "/demo/"
                cookie_val = _make_cookie()
                resp.set_cookie(
                    _DEMO_COOKIE, cookie_val,
                    httponly=True, samesite="lax", max_age=86400 * 7,
                )
                return resp
            else:
                # Invalid token → login page with error
                html = _LOGIN_PAGE.replace("__ERROR__", '<p class="err">Invalid token. Try again.</p>')
                return Response(content=html, status_code=401, media_type="text/html")

        # Check for valid session cookie
        cookie = request.cookies.get(_DEMO_COOKIE)
        if cookie and _verify_cookie(cookie):
            return None  # Authenticated → proceed

        # No auth → show login page (only for the main /demo/ page, not internal API calls)
        # Gradio internal calls expect JSON, so return 401 JSON for those
        if "/gradio_api/" in request.url.path:
            return JSONResponse(status_code=401, content={"error": "not_authenticated"})

        html = _LOGIN_PAGE.replace("__ERROR__", "")
        return Response(content=html, status_code=401, media_type="text/html")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Public paths (health, docs)
        if self._is_public(path):
            return await call_next(request)

        # Demo paths — cookie-based auth
        if path.startswith("/demo"):
            block = self._handle_demo_auth(request)
            if block is not None:
                return block
            return await call_next(request)

        # API paths — key-based auth
        key = self._extract_key(request)
        if not key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "authentication_required",
                    "message": "API key required. Pass via X-API-Key header, Authorization: Bearer <key>, or ?api_key= query param.",
                },
            )

        key_info = self.store.validate_key(key)
        if not key_info:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "invalid_api_key",
                    "message": "The provided API key is not valid or has been revoked.",
                },
            )

        request.state.api_key_info = key_info

        if path.startswith("/api/v1/keys") and key_info["role"] != "admin":
            return JSONResponse(
                status_code=403,
                content={
                    "error": "admin_required",
                    "message": "This endpoint requires an admin API key.",
                },
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self):
        self._requests: dict[str, list[tuple[float, str]]] = defaultdict(list)

    def _cleanup(self, client_ip: str, window: float):
        now = time.monotonic()
        self._requests[client_ip] = [
            (ts, pg) for ts, pg in self._requests[client_ip]
            if now - ts < window
        ]

    def is_allowed(self, client_ip: str, path_group: str, limit: int, window: float) -> tuple[bool, dict]:
        now = time.monotonic()
        self._cleanup(client_ip, window)

        count = sum(1 for ts, pg in self._requests[client_ip] if pg == path_group)
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


RATE_LIMITS = {
    "/api/v1/retro/single-step": (20, 60),
    "/api/v1/retro/multi-step": (5, 60),
    "/api/v1/molecules/": (60, 60),
    "/demo": (30, 60),
}

DEFAULT_RATE_LIMIT = (60, 60)

_rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting middleware."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Skip rate limiting for health checks and Gradio (it makes many internal calls)
        if path == "/api/v1/health" or path.startswith("/demo"):
            return await call_next(request)

        forwarded = request.headers.get("x-forwarded-for")
        client_ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")

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
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Only set X-Frame-Options on API routes (not /demo — Gradio uses iframes)
        if not request.url.path.startswith("/demo"):
            response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        if request.headers.get("x-forwarded-proto") == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_api_key() -> str:
    """Generate a secure random API key."""
    return f"rsy_{secrets.token_urlsafe(36)}"
