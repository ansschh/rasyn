"""Authentication dependency for the Gradio demo.

Flow:
  1. User visits /demo/?token=<api_key_or_demo_password>
  2. Token is validated, a session cookie is set, user is redirected to /demo/
  3. All subsequent Gradio internal calls (/demo/gradio_api/*) use the cookie
  4. If no token and no cookie → return 401 HTML page with a login form
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from fastapi import Cookie, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

# Demo access password (separate from API keys, simpler for sharing)
_DEMO_PASSWORD = os.environ.get("RASYN_DEMO_PASS", "rasyn2026")

# Secret for signing session cookies
_COOKIE_SECRET = os.environ.get("RASYN_COOKIE_SECRET", secrets.token_hex(32))
_COOKIE_NAME = "rasyn_demo_session"


def _sign_token(value: str) -> str:
    """Create an HMAC signature for a cookie value."""
    return hmac.new(_COOKIE_SECRET.encode(), value.encode(), hashlib.sha256).hexdigest()[:32]


def _make_cookie_value() -> str:
    """Create a signed session cookie value."""
    nonce = secrets.token_hex(16)
    sig = _sign_token(nonce)
    return f"{nonce}.{sig}"


def _verify_cookie(cookie: str) -> bool:
    """Verify a signed session cookie."""
    try:
        nonce, sig = cookie.split(".", 1)
        return hmac.compare_digest(sig, _sign_token(nonce))
    except (ValueError, AttributeError):
        return False


def _is_valid_token(token: str) -> bool:
    """Check if a token is valid (demo password or a valid API key)."""
    # Check demo password
    if hmac.compare_digest(token, _DEMO_PASSWORD):
        return True
    # Check if it's a valid API key
    try:
        from rasyn.api.security import get_key_store
        store = get_key_store()
        return store.validate_key(token) is not None
    except Exception:
        return False


_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Rasyn Demo - Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0f172a; color: #e2e8f0; display: flex; align-items: center;
               justify-content: center; min-height: 100vh; }
        .card { background: #1e293b; border-radius: 12px; padding: 40px;
                max-width: 420px; width: 90%; box-shadow: 0 25px 50px rgba(0,0,0,0.3); }
        h1 { font-size: 24px; margin-bottom: 8px; color: #f8fafc; }
        p { font-size: 14px; color: #94a3b8; margin-bottom: 24px; }
        label { font-size: 13px; color: #94a3b8; display: block; margin-bottom: 6px; }
        input { width: 100%; padding: 10px 14px; border-radius: 8px; border: 1px solid #334155;
                background: #0f172a; color: #f8fafc; font-size: 15px; outline: none; }
        input:focus { border-color: #3b82f6; }
        button { width: 100%; padding: 12px; border-radius: 8px; border: none; margin-top: 16px;
                 background: #3b82f6; color: white; font-size: 15px; font-weight: 600;
                 cursor: pointer; transition: background 0.2s; }
        button:hover { background: #2563eb; }
        .error { color: #f87171; font-size: 13px; margin-top: 12px; display: none; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Rasyn Retrosynthesis</h1>
        <p>Enter your access token or demo password to continue.</p>
        <form method="GET" action="/demo/">
            <label for="token">Access Token</label>
            <input type="password" name="token" id="token" placeholder="Enter token or password" required autofocus>
            <button type="submit">Access Demo</button>
        </form>
        ERRORMSG
    </div>
</body>
</html>"""


async def demo_auth_dependency(
    request: Request,
    token: str | None = Query(None),
    rasyn_demo_session: str | None = Cookie(None),
):
    """FastAPI dependency that gates access to the Gradio demo.

    Accepts:
      - ?token=<password_or_api_key> in URL (sets cookie, redirects)
      - rasyn_demo_session cookie (from previous auth)
    """
    # 1. Check for token in query string → validate, set cookie, redirect
    if token:
        if _is_valid_token(token):
            cookie_val = _make_cookie_value()
            response = RedirectResponse(url="/demo/", status_code=302)
            response.set_cookie(
                _COOKIE_NAME, cookie_val,
                httponly=True, samesite="lax", max_age=86400 * 7,  # 7 days
            )
            return response
        else:
            # Invalid token — show login with error
            html = _LOGIN_HTML.replace(
                "ERRORMSG",
                '<p class="error" style="display:block">Invalid token. Please try again.</p>'
            )
            return HTMLResponse(content=html, status_code=401)

    # 2. Check for valid session cookie
    if rasyn_demo_session and _verify_cookie(rasyn_demo_session):
        return  # Authenticated via cookie — allow access

    # 3. No auth — show login page
    html = _LOGIN_HTML.replace("ERRORMSG", "")
    return HTMLResponse(content=html, status_code=401)
