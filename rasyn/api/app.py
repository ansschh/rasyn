"""FastAPI application entry point for Rasyn retrosynthesis API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rasyn.api.routes.admin import router as admin_router
from rasyn.api.routes.analyze import router as analyze_router
from rasyn.api.routes.copilot import router as copilot_router
from rasyn.api.routes.discover import router as discover_router
from rasyn.api.routes.execute import router as execute_router
from rasyn.api.routes.jobs import router as jobs_router
from rasyn.api.routes.keys import router as keys_router
from rasyn.api.routes.learn import router as learn_router
from rasyn.api.routes.molecules import router as molecules_router
from rasyn.api.routes.retro import router as retro_router
from rasyn.api.routes.source import router as source_router
from rasyn.api.audit_middleware import AuditMiddleware
from rasyn.api.security import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from rasyn.service.model_manager import ModelManager
from rasyn.service.pipeline_service import PipelineService

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "serve.yaml"

# Allowed CORS origins — set via RASYN_CORS_ORIGINS env var (comma-separated)
# or defaults to rasyn.ai domain
_DEFAULT_ORIGINS = [
    "https://rasyn.ai",
    "https://www.rasyn.ai",
    "https://app.rasyn.ai",
]


def _get_cors_origins() -> list[str]:
    env = os.environ.get("RASYN_CORS_ORIGINS", "")
    if env.strip():
        return [o.strip() for o in env.split(",") if o.strip()]
    return _DEFAULT_ORIGINS


def load_config(path: str | Path | None = None) -> dict:
    """Load server configuration from YAML."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    logger.warning(f"Config not found at {path}, using defaults.")
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — load models on startup, cleanup on shutdown."""
    config = load_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Initialize service layer
    mm = ModelManager(config)
    app.state.model_manager = mm
    app.state.pipeline = PipelineService(mm)

    # Optionally warmup models at startup
    warmup = config.get("server", {}).get("warmup_models")
    if warmup:
        logger.info(f"Warming up models: {warmup}")
        mm.warmup(warmup if isinstance(warmup, list) else None)

    # Share pipeline with Gradio if mounted
    try:
        import rasyn.gradio_app as gradio_mod
        gradio_mod._pipeline_service = app.state.pipeline
    except Exception:
        pass

    # Initialize database tables
    try:
        from rasyn.db.init_db import init_db
        await init_db()
        logger.info("Database initialized.")
    except Exception as e:
        logger.warning(f"Database init skipped (not configured?): {e}")

    logger.info("Rasyn API ready.")
    yield
    logger.info("Rasyn API shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Rasyn Retrosynthesis API",
        version="1.0.0",
        description=(
            "AI-powered retrosynthetic analysis. Predict single-step "
            "disconnections or plan multi-step routes to purchasable starting materials."
        ),
        lifespan=lifespan,
    )

    # --- Security middleware (applied bottom-up: last added = first executed) ---

    # 1. CORS — restrict to allowed origins
    cors_origins = _get_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )

    # 2. Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # 3. Rate limiting
    app.add_middleware(RateLimitMiddleware)

    # 4. Audit logging (records all API calls)
    app.add_middleware(AuditMiddleware)

    # 5. API key authentication (outermost — runs first)
    app.add_middleware(APIKeyMiddleware)

    # --- API routes ---
    app.include_router(retro_router, prefix="/api/v1")
    app.include_router(molecules_router, prefix="/api/v1/molecules")
    app.include_router(keys_router, prefix="/api/v1")

    # v2 — job-based retrosynthesis with SSE streaming
    app.include_router(jobs_router, prefix="/api/v2")
    app.include_router(discover_router, prefix="/api/v2")
    app.include_router(source_router, prefix="/api/v2")
    app.include_router(execute_router, prefix="/api/v2")
    app.include_router(analyze_router, prefix="/api/v2")
    app.include_router(learn_router, prefix="/api/v2")
    app.include_router(admin_router, prefix="/api/v2")
    app.include_router(copilot_router, prefix="/api/v2")

    # --- Gradio demo (lazy import to avoid hard dependency) ---
    try:
        import gradio as gr
        from rasyn.gradio_app import create_gradio_app

        gradio_app = create_gradio_app()
        app = gr.mount_gradio_app(app, gradio_app, path="/demo")
        print("[rasyn] Gradio demo mounted at /demo (token-gated via middleware)")
    except ImportError:
        print("[rasyn] Gradio not installed — demo UI disabled.")
    except Exception as e:
        print(f"[rasyn] Failed to mount Gradio demo: {e}")
        import traceback
        traceback.print_exc()

    return app


# Default app instance for `uvicorn rasyn.api.app:app`
app = create_app()
