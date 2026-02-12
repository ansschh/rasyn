"""FastAPI application entry point for Rasyn retrosynthesis API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rasyn.api.routes.molecules import router as molecules_router
from rasyn.api.routes.retro import router as retro_router
from rasyn.service.model_manager import ModelManager
from rasyn.service.pipeline_service import PipelineService

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "serve.yaml"


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

    # CORS — allow all origins for development; restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(retro_router, prefix="/api/v1")
    app.include_router(molecules_router, prefix="/api/v1/molecules")

    # Mount Gradio demo (lazy import to avoid hard dependency)
    try:
        import gradio as gr
        from rasyn.gradio_app import create_gradio_app

        gradio_app = create_gradio_app()
        app = gr.mount_gradio_app(app, gradio_app, path="/demo")
        logger.info("Gradio demo mounted at /demo")
    except ImportError:
        logger.info("Gradio not installed — demo UI disabled.")
    except Exception:
        logger.exception("Failed to mount Gradio demo.")

    return app


# Default app instance for `uvicorn rasyn.api.app:app`
app = create_app()
