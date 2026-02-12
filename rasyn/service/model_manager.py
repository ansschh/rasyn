"""Model lifecycle manager — lazy loading, GPU memory, concurrency lock."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class ModelManager:
    """Loads and manages all model instances on GPU.

    Models are loaded lazily on first request and cached.
    An asyncio.Lock serializes GPU-bound inference calls.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = asyncio.Lock()

        # Cached model instances (None = not yet loaded)
        self._llm_model = None
        self._llm_tokenizer = None
        self._retro_model = None
        self._retro_tokenizer = None
        self._graph_head = None
        self._forward_model = None
        self._forward_tokenizer = None
        self._lg_vocab: dict | None = None

        self._loaded: list[str] = []

    # ------------------------------------------------------------------
    # Public lazy-load accessors
    # ------------------------------------------------------------------

    def get_llm(self):
        """Return (model, tokenizer) for the RSGPT LLM."""
        if self._llm_model is None:
            self._load_llm()
        return self._llm_model, self._llm_tokenizer

    def get_retro(self):
        """Return (model, tokenizer) for RetroTransformer v2."""
        if self._retro_model is None:
            self._load_retro()
        return self._retro_model, self._retro_tokenizer

    def get_graph_head(self):
        """Return the graph edit head model."""
        if self._graph_head is None:
            self._load_graph_head()
        return self._graph_head

    def get_forward(self):
        """Return (model, tokenizer) for the forward model."""
        if self._forward_model is None:
            self._load_forward()
        return self._forward_model, self._forward_tokenizer

    def get_lg_vocab(self) -> dict:
        """Return the leaving-group vocabulary dict."""
        if self._lg_vocab is None:
            self._load_lg_vocab()
        return self._lg_vocab

    def loaded_models(self) -> list[str]:
        """Return names of currently loaded models."""
        return list(self._loaded)

    # ------------------------------------------------------------------
    # Warmup — pre-load everything at startup
    # ------------------------------------------------------------------

    def warmup(self, models: list[str] | None = None):
        """Pre-load specified models (or all configured ones)."""
        if models is None:
            models = list(self.config.get("models", {}).keys())
        for name in models:
            try:
                if name == "llm":
                    self.get_llm()
                elif name == "retro_v2":
                    self.get_retro()
                elif name == "graph_head":
                    self.get_graph_head()
                elif name == "forward":
                    self.get_forward()
            except Exception:
                logger.exception(f"Failed to load model: {name}")

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _resolve_device(self, model_cfg: dict) -> str:
        dev = model_cfg.get("device", "auto")
        if dev == "auto":
            return self.device
        return dev

    def _load_llm(self):
        cfg = self.config.get("models", {}).get("llm", {})
        checkpoint = cfg.get("checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            logger.warning(f"LLM checkpoint not found: {checkpoint}")
            return
        device = self._resolve_device(cfg)
        logger.info(f"Loading LLM from {checkpoint} on {device}...")
        from rasyn.models.llm.model import load_trained_model
        self._llm_model, self._llm_tokenizer = load_trained_model(checkpoint, device)
        self._loaded.append("llm")
        logger.info("LLM loaded.")

    def _load_retro(self):
        cfg = self.config.get("models", {}).get("retro_v2", {})
        checkpoint = cfg.get("checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            logger.warning(f"RetroTx v2 checkpoint not found: {checkpoint}")
            return
        device = self._resolve_device(cfg)
        logger.info(f"Loading RetroTx v2 from {checkpoint} on {device}...")
        from rasyn.models.retro.model_v2 import load_retro_model_v2
        self._retro_model, self._retro_tokenizer = load_retro_model_v2(checkpoint, device)
        self._loaded.append("retro_v2")
        logger.info("RetroTx v2 loaded.")

    def _load_graph_head(self):
        cfg = self.config.get("models", {}).get("graph_head", {})
        checkpoint = cfg.get("checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            logger.warning(f"Graph head checkpoint not found: {checkpoint}")
            return
        device = self._resolve_device(cfg)
        logger.info(f"Loading graph head from {checkpoint} on {device}...")
        from rasyn.models.graph_head.model import GraphEditHead
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        ckpt_config = ckpt.get("config", {})
        model = GraphEditHead(
            hidden_dim=ckpt_config.get("hidden_dim", 32),
            lg_vocab_size=ckpt_config.get("lg_vocab_size", 170),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        self._graph_head = model
        self._loaded.append("graph_head")
        logger.info("Graph head loaded.")

    def _load_forward(self):
        cfg = self.config.get("models", {}).get("forward", {})
        checkpoint = cfg.get("checkpoint")
        if not checkpoint or not Path(checkpoint).exists():
            logger.warning(f"Forward model checkpoint not found: {checkpoint}")
            return
        device = self._resolve_device(cfg)
        logger.info(f"Loading forward model from {checkpoint} on {device}...")
        from rasyn.models.forward.model import load_forward_model
        self._forward_model, self._forward_tokenizer = load_forward_model(checkpoint, device)
        self._loaded.append("forward")
        logger.info("Forward model loaded.")

    def _load_lg_vocab(self):
        path = self.config.get("data", {}).get("lg_vocab", "data/vocab/lg_vocab.json")
        if not Path(path).exists():
            logger.warning(f"LG vocab not found: {path}")
            self._lg_vocab = {}
            return
        with open(path) as f:
            self._lg_vocab = json.load(f)
        logger.info(f"Loaded LG vocab with {len(self._lg_vocab)} entries.")
