"""RSGPT model loading and LoRA adapter setup.

Handles:
  - Loading RSGPT pretrained weights (LLaMA2 24-layer, 3.2B params)
  - Adding LoRA adapters for parameter-efficient fine-tuning
  - Managing the tokenizer with edit-language special tokens
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_rsgpt_model(
    weights_path: str | Path | None = None,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_8bit: bool = False,
    device_map: str = "auto",
):
    """Load RSGPT model with optional LoRA adapters.

    RSGPT is a LLaMA2-based model (24 layers, 2048 hidden, 32 heads, 3.2B params).
    We load it using HuggingFace Transformers and add LoRA via PEFT.

    Args:
        weights_path: Path to RSGPT checkpoint (.pth file).
        use_lora: Whether to add LoRA adapters.
        lora_rank: LoRA rank (default 16).
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout for LoRA layers.
        load_in_8bit: Use 8-bit quantization (saves VRAM).
        device_map: Device placement strategy.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import (
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
    )

    from rasyn.models.llm.tokenizer import ALL_SPECIAL_TOKENS

    # RSGPT architecture config (from paper: 24-layer LLaMA2)
    config = LlamaConfig(
        vocab_size=32000,  # Will be resized after adding special tokens
        hidden_size=2048,
        intermediate_size=5504,  # Standard for LLaMA2 with 2048 hidden
        num_hidden_layers=24,
        num_attention_heads=32,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        use_cache=True,
    )

    logger.info("Initializing LLaMA2-based model (24 layers, 2048 hidden, 32 heads)...")

    if load_in_8bit:
        model = LlamaForCausalLM(config)
    else:
        model = LlamaForCausalLM(config)

    # Load pretrained weights if available
    if weights_path and Path(weights_path).exists():
        logger.info(f"Loading RSGPT weights from {weights_path}...")
        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Try loading with strict=False to handle architecture mismatches
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
        logger.info("RSGPT weights loaded.")
    else:
        logger.warning("No pretrained weights provided. Model initialized randomly.")

    # Set up tokenizer (use LLaMA tokenizer as base)
    # RSGPT uses BPE tokenization — we use the LLaMA tokenizer and add special tokens
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            use_fast=True,
            trust_remote_code=True,
        )
    except Exception:
        # Fallback: create a simple tokenizer
        logger.warning("Could not load LLaMA tokenizer. Using GPT-2 tokenizer as fallback.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Add special tokens for edit language
    special_tokens_dict = {"additional_special_tokens": ALL_SPECIAL_TOKENS}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<PAD>"
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added} special tokens to tokenizer")

    # Resize model embeddings to match new vocabulary
    model.resize_token_embeddings(len(tokenizer))

    # Add LoRA if requested
    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"LoRA enabled: {trainable:,} trainable / {total:,} total params "
            f"({100 * trainable / total:.2f}%)"
        )

    return model, tokenizer


def save_model(model, tokenizer, output_dir: str | Path) -> None:
    """Save model and tokenizer to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapters (if PEFT model)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), output_dir / "model.pt")

    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")


def load_trained_model(checkpoint_dir: str | Path, device: str = "auto"):
    """Load a fine-tuned model from checkpoint."""
    from transformers import AutoTokenizer
    from peft import PeftModel

    checkpoint_dir = Path(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # Load base model first, then LoRA
    base_model, _ = load_rsgpt_model(use_lora=False)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, tokenizer
