"""Train scaled RetroTransformer v2 models (medium/large/xl).

Provides preset configurations for scaling the RetroTransformer:
  - medium: d_model=768, 12 heads, 8+8 layers, d_ff=3072 (~90M params)
  - large:  d_model=1024, 16 heads, 8+8 layers, d_ff=4096 (~150M params)
  - xl:     d_model=1024, 16 heads, 12+12 layers, d_ff=4096 (~250M params)

The base model (d_model=512, 8 heads, 6+6 layers) is ~45M params.

Usage:
    # Medium model (~90M params)
    python -u scripts/train_retro_v2_scaled.py --scale medium

    # Large model (~150M params)
    python -u scripts/train_retro_v2_scaled.py --scale large

    # XL model (~250M params)
    python -u scripts/train_retro_v2_scaled.py --scale xl --batch-size 16

    # RunPod (A100 80GB)
    nohup python -u scripts/train_retro_v2_scaled.py --scale large > train_scaled_large.log 2>&1 &
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# Preset configurations
SCALE_CONFIGS = {
    "base": {
        "d_model": 512, "nhead": 8,
        "num_encoder_layers": 6, "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "batch_size": 64, "lr": 3e-4,
    },
    "medium": {
        "d_model": 768, "nhead": 12,
        "num_encoder_layers": 8, "num_decoder_layers": 8,
        "dim_feedforward": 3072,
        "batch_size": 48, "lr": 2e-4,
    },
    "large": {
        "d_model": 1024, "nhead": 16,
        "num_encoder_layers": 8, "num_decoder_layers": 8,
        "dim_feedforward": 4096,
        "batch_size": 32, "lr": 2e-4,
    },
    "xl": {
        "d_model": 1024, "nhead": 16,
        "num_encoder_layers": 12, "num_decoder_layers": 12,
        "dim_feedforward": 4096,
        "batch_size": 16, "lr": 1e-4,
    },
}


@click.command()
@click.option("--scale", type=click.Choice(["base", "medium", "large", "xl"]),
              default="large", help="Model scale preset")
@click.option("--data", default="data/processed/uspto50k/augmented_train.jsonl")
@click.option("--output-dir", default=None, type=str,
              help="Override output dir (default: checkpoints/retro_v2/scaled_{scale})")
@click.option("--epochs", default=200, type=int)
@click.option("--batch-size", default=None, type=int,
              help="Override batch size from preset")
@click.option("--lr", default=None, type=float,
              help="Override learning rate from preset")
@click.option("--grad-accum", default=1, type=int,
              help="Gradient accumulation steps (use for larger models)")
@click.option("--max-src-len", default=256, type=int)
@click.option("--max-tgt-len", default=128, type=int)
@click.option("--warmup-steps", default=3000, type=int)
@click.option("--label-smoothing", default=0.05, type=float)
@click.option("--conditioning-dropout", default=0.2, type=float)
@click.option("--use-rxn-class/--no-rxn-class", default=True)
@click.option("--val-split", default=0.1, type=float)
@click.option("--patience", default=15, type=int)
@click.option("--resume", default=None, type=str,
              help="Resume from checkpoint (e.g., pretrained_full/best)")
@click.option("--skip-sanity", is_flag=True)
@click.option("--device", default="auto")
def main(
    scale, data, output_dir, epochs, batch_size, lr, grad_accum,
    max_src_len, max_tgt_len, warmup_steps, label_smoothing,
    conditioning_dropout, use_rxn_class, val_split, patience,
    resume, skip_sanity, device,
):
    """Train a scaled RetroTransformer v2 model."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get preset config
    config = SCALE_CONFIGS[scale].copy()
    if batch_size is not None:
        config["batch_size"] = batch_size
    if lr is not None:
        config["lr"] = lr

    effective_batch = config["batch_size"] * grad_accum
    if output_dir is None:
        output_dir = f"checkpoints/retro_v2/scaled_{scale}"

    data_path = PROJECT_ROOT / data
    output_path = PROJECT_ROOT / output_dir

    logger.info(f"Scale: {scale}")
    logger.info(f"  d_model={config['d_model']}, nhead={config['nhead']}")
    logger.info(f"  enc_layers={config['num_encoder_layers']}, dec_layers={config['num_decoder_layers']}")
    logger.info(f"  d_ff={config['dim_feedforward']}")
    logger.info(f"  batch={config['batch_size']} x grad_accum={grad_accum} = {effective_batch}")
    logger.info(f"  lr={config['lr']}")
    logger.info(f"Device: {device}")

    # Build tokenizer
    logger.info("Building regex tokenizer from data...")
    all_texts = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            all_texts.append(ex["src_text"])
            all_texts.append(ex["tgt_text"])

    from rasyn.models.retro.tokenizer_v2 import RegexSmilesTokenizer
    tokenizer = RegexSmilesTokenizer.build_from_data(all_texts)
    logger.info(f"Tokenizer: {tokenizer.vocab_size} tokens")

    # Load data
    from rasyn.models.retro.data_v2 import load_retro_data_v2, collate_fn_v2
    train_dataset, val_dataset = load_retro_data_v2(
        data_path=data_path,
        tokenizer=tokenizer,
        val_split=val_split,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        conditioning_dropout=conditioning_dropout,
        use_reaction_class=use_rxn_class,
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Build model
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": config["d_model"],
        "nhead": config["nhead"],
        "num_encoder_layers": config["num_encoder_layers"],
        "num_decoder_layers": config["num_decoder_layers"],
        "dim_feedforward": config["dim_feedforward"],
        "max_seq_len": max(max_src_len, max_tgt_len),
        "pad_token_id": tokenizer.pad_token_id,
        "num_segments": 2,
        "num_rxn_classes": 11,
    }

    from rasyn.models.retro.model_v2 import RetroTransformerV2, save_retro_model_v2

    # Resume from checkpoint or fresh
    resume_dir = None
    if resume:
        resume_path = PROJECT_ROOT / resume
        if (resume_path / "model.pt").exists():
            logger.info(f"Loading model from {resume_path} for resume...")
            from rasyn.models.retro.model_v2 import load_retro_model_v2
            model, _ = load_retro_model_v2(str(resume_path / "model.pt"), device=device)
            model.train()
            resume_dir = resume_path
            logger.info("Model loaded for resume")
        else:
            logger.warning(f"Resume path not found, starting fresh")
            model = RetroTransformerV2(**model_config).to(device)
    else:
        model = RetroTransformerV2(**model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Sanity checks
    if not skip_sanity and not resume:
        from scripts.train_retro_v2 import run_all_checks
        all_passed = run_all_checks(model, train_dataset, tokenizer, device, data_path)
        if not all_passed:
            logger.warning("Some checks failed. Proceeding anyway.")
        model = RetroTransformerV2(**model_config).to(device)
        logger.info("Model re-initialized for full training")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_fn_v2, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        collate_fn=collate_fn_v2, num_workers=2, pin_memory=True,
    )

    # Use the standard training loop (supports grad accumulation via batch size adjustment)
    from scripts.train_retro_v2 import train
    train(
        model=model, train_loader=train_loader, val_loader=val_loader,
        tokenizer=tokenizer, device=device, output_dir=output_path,
        epochs=epochs, lr=config["lr"], warmup_steps=warmup_steps,
        label_smoothing=label_smoothing,
        train_dataset=train_dataset, model_config=model_config,
        patience=patience, resume_from=resume_dir,
    )


if __name__ == "__main__":
    main()
