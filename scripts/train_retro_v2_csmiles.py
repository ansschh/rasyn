"""Train RetroTransformer v2 with C-SMILES (Compositional SMILES) tokenizer.

Same architecture and training loop as train_retro_v2.py, but uses
CSmilesTokenizer instead of RegexSmilesTokenizer. The C-SMILES tokenizer
decomposes bracket atoms into element-level tokens, reducing vocab from
~150-200 to ~60-80 tokens.

Usage:
    # Build augmented dataset (same as standard)
    python scripts/build_augmented_dataset.py --n-augments 5

    # Train with C-SMILES tokenizer
    python -u scripts/train_retro_v2_csmiles.py

    # RunPod
    nohup python -u scripts/train_retro_v2_csmiles.py --epochs 200 > train_csmiles.log 2>&1 &
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


@click.command()
@click.option("--data", default="data/processed/uspto50k/augmented_train.jsonl")
@click.option("--output-dir", default="checkpoints/retro_v2/csmiles")
@click.option("--epochs", default=200, type=int)
@click.option("--batch-size", default=64, type=int)
@click.option("--lr", default=3e-4, type=float)
@click.option("--d-model", default=512, type=int)
@click.option("--nhead", default=8, type=int)
@click.option("--n-layers", default=6, type=int)
@click.option("--d-ff", default=2048, type=int)
@click.option("--max-src-len", default=256, type=int)
@click.option("--max-tgt-len", default=128, type=int)
@click.option("--warmup-steps", default=2000, type=int)
@click.option("--label-smoothing", default=0.05, type=float)
@click.option("--conditioning-dropout", default=0.2, type=float)
@click.option("--use-rxn-class/--no-rxn-class", default=True)
@click.option("--val-split", default=0.1, type=float)
@click.option("--patience", default=15, type=int)
@click.option("--sanity-only", is_flag=True)
@click.option("--skip-sanity", is_flag=True)
@click.option("--device", default="auto")
def main(
    data, output_dir, epochs, batch_size, lr, d_model, nhead, n_layers, d_ff,
    max_src_len, max_tgt_len, warmup_steps, label_smoothing,
    conditioning_dropout, use_rxn_class, val_split, patience,
    sanity_only, skip_sanity, device,
):
    """Train RetroTransformer v2 with C-SMILES tokenizer."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    data_path = PROJECT_ROOT / data
    output_path = PROJECT_ROOT / output_dir

    # Build C-SMILES tokenizer from data
    logger.info("Building C-SMILES tokenizer from data...")
    all_texts = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            all_texts.append(ex["src_text"])
            all_texts.append(ex["tgt_text"])

    from rasyn.models.retro.tokenizer_csmiles import CSmilesTokenizer
    tokenizer = CSmilesTokenizer.build_from_data(all_texts)
    logger.info(f"C-SMILES Tokenizer: {tokenizer.vocab_size} tokens")

    # Compare with regex tokenizer for reference
    from rasyn.models.retro.tokenizer_v2 import RegexSmilesTokenizer
    regex_tok = RegexSmilesTokenizer.build_from_data(all_texts)
    logger.info(f"  (Regex tokenizer for reference: {regex_tok.vocab_size} tokens)")

    # Show example tokenization comparison
    sample = all_texts[0] if all_texts else "CC(=O)Oc1ccccc1C(=O)O"
    regex_tokens = regex_tok.tokenize_smiles(sample)
    csmiles_tokens = tokenizer.tokenize_smiles(sample)
    logger.info(f"  Sample: {sample[:60]}")
    logger.info(f"  Regex tokens ({len(regex_tokens)}): {regex_tokens[:15]}...")
    logger.info(f"  C-SMILES tokens ({len(csmiles_tokens)}): {csmiles_tokens[:15]}...")

    # Roundtrip check on samples
    n_check = min(1000, len(all_texts))
    failures = sum(1 for t in all_texts[:n_check] if not tokenizer.roundtrip_check(t))
    logger.info(f"  Roundtrip check: {n_check - failures}/{n_check} pass ({failures} failures)")

    # Load data using C-SMILES tokenizer
    # The data_v2 module accepts any tokenizer with the same API
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

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": n_layers,
        "num_decoder_layers": n_layers,
        "dim_feedforward": d_ff,
        "max_seq_len": max(max_src_len, max_tgt_len),
        "pad_token_id": tokenizer.pad_token_id,
        "num_segments": 2,
        "num_rxn_classes": 11,
    }

    from rasyn.models.retro.model_v2 import RetroTransformerV2
    model = RetroTransformerV2(**model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Sanity checks
    if not skip_sanity:
        from scripts.train_retro_v2 import run_all_checks
        all_passed = run_all_checks(model, train_dataset, tokenizer, device, data_path)
        if sanity_only:
            return
        if not all_passed:
            logger.warning("Some checks failed. Proceeding anyway.")
        model = RetroTransformerV2(**model_config).to(device)
        logger.info("Model re-initialized for full training")
    elif sanity_only:
        return

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn_v2, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_v2, num_workers=2, pin_memory=True,
    )

    # Use the same training loop from train_retro_v2
    from scripts.train_retro_v2 import train
    train(
        model=model, train_loader=train_loader, val_loader=val_loader,
        tokenizer=tokenizer, device=device, output_dir=output_path,
        epochs=epochs, lr=lr, warmup_steps=warmup_steps,
        label_smoothing=label_smoothing,
        train_dataset=train_dataset, model_config=model_config,
        patience=patience,
    )


if __name__ == "__main__":
    main()
