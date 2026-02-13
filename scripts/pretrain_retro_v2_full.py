"""Pre-train RetroTransformer v2 on USPTO-FULL (1.9M reactions).

Pre-trains on the larger USPTO-FULL dataset, then the checkpoint can be
fine-tuned on USPTO-50K for better performance.

Key differences from train_retro_v2.py:
  - No reaction class tokens (USPTO-FULL has no class labels)
  - Larger warmup (10K steps)
  - Fewer epochs (20) — dataset is ~40x larger
  - Tokenizer built from USPTO-FULL data (superset vocab)
  - Saved tokenizer can be reused for fine-tuning

Usage:
    # Step 1: Download + preprocess USPTO-FULL
    python -u scripts/download_data.py --datasets uspto_full
    python -u scripts/preprocess_all.py --dataset uspto_full

    # Step 2: Build augmented dataset (3x for FULL, not 20x — dataset already huge)
    python -u scripts/build_augmented_dataset.py \
        --edit-data data/processed/uspto_full/edit_conditioned_train.jsonl \
        --reactions-data data/processed/uspto_full/reactions.jsonl \
        --output data/processed/uspto_full/augmented_3x_train.jsonl \
        --n-augments 3

    # Step 3: Pre-train
    python -u scripts/pretrain_retro_v2_full.py

    # Step 4: Fine-tune on 50K
    python -u scripts/train_retro_v2.py \
        --resume checkpoints/retro_v2/pretrained_full/best \
        --data data/processed/uspto50k/augmented_train.jsonl \
        --lr 1e-4 --epochs 80 --skip-sanity

    # RunPod
    nohup python -u scripts/pretrain_retro_v2_full.py > pretrain_full.log 2>&1 &
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from rdkit import Chem, RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDKIT_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def evaluate_pretrain(model, val_loader, tokenizer, device, max_batches=100):
    """Quick validation: loss + token accuracy (no beam search for speed)."""
    model.eval()
    loss_fn = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= max_batches:
                break

            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            seg_ids = batch["segment_ids"].to(device)

            tgt_in = tgt_ids[:, :-1]
            tgt_tgt = tgt_ids[:, 1:]

            log_probs, _ = model(src_ids, tgt_in, seg_ids)
            loss = loss_fn(log_probs.reshape(-1, log_probs.size(-1)), tgt_tgt.reshape(-1))

            total_loss += loss.item()
            n_batches += 1

            preds = log_probs.argmax(dim=-1)
            mask = tgt_tgt != tokenizer.pad_token_id
            total_correct += ((preds == tgt_tgt) & mask).sum().item()
            total_tokens += mask.sum().item()

    model.train()
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_token_acc": total_correct / max(total_tokens, 1),
    }


@click.command()
@click.option("--data", default="data/processed/uspto_full/augmented_3x_train.jsonl",
              help="Path to augmented USPTO-FULL data")
@click.option("--output-dir", default="checkpoints/retro_v2/pretrained_full")
@click.option("--epochs", default=20, type=int)
@click.option("--batch-size", default=64, type=int)
@click.option("--lr", default=3e-4, type=float)
@click.option("--d-model", default=512, type=int)
@click.option("--nhead", default=8, type=int)
@click.option("--n-layers", default=6, type=int)
@click.option("--d-ff", default=2048, type=int)
@click.option("--max-src-len", default=256, type=int)
@click.option("--max-tgt-len", default=128, type=int)
@click.option("--warmup-steps", default=10000, type=int)
@click.option("--label-smoothing", default=0.05, type=float)
@click.option("--conditioning-dropout", default=0.2, type=float)
@click.option("--val-split", default=0.02, type=float,
              help="Smaller val split since dataset is huge")
@click.option("--patience", default=5, type=int)
@click.option("--eval-every", default=2, type=int)
@click.option("--device", default="auto")
def main(
    data, output_dir, epochs, batch_size, lr, d_model, nhead, n_layers, d_ff,
    max_src_len, max_tgt_len, warmup_steps, label_smoothing,
    conditioning_dropout, val_split, patience, eval_every, device,
):
    """Pre-train RetroTransformer v2 on USPTO-FULL."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    data_path = PROJECT_ROOT / data
    output_path = PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        logger.error("Run these steps first:")
        logger.error("  1. python -u scripts/download_data.py --datasets uspto_full")
        logger.error("  2. python -u scripts/preprocess_all.py --dataset uspto_full")
        logger.error("  3. python -u scripts/build_augmented_dataset.py --edit-data data/processed/uspto_full/edit_conditioned_train.jsonl --output data/processed/uspto_full/augmented_3x_train.jsonl --n-augments 3")
        return

    # Build tokenizer from data
    logger.info("Building regex tokenizer from USPTO-FULL data...")
    all_texts = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            all_texts.append(ex["src_text"])
            all_texts.append(ex["tgt_text"])

    from rasyn.models.retro.tokenizer_v2 import RegexSmilesTokenizer
    tokenizer = RegexSmilesTokenizer.build_from_data(all_texts)
    logger.info(f"Tokenizer: {tokenizer.vocab_size} tokens")

    # Load data — NO reaction class for USPTO-FULL
    from rasyn.models.retro.data_v2 import load_retro_data_v2, collate_fn_v2
    train_dataset, val_dataset = load_retro_data_v2(
        data_path=data_path,
        tokenizer=tokenizer,
        val_split=val_split,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        conditioning_dropout=conditioning_dropout,
        use_reaction_class=False,  # USPTO-FULL has no class labels
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Build model
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
        "num_rxn_classes": 11,  # Keep for compatibility with fine-tune
    }

    from rasyn.models.retro.model_v2 import RetroTransformerV2, save_retro_model_v2
    model = RetroTransformerV2(**model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn_v2, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_v2, num_workers=2, pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)

    log_file = output_path / "pretrain_log.jsonl"
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0

    logger.info(f"Pre-training for {epochs} epochs, {total_steps} total steps")
    logger.info(f"  LR: {lr}, warmup: {warmup_steps}, label_smoothing: {label_smoothing}")
    logger.info(f"  Batch size: {batch_size}, d_model: {d_model}, layers: {n_layers}")

    model.train()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_tokens = 0
        epoch_batches = 0

        for batch in train_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            seg_ids = batch["segment_ids"].to(device)

            tgt_in = tgt_ids[:, :-1]
            tgt_tgt = tgt_ids[:, 1:]

            log_probs, copy_lambda = model(src_ids, tgt_in, seg_ids)

            if label_smoothing > 0:
                nll_loss = loss_fn(log_probs.reshape(-1, log_probs.size(-1)), tgt_tgt.reshape(-1))
                smooth_loss = -log_probs.reshape(-1, log_probs.size(-1)).mean(dim=-1)
                mask_flat = (tgt_tgt.reshape(-1) != tokenizer.pad_token_id).float()
                smooth_loss = (smooth_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
                loss = (1 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
            else:
                loss = loss_fn(log_probs.reshape(-1, log_probs.size(-1)), tgt_tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_batches += 1

            preds = log_probs.argmax(dim=-1)
            mask = tgt_tgt != tokenizer.pad_token_id
            epoch_correct += ((preds == tgt_tgt) & mask).sum().item()
            epoch_tokens += mask.sum().item()

            if global_step % 200 == 0:
                avg_loss = epoch_loss / epoch_batches
                token_acc = epoch_correct / max(epoch_tokens, 1)
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time

                logger.info(
                    f"Step {global_step} | Epoch {epoch} | "
                    f"loss={loss.item():.4f} | avg={avg_loss:.4f} | "
                    f"tok_acc={token_acc:.4f} | lr={current_lr:.6f} | "
                    f"{elapsed/60:.1f}m elapsed"
                )

                with open(log_file, "a") as f:
                    f.write(json.dumps({
                        "step": global_step, "epoch": epoch,
                        "loss": loss.item(), "avg_loss": avg_loss,
                        "token_acc": token_acc, "lr": current_lr,
                    }) + "\n")

        # End of epoch
        epoch_avg_loss = epoch_loss / max(epoch_batches, 1)
        epoch_acc = epoch_correct / max(epoch_tokens, 1)
        logger.info(
            f"\n--- Epoch {epoch}/{epochs} --- "
            f"avg_loss={epoch_avg_loss:.4f} tok_acc={epoch_acc:.4f}"
        )

        # Validation
        if epoch % eval_every == 0:
            metrics = evaluate_pretrain(model, val_loader, tokenizer, device)
            logger.info(
                f"  VAL: loss={metrics['val_loss']:.4f} tok_acc={metrics['val_token_acc']:.4f}"
            )

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                epochs_without_improvement = 0
                save_retro_model_v2(
                    model, tokenizer, output_path / "best",
                    config=model_config,
                    extra={"epoch": epoch, "val_metrics": metrics, "pretrain": True},
                )
                logger.info(f"  New best! val_loss={best_val_loss:.4f}")
            else:
                epochs_without_improvement += eval_every
                logger.info(
                    f"  No improvement for {epochs_without_improvement} epochs "
                    f"(patience={patience})"
                )

            if patience > 0 and epochs_without_improvement >= patience:
                logger.info(f"\n*** EARLY STOPPING at epoch {epoch} ***")
                break

            with open(log_file, "a") as f:
                metrics["epoch"] = epoch
                metrics["type"] = "validation"
                f.write(json.dumps(metrics) + "\n")

        # Periodic checkpoint
        if epoch % 5 == 0:
            save_retro_model_v2(
                model, tokenizer, output_path / f"epoch_{epoch}",
                config=model_config, extra={"epoch": epoch},
            )

    # Final save
    save_retro_model_v2(
        model, tokenizer, output_path / "final",
        config=model_config,
        extra={"epoch": epoch, "best_val_loss": best_val_loss, "pretrain": True},
    )

    elapsed = time.time() - start_time
    logger.info(f"\nPre-training complete in {elapsed/3600:.1f} hours")
    logger.info(f"Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint: {output_path / 'best'}")
    logger.info(f"\nNext step: Fine-tune on USPTO-50K:")
    logger.info(f"  python -u scripts/train_retro_v2.py \\")
    logger.info(f"    --resume {output_path / 'best'} \\")
    logger.info(f"    --data data/processed/uspto50k/augmented_train.jsonl \\")
    logger.info(f"    --lr 1e-4 --epochs 80 --skip-sanity")


if __name__ == "__main__":
    main()
