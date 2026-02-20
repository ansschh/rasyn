"""Train RetroTransformer v2: copy mechanism + regex tokenizer + offline augmentation.

Key changes from v1:
  - RegexSmilesTokenizer (atom-level, ~2x shorter sequences)
  - Offline augmentation (canonical targets, source-only randomization)
  - RetroTransformerV2 with copy/pointer mechanism
  - Segment embeddings (product vs synthon)
  - Reaction class conditioning
  - NLL loss on combined copy+generate log-probs
  - Copy gate monitoring (fraction copied vs generated)
  - Bug checks from user's analysis integrated as sanity checks

Usage:
    # Build augmented dataset first:
    python scripts/build_augmented_dataset.py --n-augments 5

    # Run sanity checks only:
    python scripts/train_retro_v2.py --sanity-only

    # Full training:
    python scripts/train_retro_v2.py --epochs 200
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

# Import RDKit at module level so failures are loud, not silent
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)  # Suppress RDKit warnings
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


# ─────────────────────────────────────────────────────────────────────
# Sanity Checks (includes v1 checks + new bug checks)
# ─────────────────────────────────────────────────────────────────────

def sanity_check_tokenizer(tokenizer, data_path: Path) -> bool:
    """Check 1: Tokenizer round-trip on training data."""
    logger.info("=" * 60)
    logger.info("SANITY CHECK 1: Regex tokenizer round-trip")
    logger.info("=" * 60)

    failures = 0
    total = 0
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            for field in ["src_text", "tgt_text"]:
                text = ex[field]
                if not tokenizer.roundtrip_check(text):
                    failures += 1
                    if failures <= 5:
                        ids = tokenizer.encode(text)
                        decoded = tokenizer.decode(ids)
                        logger.warning(f"  FAIL: '{text[:60]}' -> '{decoded[:60]}'")
                total += 1

    rate = (total - failures) / max(total, 1) * 100
    passed = failures == 0
    logger.info(f"  Result: {total - failures}/{total} pass ({rate:.1f}%)")
    if passed:
        logger.info("  PASSED")
    else:
        logger.warning(f"  FAILED: {failures} round-trip failures")
    return passed


def sanity_check_single_memorization(model, tokenizer, device) -> bool:
    """Bug check: Train on 1 example, must achieve perfect memorization."""
    logger.info("=" * 60)
    logger.info("BUG CHECK: Single-example perfect memorization")
    logger.info("=" * 60)

    src_text = "c1ccccc1|[1*]c1ccccc1"
    tgt_text = "Clc1ccccc1"

    src_ids = torch.tensor([tokenizer.encode(src_text, max_len=128)], dtype=torch.long, device=device)
    tgt_ids = torch.tensor([tokenizer.encode(tgt_text, max_len=64)], dtype=torch.long, device=device)

    # Create a small temporary model with NO dropout
    from rasyn.models.retro.model_v2 import RetroTransformerV2
    small_model = RetroTransformerV2(
        vocab_size=tokenizer.vocab_size,
        d_model=256, nhead=4,
        num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=512, dropout=0.0,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)

    loss_fn = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

    small_model.train()
    for step in range(500):
        tgt_in = tgt_ids[:, :-1]
        tgt_tgt = tgt_ids[:, 1:]
        log_probs, _ = small_model(src_ids, tgt_in)
        loss = loss_fn(log_probs.reshape(-1, log_probs.size(-1)), tgt_tgt.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check exact match
    small_model.eval()
    pred_ids = small_model.generate_greedy(
        src_ids, tokenizer.bos_token_id, tokenizer.eos_token_id, max_len=64
    )
    pred_str = tokenizer.decode(pred_ids[0])
    exact = pred_str == tgt_text

    logger.info(f"  Pred: '{pred_str}'")
    logger.info(f"  True: '{tgt_text}'")
    logger.info(f"  Loss: {loss.item():.6f}")
    logger.info(f"  Exact match: {exact}")

    passed = exact
    if passed:
        logger.info("  PASSED: Perfect memorization achieved")
    else:
        logger.error("  FAILED: Cannot memorize 1 example — architecture bug!")
    return passed


def sanity_check_overfit(model, dataset, tokenizer, device, n=10, n_epochs=300) -> bool:
    """Check: Overfit N examples using SGD."""
    logger.info("=" * 60)
    logger.info(f"SANITY CHECK: Overfit {n} examples for {n_epochs} epochs")
    logger.info("=" * 60)

    from torch.utils.data import TensorDataset

    cached_src, cached_tgt, cached_seg = [], [], []
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        cached_src.append(item["src_ids"])
        cached_tgt.append(item["tgt_ids"])
        cached_seg.append(item["segment_ids"])

    fixed = TensorDataset(
        torch.stack(cached_src),
        torch.stack(cached_tgt),
        torch.stack(cached_seg),
    )
    loader = DataLoader(fixed, batch_size=n, shuffle=False)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)

    best_acc = 0.0
    for epoch in range(n_epochs):
        for src, tgt, seg in loader:
            src, tgt, seg = src.to(device), tgt.to(device), seg.to(device)
            tgt_in, tgt_tgt = tgt[:, :-1], tgt[:, 1:]

            log_probs, _ = model(src, tgt_in, seg)
            loss = loss_fn(log_probs.reshape(-1, log_probs.size(-1)), tgt_tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                for src, tgt, seg in loader:
                    src, tgt, seg = src.to(device), tgt.to(device), seg.to(device)
                    log_probs, _ = model(src, tgt[:, :-1], seg)
                    preds = log_probs.argmax(dim=-1)
                    mask = tgt[:, 1:] != tokenizer.pad_token_id
                    correct = ((preds == tgt[:, 1:]) & mask).sum().item()
                    total = mask.sum().item()
                    acc = correct / max(total, 1)
                    best_acc = max(best_acc, acc)
            logger.info(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.4f}")
            model.train()

    passed = best_acc >= 0.90
    if passed:
        logger.info(f"  PASSED: {best_acc:.4f} token accuracy")
    else:
        logger.warning(f"  FAILED: {best_acc:.4f} < 0.90")
    return passed


def run_all_checks(model, train_dataset, tokenizer, device, data_path) -> bool:
    """Run all sanity + bug checks."""
    results = {}
    results["tokenizer_roundtrip"] = sanity_check_tokenizer(tokenizer, data_path)
    results["single_memorization"] = sanity_check_single_memorization(model, tokenizer, device)
    results["overfit_10"] = sanity_check_overfit(model, train_dataset, tokenizer, device)

    logger.info("\n" + "=" * 60)
    logger.info("CHECK SUMMARY")
    logger.info("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_passed = False
    return all_passed


# ─────────────────────────────────────────────────────────────────────
# Evaluation Helpers
# ─────────────────────────────────────────────────────────────────────

def is_valid_smiles(smi: str) -> bool:
    if not RDKIT_AVAILABLE:
        return False
    mol = Chem.MolFromSmiles(smi.strip())
    return mol is not None


def check_validity(smiles_str: str) -> bool:
    if not RDKIT_AVAILABLE:
        return False
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    for p in parts:
        p = p.strip()
        if p and not is_valid_smiles(p):
            return False
    return len(parts) > 0


def canonicalize_and_sort(smiles_str: str) -> str:
    if not RDKIT_AVAILABLE:
        return ""
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    canon = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        mol = Chem.MolFromSmiles(p)
        if mol:
            canon.append(Chem.MolToSmiles(mol))
    return ".".join(sorted(canon)) if canon else ""


def evaluate(model, val_loader, tokenizer, device):
    """Validation: loss, token accuracy, exact match, validity, copy rate."""
    model.eval()
    loss_fn = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_exact = 0
    total_canon = 0
    total_valid = 0
    total_examples = 0
    total_copy_sum = 0.0
    total_copy_count = 0
    n_batches = 0
    validity_debug_count = 0  # Log first few validity failures

    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available — validity/canon metrics will be 0")

    with torch.no_grad():
        for batch in val_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            seg_ids = batch["segment_ids"].to(device)

            tgt_in = tgt_ids[:, :-1]
            tgt_tgt = tgt_ids[:, 1:]

            log_probs, copy_lambda = model(src_ids, tgt_in, seg_ids)
            loss = loss_fn(log_probs.reshape(-1, log_probs.size(-1)), tgt_tgt.reshape(-1))

            total_loss += loss.item()
            n_batches += 1

            # Token accuracy
            preds = log_probs.argmax(dim=-1)
            mask = tgt_tgt != tokenizer.pad_token_id
            total_correct += ((preds == tgt_tgt) & mask).sum().item()
            total_tokens += mask.sum().item()

            # Copy gate stats
            copy_mask = mask.float()
            total_copy_sum += (copy_lambda[:, :mask.shape[1]] * copy_mask).sum().item()
            total_copy_count += copy_mask.sum().item()

            # Greedy exact match + validity
            pred_ids_list = model.generate_greedy(
                src_ids, tokenizer.bos_token_id, tokenizer.eos_token_id,
                max_len=128, segment_ids=seg_ids,
            )
            for i, pred_ids in enumerate(pred_ids_list):
                pred_str = tokenizer.decode(pred_ids)
                tgt_str = tokenizer.decode(tgt_ids[i].tolist())
                if pred_str == tgt_str:
                    total_exact += 1
                valid = check_validity(pred_str)
                if valid:
                    total_valid += 1
                elif validity_debug_count < 3:
                    # Log first few validity failures for debugging
                    logger.info(f"    [validity debug] pred_str='{pred_str[:80]}' valid={valid}")
                    if RDKIT_AVAILABLE and pred_str:
                        # Test individual components
                        parts = pred_str.replace(" . ", ".").split(".")
                        for j, p in enumerate(parts[:3]):
                            mol = Chem.MolFromSmiles(p.strip())
                            logger.info(f"      component[{j}]='{p.strip()[:50]}' -> mol={mol is not None}")
                    validity_debug_count += 1
                pred_c = canonicalize_and_sort(pred_str)
                tgt_c = canonicalize_and_sort(tgt_str)
                if pred_c and tgt_c and pred_c == tgt_c:
                    total_canon += 1
                total_examples += 1

    model.train()
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_token_acc": total_correct / max(total_tokens, 1),
        "val_exact_match": total_exact / max(total_examples, 1),
        "val_canon_exact": total_canon / max(total_examples, 1),
        "val_validity": total_valid / max(total_examples, 1),
        "val_copy_rate": total_copy_sum / max(total_copy_count, 1),
        "val_examples": total_examples,
    }


# ─────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────

def train(
    model, train_loader, val_loader, tokenizer, device, output_dir: Path,
    epochs=200, lr=3e-4, warmup_steps=2000, weight_decay=0.01,
    gradient_clip=1.0, label_smoothing=0.05,
    log_every=50, eval_every=5, sample_every=10,
    train_dataset=None, model_config=None,
    patience=15, resume_from=None,
):
    """Full training loop with copy mechanism monitoring and early stopping."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # NLL loss (model outputs log-probs, not logits)
    # With label smoothing applied manually
    loss_fn = nn.NLLLoss(
        ignore_index=tokenizer.pad_token_id,
    )

    log_file = output_dir / "training_log.jsonl"
    best_val_loss = float("inf")
    best_val_exact = 0.0
    epochs_without_improvement = 0
    global_step = 0
    start_epoch = 1

    # Resume from checkpoint if provided
    if resume_from:
        resume_path = Path(resume_from)
        if (resume_path / "training_state.pt").exists():
            state = torch.load(resume_path / "training_state.pt", map_location=device, weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = state["global_step"]
            start_epoch = state["epoch"] + 1
            best_val_loss = state.get("best_val_loss", float("inf"))
            best_val_exact = state.get("best_val_exact", 0.0)
            epochs_without_improvement = state.get("epochs_without_improvement", 0)
            logger.info(f"Resumed from epoch {start_epoch - 1}, step {global_step}, best_val_loss={best_val_loss:.4f}")

    logger.info(f"Training for {epochs} epochs, {total_steps} total steps")
    logger.info(f"  LR: {lr}, warmup: {warmup_steps}, label_smoothing: {label_smoothing}")
    logger.info(f"  Early stopping patience: {patience} epochs")

    model.train()
    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_tokens = 0
        epoch_copy_sum = 0.0
        epoch_copy_count = 0
        epoch_batches = 0

        for batch in train_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            seg_ids = batch["segment_ids"].to(device)

            tgt_in = tgt_ids[:, :-1]
            tgt_tgt = tgt_ids[:, 1:]

            log_probs, copy_lambda = model(src_ids, tgt_in, seg_ids)

            # Apply label smoothing manually to log-probs
            if label_smoothing > 0:
                nll_loss = loss_fn(log_probs.reshape(-1, log_probs.size(-1)), tgt_tgt.reshape(-1))
                # Uniform distribution penalty
                smooth_loss = -log_probs.reshape(-1, log_probs.size(-1)).mean(dim=-1)
                mask_flat = (tgt_tgt.reshape(-1) != tokenizer.pad_token_id).float()
                smooth_loss = (smooth_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
                loss = (1 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
            else:
                loss = loss_fn(log_probs.reshape(-1, log_probs.size(-1)), tgt_tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_batches += 1

            # Token accuracy
            preds = log_probs.argmax(dim=-1)
            mask = tgt_tgt != tokenizer.pad_token_id
            epoch_correct += ((preds == tgt_tgt) & mask).sum().item()
            epoch_tokens += mask.sum().item()

            # Copy gate stats
            copy_mask = mask.float()
            epoch_copy_sum += (copy_lambda[:, :mask.shape[1]] * copy_mask).sum().item()
            epoch_copy_count += copy_mask.sum().item()

            if global_step % log_every == 0:
                avg_loss = epoch_loss / epoch_batches
                token_acc = epoch_correct / max(epoch_tokens, 1)
                copy_rate = epoch_copy_sum / max(epoch_copy_count, 1)
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time

                logger.info(
                    f"Step {global_step} | Epoch {epoch} | "
                    f"loss={loss.item():.4f} | avg={avg_loss:.4f} | "
                    f"tok_acc={token_acc:.4f} | copy={copy_rate:.3f} | "
                    f"lr={current_lr:.6f} | {global_step/elapsed:.1f} s/s"
                )

                with open(log_file, "a") as f:
                    f.write(json.dumps({
                        "step": global_step, "epoch": epoch,
                        "loss": loss.item(), "avg_loss": avg_loss,
                        "token_acc": token_acc, "copy_rate": copy_rate,
                        "lr": current_lr,
                    }) + "\n")

        # End of epoch
        logger.info(
            f"\n--- Epoch {epoch}/{epochs} ---"
            f" avg_loss={epoch_loss/max(epoch_batches,1):.4f}"
            f" tok_acc={epoch_correct/max(epoch_tokens,1):.4f}"
            f" copy_rate={epoch_copy_sum/max(epoch_copy_count,1):.3f}"
        )

        # Validation
        if val_loader and epoch % eval_every == 0:
            metrics = evaluate(model, val_loader, tokenizer, device)
            n_exact = int(metrics['val_exact_match'] * metrics['val_examples'])
            n_canon = int(metrics['val_canon_exact'] * metrics['val_examples'])
            logger.info(
                f"  VAL: loss={metrics['val_loss']:.4f} | "
                f"tok_acc={metrics['val_token_acc']:.4f} | "
                f"exact={metrics['val_exact_match']:.4f} ({n_exact}/{metrics['val_examples']}) | "
                f"canon={metrics['val_canon_exact']:.4f} ({n_canon}/{metrics['val_examples']}) | "
                f"validity={metrics['val_validity']:.4f} | "
                f"copy={metrics['val_copy_rate']:.3f}"
            )

            # Track improvement by val_loss OR val_exact_match
            improved = False
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                improved = True
            if metrics["val_exact_match"] > best_val_exact:
                best_val_exact = metrics["val_exact_match"]
                improved = True

            if improved:
                epochs_without_improvement = 0
                from rasyn.models.retro.model_v2 import save_retro_model_v2
                save_retro_model_v2(
                    model, tokenizer, output_dir / "best",
                    config=model_config,
                    extra={"epoch": epoch, "val_metrics": metrics},
                )
                # Save training state for resume
                torch.save({
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_exact": best_val_exact,
                    "epochs_without_improvement": 0,
                }, output_dir / "best" / "training_state.pt")
                logger.info(f"  New best! val_loss={best_val_loss:.4f} exact={best_val_exact:.4f}")
            else:
                epochs_without_improvement += eval_every
                logger.info(
                    f"  No improvement for {epochs_without_improvement} epochs "
                    f"(patience={patience}, best_loss={best_val_loss:.4f}, best_exact={best_val_exact:.4f})"
                )

            # Early stopping
            if patience > 0 and epochs_without_improvement >= patience:
                logger.info(
                    f"\n*** EARLY STOPPING at epoch {epoch} ***"
                    f"\n  No improvement for {epochs_without_improvement} epochs."
                    f"\n  Best val_loss={best_val_loss:.4f}, best_exact={best_val_exact:.4f}"
                )
                # Save final state before stopping
                from rasyn.models.retro.model_v2 import save_retro_model_v2
                save_retro_model_v2(
                    model, tokenizer, output_dir / "early_stopped",
                    config=model_config,
                    extra={"epoch": epoch, "best_val_loss": best_val_loss,
                           "best_val_exact": best_val_exact, "early_stopped": True},
                )
                return

            with open(log_file, "a") as f:
                metrics["epoch"] = epoch
                metrics["type"] = "validation"
                metrics["epochs_without_improvement"] = epochs_without_improvement
                f.write(json.dumps(metrics) + "\n")

        # Sample generations
        if train_dataset and epoch % sample_every == 0:
            model.eval()
            import random
            indices = random.sample(range(len(train_dataset)), min(5, len(train_dataset)))
            logger.info(f"\n  Samples (epoch {epoch}):")
            for idx in indices:
                item = train_dataset[idx]
                src = item["src_ids"].unsqueeze(0).to(device)
                seg = item["segment_ids"].unsqueeze(0).to(device)
                pred_ids = model.generate_greedy(
                    src, tokenizer.bos_token_id, tokenizer.eos_token_id,
                    max_len=128, segment_ids=seg,
                )
                pred = tokenizer.decode(pred_ids[0])
                true = tokenizer.decode(item["tgt_ids"].tolist())
                valid = check_validity(pred) if pred else False
                match = "EXACT" if pred == true else ("valid" if valid else "invalid")
                logger.info(f"    [{match}] P: {pred[:70]}")
                logger.info(f"           T: {true[:70]}")
            model.train()

        # Periodic checkpoint
        if epoch % 20 == 0:
            from rasyn.models.retro.model_v2 import save_retro_model_v2
            save_retro_model_v2(
                model, tokenizer, output_dir / f"epoch_{epoch}",
                config=model_config, extra={"epoch": epoch},
            )

    # Final save
    from rasyn.models.retro.model_v2 import save_retro_model_v2
    save_retro_model_v2(
        model, tokenizer, output_dir / "final",
        config=model_config, extra={"epoch": epochs, "best_val_loss": best_val_loss},
    )

    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete in {elapsed/3600:.1f} hours")
    logger.info(f"Best val_loss: {best_val_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--data", default="data/processed/uspto50k/augmented_train.jsonl")
@click.option("--output-dir", default="checkpoints/retro_v2/uspto50k")
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
@click.option("--patience", default=15, type=int, help="Early stopping patience (0=disable)")
@click.option("--resume", default=None, type=str, help="Resume from checkpoint dir")
@click.option("--sanity-only", is_flag=True)
@click.option("--skip-sanity", is_flag=True)
@click.option("--device", default="auto")
def main(
    data, output_dir, epochs, batch_size, lr, d_model, nhead, n_layers, d_ff,
    max_src_len, max_tgt_len, warmup_steps, label_smoothing,
    conditioning_dropout, use_rxn_class, val_split, patience, resume,
    sanity_only, skip_sanity, device,
):
    """Train RetroTransformer v2."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    data_path = PROJECT_ROOT / data
    output_dir = PROJECT_ROOT / output_dir

    # Build tokenizer from data
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

    # Resume from checkpoint or build fresh
    resume_dir = None
    if resume:
        resume_path = PROJECT_ROOT / resume
        if (resume_path / "model.pt").exists():
            logger.info(f"Loading model from {resume_path} for resume...")
            from rasyn.models.retro.model_v2 import load_retro_model_v2
            model, ckpt_tokenizer = load_retro_model_v2(str(resume_path / "model.pt"), device=device)
            # Use checkpoint's tokenizer to preserve token-ID alignment with
            # pretrained embeddings.  Rebuild data with this tokenizer below.
            if ckpt_tokenizer.vocab_size != tokenizer.vocab_size:
                logger.warning(
                    f"Tokenizer mismatch: checkpoint has {ckpt_tokenizer.vocab_size} tokens, "
                    f"data-built has {tokenizer.vocab_size}. Using checkpoint tokenizer."
                )
            tokenizer = ckpt_tokenizer
            # Reload datasets with the checkpoint tokenizer
            train_dataset, val_dataset = load_retro_data_v2(
                data_path=data_path,
                tokenizer=tokenizer,
                val_split=val_split,
                max_src_len=max_src_len,
                max_tgt_len=max_tgt_len,
                conditioning_dropout=conditioning_dropout,
                use_reaction_class=use_rxn_class,
            )
            logger.info(f"Reloaded data with checkpoint tokenizer: Train={len(train_dataset)}, Val={len(val_dataset)}")
            model.train()
            resume_dir = resume_path
            logger.info("Model loaded for resume")
        else:
            logger.warning(f"Resume path {resume_path} not found, starting fresh")
            model = RetroTransformerV2(**model_config).to(device)
    else:
        model = RetroTransformerV2(**model_config).to(device)

    # Sanity checks (skip if resuming)
    if not skip_sanity and not resume:
        all_passed = run_all_checks(model, train_dataset, tokenizer, device, data_path)
        if sanity_only:
            return
        if not all_passed:
            logger.warning("Some checks failed. Proceeding anyway.")
        # Re-init model after sanity (overfit modified weights)
        model = RetroTransformerV2(**model_config).to(device)
        logger.info("Model re-initialized for full training")
    elif sanity_only:
        # Still allow sanity-only even when resuming
        model_fresh = RetroTransformerV2(**model_config).to(device)
        run_all_checks(model_fresh, train_dataset, tokenizer, device, data_path)
        return

    logger.info(f"RDKit available: {RDKIT_AVAILABLE}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn_v2, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_v2, num_workers=2, pin_memory=True,
    )

    # Train
    train(
        model=model, train_loader=train_loader, val_loader=val_loader,
        tokenizer=tokenizer, device=device, output_dir=output_dir,
        epochs=epochs, lr=lr, warmup_steps=warmup_steps,
        label_smoothing=label_smoothing,
        train_dataset=train_dataset, model_config=model_config,
        patience=patience, resume_from=resume_dir,
    )


if __name__ == "__main__":
    main()
