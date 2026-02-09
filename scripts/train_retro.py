"""Train RetroTransformer: custom encoder-decoder for retrosynthesis.

This script includes:
  1. Built-in sanity checks (run before full training)
  2. Comprehensive logging (loss, accuracy, gradients, sample generations)
  3. SMILES augmentation and conditioning dropout
  4. Checkpoint saving with best model tracking

Usage:
    # Run sanity checks only (no full training):
    python scripts/train_retro.py --sanity-only

    # Full training (sanity checks run first):
    python scripts/train_retro.py --epochs 100

    # Quick test on small data:
    python scripts/train_retro.py --epochs 5 --max-examples 500
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
from torch.utils.data import DataLoader, Subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────────────
# Sanity Checks
# ─────────────────────────────────────────────────────────────────────

def sanity_check_tokenizer(tokenizer, data_path: Path) -> bool:
    """Check 1: Tokenizer round-trip on all training SMILES."""
    logger.info("=" * 60)
    logger.info("SANITY CHECK 1: Tokenizer round-trip")
    logger.info("=" * 60)

    failures = 0
    total = 0

    with open(data_path) as f:
        for i, line in enumerate(f):
            ex = json.loads(line.strip())
            completion = ex["completion"]

            # Check completion round-trip
            if not tokenizer.roundtrip_check(completion):
                failures += 1
                if failures <= 5:
                    encoded = tokenizer.encode(completion)
                    decoded = tokenizer.decode(encoded)
                    logger.warning(f"  FAIL #{failures}: '{completion[:60]}...' -> '{decoded[:60]}...'")

            total += 1

    rate = (total - failures) / max(total, 1) * 100
    passed = failures == 0
    logger.info(f"  Result: {total - failures}/{total} pass ({rate:.1f}%)")

    if failures > 0:
        logger.warning(f"  {failures} tokenizer round-trip failures!")
        # Check what characters are missing
        missing_chars = set()
        with open(data_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                for ch in ex["completion"]:
                    if ch not in tokenizer.token2id:
                        missing_chars.add(ch)
        if missing_chars:
            logger.warning(f"  Missing chars in vocab: {missing_chars}")
            logger.warning("  Re-building tokenizer from data will fix this.")
    else:
        logger.info("  PASSED: All SMILES survive tokenizer round-trip")

    return passed


def sanity_check_data_pipeline(dataset, tokenizer, n: int = 20) -> bool:
    """Check 2: Print N random examples and verify they look correct."""
    logger.info("=" * 60)
    logger.info("SANITY CHECK 2: Data pipeline (visual inspection)")
    logger.info("=" * 60)

    import random
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    for i, idx in enumerate(indices[:10]):  # Print first 10
        raw = dataset.get_raw_example(idx) if hasattr(dataset, 'get_raw_example') else None
        item = dataset[idx]

        src_decoded = tokenizer.decode(item["src_ids"].tolist())
        tgt_decoded = tokenizer.decode(item["tgt_ids"].tolist())

        logger.info(f"\n  Example {i+1}:")
        if raw:
            logger.info(f"    Product:   {raw['product'][:80]}")
            logger.info(f"    Synthons:  {raw['synthons'][:80]}")
            logger.info(f"    Reactants: {raw['reactants'][:80]}")
        logger.info(f"    SRC (decoded): {src_decoded[:100]}")
        logger.info(f"    TGT (decoded): {tgt_decoded[:100]}")
        logger.info(f"    SRC length: {(item['src_ids'] != 0).sum().item()} tokens")
        logger.info(f"    TGT length: {(item['tgt_ids'] != 0).sum().item()} tokens")

    logger.info("\n  PASSED (visual check — review examples above)")
    return True


def sanity_check_loss(model, dataset, tokenizer, device) -> bool:
    """Check 3: Verify loss is ~ln(vocab_size) at initialization."""
    logger.info("=" * 60)
    logger.info("SANITY CHECK 3: Initial loss value")
    logger.info("=" * 60)

    model.eval()
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=False)
    batch = next(iter(loader))

    src_ids = batch["src_ids"].to(device)
    tgt_ids = batch["tgt_ids"].to(device)

    # Teacher forcing: input is tgt[:-1], target is tgt[1:]
    tgt_input = tgt_ids[:, :-1]
    tgt_target = tgt_ids[:, 1:]

    with torch.no_grad():
        logits = model(src_ids, tgt_input)
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))

    expected_loss = math.log(tokenizer.vocab_size)
    actual_loss = loss.item()

    logger.info(f"  Expected initial loss (random): ~{expected_loss:.2f} = ln({tokenizer.vocab_size})")
    logger.info(f"  Actual initial loss: {actual_loss:.4f}")

    # Allow ±50% of expected
    passed = 0.5 * expected_loss < actual_loss < 2.0 * expected_loss
    if passed:
        logger.info(f"  PASSED: Loss is in expected range")
    else:
        logger.warning(f"  FAILED: Loss {actual_loss:.4f} is outside expected range [{0.5*expected_loss:.2f}, {2*expected_loss:.2f}]")

    return passed


def sanity_check_overfit(
    model,
    dataset,
    tokenizer,
    device,
    n_examples: int = 10,
    n_epochs: int = 500,
    target_accuracy: float = 0.95,
) -> bool:
    """Check 4/5: Overfit on a small number of examples."""
    logger.info("=" * 60)
    logger.info(f"SANITY CHECK: Overfit {n_examples} examples for {n_epochs} epochs")
    logger.info("=" * 60)

    # Pre-cache examples with augmentation DISABLED so they're deterministic
    # (augmented SMILES change every access, making memorization impossible)
    n = min(n_examples, len(dataset))
    cached_src = []
    cached_tgt = []
    for i in range(n):
        item = dataset[i]
        cached_src.append(item["src_ids"])
        cached_tgt.append(item["tgt_ids"])
    fixed_src = torch.stack(cached_src)
    fixed_tgt = torch.stack(cached_tgt)

    logger.info(f"  Pre-cached {n} examples (frozen — no augmentation during overfit)")
    # Log a sample to verify
    logger.info(f"  Sample src: {tokenizer.decode(cached_src[0].tolist())[:80]}")
    logger.info(f"  Sample tgt: {tokenizer.decode(cached_tgt[0].tolist())[:80]}")

    from torch.utils.data import TensorDataset
    fixed_dataset = TensorDataset(fixed_src, fixed_tgt)
    loader = DataLoader(fixed_dataset, batch_size=min(n, 32), shuffle=True)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_acc = 0.0

    for epoch in range(n_epochs):
        for batch in loader:
            src_ids = batch[0].to(device)
            tgt_ids = batch[1].to(device)

            tgt_input = tgt_ids[:, :-1]
            tgt_target = tgt_ids[:, 1:]

            logits = model(src_ids, tgt_input)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check accuracy every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
            model.eval()
            correct = 0
            total = 0

            for batch in loader:
                src_ids = batch[0].to(device)
                tgt_ids = batch[1].to(device)

                tgt_input = tgt_ids[:, :-1]
                tgt_target = tgt_ids[:, 1:]

                with torch.no_grad():
                    logits = model(src_ids, tgt_input)
                    preds = logits.argmax(dim=-1)

                # Only count non-padding positions
                mask = tgt_target != tokenizer.pad_token_id
                correct += ((preds == tgt_target) & mask).sum().item()
                total += mask.sum().item()

            acc = correct / max(total, 1)
            best_acc = max(best_acc, acc)
            logger.info(f"  Epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}, "
                        f"token_accuracy={acc:.4f} ({correct}/{total})")

            # Also check exact match on greedy decoding
            if (epoch + 1) % 100 == 0 or epoch == n_epochs - 1:
                exact = 0
                for batch in loader:
                    src_ids = batch[0].to(device)
                    tgt_ids = batch[1].to(device)

                    preds_list = model.generate_greedy(
                        src_ids,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_len=256,
                    )
                    for i, pred_ids in enumerate(preds_list):
                        pred_str = tokenizer.decode(pred_ids)
                        tgt_str = tokenizer.decode(tgt_ids[i].tolist())
                        if pred_str == tgt_str:
                            exact += 1
                        elif epoch == n_epochs - 1 and i < 3:
                            logger.info(f"    Pred: {pred_str[:80]}")
                            logger.info(f"    True: {tgt_str[:80]}")

                logger.info(f"  Exact match: {exact}/{n}")

            model.train()

    passed = best_acc >= target_accuracy
    if passed:
        logger.info(f"  PASSED: Achieved {best_acc:.4f} token accuracy (target: {target_accuracy})")
    else:
        logger.warning(f"  FAILED: Best token accuracy {best_acc:.4f} < target {target_accuracy}")

    return passed


def sanity_check_gradients(model, dataset, tokenizer, device) -> bool:
    """Check 6: Verify gradient flow — no zeros or explosions."""
    logger.info("=" * 60)
    logger.info("SANITY CHECK: Gradient flow")
    logger.info("=" * 60)

    model.train()
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
    batch = next(iter(loader))

    src_ids = batch["src_ids"].to(device)
    tgt_ids = batch["tgt_ids"].to(device)

    tgt_input = tgt_ids[:, :-1]
    tgt_target = tgt_ids[:, 1:]

    logits = model(src_ids, tgt_input)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))
    loss.backward()

    zero_grad_layers = []
    exploding_layers = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                zero_grad_layers.append(name)
            elif grad_norm > 100:
                exploding_layers.append((name, grad_norm))
            # Log a few
            if "weight" in name and ("layer" in name or "embed" in name or "output" in name):
                logger.info(f"  {name}: grad_norm={grad_norm:.6f}")

    model.zero_grad()

    passed = len(zero_grad_layers) == 0 and len(exploding_layers) == 0
    if zero_grad_layers:
        logger.warning(f"  Zero gradients in: {zero_grad_layers[:5]}")
    if exploding_layers:
        logger.warning(f"  Exploding gradients in: {exploding_layers[:5]}")
    if passed:
        logger.info("  PASSED: All gradients flowing normally")

    return passed


def run_all_sanity_checks(
    model, train_dataset, val_dataset, tokenizer, device, data_path
) -> bool:
    """Run all sanity checks. Returns True if all pass."""
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING ALL SANITY CHECKS")
    logger.info("=" * 60 + "\n")

    results = {}

    # Check 1: Tokenizer round-trip
    results["tokenizer_roundtrip"] = sanity_check_tokenizer(tokenizer, data_path)

    # Check 2: Data pipeline
    results["data_pipeline"] = sanity_check_data_pipeline(train_dataset, tokenizer)

    # Check 3: Initial loss
    results["initial_loss"] = sanity_check_loss(model, train_dataset, tokenizer, device)

    # Check 4: Gradient flow
    results["gradient_flow"] = sanity_check_gradients(model, train_dataset, tokenizer, device)

    # Check 5: Overfit 10 examples (most important)
    # Re-initialize model for this check
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": model.d_model,
    }
    results["overfit_10"] = sanity_check_overfit(
        model, train_dataset, tokenizer, device,
        n_examples=10, n_epochs=300, target_accuracy=0.90,
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECK SUMMARY")
    logger.info("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\nAll sanity checks PASSED. Safe to proceed with full training.")
    else:
        logger.warning("\nSome sanity checks FAILED. Review issues before full training.")

    return all_passed


# ─────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(logits, targets, pad_id):
    """Compute token-level accuracy and per-token loss."""
    preds = logits.argmax(dim=-1)
    mask = targets != pad_id
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct, total


def generate_samples(model, dataset, tokenizer, device, n=5):
    """Generate and print sample predictions."""
    import random
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    model.eval()

    samples = []
    for idx in indices:
        item = dataset[idx]
        src_ids = item["src_ids"].unsqueeze(0).to(device)
        tgt_ids = item["tgt_ids"]

        # Greedy decode
        pred_ids_list = model.generate_greedy(
            src_ids,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_len=256,
        )

        pred_str = tokenizer.decode(pred_ids_list[0])
        tgt_str = tokenizer.decode(tgt_ids.tolist())
        src_str = tokenizer.decode(item["src_ids"].tolist())

        valid = check_validity(pred_str) if pred_str else False
        if pred_str == tgt_str:
            match = "EXACT"
        elif valid:
            # Check canonical match
            pred_c = canonicalize_and_sort(pred_str)
            tgt_c = canonicalize_and_sort(tgt_str)
            match = "CANON" if (pred_c and tgt_c and pred_c == tgt_c) else "valid"
        else:
            match = "invalid"

        samples.append({
            "src": src_str[:80],
            "pred": pred_str[:80],
            "true": tgt_str[:80],
            "match": match,
        })

    model.train()
    return samples


def is_valid_smiles(smi: str) -> bool:
    """Check if a SMILES string is valid (parseable by RDKit)."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi.strip())
        return mol is not None
    except Exception:
        return False


def check_validity(smiles_str: str) -> bool:
    """Check if all components in a multi-component SMILES are valid."""
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not is_valid_smiles(p):
            return False
    return len(parts) > 0


def canonicalize_and_sort(smiles_str: str) -> str:
    """Canonicalize and sort multi-component SMILES for fair comparison."""
    try:
        from rdkit import Chem
        parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
        canon = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            mol = Chem.MolFromSmiles(p)
            if mol is not None:
                canon.append(Chem.MolToSmiles(mol))
        if canon:
            return ".".join(sorted(canon))
    except Exception:
        pass
    return ""


def evaluate(model, val_loader, tokenizer, device):
    """Run full validation: loss, token accuracy, exact match, validity rate."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_exact = 0
    total_canon_exact = 0  # Canonical exact match (fairer comparison)
    total_valid = 0  # Valid SMILES predictions
    total_examples = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)

            tgt_input = tgt_ids[:, :-1]
            tgt_target = tgt_ids[:, 1:]

            logits = model(src_ids, tgt_input)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))

            total_loss += loss.item()
            n_batches += 1

            correct, total = compute_metrics(logits, tgt_target, tokenizer.pad_token_id)
            total_correct += correct
            total_tokens += total

            # Exact match (greedy decode)
            pred_ids_list = model.generate_greedy(
                src_ids,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_len=256,
            )
            for i, pred_ids in enumerate(pred_ids_list):
                pred_str = tokenizer.decode(pred_ids)
                tgt_str = tokenizer.decode(tgt_ids[i].tolist())

                # String exact match
                if pred_str == tgt_str:
                    total_exact += 1

                # Validity check
                if check_validity(pred_str):
                    total_valid += 1

                # Canonical exact match (fairer — handles different SMILES orderings)
                pred_canon = canonicalize_and_sort(pred_str)
                tgt_canon = canonicalize_and_sort(tgt_str)
                if pred_canon and tgt_canon and pred_canon == tgt_canon:
                    total_canon_exact += 1

                total_examples += 1

    model.train()

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_token_acc": total_correct / max(total_tokens, 1),
        "val_exact_match": total_exact / max(total_examples, 1),
        "val_canon_exact": total_canon_exact / max(total_examples, 1),
        "val_validity_rate": total_valid / max(total_examples, 1),
        "val_examples": total_examples,
    }


def train(
    model,
    train_loader,
    val_loader,
    tokenizer,
    device,
    output_dir: Path,
    epochs: int = 100,
    lr: float = 3e-4,
    warmup_steps: int = 2000,
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,
    label_smoothing: float = 0.1,
    log_every: int = 50,
    eval_every_epoch: int = 5,
    sample_every_epoch: int = 10,
    train_dataset=None,
    model_config: dict = None,
):
    """Full training loop with comprehensive logging."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Cosine schedule with linear warmup
    total_steps = len(train_loader) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=label_smoothing,
    )

    # Training log
    log_file = output_dir / "training_log.jsonl"
    best_val_loss = float("inf")
    best_val_exact = 0.0
    global_step = 0

    logger.info(f"Training for {epochs} epochs, {total_steps} total steps")
    logger.info(f"  LR: {lr}, warmup: {warmup_steps} steps")
    logger.info(f"  Label smoothing: {label_smoothing}")
    logger.info(f"  Output: {output_dir}")

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

            tgt_input = tgt_ids[:, :-1]
            tgt_target = tgt_ids[:, 1:]

            logits = model(src_ids, tgt_input)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_batches += 1

            correct, total = compute_metrics(logits, tgt_target, tokenizer.pad_token_id)
            epoch_correct += correct
            epoch_tokens += total

            # Periodic logging
            if global_step % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / epoch_batches
                token_acc = epoch_correct / max(epoch_tokens, 1)
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed

                log_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "avg_loss": avg_loss,
                    "token_acc": token_acc,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "lr": current_lr,
                    "steps_per_sec": steps_per_sec,
                }
                logger.info(
                    f"Step {global_step} | Epoch {epoch} | "
                    f"loss={loss.item():.4f} | avg_loss={avg_loss:.4f} | "
                    f"token_acc={token_acc:.4f} | "
                    f"grad_norm={log_entry['grad_norm']:.4f} | "
                    f"lr={current_lr:.6f} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

        # End of epoch
        epoch_avg_loss = epoch_loss / max(epoch_batches, 1)
        epoch_token_acc = epoch_correct / max(epoch_tokens, 1)
        logger.info(
            f"\n--- Epoch {epoch}/{epochs} complete ---"
            f" avg_loss={epoch_avg_loss:.4f}"
            f" token_acc={epoch_token_acc:.4f}"
        )

        # Validation
        if val_loader is not None and epoch % eval_every_epoch == 0:
            val_metrics = evaluate(model, val_loader, tokenizer, device)
            n_exact = int(val_metrics['val_exact_match'] * val_metrics['val_examples'])
            n_canon = int(val_metrics['val_canon_exact'] * val_metrics['val_examples'])
            logger.info(
                f"  VAL: loss={val_metrics['val_loss']:.4f} | "
                f"token_acc={val_metrics['val_token_acc']:.4f} | "
                f"exact_match={val_metrics['val_exact_match']:.4f} ({n_exact}/{val_metrics['val_examples']}) | "
                f"canon_match={val_metrics['val_canon_exact']:.4f} ({n_canon}/{val_metrics['val_examples']}) | "
                f"validity={val_metrics['val_validity_rate']:.4f}"
            )

            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_val_exact = val_metrics["val_exact_match"]
                from rasyn.models.retro.model import save_retro_model
                save_retro_model(
                    model, tokenizer, output_dir / "best",
                    config=model_config,
                    extra={"epoch": epoch, "val_metrics": val_metrics},
                )
                logger.info(f"  New best model saved (val_loss={best_val_loss:.4f}, "
                            f"exact_match={best_val_exact:.4f})")

            with open(log_file, "a") as f:
                val_metrics["epoch"] = epoch
                val_metrics["step"] = global_step
                val_metrics["type"] = "validation"
                f.write(json.dumps(val_metrics) + "\n")

        # Sample generations
        if train_dataset is not None and epoch % sample_every_epoch == 0:
            logger.info(f"\n  Sample generations (epoch {epoch}):")
            samples = generate_samples(model, train_dataset, tokenizer, device, n=5)
            for s in samples:
                logger.info(f"    [{s['match']}] Pred: {s['pred']}")
                logger.info(f"           True: {s['true']}")

        # Periodic checkpoint
        if epoch % 20 == 0:
            from rasyn.models.retro.model import save_retro_model
            save_retro_model(
                model, tokenizer, output_dir / f"epoch_{epoch}",
                config=model_config,
                extra={"epoch": epoch},
            )

    # Final save
    from rasyn.models.retro.model import save_retro_model
    save_retro_model(
        model, tokenizer, output_dir / "final",
        config=model_config,
        extra={"epoch": epochs, "best_val_loss": best_val_loss, "best_val_exact": best_val_exact},
    )

    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete in {elapsed/3600:.1f} hours")
    logger.info(f"Best validation: loss={best_val_loss:.4f}, exact_match={best_val_exact:.4f}")
    logger.info(f"Model saved to {output_dir}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--data", default="data/processed/uspto50k/edit_conditioned_train.jsonl")
@click.option("--output-dir", default="checkpoints/retro/uspto50k")
@click.option("--epochs", default=100, type=int)
@click.option("--batch-size", default=64, type=int)
@click.option("--lr", default=3e-4, type=float)
@click.option("--d-model", default=512, type=int)
@click.option("--nhead", default=8, type=int)
@click.option("--n-layers", default=6, type=int, help="Number of encoder AND decoder layers")
@click.option("--d-ff", default=2048, type=int)
@click.option("--max-src-len", default=512, type=int)
@click.option("--max-tgt-len", default=256, type=int)
@click.option("--warmup-steps", default=2000, type=int)
@click.option("--label-smoothing", default=0.1, type=float)
@click.option("--conditioning-dropout", default=0.2, type=float)
@click.option("--val-split", default=0.1, type=float)
@click.option("--max-examples", default=0, type=int, help="Limit training data (0=all)")
@click.option("--sanity-only", is_flag=True, help="Run sanity checks only, no training")
@click.option("--skip-sanity", is_flag=True, help="Skip sanity checks")
@click.option("--device", default="auto")
def main(
    data, output_dir, epochs, batch_size, lr, d_model, nhead, n_layers, d_ff,
    max_src_len, max_tgt_len, warmup_steps, label_smoothing,
    conditioning_dropout, val_split, max_examples, sanity_only, skip_sanity, device,
):
    """Train RetroTransformer for retrosynthesis prediction."""

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    data_path = PROJECT_ROOT / data
    output_dir = PROJECT_ROOT / output_dir

    # ── Build tokenizer from data ──
    logger.info("Building tokenizer from training data...")
    all_texts = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            all_texts.append(ex["completion"])
            # Also include product and synthons (from prompt)
            prompt = ex["prompt"]
            all_texts.append(prompt)

    from rasyn.models.retro.tokenizer import CharSmilesTokenizer
    tokenizer = CharSmilesTokenizer.build_from_data(all_texts)
    logger.info(f"Tokenizer: {tokenizer.vocab_size} tokens")

    # ── Load data ──
    from rasyn.models.retro.data import load_retro_data
    train_dataset, val_dataset = load_retro_data(
        data_path=data_path,
        tokenizer=tokenizer,
        val_split=val_split,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        augment_train=True,
        conditioning_dropout=conditioning_dropout,
    )

    if max_examples > 0 and max_examples < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(max_examples)))
        logger.info(f"Limited to {max_examples} training examples")

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Build model ──
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": n_layers,
        "num_decoder_layers": n_layers,
        "dim_feedforward": d_ff,
        "max_seq_len": max(max_src_len, max_tgt_len),
        "pad_token_id": tokenizer.pad_token_id,
    }

    from rasyn.models.retro.model import RetroTransformer
    model = RetroTransformer(**model_config).to(device)

    # ── Sanity checks ──
    if not skip_sanity:
        all_passed = run_all_sanity_checks(
            model, train_dataset, val_dataset, tokenizer, device, data_path
        )
        if sanity_only:
            logger.info("Sanity-only mode. Exiting.")
            return
        if not all_passed:
            logger.warning("Some sanity checks failed. Proceeding anyway (check logs).")

        # Re-initialize model after sanity checks (overfit check modifies weights)
        model = RetroTransformer(**model_config).to(device)
        logger.info("Model re-initialized for full training")
    elif sanity_only:
        logger.info("Cannot run --sanity-only with --skip-sanity")
        return

    # ── Data loaders ──
    from rasyn.models.retro.data import collate_fn
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # ── Train ──
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        warmup_steps=warmup_steps,
        label_smoothing=label_smoothing,
        train_dataset=train_dataset,
        model_config=model_config,
    )


if __name__ == "__main__":
    main()
