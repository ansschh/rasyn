"""Train the forward reaction prediction model.

Trains a small encoder-decoder Transformer to predict products from reactants.
Used by the round-trip verifier to check if predicted reactants actually
produce the target product.

Usage:
    python scripts/train_forward.py
    python scripts/train_forward.py --epochs 30 --batch-size 32
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent

# Simple character-level SMILES tokenizer
SMILES_CHARS = list(
    "CNOSFPIBrcslnop=#@+\\/-().[]0123456789%"
    "HKZeAadgimtfu"  # Less common elements/chars
)
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
VOCAB = SPECIAL + SMILES_CHARS


def build_vocab():
    """Build character-level vocabulary for SMILES."""
    char2idx = {c: i for i, c in enumerate(VOCAB)}
    idx2char = {i: c for i, c in enumerate(VOCAB)}
    return char2idx, idx2char


def tokenize_smiles(smiles: str, char2idx: dict, max_len: int = 256) -> list[int]:
    """Tokenize a SMILES string to character-level token IDs."""
    tokens = [char2idx.get("<bos>", 1)]
    for ch in smiles:
        tokens.append(char2idx.get(ch, char2idx["<unk>"]))
    tokens.append(char2idx.get("<eos>", 2))
    tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(char2idx["<pad>"])
    return tokens


def detokenize(token_ids: list[int], idx2char: dict) -> str:
    """Convert token IDs back to SMILES string."""
    chars = []
    for tid in token_ids:
        ch = idx2char.get(tid, "")
        if ch in ("<eos>", "<pad>"):
            break
        if ch not in ("<bos>", "<unk>"):
            chars.append(ch)
    return "".join(chars)


class ForwardReactionDataset(Dataset):
    """Dataset for forward reaction prediction (reactants -> product)."""

    def __init__(self, data_path: str | Path, char2idx: dict, max_len: int = 256):
        self.char2idx = char2idx
        self.max_len = max_len
        self.examples = []

        with open(data_path) as f:
            for line in f:
                record = json.loads(line.strip())
                product = record.get("product_smiles", "")
                reactants = record.get("reactants_smiles", [])
                if product and reactants:
                    reactants_str = ".".join(sorted(reactants))
                    self.examples.append((reactants_str, product))

        logger.info(f"Loaded {len(self.examples)} forward reaction examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        reactants, product = self.examples[idx]
        src = tokenize_smiles(reactants, self.char2idx, self.max_len)
        tgt = tokenize_smiles(product, self.char2idx, self.max_len)
        return {
            "src_ids": torch.tensor(src, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt, dtype=torch.long),
        }


@click.command()
@click.option("--data", default="data/processed/uspto50k/reactions.jsonl")
@click.option("--output-dir", default="checkpoints/forward/uspto50k")
@click.option("--d-model", default=512, type=int)
@click.option("--nhead", default=8, type=int)
@click.option("--num-layers", default=4, type=int, help="Encoder and decoder layers")
@click.option("--epochs", default=30, type=int)
@click.option("--batch-size", default=32, type=int)
@click.option("--lr", default=1e-4, type=float)
@click.option("--max-len", default=256, type=int)
@click.option("--val-split", default=0.05, type=float)
@click.option("--device", default="auto")
def main(data, output_dir, d_model, nhead, num_layers, epochs, batch_size, lr, max_len, val_split, device):
    """Train forward reaction prediction model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    char2idx, idx2char = build_vocab()
    vocab_size = len(VOCAB)
    logger.info(f"Vocabulary size: {vocab_size}")

    # Load data
    dataset = ForwardReactionDataset(data, char2idx, max_len)

    # Train/val split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    from rasyn.models.forward.model import ForwardTransformer

    model = ForwardTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=d_model * 4,
        max_seq_len=max_len,
        pad_token_id=char2idx["<pad>"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Forward model: {total_params:,} parameters")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx["<pad>"])

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)

            # Teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_input = tgt_ids[:, :-1]
            tgt_target = tgt_ids[:, 1:]

            logits = model(src_ids, tgt_input)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train_loss = train_loss / max(train_steps, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                src_ids = batch["src_ids"].to(device)
                tgt_ids = batch["tgt_ids"].to(device)
                tgt_input = tgt_ids[:, :-1]
                tgt_target = tgt_ids[:, 1:]

                logits = model(src_ids, tgt_input)
                loss = criterion(logits.reshape(-1, vocab_size), tgt_target.reshape(-1))
                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.0f}s"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_encoder_layers": num_layers,
                    "num_decoder_layers": num_layers,
                },
                "char2idx": char2idx,
                "idx2char": idx2char,
                "epoch": epoch,
                "val_loss": avg_val_loss,
            }, output_dir / "best_model.pt")
            logger.info(f"  -> New best model saved (val_loss={avg_val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_encoder_layers": num_layers,
                    "num_decoder_layers": num_layers,
                },
                "char2idx": char2idx,
                "idx2char": idx2char,
                "epoch": epoch,
                "val_loss": avg_val_loss,
            }, output_dir / f"checkpoint_epoch{epoch}.pt")

    total_time = time.time() - start_time
    logger.info(f"Training complete in {total_time:.0f}s. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
