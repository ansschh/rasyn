"""Train the Graph Edit Head model.

Usage:
    python scripts/train_graph_head.py
    python scripts/train_graph_head.py --config configs/graph_head.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import yaml

from rasyn.models.graph_head.train import train_graph_head

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


@click.command()
@click.option("--config", type=click.Path(exists=True), default=None)
@click.option("--dataset", default="uspto50k")
@click.option("--epochs", default=100, type=int)
@click.option("--batch-size", default=32, type=int)
@click.option("--lr", default=1e-3, type=float)
@click.option("--hidden-dim", default=32, type=int)
@click.option("--device", default="auto")
def main(config, dataset, epochs, batch_size, lr, hidden_dim, device):
    """Train the graph edit head on preprocessed reaction data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load config if provided
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        dataset = cfg.get("dataset", dataset)
        epochs = cfg.get("epochs", epochs)
        batch_size = cfg.get("batch_size", batch_size)
        lr = cfg.get("lr", lr)
        hidden_dim = cfg.get("hidden_dim", hidden_dim)
        device = cfg.get("device", device)

    data_dir = PROJECT_ROOT / "data" / "processed" / dataset
    vocab_dir = PROJECT_ROOT / "data" / "vocab"
    output_dir = PROJECT_ROOT / "checkpoints" / "graph_head" / dataset

    train_path = data_dir / "reactions.jsonl"
    val_path = data_dir / "reactions.jsonl"  # TODO: split train/val properly
    lg_vocab_path = vocab_dir / "lg_vocab.json"
    cog_path = vocab_dir / "lg_cog.npy"

    # Check files exist
    for p in [train_path, lg_vocab_path]:
        if not p.exists():
            logger.error(f"Required file not found: {p}. Run preprocess_all.py first.")
            return

    best_model = train_graph_head(
        train_records_path=train_path,
        val_records_path=val_path,
        lg_vocab_path=lg_vocab_path,
        output_dir=output_dir,
        cog_path=cog_path if cog_path.exists() else None,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        device=device,
    )

    logger.info(f"Best model saved to: {best_model}")


if __name__ == "__main__":
    main()
