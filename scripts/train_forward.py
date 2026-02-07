"""Train the forward reaction prediction model.

Usage:
    python scripts/train_forward.py --config configs/verifier.yaml
"""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", default="configs/verifier.yaml")
@click.option("--data-dir", default="data/processed/uspto50k")
@click.option("--output-dir", default="checkpoints/forward")
@click.option("--epochs", default=50, type=int)
@click.option("--batch-size", default=32, type=int)
@click.option("--lr", default=1e-4, type=float)
@click.option("--device", default="auto")
def main(config, data_dir, output_dir, epochs, batch_size, lr, device):
    """Train the forward reaction prediction model."""
    logging.basicConfig(level=logging.INFO)

    logger.info("Forward model training not yet fully implemented.")
    logger.info("The verifier ensemble falls back to Tanimoto similarity.")
    logger.info("Implement this after the core pipeline is validated.")

    # TODO: Implement forward model training
    # 1. Load preprocessed reactions (reactants -> product direction)
    # 2. Build character-level or BPE tokenizer for SMILES
    # 3. Train ForwardTransformer with teacher forcing
    # 4. Save checkpoint


if __name__ == "__main__":
    main()
