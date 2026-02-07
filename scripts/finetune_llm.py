"""Fine-tune RSGPT with edit conditioning.

Usage:
    python scripts/finetune_llm.py
    python scripts/finetune_llm.py --config configs/llm.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import yaml

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


@click.command()
@click.option("--config", type=click.Path(exists=True), default=None)
@click.option("--dataset", default="uspto50k")
@click.option("--weights-path", default=None, help="Path to RSGPT pretrained weights")
@click.option("--epochs", default=3, type=int)
@click.option("--batch-size", default=2, type=int)
@click.option("--grad-accum", default=8, type=int)
@click.option("--lr", default=2e-5, type=float)
@click.option("--lora-rank", default=16, type=int)
def main(config, dataset, weights_path, epochs, batch_size, grad_accum, lr, lora_rank):
    """Fine-tune RSGPT with edit conditioning on preprocessed data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        dataset = cfg.get("dataset", dataset)
        weights_path = cfg.get("weights_path", weights_path)
        epochs = cfg.get("epochs", epochs)
        batch_size = cfg.get("batch_size", batch_size)
        grad_accum = cfg.get("gradient_accumulation_steps", grad_accum)
        lr = cfg.get("learning_rate", lr)
        lora_rank = cfg.get("lora_rank", lora_rank)

    data_dir = PROJECT_ROOT / "data" / "processed" / dataset
    train_path = data_dir / "edit_conditioned_train.jsonl"
    output_dir = PROJECT_ROOT / "checkpoints" / "llm" / dataset

    if weights_path is None:
        weights_path = PROJECT_ROOT / "weights" / "rsgpt" / "finetune_50k.pth"

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}. Run preprocess_all.py first.")
        return

    # Load model
    from rasyn.models.llm.model import load_rsgpt_model
    model, tokenizer = load_rsgpt_model(
        weights_path=weights_path if Path(weights_path).exists() else None,
        use_lora=True,
        lora_rank=lora_rank,
    )

    # Fine-tune
    from rasyn.models.llm.finetune import finetune_rsgpt
    finetune_rsgpt(
        model=model,
        tokenizer=tokenizer,
        train_data_path=train_path,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
    )

    logger.info(f"Fine-tuning complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
