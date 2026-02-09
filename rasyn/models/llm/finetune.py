"""Edit-conditioned fine-tuning for RSGPT.

Fine-tunes the RSGPT model with LoRA to follow edit conditioning:
  - Input: edit-conditioned prompt (product + edit + synthons + LG hints)
  - Output: reactant SMILES

Uses HuggingFace Trainer with gradient checkpointing for memory efficiency
on a single A100 80GB.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EditConditionedDataset(Dataset):
    """Dataset for edit-conditioned LLM fine-tuning."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path) as f:
            for line in f:
                example = json.loads(line.strip())
                self.examples.append(example)

        logger.info(f"Loaded {len(self.examples)} training examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        prompt = example["prompt"]
        completion = example["completion"]

        # Build full sequence: prompt + completion + EOS
        full_text = f"{prompt} {completion}{self.tokenizer.eos_token}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: mask the prompt tokens (only compute loss on completion)
        prompt_encoding = self.tokenizer(
            prompt + " ",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_length = prompt_encoding["attention_mask"].sum().item()

        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Ignore prompt in loss computation
        labels[attention_mask == 0] = -100  # Ignore padding in loss computation

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def finetune_rsgpt(
    model,
    tokenizer,
    train_data_path: str | Path,
    val_data_path: str | Path | None = None,
    output_dir: str | Path = "checkpoints/llm",
    # Training hyperparameters
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_length: int = 512,
    fp16: bool = False,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    save_steps: int = 500,
    logging_steps: int = 50,
) -> Path:
    """Fine-tune RSGPT with edit conditioning using HuggingFace Trainer.

    Args:
        model: RSGPT model (with or without LoRA).
        tokenizer: Tokenizer with edit-language special tokens.
        train_data_path: Path to edit-conditioned training JSONL.
        val_data_path: Optional validation data path.
        output_dir: Directory for checkpoints.

    Returns:
        Path to the output directory.
    """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        # Disable cache when using gradient checkpointing
        if hasattr(model, "config"):
            model.config.use_cache = False

    # Datasets
    train_dataset = EditConditionedDataset(train_data_path, tokenizer, max_length)

    val_dataset = None
    if val_data_path and Path(val_data_path).exists():
        val_dataset = EditConditionedDataset(val_data_path, tokenizer, max_length)

    # Training arguments (optimized for 1x A100 80GB)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=save_steps if val_dataset else None,
        load_best_model_at_end=val_dataset is not None,
        report_to="none",
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("Starting fine-tuning...")
    logger.info(f"  Train examples: {len(train_dataset)}")
    logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Learning rate: {learning_rate}")

    trainer.train()

    # Save final model
    from rasyn.models.llm.model import save_model
    save_model(model, tokenizer, output_dir / "final")

    logger.info(f"Fine-tuning complete. Model saved to {output_dir / 'final'}")
    return output_dir
