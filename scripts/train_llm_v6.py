"""LLM Training V6: Aggressive hyperparameters for edit-conditioned fine-tuning.

Changes from v5:
  - LoRA rank 64 (from 16) -> 4x more capacity
  - LoRA alpha 128 (from 32) -> maintain scaling factor
  - Learning rate 1e-4 (from 2e-5) -> 5x faster convergence
  - 30 epochs (from 5)
  - Target modules include gate/up/down projections for more adaptability
  - Warmup steps 500 (fixed, not ratio-based)
  - Save every 2000 steps
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import torch
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class EditConditionedDataset(Dataset):
    """Dataset for edit-conditioned LLM fine-tuning."""

    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(data_path) as f:
            for line in f:
                self.examples.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(self.examples)} training examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex["prompt"]
        completion = ex["completion"]

        full_text = f"{prompt} {completion}{self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: mask prompt + padding
        prompt_enc = self.tokenizer(
            prompt + " ",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_length = prompt_enc["attention_mask"].sum().item()

        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Mask prompt
        labels[attention_mask == 0] = -100  # Mask padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@click.command()
@click.option("--data", default="data/processed/uspto50k/edit_conditioned_train.jsonl",
              help="Training data path (relative to project root)")
@click.option("--output-dir", default="checkpoints/llm/uspto50k_v6",
              help="Output directory for checkpoints")
@click.option("--epochs", default=30, type=int, help="Number of training epochs")
@click.option("--save-steps", default=2000, type=int, help="Save checkpoint every N steps")
def main(data, output_dir, epochs, save_steps):
    from transformers import (
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    from rasyn.models.llm.tokenizer import ALL_SPECIAL_TOKENS

    weights_path = PROJECT_ROOT / "weights" / "rsgpt" / "finetune_50k.pth"
    train_path = PROJECT_ROOT / data
    output_dir = PROJECT_ROOT / output_dir

    # ---------- Model ----------
    logger.info("Loading RSGPT base model...")
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=24,
        num_attention_heads=32,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        use_cache=True,
    )
    model = LlamaForCausalLM(config)

    # Load pretrained weights
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module.model."):
            cleaned[k[len("module.model."):]] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    logger.info(f"Loaded weights: {len(cleaned)} keys, {len(missing)} missing, {len(unexpected)} unexpected")

    # Load tokenizer
    tokenizer_path = weights_path.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    new_tokens = [t for t in ALL_SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        logger.info(f"Added {len(new_tokens)} edit-language tokens")

    # Resize embeddings
    logger.info(f"Resizing embeddings: {config.vocab_size} -> {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # ---------- LoRA (rank 64, full coverage) ----------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    # Gradient checkpointing
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ---------- Dataset ----------
    train_dataset = EditConditionedDataset(train_path, tokenizer, max_length=512)

    # ---------- Training ----------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=500,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        save_steps=save_steps,
        save_total_limit=5,
        report_to="none",
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    logger.info(f"Starting V6 training (rank=64, lr=1e-4, {epochs} epochs)...")
    logger.info(f"  Train examples: {len(train_dataset)}")
    logger.info(f"  Steps per epoch: {len(train_dataset) // 16}")
    logger.info(f"  Total steps: {len(train_dataset) // 16 * epochs}")

    trainer.train()

    # Save
    from rasyn.models.llm.model import save_model
    save_model(model, tokenizer, output_dir / "final")
    logger.info(f"Training complete! Model saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
