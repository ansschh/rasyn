"""LLM Training V7: R-SMILES aligned edit-conditioned fine-tuning.

Changes from v6:
  - Training data uses R-SMILES (root-aligned SMILES at reaction center)
  - This reduces edit distance between product and reactants by ~50%
  - Fewer epochs needed (15 vs 30) since the task is fundamentally easier
  - Same LoRA config (rank=64, alpha=128, 7 target modules)
  - Same model architecture (LLaMA2 24L, 3.2B params)

Usage:
    # Standard training with R-SMILES 5x augmented data
    python -u scripts/train_llm_v7_rsmiles.py

    # With 10x augmented data (fewer epochs)
    python -u scripts/train_llm_v7_rsmiles.py \
        --train-data data/processed/uspto50k/r_smiles_llm_10x.jsonl \
        --epochs 10

    # RunPod
    nohup python -u scripts/train_llm_v7_rsmiles.py > train_llm_v7.log 2>&1 &
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


class RSmilesLLMDataset(Dataset):
    """Dataset for R-SMILES edit-conditioned LLM fine-tuning."""

    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(data_path) as f:
            for line in f:
                self.examples.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(self.examples)} R-SMILES training examples from {data_path}")

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

        # Labels: mask prompt + padding (only train on completion)
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
@click.option("--train-data",
              default="data/processed/uspto50k/r_smiles_llm_train.jsonl",
              help="Path to R-SMILES LLM training data")
@click.option("--output-dir", default="checkpoints/llm/uspto50k_v7_rsmiles")
@click.option("--weights-path", default="weights/rsgpt/finetune_50k.pth")
@click.option("--epochs", default=15, type=int,
              help="Training epochs (fewer needed with R-SMILES)")
@click.option("--batch-size", default=2, type=int)
@click.option("--grad-accum", default=8, type=int)
@click.option("--lr", default=1e-4, type=float)
@click.option("--warmup-steps", default=500, type=int)
@click.option("--max-length", default=512, type=int)
@click.option("--lora-r", default=64, type=int)
@click.option("--lora-alpha", default=128, type=int)
@click.option("--save-steps", default=2000, type=int)
def main(
    train_data, output_dir, weights_path, epochs, batch_size, grad_accum,
    lr, warmup_steps, max_length, lora_r, lora_alpha, save_steps,
):
    """Train LLM v7 with R-SMILES aligned data."""
    from transformers import (
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from rasyn.models.llm.tokenizer import ALL_SPECIAL_TOKENS

    train_path = PROJECT_ROOT / train_data
    output_path = PROJECT_ROOT / output_dir
    weights = PROJECT_ROOT / weights_path

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
    state_dict = torch.load(str(weights), map_location="cpu", weights_only=False)
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
    tokenizer_path = weights.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    new_tokens = [t for t in ALL_SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        logger.info(f"Added {len(new_tokens)} edit-language tokens")

    # Resize embeddings
    logger.info(f"Resizing embeddings: {config.vocab_size} -> {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # ---------- LoRA ----------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
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
    train_dataset = RSmilesLLMDataset(train_path, tokenizer, max_length=max_length)

    # ---------- Training ----------
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
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

    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(train_dataset) // effective_batch

    logger.info(f"Starting V7 R-SMILES training (rank={lora_r}, lr={lr}, {epochs} epochs)...")
    logger.info(f"  Train examples: {len(train_dataset)}")
    logger.info(f"  Effective batch: {effective_batch}")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Total steps: {steps_per_epoch * epochs}")

    trainer.train()

    # Save
    from rasyn.models.llm.model import save_model
    save_model(model, tokenizer, output_path / "final")
    logger.info(f"V7 R-SMILES training complete! Model saved to {output_path / 'final'}")


if __name__ == "__main__":
    main()
