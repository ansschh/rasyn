"""RL fine-tuning of LLM (RSGPT) with PPO and chemical rewards via trl.

Loads a SFT-trained LLM checkpoint (v6 or v7), adds a value head, and
runs PPO training with chemical rewards.

Usage:
    # Standard RL from v7 (R-SMILES)
    python -u scripts/train_llm_rl.py \
        --sft-checkpoint checkpoints/llm/uspto50k_v7_rsmiles/final

    # With forward model
    python -u scripts/train_llm_rl.py \
        --sft-checkpoint checkpoints/llm/uspto50k_v7_rsmiles/final \
        --forward-checkpoint checkpoints/forward/uspto50k/best_model.pt

    # RunPod
    nohup python -u scripts/train_llm_rl.py \
        --sft-checkpoint checkpoints/llm/uspto50k_v7_rsmiles/final \
        > train_llm_rl.log 2>&1 &
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import click
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def load_prompts(data_path: Path, max_samples: int = 0) -> list[dict]:
    """Load training prompts from edit-conditioned dataset."""
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            prompt = ex["prompt"]
            completion = ex["completion"]

            # Extract product from prompt
            prod_match = re.search(r"<PROD>\s+(.+?)\s+<EDIT>", prompt)
            product = prod_match.group(1).strip() if prod_match else ""

            if product and completion:
                examples.append({
                    "prompt": prompt,
                    "product": product,
                    "gt_reactants": completion,
                })

            if max_samples > 0 and len(examples) >= max_samples:
                break

    return examples


@click.command()
@click.option("--sft-checkpoint", required=True,
              help="Path to SFT-trained LLM checkpoint directory")
@click.option("--forward-checkpoint", default=None,
              help="Optional forward model for round-trip rewards")
@click.option("--train-data", default="data/processed/uspto50k/r_smiles_llm_train.jsonl",
              help="Training data (use augment_idx=0 only for RL)")
@click.option("--output-dir", default="checkpoints/llm/rl")
@click.option("--epochs", default=3, type=int)
@click.option("--batch-size", default=4, type=int)
@click.option("--lr", default=1e-5, type=float)
@click.option("--kl-coeff", default=0.05, type=float)
@click.option("--temperature", default=0.8, type=float)
@click.option("--max-new-tokens", default=256, type=int)
@click.option("--max-samples", default=0, type=int, help="0=all")
@click.option("--device", default="auto")
def main(
    sft_checkpoint, forward_checkpoint, train_data, output_dir, epochs,
    batch_size, lr, kl_coeff, temperature, max_new_tokens, max_samples, device,
):
    """RL fine-tune LLM with PPO and chemical rewards."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = PROJECT_ROOT / train_data

    # Load forward model for rewards (optional)
    fwd_model = fwd_tokenizer = None
    if forward_checkpoint:
        logger.info(f"Loading forward model from {forward_checkpoint}...")
        from rasyn.models.forward.model import load_forward_model
        fwd_model, fwd_tokenizer = load_forward_model(forward_checkpoint, device=device)
        logger.info("Forward model loaded")

    # Build reward function
    from rasyn.rl.rewards import ChemicalRewardFunction
    reward_fn = ChemicalRewardFunction(
        forward_model=fwd_model,
        forward_tokenizer=fwd_tokenizer,
        device=device,
        w_validity=0.3,
        w_roundtrip=0.3 if fwd_model else 0.0,
        w_tanimoto=0.2 if fwd_model else 0.5,
        w_exact=0.2,
    )

    # Setup PPO trainer
    from rasyn.rl.ppo_llm_trainer import LLMPPOTrainer
    ppo_trainer = LLMPPOTrainer(
        model_name_or_path=sft_checkpoint,
        reward_fn=reward_fn,
        lr=lr,
        kl_coeff=kl_coeff,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    logger.info("Setting up PPO trainer (loading model)...")
    ppo_trainer.setup()

    # Load training data (only canonical examples, not augmented)
    logger.info(f"Loading training prompts from {data_path}...")
    all_examples = load_prompts(data_path, max_samples)

    # Filter to canonical only (augment_idx=0) to avoid redundancy
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            if ex.get("metadata", {}).get("augment_idx", 0) == 0:
                prompt = ex["prompt"]
                completion = ex["completion"]
                prod_match = re.search(r"<PROD>\s+(.+?)\s+<EDIT>", prompt)
                product = prod_match.group(1).strip() if prod_match else ""
                if product and completion:
                    examples.append({
                        "prompt": prompt,
                        "product": product,
                        "gt_reactants": completion,
                    })
            if max_samples > 0 and len(examples) >= max_samples:
                break

    if not examples:
        examples = all_examples
    logger.info(f"Loaded {len(examples)} training prompts (canonical only)")

    # PPO training loop
    logger.info(f"\nStarting LLM PPO training:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  LR: {lr}, KL coeff: {kl_coeff}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Forward model: {'yes' if fwd_model else 'no'}")

    start_time = time.time()
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_rewards = []

        # Shuffle examples each epoch
        import random
        random.shuffle(examples)

        for i in tqdm(range(0, len(examples), batch_size), desc=f"PPO Epoch {epoch}"):
            batch = examples[i:i + batch_size]
            prompts = [ex["prompt"] for ex in batch]
            products = [ex["product"] for ex in batch]
            gt_reactants = [ex["gt_reactants"] for ex in batch]

            stats = ppo_trainer.train_step(prompts, products, gt_reactants)
            epoch_rewards.append(stats["mean_reward"])
            global_step += 1

            if global_step % 50 == 0:
                elapsed = time.time() - start_time
                avg_reward = sum(epoch_rewards[-50:]) / len(epoch_rewards[-50:])
                logger.info(
                    f"Step {global_step} | Epoch {epoch} | "
                    f"avg_reward={avg_reward:.4f} | "
                    f"{elapsed/60:.1f}m elapsed"
                )

        # End of epoch
        avg_epoch_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        logger.info(
            f"\n--- PPO Epoch {epoch}/{epochs} --- "
            f"avg_reward={avg_epoch_reward:.4f} "
            f"({len(epoch_rewards)} batches)"
        )

        # Save checkpoint
        ppo_trainer.save(str(output_path / f"epoch_{epoch}"))

    # Save final
    ppo_trainer.save(str(output_path / "final"))
    elapsed = time.time() - start_time
    logger.info(f"\nLLM RL fine-tuning complete in {elapsed/60:.1f} minutes")
    logger.info(f"Model saved to {output_path / 'final'}")


if __name__ == "__main__":
    main()
