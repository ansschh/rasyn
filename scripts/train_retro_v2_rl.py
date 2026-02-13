"""RL fine-tuning of RetroTransformer v2 with PPO and chemical rewards.

Loads a pretrained SFT checkpoint, wraps it with a value head, and runs
PPO training with chemical rewards (validity, round-trip, Tanimoto, exact match).

Usage:
    # Standard RL fine-tuning
    python -u scripts/train_retro_v2_rl.py \
        --sft-checkpoint checkpoints/retro_v2/uspto50k/best/model.pt

    # With forward model for round-trip rewards
    python -u scripts/train_retro_v2_rl.py \
        --sft-checkpoint checkpoints/retro_v2/r_smiles/best/model.pt \
        --forward-checkpoint checkpoints/forward/uspto50k/best_model.pt

    # RunPod
    nohup python -u scripts/train_retro_v2_rl.py \
        --sft-checkpoint checkpoints/retro_v2/r_smiles/best/model.pt \
        --forward-checkpoint checkpoints/forward/uspto50k/best_model.pt \
        > train_rl.log 2>&1 &
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


@click.command()
@click.option("--sft-checkpoint", required=True,
              help="Path to SFT-trained RetroTx v2 checkpoint (model.pt)")
@click.option("--forward-checkpoint", default=None,
              help="Optional forward model for round-trip rewards")
@click.option("--data", default="data/processed/uspto50k/augmented_train.jsonl")
@click.option("--output-dir", default="checkpoints/retro_v2/rl")
@click.option("--epochs", default=5, type=int)
@click.option("--batch-size", default=32, type=int)
@click.option("--lr", default=1e-5, type=float)
@click.option("--kl-coeff", default=0.05, type=float)
@click.option("--clip-range", default=0.2, type=float)
@click.option("--temperature", default=0.8, type=float)
@click.option("--max-gen-len", default=128, type=int)
@click.option("--val-split", default=0.1, type=float)
@click.option("--device", default="auto")
def main(
    sft_checkpoint, forward_checkpoint, data, output_dir, epochs,
    batch_size, lr, kl_coeff, clip_range, temperature, max_gen_len,
    val_split, device,
):
    """RL fine-tune RetroTransformer v2 with PPO."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = PROJECT_ROOT / data
    output_path = PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Load SFT model
    logger.info(f"Loading SFT checkpoint from {sft_checkpoint}...")
    from rasyn.models.retro.model_v2 import load_retro_model_v2, save_retro_model_v2
    sft_model, tokenizer = load_retro_model_v2(sft_checkpoint, device=device)

    # Create policy model with value head
    from rasyn.models.retro.model_v2_rl import RetroTransformerV2WithValueHead
    policy_model = RetroTransformerV2WithValueHead(
        base_model=sft_model,
        d_model=sft_model.src_embedding.embedding_dim,
    ).to(device)

    # Create frozen reference model (deep copy of SFT model)
    logger.info("Creating frozen reference model...")
    ref_model = copy.deepcopy(sft_model).to(device)
    ref_model.eval()

    total_params = sum(p.numel() for p in policy_model.parameters())
    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    logger.info(f"Policy model: {total_params:,} total, {trainable:,} trainable")

    # Load forward model for round-trip rewards (optional)
    fwd_model = fwd_tokenizer = None
    if forward_checkpoint:
        logger.info(f"Loading forward model from {forward_checkpoint}...")
        from rasyn.models.forward.model import load_forward_model
        fwd_model, fwd_tokenizer = load_forward_model(forward_checkpoint, device=device)
        logger.info("Forward model loaded for round-trip rewards")

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

    # Load data
    logger.info("Loading training data...")
    from rasyn.models.retro.data_v2 import load_retro_data_v2, collate_fn_v2
    train_dataset, val_dataset = load_retro_data_v2(
        data_path=data_path,
        tokenizer=tokenizer,
        val_split=val_split,
        max_src_len=256,
        max_tgt_len=128,
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # We need to augment the collate function to include product/reactant strings
    def rl_collate_fn(batch):
        base_batch = collate_fn_v2(batch)
        # Decode src_text to get product strings
        products = []
        gt_reactants = []
        for item in batch:
            src_str = tokenizer.decode(item["src_ids"].tolist())
            tgt_str = tokenizer.decode(item["tgt_ids"].tolist())
            # Extract product (before |)
            product = src_str.split("|")[0].strip()
            products.append(product)
            gt_reactants.append(tgt_str)
        base_batch["products"] = products
        base_batch["gt_reactants"] = gt_reactants
        return base_batch

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=rl_collate_fn, num_workers=2, pin_memory=True,
    )

    # Build PPO trainer
    from rasyn.rl.ppo_trainer import RetroTransformerPPOTrainer
    ppo_trainer = RetroTransformerPPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        lr=lr,
        kl_coeff=kl_coeff,
        clip_range=clip_range,
        temperature=temperature,
        max_gen_len=max_gen_len,
    )

    # Save callback
    def save_checkpoint(epoch):
        ckpt_dir = output_path / f"epoch_{epoch}"
        save_retro_model_v2(
            policy_model.base_model, tokenizer, ckpt_dir,
            extra={"epoch": epoch, "rl": True},
        )
        # Also save value head
        torch.save(
            policy_model.value_head.state_dict(),
            ckpt_dir / "value_head.pt",
        )
        logger.info(f"RL checkpoint saved to {ckpt_dir}")

    # Train
    logger.info(f"\nStarting PPO training:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  LR: {lr}")
    logger.info(f"  KL coeff: {kl_coeff}")
    logger.info(f"  Clip range: {clip_range}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Forward model: {'yes' if fwd_model else 'no'}")

    ppo_trainer.train(
        train_loader=train_loader,
        epochs=epochs,
        log_every=10,
        save_fn=save_checkpoint,
    )

    # Save final
    save_retro_model_v2(
        policy_model.base_model, tokenizer, output_path / "final",
        extra={"epochs": epochs, "rl": True, "kl_coeff": kl_coeff},
    )
    logger.info(f"\nRL fine-tuning complete! Model saved to {output_path / 'final'}")


if __name__ == "__main__":
    main()
