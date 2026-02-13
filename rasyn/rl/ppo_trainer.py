"""PPO trainer for RetroTransformer v2.

Implements Proximal Policy Optimization (PPO) with chemical rewards
for fine-tuning the RetroTransformer v2 model. Uses a frozen reference
model for KL penalty to prevent mode collapse.

Usage:
    from rasyn.rl.ppo_trainer import RetroTransformerPPOTrainer
    from rasyn.rl.rewards import ChemicalRewardFunction

    trainer = RetroTransformerPPOTrainer(
        policy_model=model_with_value_head,
        ref_model=frozen_ref_model,
        tokenizer=tokenizer,
        reward_fn=reward_function,
    )
    trainer.train(train_loader, epochs=5)
"""

from __future__ import annotations

import copy
import logging
import time

import torch
import torch.nn as nn

from rasyn.rl.rewards import ChemicalRewardFunction

logger = logging.getLogger(__name__)


class RetroTransformerPPOTrainer:
    """PPO trainer for RetroTransformer v2.

    Args:
        policy_model: RetroTransformerV2WithValueHead (trainable).
        ref_model: Frozen reference RetroTransformerV2 for KL penalty.
        tokenizer: RegexSmilesTokenizer or CSmilesTokenizer.
        reward_fn: ChemicalRewardFunction for computing rewards.
        lr: Learning rate for policy + value head.
        kl_coeff: KL penalty coefficient.
        clip_range: PPO clipping range.
        value_coeff: Value loss coefficient.
        entropy_coeff: Entropy bonus coefficient.
        max_grad_norm: Gradient clipping norm.
        temperature: Sampling temperature for rollout generation.
        max_gen_len: Maximum generation length.
    """

    def __init__(
        self,
        policy_model,
        ref_model,
        tokenizer,
        reward_fn: ChemicalRewardFunction,
        lr: float = 1e-5,
        kl_coeff: float = 0.05,
        clip_range: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 1.0,
        temperature: float = 0.8,
        max_gen_len: int = 128,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn

        self.kl_coeff = kl_coeff
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.max_gen_len = max_gen_len

        self.optimizer = torch.optim.Adam(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=lr,
        )

        # Freeze reference model
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False

    def generate_rollouts(
        self,
        src_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        products: list[str],
        gt_reactants: list[str] | None = None,
    ) -> dict:
        """Generate rollouts: sample sequences, compute rewards.

        Args:
            src_ids: Source token IDs [batch, src_len].
            segment_ids: Segment IDs [batch, src_len].
            products: Product SMILES strings.
            gt_reactants: Optional ground truth reactants.

        Returns:
            Dict with sequences, logprobs, values, rewards, advantages.
        """
        self.policy.eval()

        with torch.no_grad():
            sequences, logprobs, values = self.policy.generate_with_logprobs(
                src_ids=src_ids,
                bos_id=self.tokenizer.bos_token_id,
                eos_id=self.tokenizer.eos_token_id,
                max_len=self.max_gen_len,
                temperature=self.temperature,
                segment_ids=segment_ids,
            )

        # Decode predictions
        predictions = []
        for seq in sequences:
            pred_str = self.tokenizer.decode(seq)
            predictions.append(pred_str)

        # Compute rewards
        rewards_list = self.reward_fn.compute_batch_rewards(
            products=products,
            predictions=predictions,
            gt_reactants=gt_reactants,
        )
        reward_values = torch.tensor(
            [r["total"] for r in rewards_list],
            dtype=torch.float32, device=src_ids.device,
        )

        # Compute advantages (simple: reward - value baseline)
        advantages = reward_values - values.detach()

        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.policy.train()

        return {
            "sequences": sequences,
            "logprobs": logprobs,
            "values": values.detach(),
            "rewards": reward_values,
            "advantages": advantages,
            "predictions": predictions,
            "reward_details": rewards_list,
        }

    def ppo_step(self, rollouts: dict, src_ids: torch.Tensor, segment_ids: torch.Tensor) -> dict:
        """Perform one PPO update step.

        Args:
            rollouts: Output from generate_rollouts().
            src_ids: Source token IDs [batch, src_len].
            segment_ids: Segment IDs [batch, src_len].

        Returns:
            Dict with loss components.
        """
        device = src_ids.device
        old_logprobs = rollouts["logprobs"]
        advantages = rollouts["advantages"]
        old_values = rollouts["values"]
        rewards = rollouts["rewards"]

        # Reconstruct target tensor from sequences
        max_seq_len = max(len(s) for s in rollouts["sequences"])
        batch_size = len(rollouts["sequences"])

        tgt_ids = torch.full(
            (batch_size, max_seq_len + 1),
            self.tokenizer.pad_token_id,
            dtype=torch.long, device=device,
        )
        tgt_ids[:, 0] = self.tokenizer.bos_token_id
        for i, seq in enumerate(rollouts["sequences"]):
            for j, tok_id in enumerate(seq):
                tgt_ids[i, j + 1] = tok_id

        # Forward pass to get new log-probs and values
        log_probs, _, new_values = self.policy(src_ids, tgt_ids[:, :-1], segment_ids)

        # Gather log-probs for chosen actions
        new_logprobs = torch.zeros_like(old_logprobs)
        for i, seq in enumerate(rollouts["sequences"]):
            for j, tok_id in enumerate(seq):
                if j < log_probs.size(1):
                    new_logprobs[i, j] = log_probs[i, j, tok_id]

        # Compute sequence-level log-probs (sum over tokens)
        seq_masks = torch.zeros_like(old_logprobs)
        for i, seq in enumerate(rollouts["sequences"]):
            seq_masks[i, :len(seq)] = 1.0

        new_seq_logprobs = (new_logprobs * seq_masks).sum(dim=1)
        old_seq_logprobs = (old_logprobs * seq_masks).sum(dim=1)

        # Policy ratio
        ratio = torch.exp(new_seq_logprobs - old_seq_logprobs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (MSE)
        value_loss = 0.5 * ((new_values - rewards) ** 2).mean()

        # Entropy bonus (approximate from log-probs)
        entropy = -(torch.exp(new_logprobs) * new_logprobs * seq_masks).sum(dim=1).mean()

        # KL penalty against reference model
        with torch.no_grad():
            ref_logprobs, _ = self.ref(src_ids, tgt_ids[:, :-1], segment_ids)
            ref_token_lp = torch.zeros_like(old_logprobs)
            for i, seq in enumerate(rollouts["sequences"]):
                for j, tok_id in enumerate(seq):
                    if j < ref_logprobs.size(1):
                        ref_token_lp[i, j] = ref_logprobs[i, j, tok_id]

        kl = ((old_logprobs - ref_token_lp) * seq_masks).sum(dim=1).mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.value_coeff * value_loss
            - self.entropy_coeff * entropy
            + self.kl_coeff * kl
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl": kl.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "mean_ratio": ratio.mean().item(),
        }

    def train(
        self,
        train_loader,
        epochs: int = 5,
        log_every: int = 10,
        save_fn=None,
    ):
        """Run PPO training loop.

        Args:
            train_loader: DataLoader yielding batches with src_ids, tgt_ids,
                segment_ids, and metadata (products, gt_reactants).
            epochs: Number of PPO epochs.
            log_every: Log frequency (in batches).
            save_fn: Optional callback to save checkpoints.
        """
        logger.info(f"Starting PPO training for {epochs} epochs")
        logger.info(f"  KL coeff: {self.kl_coeff}, clip: {self.clip_range}")
        logger.info(f"  Temperature: {self.temperature}")

        global_step = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_stats = {
                "total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                "mean_reward": 0.0, "kl": 0.0, "n_batches": 0,
            }

            for batch in train_loader:
                src_ids = batch["src_ids"].to(next(self.policy.parameters()).device)
                seg_ids = batch["segment_ids"].to(src_ids.device)

                # Need product and gt_reactants strings
                products = batch.get("products", [""] * src_ids.size(0))
                gt_reactants = batch.get("gt_reactants", None)

                # Generate rollouts
                rollouts = self.generate_rollouts(
                    src_ids, seg_ids, products, gt_reactants,
                )

                # PPO update
                stats = self.ppo_step(rollouts, src_ids, seg_ids)
                global_step += 1

                for k in epoch_stats:
                    if k != "n_batches" and k in stats:
                        epoch_stats[k] += stats[k]
                epoch_stats["n_batches"] += 1

                if global_step % log_every == 0:
                    n = epoch_stats["n_batches"]
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Step {global_step} | Epoch {epoch} | "
                        f"loss={stats['total_loss']:.4f} | "
                        f"reward={stats['mean_reward']:.4f} | "
                        f"kl={stats['kl']:.4f} | "
                        f"{elapsed/60:.1f}m"
                    )

            # End of epoch
            n = max(epoch_stats["n_batches"], 1)
            logger.info(
                f"\n--- PPO Epoch {epoch}/{epochs} --- "
                f"avg_loss={epoch_stats['total_loss']/n:.4f} "
                f"avg_reward={epoch_stats['mean_reward']/n:.4f} "
                f"avg_kl={epoch_stats['kl']/n:.4f}"
            )

            if save_fn:
                save_fn(epoch)

        logger.info(f"PPO training complete in {(time.time()-start_time)/60:.1f} minutes")
