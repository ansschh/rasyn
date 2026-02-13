"""PPO trainer for LLM (RSGPT) using HuggingFace trl.

Uses trl.PPOTrainer with AutoModelForCausalLMWithValueHead for PPO
fine-tuning of the LLM with chemical rewards. This is a thin wrapper
around trl that integrates our ChemicalRewardFunction.

Usage:
    from rasyn.rl.ppo_llm_trainer import LLMPPOTrainer

    trainer = LLMPPOTrainer(
        model_name_or_path="checkpoints/llm/uspto50k_v7_rsmiles/final",
        reward_fn=reward_function,
    )
    trainer.train(dataset, epochs=3)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from rasyn.rl.rewards import ChemicalRewardFunction

logger = logging.getLogger(__name__)


class LLMPPOTrainer:
    """PPO trainer for LLM using trl library.

    Args:
        model_name_or_path: Path to SFT-trained LoRA model.
        base_weights_path: Path to RSGPT base weights.
        reward_fn: ChemicalRewardFunction instance.
        lr: Learning rate.
        kl_coeff: KL penalty coefficient.
        batch_size: Mini-batch size for PPO.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model_name_or_path: str,
        base_weights_path: str | None = None,
        reward_fn: ChemicalRewardFunction | None = None,
        lr: float = 1e-5,
        kl_coeff: float = 0.05,
        batch_size: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
    ):
        self.reward_fn = reward_fn
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.batch_size = batch_size

        self._model_path = model_name_or_path
        self._base_weights_path = base_weights_path

    def setup(self):
        """Initialize model, tokenizer, and trl PPOTrainer.

        Call this before training. Separated from __init__ to allow
        configuration before heavy model loading.
        """
        from transformers import AutoTokenizer
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
        from peft import PeftModel
        from rasyn.models.llm.model import load_rsgpt_model

        model_path = Path(self._model_path)

        # Load base model + LoRA
        if (model_path / "adapter_config.json").exists():
            base_weights = self._base_weights_path
            if base_weights is None:
                project_root = Path(__file__).parent.parent.parent
                base_weights = str(project_root / "weights" / "rsgpt" / "finetune_50k.pth")

            base_model, _ = load_rsgpt_model(
                weights_path=base_weights, use_lora=False,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            base_model.resize_token_embeddings(len(self.tokenizer))
            sft_model = PeftModel.from_pretrained(base_model, str(model_path))

            # Merge LoRA for trl compatibility
            sft_model = sft_model.merge_and_unload()
        else:
            from transformers import LlamaForCausalLM
            sft_model = LlamaForCausalLM.from_pretrained(str(model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Wrap with value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model)

        # PPO config
        ppo_config = PPOConfig(
            learning_rate=self.lr,
            batch_size=self.batch_size,
            mini_batch_size=self.batch_size,
            init_kl_coef=self.kl_coeff,
            log_with=None,
        )

        self.ppo_trainer = PPOTrainer(
            model=self.model,
            config=ppo_config,
            tokenizer=self.tokenizer,
        )

        logger.info("LLM PPO Trainer initialized")
        logger.info(f"  Model: {self._model_path}")
        logger.info(f"  LR: {self.lr}, KL coeff: {self.kl_coeff}")

    def compute_rewards(
        self,
        prompts: list[str],
        responses: list[str],
        products: list[str],
        gt_reactants: list[str | None] | None = None,
    ) -> list[torch.Tensor]:
        """Compute rewards for a batch of generated responses.

        Args:
            prompts: Input prompts.
            responses: Generated completions.
            products: Product SMILES (extracted from prompts).
            gt_reactants: Optional ground truth reactants.

        Returns:
            List of scalar reward tensors.
        """
        if self.reward_fn is None:
            return [torch.tensor(0.0) for _ in responses]

        rewards_list = self.reward_fn.compute_batch_rewards(
            products=products,
            predictions=responses,
            gt_reactants=gt_reactants,
        )

        return [torch.tensor(r["total"], dtype=torch.float32) for r in rewards_list]

    def train_step(
        self,
        prompts: list[str],
        products: list[str],
        gt_reactants: list[str | None] | None = None,
    ) -> dict:
        """Single PPO training step.

        Args:
            prompts: Edit-conditioned prompts.
            products: Product SMILES.
            gt_reactants: Optional ground truth.

        Returns:
            Dict with training stats.
        """
        # Tokenize prompts
        from rasyn.models.llm.generate import tokenize_prompt_for_inference

        query_tensors = []
        for prompt in prompts:
            inputs = tokenize_prompt_for_inference(
                prompt, self.tokenizer, max_length=512,
                device=self.model.pretrained_model.device,
            )
            query_tensors.append(inputs["input_ids"].squeeze(0))

        # Generate responses
        response_tensors = []
        for query in query_tensors:
            response = self.ppo_trainer.generate(
                query,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            response_tensors.append(response.squeeze(0)[len(query):])

        # Decode responses
        responses = []
        for resp in response_tensors:
            text = self.tokenizer.decode(resp, skip_special_tokens=True)
            # Extract after <OUT> if present
            if "<OUT>" in text:
                text = text.split("<OUT>")[-1].strip()
            responses.append(text)

        # Compute rewards
        rewards = self.compute_rewards(prompts, responses, products, gt_reactants)

        # PPO step
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)

        return {
            "mean_reward": sum(r.item() for r in rewards) / len(rewards),
            "ppo_stats": stats,
            "responses": responses,
        }

    def save(self, output_dir: str):
        """Save the PPO-trained model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        logger.info(f"PPO model saved to {output_path}")
