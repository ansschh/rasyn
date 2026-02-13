"""RetroTransformer v2 with value head for PPO training.

Wraps the base RetroTransformerV2 model with a value head that estimates
the expected return from the encoder state. Used by the PPO trainer to
compute advantages for policy gradient updates.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from rasyn.models.retro.model_v2 import RetroTransformerV2

logger = logging.getLogger(__name__)


class RetroTransformerV2WithValueHead(nn.Module):
    """RetroTransformerV2 wrapper with a value head for PPO.

    The value head takes the mean-pooled encoder output and predicts a
    scalar value estimate. The base model produces log-probs and copy
    gate outputs as usual.

    Args:
        base_model: Pretrained RetroTransformerV2 model.
        d_model: Hidden size (should match base model).
        freeze_base: If True, freeze the base model initially.
    """

    def __init__(
        self,
        base_model: RetroTransformerV2,
        d_model: int = 512,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.base_model = base_model

        # Value head: encoder output -> scalar value
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            logger.info("Base model frozen (only value head is trainable)")

    def _encode_for_value(
        self,
        src_ids: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source and return memory + padding mask.

        Uses the base model's _encode() method which handles embedding,
        segment embedding, positional encoding, and transformer encoding.
        """
        memory, src_padding_mask = self.base_model._encode(src_ids, segment_ids)
        return memory, src_padding_mask

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning log-probs, copy gate, and value estimates.

        Args:
            src_ids: Source token IDs [batch, src_len].
            tgt_ids: Target token IDs [batch, tgt_len] (teacher forcing).
            segment_ids: Segment IDs [batch, src_len].

        Returns:
            Tuple of:
                log_probs: [batch, tgt_len, vocab_size]
                copy_lambda: [batch, tgt_len] (copy gate values)
                values: [batch] (scalar value estimates)
        """
        # Get encoder output for value head
        memory, src_padding_mask = self._encode_for_value(src_ids, segment_ids)

        # Value estimate: mean pool encoder output (excluding padding)
        non_pad_mask = (~src_padding_mask).float().unsqueeze(-1)  # [batch, src_len, 1]
        pooled = (memory * non_pad_mask).sum(dim=1) / non_pad_mask.sum(dim=1).clamp(min=1)
        values = self.value_head(pooled).squeeze(-1)  # [batch]

        # Standard forward pass for log-probs and copy gate
        log_probs, copy_lambda = self.base_model(src_ids, tgt_ids, segment_ids)

        return log_probs, copy_lambda, values

    def generate_with_logprobs(
        self,
        src_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 128,
        temperature: float = 1.0,
        segment_ids: torch.Tensor | None = None,
    ) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
        """Sample sequences and collect per-token log-probs.

        Uses the base model's encoder and a simplified autoregressive decoder
        (generation head only, no copy mechanism) for sampling during PPO rollouts.

        Args:
            src_ids: Source token IDs [batch, src_len].
            bos_id: BOS token ID.
            eos_id: EOS token ID.
            max_len: Maximum generation length.
            temperature: Sampling temperature.
            segment_ids: Optional segment IDs.

        Returns:
            Tuple of:
                sequences: List of generated token ID lists.
                all_logprobs: Tensor of per-token log-probs [batch, max_len].
                values: Scalar value estimates [batch].
        """
        batch_size = src_ids.size(0)
        device = src_ids.device

        # Encode source using base model's _encode method
        memory, src_padding_mask = self._encode_for_value(src_ids, segment_ids)

        # Value estimate
        non_pad_mask = (~src_padding_mask).float().unsqueeze(-1)
        pooled = (memory * non_pad_mask).sum(dim=1) / non_pad_mask.sum(dim=1).clamp(min=1)
        values = self.value_head(pooled).squeeze(-1)

        # Autoregressive sampling using base model's components
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        all_logprobs = torch.zeros(batch_size, max_len, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len):
            tgt_emb = self.base_model.embedding(generated) * math.sqrt(self.base_model.d_model)
            tgt_emb = self.base_model.pos_encoding(tgt_emb)
            tgt_len = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

            # Run through decoder layers
            output = tgt_emb
            for layer in self.base_model.decoder_layers:
                output = layer(
                    output, memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask,
                )
            output = self.base_model.decoder_norm(output)

            last_hidden = output[:, -1, :]  # [batch, d_model]

            # Generation distribution (simplified — no copy mechanism for sampling)
            gen_logits = self.base_model.output_proj(last_hidden)
            gen_logits = gen_logits / max(temperature, 1e-8)
            log_probs = torch.log_softmax(gen_logits, dim=-1)

            # Sample
            probs = torch.exp(log_probs)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            # Record log-prob of chosen action
            chosen_logprob = log_probs.gather(1, next_token).squeeze(-1)
            all_logprobs[:, step] = chosen_logprob * (~finished).float()

            # Update finished mask
            finished = finished | (next_token.squeeze(-1) == eos_id)

            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        # Convert to list of lists (excluding BOS)
        sequences = []
        for i in range(batch_size):
            seq = generated[i, 1:].tolist()
            if eos_id in seq:
                seq = seq[:seq.index(eos_id)]
            sequences.append(seq)

        return sequences, all_logprobs[:, :step + 1], values
