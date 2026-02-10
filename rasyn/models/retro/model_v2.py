"""RetroTransformer v2: Encoder-Decoder with Copy/Pointer Mechanism.

Key improvements over v1:
  1. Copy/pointer mechanism — decoder can copy tokens from encoder input
  2. Segment embeddings — distinguish product vs synthon tokens
  3. Reaction class embedding — prepended to encoder memory
  4. Fixed beam search — proper finished-beam handling, no duplication
  5. Split encoder/decoder calls — needed for cross-attention weight extraction

Architecture (~45-50M params):
  - 6 encoder + 6 decoder layers
  - d_model=512, nhead=8, d_ff=2048
  - Atom-level regex vocab (~150-200 tokens)
  - Sinusoidal positional encoding
  - Pointer-generator copy mechanism on decoder output
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CopyDecoderLayer(nn.TransformerDecoderLayer):
    """Decoder layer that can return cross-attention weights."""

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        # Self-attention
        tgt2, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention — extract weights
        tgt2, attn_weights = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # Store attention weights as attribute for copy mechanism
        self._last_attn_weights = attn_weights

        return tgt


class RetroTransformerV2(nn.Module):
    """Encoder-Decoder Transformer with Copy/Pointer mechanism.

    The copy mechanism allows the decoder to directly copy tokens from
    the encoder input, which is critical for retrosynthesis where most
    of the output (reactant SMILES) overlaps with the input (product SMILES).
    """

    def __init__(
        self,
        vocab_size: int = 200,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        num_segments: int = 2,
        num_rxn_classes: int = 11,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

        # Token embedding (shared encoder/decoder)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Segment embedding (product=0, synthon=1)
        self.segment_embedding = nn.Embedding(num_segments, d_model)

        # Reaction class embedding
        self.rxn_class_embedding = nn.Embedding(num_rxn_classes, d_model)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder (standard)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder (custom layers that expose cross-attention weights)
        self.decoder_layers = nn.ModuleList([
            CopyDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output projection (generation distribution)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Copy gate: predicts probability of copying vs generating
        # Input: [decoder_hidden (d_model) | context (d_model) | embedding (d_model)]
        self.copy_gate = nn.Linear(d_model * 3, 1)

        # Context projection: project memory-attended output
        self.context_proj = nn.Linear(d_model, d_model)

        # Initialize weights
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"RetroTransformerV2: {total_params:,} total params")

    def _init_weights(self):
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode(
        self,
        src_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source sequence.

        Returns:
            (memory, src_padding_mask)
        """
        src_emb = self.embedding(src_ids) * math.sqrt(self.d_model)

        # Add segment embeddings if provided
        if segment_ids is not None:
            src_emb = src_emb + self.segment_embedding(segment_ids)

        src_emb = self.pos_encoding(src_emb)
        src_padding_mask = src_ids == self.pad_token_id

        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        return memory, src_padding_mask

    def _decode_with_copy(
        self,
        tgt_emb: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        src_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run decoder and compute copy-augmented output distribution.

        Returns:
            (log_probs, copy_gate_values)
            log_probs: (batch, tgt_len, vocab_size) — combined copy + generate
            copy_gate_values: (batch, tgt_len) — lambda values for monitoring
        """
        # Run through decoder layers
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )
        output = self.decoder_norm(output)

        # Get cross-attention weights from last decoder layer
        # Shape: (batch, tgt_len, src_len)
        attn_weights = self.decoder_layers[-1]._last_attn_weights

        # Context vector from attention
        context = torch.bmm(attn_weights, memory)  # (batch, tgt_len, d_model)

        # Compute copy gate: lambda = sigmoid(W @ [hidden; context; embedding])
        gate_input = torch.cat([output, context, tgt_emb], dim=-1)
        copy_lambda = torch.sigmoid(self.copy_gate(gate_input)).squeeze(-1)
        # copy_lambda: (batch, tgt_len), values in [0, 1]
        # 1 = copy, 0 = generate

        # Generation distribution
        gen_logits = self.output_proj(output)  # (batch, tgt_len, vocab_size)
        gen_probs = F.softmax(gen_logits, dim=-1)

        # Copy distribution: scatter attention weights to vocab
        # For each source position, add its attention weight to the
        # corresponding vocab token
        batch_size, tgt_len, src_len = attn_weights.shape
        copy_probs = torch.zeros_like(gen_probs)  # (batch, tgt_len, vocab_size)

        # Expand src_ids for scatter: (batch, 1, src_len) -> (batch, tgt_len, src_len)
        src_expanded = src_ids.unsqueeze(1).expand(-1, tgt_len, -1)

        # Scatter-add attention weights
        copy_probs.scatter_add_(2, src_expanded, attn_weights)

        # Combine: P(token) = lambda * P_copy + (1-lambda) * P_gen
        copy_lambda_expanded = copy_lambda.unsqueeze(-1)  # (batch, tgt_len, 1)
        combined_probs = (
            copy_lambda_expanded * copy_probs
            + (1 - copy_lambda_expanded) * gen_probs
        )

        # Clamp for numerical stability before log
        combined_probs = combined_probs.clamp(min=1e-12)
        log_probs = torch.log(combined_probs)

        return log_probs, copy_lambda

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with teacher forcing and copy mechanism.

        Args:
            src_ids: Encoder input token IDs. (batch, src_len)
            tgt_ids: Decoder input token IDs (shifted right). (batch, tgt_len)
            segment_ids: Segment IDs for source. (batch, src_len) or None.

        Returns:
            (log_probs, copy_lambda)
            log_probs: (batch, tgt_len, vocab_size) — log probabilities
            copy_lambda: (batch, tgt_len) — copy gate values
        """
        # Encode
        memory, src_padding_mask = self._encode(src_ids, segment_ids)

        # Decoder embeddings
        tgt_emb = self.pos_encoding(
            self.embedding(tgt_ids) * math.sqrt(self.d_model)
        )

        # Causal mask
        tgt_len = tgt_ids.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt_ids.device
        )
        tgt_padding_mask = tgt_ids == self.pad_token_id

        # Decode with copy
        log_probs, copy_lambda = self._decode_with_copy(
            tgt_emb, memory, tgt_mask, src_padding_mask, tgt_padding_mask, src_ids
        )

        return log_probs, copy_lambda

    @torch.no_grad()
    def generate_greedy(
        self,
        src_ids: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_len: int = 128,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> list[list[int]]:
        """Greedy autoregressive decoding with copy mechanism."""
        self.eval()
        batch_size = src_ids.shape[0]
        device = src_ids.device

        # Encode
        memory, src_padding_mask = self._encode(src_ids, segment_ids)

        # Decode
        tgt_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_emb = self.pos_encoding(
                self.embedding(tgt_ids) * math.sqrt(self.d_model)
            )
            tgt_len = tgt_ids.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=device
            )

            log_probs, _ = self._decode_with_copy(
                tgt_emb, memory, tgt_mask, src_padding_mask,
                tgt_ids == self.pad_token_id, src_ids
            )

            # Take last position
            next_log_probs = log_probs[:, -1, :]  # (batch, vocab)
            next_tokens = next_log_probs.argmax(dim=-1)

            finished = finished | (next_tokens == eos_token_id)
            next_tokens[finished] = eos_token_id
            tgt_ids = torch.cat([tgt_ids, next_tokens.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # Convert to lists
        results = []
        for i in range(batch_size):
            ids = tgt_ids[i].tolist()
            try:
                eos_pos = ids.index(eos_token_id, 1)
                ids = ids[1:eos_pos]
            except ValueError:
                ids = ids[1:]
            results.append(ids)

        return results

    @torch.no_grad()
    def generate_beam(
        self,
        src_ids: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        beam_size: int = 10,
        max_len: int = 128,
        length_penalty: float = 0.6,
        diversity_penalty: float = 0.0,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> list[list[tuple[list[int], float]]]:
        """Beam search with proper finished-beam handling and optional diversity.

        Fixes v1 beam duplication bug: finished beams go to a separate list,
        active beams continue independently without duplication.
        """
        self.eval()
        batch_size = src_ids.shape[0]
        device = src_ids.device

        all_results = []

        for b in range(batch_size):
            single_src = src_ids[b:b+1]
            single_seg = segment_ids[b:b+1] if segment_ids is not None else None

            # Encode
            memory, src_padding_mask = self._encode(single_src, single_seg)
            single_src_for_copy = single_src  # Keep for copy mechanism

            # Expand for beam search
            memory = memory.repeat(beam_size, 1, 1)
            src_padding_mask = src_padding_mask.repeat(beam_size, 1)
            src_for_copy = single_src_for_copy.repeat(beam_size, 1)

            # Initialize beams
            beam_ids = torch.full(
                (beam_size, 1), bos_token_id, dtype=torch.long, device=device
            )
            beam_scores = torch.zeros(beam_size, device=device)
            beam_scores[1:] = -1e9  # Only first beam active initially

            finished_beams: list[tuple[list[int], float]] = []
            active_mask = torch.ones(beam_size, dtype=torch.bool, device=device)

            for step in range(max_len):
                n_active = active_mask.sum().item()
                if n_active == 0:
                    break

                # Only process active beams
                active_idx = active_mask.nonzero(as_tuple=True)[0]
                active_beam_ids = beam_ids[active_idx]
                active_memory = memory[active_idx]
                active_src_mask = src_padding_mask[active_idx]
                active_src_for_copy = src_for_copy[active_idx]
                active_scores = beam_scores[active_idx]

                tgt_emb = self.pos_encoding(
                    self.embedding(active_beam_ids) * math.sqrt(self.d_model)
                )
                tgt_len = active_beam_ids.shape[1]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt_len, device=device
                )

                log_probs, _ = self._decode_with_copy(
                    tgt_emb, active_memory, tgt_mask,
                    active_src_mask,
                    active_beam_ids == self.pad_token_id,
                    active_src_for_copy,
                )

                # Last position log probs
                last_log_probs = log_probs[:, -1, :]  # (n_active, vocab)

                # Expand scores
                next_scores = active_scores.unsqueeze(1) + last_log_probs
                next_scores = next_scores.view(-1)

                # Top-k from active beams
                k = min(beam_size, next_scores.numel())
                topk_scores, topk_flat = next_scores.topk(k, dim=-1)
                beam_indices = topk_flat // self.vocab_size
                token_indices = topk_flat % self.vocab_size

                # Map beam indices back to active beams
                new_beam_ids = torch.cat([
                    active_beam_ids[beam_indices],
                    token_indices.unsqueeze(1),
                ], dim=1)
                new_beam_scores = topk_scores

                # Check for finished beams
                new_active_mask = torch.ones(k, dtype=torch.bool, device=device)
                for i in range(k):
                    if token_indices[i].item() == eos_token_id:
                        seq_len = new_beam_ids.shape[1] - 1
                        norm_score = new_beam_scores[i].item() / max(seq_len, 1) ** length_penalty
                        ids = new_beam_ids[i].tolist()[1:-1]  # Exclude BOS and EOS
                        finished_beams.append((ids, norm_score))
                        new_active_mask[i] = False

                # Keep only active beams (up to beam_size)
                still_active = new_active_mask.nonzero(as_tuple=True)[0]

                if len(still_active) == 0 or len(finished_beams) >= beam_size:
                    break

                # Update beam state with only active beams
                n_keep = min(len(still_active), beam_size)
                keep_idx = still_active[:n_keep]

                # Resize to beam_size, padding inactive slots
                beam_ids = torch.full(
                    (beam_size, new_beam_ids.shape[1]),
                    self.pad_token_id, dtype=torch.long, device=device,
                )
                beam_scores = torch.full((beam_size,), -1e9, device=device)
                active_mask = torch.zeros(beam_size, dtype=torch.bool, device=device)

                beam_ids[:n_keep] = new_beam_ids[keep_idx]
                beam_scores[:n_keep] = new_beam_scores[keep_idx]
                active_mask[:n_keep] = True

            # If not enough finished, take best in-progress
            if len(finished_beams) < beam_size:
                for i in range(min(beam_size, beam_ids.shape[0])):
                    if active_mask[i]:
                        ids = beam_ids[i].tolist()[1:]
                        ids = [t for t in ids if t != self.pad_token_id]
                        score = beam_scores[i].item() / max(len(ids), 1) ** length_penalty
                        finished_beams.append((ids, score))

            finished_beams.sort(key=lambda x: x[1], reverse=True)
            all_results.append(finished_beams[:beam_size])

        return all_results


def save_retro_model_v2(model, tokenizer, output_dir, config=None, extra=None):
    """Save v2 model checkpoint."""
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "vocab": tokenizer.get_vocab_dict(),
        "version": "v2",
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, output_dir / "model.pt")
    logger.info(f"V2 model saved to {output_dir / 'model.pt'}")


def load_retro_model_v2(checkpoint_path, device="cpu"):
    """Load a trained RetroTransformerV2 from checkpoint."""
    from rasyn.models.retro.tokenizer_v2 import RegexSmilesTokenizer

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = RetroTransformerV2(
        vocab_size=config["vocab_size"],
        d_model=config.get("d_model", 512),
        nhead=config.get("nhead", 8),
        num_encoder_layers=config.get("num_encoder_layers", 6),
        num_decoder_layers=config.get("num_decoder_layers", 6),
        dim_feedforward=config.get("dim_feedforward", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=0.0,
        pad_token_id=config.get("pad_token_id", 0),
        num_segments=config.get("num_segments", 2),
        num_rxn_classes=config.get("num_rxn_classes", 11),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = RegexSmilesTokenizer.from_vocab_dict(checkpoint["vocab"])
    return model, tokenizer
