"""RetroTransformer: Encoder-Decoder Transformer for retrosynthesis.

Architecture mirrors our proven ForwardTransformer but optimized for the
harder retrosynthesis direction. Character-level SMILES tokenization avoids
all BPE artifacts that plagued the RSGPT fine-tuning approach.

Input:  product SMILES [| synthon1 . synthon2] (conditioning optional)
Output: reactant1 . reactant2

Default config (~45M params):
  - 6 encoder + 6 decoder layers
  - d_model=512, nhead=8, d_ff=2048
  - Character-level vocab (~50 tokens)
  - Sinusoidal positional encoding
"""

from __future__ import annotations

import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (no learned parameters)."""

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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to embeddings. x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RetroTransformer(nn.Module):
    """Encoder-Decoder Transformer for retrosynthesis prediction.

    Same proven architecture as our ForwardTransformer, with sinusoidal
    positional encoding and proper weight initialization.
    """

    def __init__(
        self,
        vocab_size: int = 50,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

        # Token embedding (shared between encoder and decoder)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Sinusoidal positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"RetroTransformer: {trainable_params:,} trainable / {total_params:,} total params")

    def _init_weights(self):
        """Xavier uniform initialization for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            src_ids: Encoder input token IDs. Shape: (batch, src_len)
            tgt_ids: Decoder input token IDs (shifted right). Shape: (batch, tgt_len)

        Returns:
            Logits over vocabulary. Shape: (batch, tgt_len, vocab_size)
        """
        # Embeddings + positional encoding
        src_emb = self.pos_encoding(
            self.embedding(src_ids) * math.sqrt(self.d_model)
        )
        tgt_emb = self.pos_encoding(
            self.embedding(tgt_ids) * math.sqrt(self.d_model)
        )

        # Masks
        tgt_len = tgt_ids.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt_ids.device
        )
        src_padding_mask = src_ids == self.pad_token_id
        tgt_padding_mask = tgt_ids == self.pad_token_id

        # Transformer
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        return self.output_proj(output)

    @torch.no_grad()
    def generate_greedy(
        self,
        src_ids: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_len: int = 256,
    ) -> list[list[int]]:
        """Greedy autoregressive decoding.

        Args:
            src_ids: Encoder input. Shape: (batch, src_len)
            bos_token_id: Beginning-of-sequence token ID.
            eos_token_id: End-of-sequence token ID.
            max_len: Maximum output length.

        Returns:
            List of token ID lists (one per batch element).
        """
        self.eval()
        batch_size = src_ids.shape[0]
        device = src_ids.device

        # Encode
        src_emb = self.pos_encoding(
            self.embedding(src_ids) * math.sqrt(self.d_model)
        )
        src_padding_mask = src_ids == self.pad_token_id
        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_padding_mask,
        )

        # Decode autoregressively
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

            output = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            logits = self.output_proj(output[:, -1, :])  # (batch, vocab)
            next_tokens = logits.argmax(dim=-1)  # (batch,)

            # Mark finished sequences
            finished = finished | (next_tokens == eos_token_id)
            next_tokens[finished] = eos_token_id

            tgt_ids = torch.cat([tgt_ids, next_tokens.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # Convert to lists, stopping at EOS
        results = []
        for i in range(batch_size):
            ids = tgt_ids[i].tolist()
            # Trim after first EOS (skip BOS at position 0)
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
        beam_size: int = 5,
        max_len: int = 256,
        length_penalty: float = 0.6,
    ) -> list[list[tuple[list[int], float]]]:
        """Beam search decoding.

        Args:
            src_ids: Encoder input. Shape: (batch, src_len) — batch=1 recommended.
            beam_size: Number of beams.
            max_len: Maximum output length.
            length_penalty: Length normalization exponent.

        Returns:
            For each batch element: list of (token_ids, score) tuples sorted by score.
        """
        self.eval()
        batch_size = src_ids.shape[0]
        device = src_ids.device

        all_results = []

        for b in range(batch_size):
            single_src = src_ids[b:b+1]  # (1, src_len)

            # Encode
            src_emb = self.pos_encoding(
                self.embedding(single_src) * math.sqrt(self.d_model)
            )
            src_padding_mask = single_src == self.pad_token_id
            memory = self.transformer.encoder(
                src_emb,
                src_key_padding_mask=src_padding_mask,
            )

            # Expand memory for beam search
            memory = memory.repeat(beam_size, 1, 1)  # (beam, src_len, d)
            src_padding_mask = src_padding_mask.repeat(beam_size, 1)

            # Initialize beams
            beam_ids = torch.full(
                (beam_size, 1), bos_token_id, dtype=torch.long, device=device
            )
            beam_scores = torch.zeros(beam_size, device=device)
            beam_scores[1:] = -1e9  # Only first beam active initially

            finished_beams: list[tuple[list[int], float]] = []

            for step in range(max_len):
                tgt_emb = self.pos_encoding(
                    self.embedding(beam_ids) * math.sqrt(self.d_model)
                )
                tgt_len = beam_ids.shape[1]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt_len, device=device
                )

                output = self.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask,
                )

                logits = self.output_proj(output[:, -1, :])  # (beam, vocab)
                log_probs = torch.log_softmax(logits, dim=-1)

                # Expand scores
                next_scores = beam_scores.unsqueeze(1) + log_probs  # (beam, vocab)
                next_scores = next_scores.view(-1)  # (beam * vocab)

                # Top-k
                topk_scores, topk_indices = next_scores.topk(beam_size, dim=-1)
                beam_indices = topk_indices // self.vocab_size
                token_indices = topk_indices % self.vocab_size

                # Update beams
                beam_ids = torch.cat([
                    beam_ids[beam_indices],
                    token_indices.unsqueeze(1),
                ], dim=1)
                beam_scores = topk_scores

                # Check for finished beams
                active_mask = torch.ones(beam_size, dtype=torch.bool, device=device)
                for i in range(beam_size):
                    if token_indices[i].item() == eos_token_id:
                        seq_len = beam_ids.shape[1] - 1  # exclude BOS
                        norm_score = beam_scores[i].item() / (seq_len ** length_penalty)
                        ids = beam_ids[i].tolist()[1:-1]  # exclude BOS and EOS
                        finished_beams.append((ids, norm_score))
                        active_mask[i] = False

                if not active_mask.any() or len(finished_beams) >= beam_size:
                    break

                # Keep only active beams
                if not active_mask.all():
                    active_indices = active_mask.nonzero(as_tuple=True)[0]
                    if len(active_indices) == 0:
                        break
                    # Pad back to beam_size by repeating best active
                    while len(active_indices) < beam_size:
                        active_indices = torch.cat([
                            active_indices,
                            active_indices[:beam_size - len(active_indices)],
                        ])
                    beam_ids = beam_ids[active_indices[:beam_size]]
                    beam_scores = beam_scores[active_indices[:beam_size]]

            # If no beams finished, take best in-progress
            if not finished_beams:
                for i in range(beam_size):
                    ids = beam_ids[i].tolist()[1:]
                    score = beam_scores[i].item() / max(len(ids), 1) ** length_penalty
                    finished_beams.append((ids, score))

            finished_beams.sort(key=lambda x: x[1], reverse=True)
            all_results.append(finished_beams[:beam_size])

        return all_results


def save_retro_model(model, tokenizer, output_dir, config=None, extra=None):
    """Save model checkpoint with config and tokenizer."""
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
        "vocab": tokenizer.get_vocab_dict(),
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, output_dir / "model.pt")
    logger.info(f"Model saved to {output_dir / 'model.pt'}")


def load_retro_model(checkpoint_path, device="cpu"):
    """Load a trained RetroTransformer from checkpoint."""
    from rasyn.models.retro.tokenizer import CharSmilesTokenizer

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = RetroTransformer(
        vocab_size=config["vocab_size"],
        d_model=config.get("d_model", 512),
        nhead=config.get("nhead", 8),
        num_encoder_layers=config.get("num_encoder_layers", 6),
        num_decoder_layers=config.get("num_decoder_layers", 6),
        dim_feedforward=config.get("dim_feedforward", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=0.0,  # No dropout at inference
        pad_token_id=config.get("pad_token_id", 0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = CharSmilesTokenizer.from_vocab_dict(checkpoint["vocab"])

    return model, tokenizer
