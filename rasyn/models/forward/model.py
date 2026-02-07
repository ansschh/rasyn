"""Forward reaction prediction model for round-trip verification.

A small encoder-decoder Transformer that predicts products from reactants.
Used by the verifier ensemble to check if predicted reactants actually
produce the target product.

For MVP: this is a stub. The RoundTripVerifier falls back to Tanimoto
similarity when no forward model is loaded.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ForwardTransformer(nn.Module):
    """Small encoder-decoder Transformer for forward reaction prediction.

    Input: tokenized reactant SMILES
    Output: tokenized product SMILES
    """

    def __init__(
        self,
        vocab_size: int = 600,
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

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            src_ids: Source token IDs (reactants). Shape: (batch, src_len)
            tgt_ids: Target token IDs (product). Shape: (batch, tgt_len)

        Returns:
            Logits over vocabulary. Shape: (batch, tgt_len, vocab_size)
        """
        batch_size, src_len = src_ids.shape
        _, tgt_len = tgt_ids.shape

        # Embeddings + positional encoding
        src_pos = torch.arange(src_len, device=src_ids.device).unsqueeze(0)
        tgt_pos = torch.arange(tgt_len, device=tgt_ids.device).unsqueeze(0)

        src_emb = self.embedding(src_ids) + self.pos_encoding(src_pos)
        tgt_emb = self.embedding(tgt_ids) + self.pos_encoding(tgt_pos)

        # Causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt_ids.device)

        # Padding masks
        if src_padding_mask is None:
            src_padding_mask = src_ids == self.pad_token_id
        if tgt_padding_mask is None:
            tgt_padding_mask = tgt_ids == self.pad_token_id

        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        return self.output_proj(output)


def load_forward_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> ForwardTransformer:
    """Load a trained forward model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = ForwardTransformer(
        vocab_size=config.get("vocab_size", 600),
        d_model=config.get("d_model", 512),
        nhead=config.get("nhead", 8),
        num_encoder_layers=config.get("num_encoder_layers", 6),
        num_decoder_layers=config.get("num_decoder_layers", 6),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
