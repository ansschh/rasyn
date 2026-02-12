"""Forward reaction prediction model for round-trip verification.

A small encoder-decoder Transformer that predicts products from reactants.
Used by the verifier ensemble to check if predicted reactants actually
produce the target product.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ForwardTokenizer:
    """Simple char-level tokenizer for the forward model."""

    def __init__(self, char2idx: dict, idx2char: dict):
        self.char2idx = char2idx
        self.idx2char = {int(k): v for k, v in idx2char.items()}
        self.pad_token_id = char2idx.get("<pad>", 0)
        self.bos_token_id = char2idx.get("<bos>", 1)
        self.eos_token_id = char2idx.get("<eos>", 2)
        self.unk_token_id = char2idx.get("<unk>", 3)
        self.vocab_size = len(char2idx)

    def encode(self, smiles: str, max_len: int = 256) -> list[int]:
        tokens = [self.bos_token_id]
        for ch in smiles:
            tokens.append(self.char2idx.get(ch, self.unk_token_id))
        tokens.append(self.eos_token_id)
        tokens = tokens[:max_len]
        while len(tokens) < max_len:
            tokens.append(self.pad_token_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        chars = []
        for tid in token_ids:
            ch = self.idx2char.get(tid, "")
            if ch in ("<eos>", "<pad>"):
                break
            if ch not in ("<bos>", "<unk>"):
                chars.append(ch)
        return "".join(chars)


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

    def _encode(self, src_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source sequence, return (memory, src_padding_mask)."""
        src_len = src_ids.shape[1]
        src_pos = torch.arange(src_len, device=src_ids.device).unsqueeze(0)
        src_emb = self.embedding(src_ids) + self.pos_encoding(src_pos)
        src_padding_mask = src_ids == self.pad_token_id

        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_padding_mask
        )
        return memory, src_padding_mask

    @torch.no_grad()
    def generate_greedy(
        self,
        src_ids: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_len: int = 256,
    ) -> list[list[int]]:
        """Autoregressive greedy decoding.

        Args:
            src_ids: (batch, src_len) source token IDs.
            bos_token_id: Begin-of-sequence token ID.
            eos_token_id: End-of-sequence token ID.
            max_len: Maximum output length.

        Returns:
            List of token ID lists (one per batch element, without BOS/EOS).
        """
        self.eval()
        batch_size = src_ids.shape[0]
        device = src_ids.device

        memory, src_padding_mask = self._encode(src_ids)

        tgt_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_len = tgt_ids.shape[1]
            tgt_pos = torch.arange(tgt_len, device=device).unsqueeze(0)
            tgt_emb = self.embedding(tgt_ids) + self.pos_encoding(tgt_pos)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=device
            )
            tgt_padding_mask = tgt_ids == self.pad_token_id

            output = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

            logits = self.output_proj(output[:, -1, :])
            next_tokens = logits.argmax(dim=-1)

            finished = finished | (next_tokens == eos_token_id)
            next_tokens[finished] = eos_token_id
            tgt_ids = torch.cat([tgt_ids, next_tokens.unsqueeze(1)], dim=1)

            if finished.all():
                break

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


def load_forward_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[ForwardTransformer, ForwardTokenizer]:
    """Load a trained forward model and tokenizer from checkpoint.

    Returns:
        Tuple of (model, tokenizer).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = ForwardTransformer(
        vocab_size=config.get("vocab_size", 600),
        d_model=config.get("d_model", 512),
        nhead=config.get("nhead", 8),
        num_encoder_layers=config.get("num_encoder_layers", 6),
        num_decoder_layers=config.get("num_decoder_layers", 6),
        dim_feedforward=config.get("dim_feedforward", 2048),
        max_seq_len=config.get("max_seq_len", 256),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = ForwardTokenizer(
        char2idx=checkpoint["char2idx"],
        idx2char=checkpoint["idx2char"],
    )

    return model, tokenizer
