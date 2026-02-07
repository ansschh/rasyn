"""Leaving Group Predictor (LGP).

Predicts which leaving group should be attached to each synthon at the
reaction center attachment point.

Architecture:
  1. LG co-occurrence graph (LGCoG) is processed by a 1-layer GNN to produce
     enriched LG embeddings that encode co-occurrence patterns.
  2. Synthon embeddings at attachment points are computed from atom + bond embeddings.
  3. An adaptor MLP maps synthon embeddings to the LG embedding space.
  4. Prediction is done via softmax over LG vocabulary (cross-entropy loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LGCoGEncoder(nn.Module):
    """GNN encoder for the Leaving Group Co-occurrence Graph.

    Takes the LGCoG adjacency (normalized co-occurrence probabilities)
    and produces enriched LG embeddings.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super().__init__()
        self.lg_embedding = nn.Embedding(vocab_size, embed_dim)
        # Simple graph convolution on the co-occurrence graph
        self.weight = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, cog_adj: torch.Tensor) -> torch.Tensor:
        """Encode LG vocabulary through co-occurrence graph.

        Args:
            cog_adj: [K, K] normalized co-occurrence adjacency matrix.

        Returns:
            [K, embed_dim] enriched LG embeddings.
        """
        K = cog_adj.shape[0]
        # Initial embeddings from vocabulary indices
        indices = torch.arange(K, device=cog_adj.device)
        h = self.lg_embedding(indices)  # [K, embed_dim]

        # Graph convolution: h' = sigma(A * W * h + h)
        transformed = self.weight(h)
        aggregated = torch.mm(cog_adj, transformed)
        h = torch.tanh(aggregated + h)  # Residual connection

        return h


class LeavingGroupPredictor(nn.Module):
    """Predict leaving groups for synthon attachment points.

    Takes atom embeddings at the attachment point + bond embedding at the
    reaction center, and classifies over the LG vocabulary.
    """

    def __init__(
        self,
        atom_dim: int = 32,
        bond_energy_dim: int = 1,
        lg_vocab_size: int = 170,
        hidden_dim: int = 128,
        lg_embed_dim: int = 32,
        use_cog: bool = True,
    ):
        super().__init__()
        self.lg_vocab_size = lg_vocab_size
        self.use_cog = use_cog

        # Synthon embedding: atom pair (sum) + bond energy = atom_dim + bond_energy_dim
        synthon_input_dim = atom_dim + bond_energy_dim

        # Adaptor MLP: maps synthon embedding to prediction space
        self.adaptor = nn.Sequential(
            nn.Linear(synthon_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lg_vocab_size),
        )

        # LGCoG encoder (optional)
        if use_cog:
            self.cog_encoder = LGCoGEncoder(lg_vocab_size, lg_embed_dim)
            # When using CoG, we project to LG embedding space instead
            self.adaptor_cog = nn.Sequential(
                nn.Linear(synthon_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, lg_embed_dim),
            )

        self.cog_adj = None  # Set during training setup

    def set_cog_adjacency(self, cog_adj: torch.Tensor) -> None:
        """Register the co-occurrence adjacency matrix."""
        self.cog_adj = cog_adj

    def forward(
        self,
        atom_embeddings: torch.Tensor,
        attachment_atom_indices: list[int],
        bond_energies_at_rc: list[float],
    ) -> torch.Tensor:
        """Predict LG probabilities for each attachment point.

        Args:
            atom_embeddings: [N, atom_dim] atom embeddings from MPNN.
            attachment_atom_indices: List of atom indices at synthon attachment points.
            bond_energies_at_rc: Bond energy values at corresponding reaction center bonds.

        Returns:
            [num_attachments, lg_vocab_size] log-probabilities over LG vocabulary.
        """
        synthon_embeddings = []
        for atom_idx, energy in zip(attachment_atom_indices, bond_energies_at_rc):
            atom_emb = atom_embeddings[atom_idx]
            energy_tensor = torch.tensor([energy], device=atom_emb.device, dtype=atom_emb.dtype)
            synthon_emb = torch.cat([atom_emb, energy_tensor], dim=-1)
            synthon_embeddings.append(synthon_emb)

        if not synthon_embeddings:
            return torch.zeros((0, self.lg_vocab_size), device=atom_embeddings.device)

        synthon_batch = torch.stack(synthon_embeddings)  # [num_attach, atom_dim + 1]

        if self.use_cog and self.cog_adj is not None:
            # Project synthon to LG embedding space
            synthon_proj = self.adaptor_cog(synthon_batch)  # [num_attach, lg_embed_dim]

            # Get enriched LG embeddings from CoG
            lg_embs = self.cog_encoder(self.cog_adj)  # [K, lg_embed_dim]

            # Similarity-based prediction
            logits = torch.mm(synthon_proj, lg_embs.t())  # [num_attach, K]
        else:
            # Direct classification
            logits = self.adaptor(synthon_batch)  # [num_attach, K]

        return F.log_softmax(logits, dim=-1)

    def predict_top_m(
        self,
        atom_embeddings: torch.Tensor,
        attachment_atom_indices: list[int],
        bond_energies_at_rc: list[float],
        m: int = 3,
    ) -> list[list[tuple[int, float]]]:
        """Predict top-M leaving groups per attachment point.

        Returns:
            List of lists, each containing (lg_vocab_idx, probability) tuples.
        """
        log_probs = self.forward(atom_embeddings, attachment_atom_indices, bond_energies_at_rc)
        probs = torch.exp(log_probs)

        results = []
        for i in range(probs.shape[0]):
            top_m_vals, top_m_idxs = torch.topk(probs[i], min(m, self.lg_vocab_size))
            results.append([
                (idx.item(), val.item())
                for idx, val in zip(top_m_idxs, top_m_vals)
            ])
        return results
