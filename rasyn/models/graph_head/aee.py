"""Atom Embedding Enhancement (AEE) module — contrastive training regularizer.

From Retro-MTGR: the AEE module trains the MPNN to produce atom embeddings
where a product molecule and its combined synthons (positive pair) have similar
graph-level representations, while random other molecules (negative samples)
are pushed apart.

This is a TRAINING-ONLY module. It is removed at inference time.

Contrastive loss:
  L_contrast = -log(sim(prod, synthon) / (sim(prod, synthon) + sim(prod, neg)))
  where sim() = cosine similarity
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasyn.models.graph_head.mpnn import MPNN


class AtomEmbeddingEnhancement(nn.Module):
    """Contrastive learning module for atom embedding regularization.

    During training, pulls product and synthon embeddings together
    while pushing product and negative molecule embeddings apart.
    """

    def __init__(self, hidden_dim: int = 32, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def compute_loss(
        self,
        product_emb: torch.Tensor,
        synthon_emb: torch.Tensor,
        negative_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss (InfoNCE-style).

        Args:
            product_emb: [B, hidden_dim] graph-level product embeddings.
            synthon_emb: [B, hidden_dim] graph-level synthon embeddings (positive).
            negative_emb: [B, hidden_dim] graph-level negative molecule embeddings.

        Returns:
            Scalar contrastive loss.
        """
        # Project
        prod_proj = F.normalize(self.projector(product_emb), dim=-1)
        synth_proj = F.normalize(self.projector(synthon_emb), dim=-1)
        neg_proj = F.normalize(self.projector(negative_emb), dim=-1)

        # Cosine similarity
        pos_sim = (prod_proj * synth_proj).sum(dim=-1) / self.temperature
        neg_sim = (prod_proj * neg_proj).sum(dim=-1) / self.temperature

        # InfoNCE loss
        logits = torch.stack([pos_sim, neg_sim], dim=-1)  # [B, 2]
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(
        self,
        mpnn: MPNN,
        product_data,
        synthon_data,
        negative_data,
    ) -> torch.Tensor:
        """Full forward pass: encode all three molecules and compute loss.

        Args:
            mpnn: Shared MPNN encoder.
            product_data: PyG Batch of product graphs.
            synthon_data: PyG Batch of combined synthon graphs.
            negative_data: PyG Batch of random negative molecules.

        Returns:
            Contrastive loss scalar.
        """
        # Encode with shared MPNN
        prod_h = mpnn(product_data.x, product_data.edge_index, product_data.edge_attr)
        synth_h = mpnn(synthon_data.x, synthon_data.edge_index, synthon_data.edge_attr)
        neg_h = mpnn(negative_data.x, negative_data.edge_index, negative_data.edge_attr)

        # Graph-level readout
        prod_emb = mpnn.graph_readout(prod_h, product_data.batch)
        synth_emb = mpnn.graph_readout(synth_h, synthon_data.batch)
        neg_emb = mpnn.graph_readout(neg_h, negative_data.batch)

        return self.compute_loss(prod_emb, synth_emb, neg_emb)
