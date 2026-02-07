"""Reaction Center Predictor (RCP): bond-level binary classifier.

Given atom embeddings from the MPNN, predicts which bonds in the product
molecule are reaction center bonds (i.e., bonds that were formed during
the forward reaction and should be broken for retrosynthesis).

Bond embedding: b_ij = [a_i + a_j || bond_energy]
Classifier: MLP(33 -> 16 -> 1) -> sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data


class ReactionCenterPredictor(nn.Module):
    """Predict reaction center bonds from atom embeddings.

    For each bond in the product graph, compute a score indicating
    whether it is a reaction center bond.
    """

    def __init__(self, atom_dim: int = 32, energy_dim: int = 1):
        super().__init__()
        input_dim = atom_dim + energy_dim  # 32 + 1 = 33
        self.energy_proj = nn.Linear(1, energy_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        atom_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        bond_energies: torch.Tensor,
    ) -> torch.Tensor:
        """Predict reaction center scores for each bond.

        Args:
            atom_embeddings: [N, atom_dim] from MPNN.
            edge_index: [2, E] bond connectivity (bidirectional).
            bond_energies: [E, 1] normalized bond energies.

        Returns:
            [E//2] confidence scores for each unique bond (undirected).
        """
        src, dst = edge_index

        # Bond embedding: sum of atom pair embeddings + energy projection
        bond_emb = atom_embeddings[src] + atom_embeddings[dst]
        energy_feat = self.energy_proj(bond_energies)
        bond_feat = torch.cat([bond_emb, energy_feat], dim=-1)

        logits = self.classifier(bond_feat).squeeze(-1)
        scores = torch.sigmoid(logits)

        # Since edges are bidirectional, take only unique (i < j) pairs
        # The first half of edges are (i,j) and second half are (j,i)
        # due to how we construct edge_index in featurize.py
        n_unique = scores.shape[0] // 2
        # Average both directions for stability
        scores_forward = scores[:n_unique]
        scores_backward = scores[n_unique:]
        unique_scores = (scores_forward + scores_backward) / 2

        return unique_scores

    def get_bond_labels(
        self,
        data: Data,
        reaction_center_bonds: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Create binary labels for bond classification.

        Args:
            data: PyG Data object with edge_index.
            reaction_center_bonds: List of (atom_idx_i, atom_idx_j) tuples
                that are reaction center bonds.

        Returns:
            [E//2] binary labels (1 = reaction center, 0 = not).
        """
        rc_set = {(min(i, j), max(i, j)) for i, j in reaction_center_bonds}
        n_unique = data.edge_index.shape[1] // 2
        labels = torch.zeros(n_unique, dtype=torch.float)

        for k in range(n_unique):
            i = data.edge_index[0, k].item()
            j = data.edge_index[1, k].item()
            if (min(i, j), max(i, j)) in rc_set:
                labels[k] = 1.0

        return labels

    def predict_top_k(
        self,
        atom_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        bond_energies: torch.Tensor,
        k: int = 10,
    ) -> list[tuple[int, int, float]]:
        """Predict top-K reaction center bonds.

        Returns:
            List of (atom_idx_i, atom_idx_j, score) sorted by descending score.
        """
        scores = self.forward(atom_embeddings, edge_index, bond_energies)
        n_unique = edge_index.shape[1] // 2

        # Get atom indices for each unique bond
        top_k = min(k, n_unique)
        top_scores, top_indices = torch.topk(scores, top_k)

        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            i = edge_index[0, idx].item()
            j = edge_index[1, idx].item()
            results.append((min(i, j), max(i, j), score))

        return results
