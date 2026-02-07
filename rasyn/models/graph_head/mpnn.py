"""2-layer Message Passing Neural Network (MPNN) encoder.

Follows Retro-MTGR architecture:
  - Atom MLP: 28 -> 64 -> 32 (ReLU activations)
  - 2-layer MPNN: message = W2 * [neighbor_feat || bond_feat] + b
                  update = sigmoid(W1 * aggregated_msg + W3 * self_feat)
  - Output: 32-dim atom embeddings
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class RetroMPNNConv(MessagePassing):
    """Single MPNN convolution layer matching Retro-MTGR equations.

    Message:  m_{i->j} = W2 * [h_j || e_{ij}] + b
    Aggregate: agg_i = sum_{j in N(i)} m_{j->i}
    Update:    h_i' = sigmoid(W1 * agg_i + W3 * h_i)
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 5):
        super().__init__(aggr="add")
        self.msg_transform = nn.Linear(in_dim + edge_dim, out_dim)
        self.msg_weight = nn.Linear(out_dim, out_dim, bias=False)
        self.self_weight = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] atom features.
            edge_index: [2, E] edge indices.
            edge_attr: [E, edge_dim] bond features.

        Returns:
            [N, out_dim] updated atom embeddings.
        """
        aggr = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = torch.sigmoid(self.msg_weight(aggr) + self.self_weight(x))
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Compute messages from neighbors."""
        return self.msg_transform(torch.cat([x_j, edge_attr], dim=-1))


class AtomEncoder(nn.Module):
    """MLP that maps raw 28-dim atom features to dense 32-dim embeddings."""

    def __init__(self, in_dim: int = 28, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MPNN(nn.Module):
    """Full 2-layer MPNN encoder for molecular graphs.

    Pipeline: AtomEncoder(28 -> 32) -> MPNNConv1(32 -> 32) -> MPNNConv2(32 -> 32)
    """

    def __init__(
        self,
        atom_in_dim: int = 28,
        hidden_dim: int = 32,
        edge_dim: int = 5,
        n_layers: int = 2,
        encoder_hidden: int = 64,
    ):
        super().__init__()
        self.atom_encoder = AtomEncoder(atom_in_dim, encoder_hidden, hidden_dim)
        self.convs = nn.ModuleList([
            RetroMPNNConv(hidden_dim, hidden_dim, edge_dim)
            for _ in range(n_layers)
        ])
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a molecular graph into atom-level embeddings.

        Args:
            x: [N, atom_in_dim] raw atom features.
            edge_index: [2, E] bond connectivity.
            edge_attr: [E, edge_dim] bond features.

        Returns:
            [N, hidden_dim] atom embeddings.
        """
        h = self.atom_encoder(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        return h

    def graph_readout(
        self,
        h: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean-pool atom embeddings to get a graph-level embedding.

        Args:
            h: [N, hidden_dim] atom embeddings.
            batch: [N] batch assignment vector (for batched graphs).

        Returns:
            [B, hidden_dim] graph-level embeddings (B = number of graphs in batch).
        """
        if batch is None:
            return h.mean(dim=0, keepdim=True)
        from torch_geometric.utils import scatter
        return scatter(h, batch, dim=0, reduce="mean")
