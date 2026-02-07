"""Full Graph Edit Head: combines MPNN + RCP + AEE + LGP.

This is the main model class for the graph edit head component.
It takes a product molecule graph and outputs:
  1. Top-K reaction center bond predictions (from RCP)
  2. Top-M leaving group predictions per synthon attachment point (from LGP)
  3. Edit hypotheses formatted for LLM conditioning

Training loss: L = w_bond * L_bond + w_contrast * L_contrast + w_lg * L_lg
Default weights from Retro-MTGR: 0.6, 0.2, 0.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.data import Data

from rasyn.models.graph_head.aee import AtomEmbeddingEnhancement
from rasyn.models.graph_head.lgp import LeavingGroupPredictor
from rasyn.models.graph_head.mpnn import MPNN
from rasyn.models.graph_head.rcp import ReactionCenterPredictor
from rasyn.schema import EditHypothesis

logger = logging.getLogger(__name__)


@dataclass
class GraphHeadOutput:
    """Output of the full graph edit head."""

    bond_scores: torch.Tensor          # [E//2] scores per unique bond
    bond_labels: torch.Tensor | None   # [E//2] ground-truth labels (training only)
    lg_log_probs: torch.Tensor | None  # [num_attach, K] LG predictions
    lg_labels: torch.Tensor | None     # [num_attach] ground-truth LG indices (training)
    contrastive_loss: torch.Tensor | None  # Scalar AEE loss (training only)
    atom_embeddings: torch.Tensor      # [N, hidden_dim] for downstream use


class GraphEditHead(nn.Module):
    """Complete graph edit head for retrosynthesis.

    Architecture:
        MPNN encoder -> RCP (bond classifier) + LGP (LG predictor) + AEE (training only)
    """

    def __init__(
        self,
        atom_in_dim: int = 28,
        hidden_dim: int = 32,
        edge_dim: int = 5,
        n_mpnn_layers: int = 2,
        lg_vocab_size: int = 170,
        lg_hidden_dim: int = 128,
        use_cog: bool = True,
        use_aee: bool = True,
        # Loss weights
        w_bond: float = 0.6,
        w_contrast: float = 0.2,
        w_lg: float = 0.2,
    ):
        super().__init__()

        self.mpnn = MPNN(
            atom_in_dim=atom_in_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            n_layers=n_mpnn_layers,
        )

        self.rcp = ReactionCenterPredictor(
            atom_dim=hidden_dim,
            energy_dim=1,
        )

        self.lgp = LeavingGroupPredictor(
            atom_dim=hidden_dim,
            bond_energy_dim=1,
            lg_vocab_size=lg_vocab_size,
            hidden_dim=lg_hidden_dim,
            use_cog=use_cog,
        )

        self.use_aee = use_aee
        if use_aee:
            self.aee = AtomEmbeddingEnhancement(hidden_dim=hidden_dim)

        self.w_bond = w_bond
        self.w_contrast = w_contrast
        self.w_lg = w_lg

        # Loss functions
        self.bond_loss_fn = nn.BCELoss()
        self.lg_loss_fn = nn.NLLLoss()

    def forward(
        self,
        data: Data,
        reaction_center_bonds: list[tuple[int, int]] | None = None,
        attachment_atoms: list[int] | None = None,
        attachment_energies: list[float] | None = None,
        lg_labels: list[int] | None = None,
    ) -> GraphHeadOutput:
        """Forward pass through the graph edit head.

        Args:
            data: PyG Data object for the product molecule.
            reaction_center_bonds: Ground-truth RC bonds (training only).
            attachment_atoms: Atom indices at synthon attachment points (training/LG pred).
            attachment_energies: Bond energies at RC bonds (training/LG pred).
            lg_labels: Ground-truth LG vocabulary indices (training only).

        Returns:
            GraphHeadOutput with all predictions and optional losses.
        """
        # Encode with MPNN
        atom_embeddings = self.mpnn(data.x, data.edge_index, data.edge_attr)

        # Extract bond energies from edge attributes (5th feature is energy)
        bond_energies = data.edge_attr[:, 4:5]  # [E, 1]

        # Predict reaction center bonds
        bond_scores = self.rcp(atom_embeddings, data.edge_index, bond_energies)

        # Get bond labels if in training mode
        bond_labels_tensor = None
        if reaction_center_bonds is not None:
            bond_labels_tensor = self.rcp.get_bond_labels(data, reaction_center_bonds)
            bond_labels_tensor = bond_labels_tensor.to(bond_scores.device)

        # Predict leaving groups (if attachment points provided)
        lg_log_probs = None
        lg_labels_tensor = None
        if attachment_atoms is not None and attachment_energies is not None:
            lg_log_probs = self.lgp(atom_embeddings, attachment_atoms, attachment_energies)
            if lg_labels is not None:
                lg_labels_tensor = torch.tensor(lg_labels, dtype=torch.long, device=atom_embeddings.device)

        return GraphHeadOutput(
            bond_scores=bond_scores,
            bond_labels=bond_labels_tensor,
            lg_log_probs=lg_log_probs,
            lg_labels=lg_labels_tensor,
            contrastive_loss=None,  # Computed separately via AEE
            atom_embeddings=atom_embeddings,
        )

    def compute_loss(
        self,
        output: GraphHeadOutput,
        contrastive_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the combined training loss.

        Returns:
            Dict with 'total', 'bond', 'lg', and 'contrast' loss values.
        """
        losses = {}

        # Bond classification loss (BCE)
        if output.bond_labels is not None:
            losses["bond"] = self.bond_loss_fn(output.bond_scores, output.bond_labels)
        else:
            losses["bond"] = torch.tensor(0.0, device=output.bond_scores.device)

        # LG classification loss (NLL)
        if output.lg_log_probs is not None and output.lg_labels is not None:
            losses["lg"] = self.lg_loss_fn(output.lg_log_probs, output.lg_labels)
        else:
            losses["lg"] = torch.tensor(0.0, device=output.bond_scores.device)

        # Contrastive loss (from AEE)
        if contrastive_loss is not None:
            losses["contrast"] = contrastive_loss
        else:
            losses["contrast"] = torch.tensor(0.0, device=output.bond_scores.device)

        # Weighted total
        losses["total"] = (
            self.w_bond * losses["bond"]
            + self.w_lg * losses["lg"]
            + self.w_contrast * losses["contrast"]
        )

        return losses

    @torch.no_grad()
    def predict(
        self,
        data: Data,
        top_k_bonds: int = 10,
        top_m_lgs: int = 3,
        lg_vocab: dict[str, int] | None = None,
    ) -> list[EditHypothesis]:
        """Generate edit hypotheses for a product molecule.

        Args:
            data: PyG Data object for the product.
            top_k_bonds: Number of bond candidates to return.
            top_m_lgs: Number of LG candidates per attachment point.
            lg_vocab: LG index-to-SMILES mapping for human-readable output.

        Returns:
            List of EditHypothesis objects, sorted by confidence.
        """
        self.eval()

        atom_embeddings = self.mpnn(data.x, data.edge_index, data.edge_attr)
        bond_energies = data.edge_attr[:, 4:5]

        # Get top-K bonds
        top_bonds = self.rcp.predict_top_k(
            atom_embeddings, data.edge_index, bond_energies, k=top_k_bonds,
        )

        # Invert LG vocab for index -> SMILES lookup
        idx_to_lg = {}
        if lg_vocab:
            idx_to_lg = {v: k for k, v in lg_vocab.items()}

        hypotheses = []
        for atom_i, atom_j, bond_score in top_bonds:
            # Get bond energy for this bond
            energy = 0.5  # Default
            for k in range(data.edge_index.shape[1]):
                if data.edge_index[0, k] == atom_i and data.edge_index[1, k] == atom_j:
                    energy = data.edge_attr[k, 4].item()
                    break

            # Predict LGs at both attachment points
            attachment_atoms = [atom_i, atom_j]
            attachment_energies = [energy, energy]
            lg_predictions = self.lgp.predict_top_m(
                atom_embeddings, attachment_atoms, attachment_energies, m=top_m_lgs,
            )

            # Format LG options
            lg_options = []
            for preds_per_point in lg_predictions:
                options = [idx_to_lg.get(idx, f"LG_{idx}") for idx, _ in preds_per_point]
                lg_options.append(options)

            # Build edit tokens
            lg_hints = " ".join(
                "[" + ",".join(opts) + "]" for opts in lg_options
            )
            edit_tokens = (
                f"<EDIT> DISCONNECT {atom_i}-{atom_j} "
                f"<LG_HINTS> {lg_hints}"
            )

            hypotheses.append(EditHypothesis(
                reaction_center_bonds=[(atom_i, atom_j)],
                synthon_smiles=[],  # Filled in by pipeline after fragmentation
                leaving_group_options=lg_options,
                confidence=bond_score,
                edit_tokens=edit_tokens,
            ))

        return hypotheses
