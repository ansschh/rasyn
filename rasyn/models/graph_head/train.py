"""Training loop for the Graph Edit Head.

Handles:
- Loading preprocessed reaction data
- Building PyG datasets with reaction center + LG labels
- Training with combined loss (bond + LG + contrastive)
- Validation and checkpointing
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
except ImportError:
    raise ImportError("torch_geometric is required. Install with: pip install torch-geometric")

from rasyn.models.graph_head.model import GraphEditHead
from rasyn.preprocess.featurize import mol_to_pyg_data, get_bond_energy
from rasyn.preprocess.canonicalize import get_atom_map_to_idx

from rdkit import Chem

logger = logging.getLogger(__name__)


class RetroDataset:
    """Dataset of product graphs with reaction center + LG labels."""

    def __init__(
        self,
        records_path: str | Path,
        lg_vocab_path: str | Path,
        max_samples: int | None = None,
    ):
        self.records = []
        self.lg_vocab = {}

        # Load LG vocabulary
        with open(lg_vocab_path) as f:
            self.lg_vocab = json.load(f)

        # Load preprocessed records
        with open(records_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                record = json.loads(line.strip())
                self.records.append(record)

        logger.info(f"Loaded {len(self.records)} records, LG vocab size: {len(self.lg_vocab)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict | None:
        """Build a training sample from a reaction record.

        Returns a dict with:
          - 'product_data': PyG Data for product graph
          - 'rc_bonds': list of (atom_idx_i, atom_idx_j) reaction center bonds
          - 'attachment_atoms': atom indices at synthon attachment points
          - 'attachment_energies': bond energies at RC bonds
          - 'lg_labels': LG vocabulary indices for each attachment point
          - 'product_smiles': for negative sampling in AEE
        """
        record = self.records[idx]
        product_smi = record.get("rxn_smiles", "").split(">>")[-1].split(">")[-1]

        # Build product graph (with atom mapping preserved for label alignment)
        product_data = mol_to_pyg_data(product_smi)
        if product_data is None:
            return None

        product_mol = Chem.MolFromSmiles(product_smi)
        if product_mol is None:
            return None

        # Get reaction center bonds (stored as atom indices in product)
        rc_bonds = [tuple(b) for b in record.get("changed_bonds", [])]

        # Get attachment atoms and their bond energies
        attachment_atoms = []
        attachment_energies = []
        for (ai, aj) in rc_bonds:
            energy = get_bond_energy(product_mol, ai, aj)
            attachment_atoms.extend([ai, aj])
            attachment_energies.extend([energy, energy])

        # Get LG labels
        leaving_groups = record.get("leaving_groups", [])
        unk_idx = self.lg_vocab.get("<UNK>", 0)
        lg_labels = [self.lg_vocab.get(lg, unk_idx) for lg in leaving_groups]

        # Pad or truncate LG labels to match attachment atoms
        while len(lg_labels) < len(attachment_atoms):
            lg_labels.append(unk_idx)
        lg_labels = lg_labels[:len(attachment_atoms)]

        return {
            "product_data": product_data,
            "rc_bonds": rc_bonds,
            "attachment_atoms": attachment_atoms,
            "attachment_energies": attachment_energies,
            "lg_labels": lg_labels,
            "product_smiles": record.get("product_smiles", ""),
        }


def collate_fn(batch: list[dict | None]) -> dict | None:
    """Custom collation — handles variable-size labels."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return batch  # Return as list; we process individually due to variable labels


def train_epoch(
    model: GraphEditHead,
    dataset: RetroDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_losses = {"total": 0.0, "bond": 0.0, "lg": 0.0, "contrast": 0.0}
    n_samples = 0

    # Shuffle indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    pbar = tqdm(range(0, len(indices), batch_size), desc="Training")
    for start in pbar:
        batch_indices = indices[start:start + batch_size]
        batch_losses = {"total": 0.0, "bond": 0.0, "lg": 0.0, "contrast": 0.0}
        valid_in_batch = 0

        optimizer.zero_grad()

        for idx in batch_indices:
            sample = dataset[idx]
            if sample is None:
                continue

            product_data = sample["product_data"].to(device)
            rc_bonds = sample["rc_bonds"]
            attachment_atoms = sample["attachment_atoms"]
            attachment_energies = sample["attachment_energies"]
            lg_labels = sample["lg_labels"]

            # Forward pass
            output = model(
                product_data,
                reaction_center_bonds=rc_bonds,
                attachment_atoms=attachment_atoms if attachment_atoms else None,
                attachment_energies=attachment_energies if attachment_energies else None,
                lg_labels=lg_labels if lg_labels else None,
            )

            # Compute loss
            losses = model.compute_loss(output)

            # Accumulate
            for key in batch_losses:
                if key in losses:
                    batch_losses[key] += losses[key].item()

            # Backward (accumulate gradients)
            losses["total"].backward()
            valid_in_batch += 1

        if valid_in_batch > 0:
            # Scale gradients by batch size
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= valid_in_batch

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for key in total_losses:
                total_losses[key] += batch_losses[key]
            n_samples += valid_in_batch

        pbar.set_postfix({
            "loss": f"{total_losses['total'] / max(n_samples, 1):.4f}",
            "bond": f"{total_losses['bond'] / max(n_samples, 1):.4f}",
        })

    # Average losses
    for key in total_losses:
        total_losses[key] /= max(n_samples, 1)

    return total_losses


@torch.no_grad()
def validate(
    model: GraphEditHead,
    dataset: RetroDataset,
    device: torch.device,
    max_samples: int = 1000,
) -> dict[str, float]:
    """Validate the model on a dataset."""
    model.eval()

    total_losses = {"total": 0.0, "bond": 0.0, "lg": 0.0}
    correct_bonds = 0
    total_bonds = 0
    n_samples = 0

    indices = list(range(min(len(dataset), max_samples)))

    for idx in indices:
        sample = dataset[idx]
        if sample is None:
            continue

        product_data = sample["product_data"].to(device)
        rc_bonds = sample["rc_bonds"]

        output = model(
            product_data,
            reaction_center_bonds=rc_bonds,
            attachment_atoms=sample["attachment_atoms"] if sample["attachment_atoms"] else None,
            attachment_energies=sample["attachment_energies"] if sample["attachment_energies"] else None,
            lg_labels=sample["lg_labels"] if sample["lg_labels"] else None,
        )

        losses = model.compute_loss(output)
        for key in total_losses:
            if key in losses:
                total_losses[key] += losses[key].item()

        # Bond prediction accuracy (top-1)
        if output.bond_labels is not None and output.bond_scores.numel() > 0:
            pred_top1 = output.bond_scores.argmax().item()
            true_positives = output.bond_labels.nonzero(as_tuple=True)[0]
            if len(true_positives) > 0 and pred_top1 in true_positives.tolist():
                correct_bonds += 1
            total_bonds += 1

        n_samples += 1

    for key in total_losses:
        total_losses[key] /= max(n_samples, 1)

    total_losses["bond_acc_top1"] = correct_bonds / max(total_bonds, 1)

    return total_losses


def train_graph_head(
    train_records_path: str | Path,
    val_records_path: str | Path,
    lg_vocab_path: str | Path,
    output_dir: str | Path,
    cog_path: str | Path | None = None,
    # Hyperparameters
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 15,
    hidden_dim: int = 32,
    device: str = "auto",
    max_train_samples: int | None = None,
) -> Path:
    """Full training pipeline for the graph edit head.

    Returns:
        Path to the best model checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Load LG vocab to get size
    with open(lg_vocab_path) as f:
        lg_vocab = json.load(f)
    lg_vocab_size = len(lg_vocab)

    # Datasets
    train_dataset = RetroDataset(train_records_path, lg_vocab_path, max_train_samples)
    val_dataset = RetroDataset(val_records_path, lg_vocab_path)

    # Model
    model = GraphEditHead(
        hidden_dim=hidden_dim,
        lg_vocab_size=lg_vocab_size,
        use_cog=cog_path is not None,
        use_aee=True,
    ).to(device)

    # Load CoG if available
    if cog_path:
        cog_adj = torch.tensor(np.load(str(cog_path)), dtype=torch.float).to(device)
        model.lgp.set_cog_adjacency(cog_adj)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_losses = train_epoch(model, train_dataset, optimizer, device, batch_size)
        logger.info(
            f"  Train - total: {train_losses['total']:.4f}, "
            f"bond: {train_losses['bond']:.4f}, lg: {train_losses['lg']:.4f}"
        )

        # Validate
        val_losses = validate(model, val_dataset, device)
        logger.info(
            f"  Val   - total: {val_losses['total']:.4f}, "
            f"bond: {val_losses['bond']:.4f}, "
            f"bond_acc_top1: {val_losses.get('bond_acc_top1', 0):.4f}"
        )

        scheduler.step(val_losses["total"])

        # Checkpointing
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_epoch = epoch
            no_improve = 0
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": {
                    "hidden_dim": hidden_dim,
                    "lg_vocab_size": lg_vocab_size,
                },
            }, checkpoint_path)
            logger.info(f"  Saved best model (epoch {epoch}, val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break

    logger.info(f"\nTraining complete. Best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")
    return output_dir / "best_model.pt"
