"""Build Leaving Group Co-occurrence Graph (LGCoG).

From Retro-MTGR: LGs that appear together in the same reaction encode
complementary information. The co-occurrence graph is used as input to
a GNN that produces enriched LG embeddings for the LG predictor.

Construction:
  1. For each reaction, identify the pair/set of leaving groups.
  2. Build co-occurrence count matrix U[i,j].
  3. Normalize to probability: p_ij = U[i,j] / sum(U).
  4. This becomes the adjacency matrix for the LGCoG.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

from rasyn.preprocess.extract_edits import extract_edits_from_reaction

logger = logging.getLogger(__name__)


def build_lg_cooccurrence(
    reaction_smiles_list: list[str],
    lg_to_idx: dict[str, int],
) -> np.ndarray:
    """Build the LG co-occurrence matrix.

    Args:
        reaction_smiles_list: List of atom-mapped reaction SMILES.
        lg_to_idx: LG vocabulary mapping (from build_lg_vocab).

    Returns:
        Normalized co-occurrence matrix of shape [vocab_size, vocab_size].
    """
    vocab_size = len(lg_to_idx)
    cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    unk_idx = lg_to_idx.get("<UNK>", -1)

    processed = 0
    for rxn_smi in reaction_smiles_list:
        labels = extract_edits_from_reaction(rxn_smi)
        if labels is None:
            continue

        # Get LG indices for this reaction
        lg_indices = []
        for lg in labels.leaving_groups:
            idx = lg_to_idx.get(lg, unk_idx)
            if idx >= 0:
                lg_indices.append(idx)

        # Self-occurrence (diagonal)
        for idx in lg_indices:
            cooccurrence[idx, idx] += 1

        # Co-occurrence between pairs
        for i, j in combinations(lg_indices, 2):
            cooccurrence[i, j] += 1
            cooccurrence[j, i] += 1

        processed += 1

    # Normalize to probability
    total = cooccurrence.sum()
    if total > 0:
        cooccurrence /= total

    logger.info(
        f"Built LGCoG from {processed} reactions. "
        f"Matrix shape: {cooccurrence.shape}, "
        f"Non-zero entries: {np.count_nonzero(cooccurrence)}"
    )

    return cooccurrence


def save_lg_cog(cog_matrix: np.ndarray, path: str | Path) -> None:
    """Save co-occurrence matrix as numpy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), cog_matrix)
    logger.info(f"Saved LGCoG matrix {cog_matrix.shape} to {path}")


def load_lg_cog(path: str | Path) -> np.ndarray:
    """Load co-occurrence matrix from numpy file."""
    return np.load(str(path))


def cog_to_edge_index_and_weights(
    cog_matrix: np.ndarray,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert co-occurrence matrix to sparse edge representation for GNN.

    Args:
        cog_matrix: [K, K] normalized co-occurrence matrix.
        threshold: Minimum edge weight to include (filters noise).

    Returns:
        Tuple of (edge_index [2, num_edges], edge_weights [num_edges]).
    """
    rows, cols = np.where(cog_matrix > threshold)
    weights = cog_matrix[rows, cols]

    edge_index = np.stack([rows, cols], axis=0)  # [2, num_edges]
    return edge_index, weights
