"""Build leaving group vocabulary from a preprocessed reaction dataset.

Enumerates all leaving groups and tracks their frequencies.
Following Retro-MTGR: build the vocab on a LARGER corpus (e.g. USPTO-FULL)
than your training split to reduce OOV issues on test.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from rasyn.preprocess.extract_edits import extract_edits_from_reaction

logger = logging.getLogger(__name__)


def build_lg_vocabulary(
    reaction_smiles_list: list[str],
    min_count: int = 1,
    special_tokens: list[str] | None = None,
) -> tuple[dict[str, int], Counter]:
    """Build an LG vocabulary from a list of reaction SMILES.

    Args:
        reaction_smiles_list: List of atom-mapped reaction SMILES strings.
        min_count: Minimum occurrence count to include an LG in the vocabulary.
        special_tokens: Extra tokens to prepend (e.g. ['<PAD>', '<UNK>']).

    Returns:
        Tuple of (lg_to_idx dict, lg_counter).
    """
    if special_tokens is None:
        special_tokens = ["<PAD>", "<UNK>"]

    lg_counter: Counter = Counter()
    processed = 0
    failed = 0

    for rxn_smi in reaction_smiles_list:
        labels = extract_edits_from_reaction(rxn_smi)
        if labels is None:
            failed += 1
            continue
        for lg in labels.leaving_groups:
            lg_counter[lg] += 1
        processed += 1

    logger.info(
        f"Processed {processed} reactions ({failed} failed). "
        f"Found {len(lg_counter)} unique LGs."
    )

    # Build vocabulary: special tokens first, then LGs sorted by frequency
    lg_to_idx = {}
    for i, token in enumerate(special_tokens):
        lg_to_idx[token] = i

    offset = len(special_tokens)
    for lg, count in lg_counter.most_common():
        if count >= min_count:
            lg_to_idx[lg] = offset
            offset += 1

    logger.info(
        f"Vocabulary size: {len(lg_to_idx)} "
        f"(including {len(special_tokens)} special tokens)"
    )

    return lg_to_idx, lg_counter


def save_lg_vocab(lg_to_idx: dict[str, int], path: str | Path) -> None:
    """Save LG vocabulary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(lg_to_idx, f, indent=2)
    logger.info(f"Saved LG vocabulary ({len(lg_to_idx)} entries) to {path}")


def load_lg_vocab(path: str | Path) -> dict[str, int]:
    """Load LG vocabulary from a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_lg_counts(lg_counter: Counter, path: str | Path) -> None:
    """Save LG frequency counts to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(lg_counter.most_common()), f, indent=2)
    logger.info(f"Saved LG counts to {path}")
