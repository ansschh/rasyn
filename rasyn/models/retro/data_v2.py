"""Dataset for RetroTransformer v2 with offline augmentation.

Key differences from v1 (data.py):
  - Loads pre-augmented data (no on-the-fly SMILES randomization)
  - Targets are always canonical (no target randomization)
  - Supports reaction class conditioning (prepended token)
  - Uses RegexSmilesTokenizer (atom-level, shorter sequences)
  - Computes segment IDs for product vs synthon separation
  - Conditioning dropout still applied (20%)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RetroDatasetV2(Dataset):
    """Dataset for RetroTransformer v2 training.

    Loads offline-augmented data with canonical targets.
    No on-the-fly SMILES randomization — augmentation was done at preprocessing time.

    Features:
      - Reaction class conditioning (prepend <RXN_k> token to source)
      - Conditioning dropout (20% — sometimes drop synthon conditioning)
      - Segment IDs for product vs synthon separation
      - Atom-level regex tokenization
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_src_len: int = 256,
        max_tgt_len: int = 128,
        conditioning_dropout: float = 0.2,
        use_reaction_class: bool = True,
    ):
        """Load pre-augmented dataset.

        Args:
            data_path: Path to augmented_train.jsonl.
            tokenizer: RegexSmilesTokenizer instance.
            max_src_len: Maximum encoder input length (tokens, not chars).
            max_tgt_len: Maximum decoder output length (tokens, not chars).
            conditioning_dropout: Probability of dropping synthon conditioning.
            use_reaction_class: Whether to prepend reaction class token.
        """
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.conditioning_dropout = conditioning_dropout
        self.use_reaction_class = use_reaction_class

        # Load examples
        self.examples: list[dict] = []
        with open(data_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                self.examples.append(ex)

        logger.info(
            f"RetroDatasetV2: {len(self.examples)} examples loaded from {data_path}"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        src_text = ex["src_text"]
        tgt_text = ex["tgt_text"]
        rxn_class = ex.get("reaction_class", 0)

        # Parse product and synthons from src_text
        if "|" in src_text:
            product, synthons = src_text.split("|", 1)
        else:
            product = src_text
            synthons = ""

        # Conditioning dropout: sometimes drop synthon conditioning
        use_conditioning = synthons and (random.random() > self.conditioning_dropout)

        # Build encoder input
        if use_conditioning:
            src_input = f"{product}|{synthons}"
        else:
            src_input = product

        # Prepend reaction class token if available
        if self.use_reaction_class and rxn_class and 1 <= rxn_class <= 10:
            rxn_token = self.tokenizer.get_rxn_class_token(rxn_class)
            src_input = f"{rxn_token} {src_input}"

        # Tokenize
        src_ids = self.tokenizer.encode(src_input, max_len=self.max_src_len)
        tgt_ids = self.tokenizer.encode(tgt_text, max_len=self.max_tgt_len)

        # Compute segment IDs for source (0=product, 1=synthon)
        segment_ids = self.tokenizer.get_segment_ids(src_ids)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
        }


def collate_fn_v2(batch: list[dict]) -> dict:
    """Collate batch — tensors are already padded to max_len."""
    return {
        "src_ids": torch.stack([b["src_ids"] for b in batch]),
        "tgt_ids": torch.stack([b["tgt_ids"] for b in batch]),
        "segment_ids": torch.stack([b["segment_ids"] for b in batch]),
    }


def load_retro_data_v2(
    data_path: str | Path,
    tokenizer,
    val_split: float = 0.1,
    max_src_len: int = 256,
    max_tgt_len: int = 128,
    conditioning_dropout: float = 0.2,
    use_reaction_class: bool = True,
    seed: int = 42,
) -> tuple[RetroDatasetV2, RetroDatasetV2]:
    """Load augmented data and split into train/val.

    IMPORTANT: We split by unique rxn_id BEFORE augmentation copies,
    so no augmented variant of a val reaction leaks into training.

    Returns:
        (train_dataset, val_dataset)
    """
    # Load all examples
    all_examples = []
    with open(data_path) as f:
        for line in f:
            all_examples.append(json.loads(line.strip()))

    # Group by rxn_id to prevent data leakage
    rxn_groups: dict[str, list[dict]] = {}
    for ex in all_examples:
        rxn_id = ex.get("rxn_id", str(id(ex)))
        if rxn_id not in rxn_groups:
            rxn_groups[rxn_id] = []
        rxn_groups[rxn_id].append(ex)

    # Split unique reaction IDs
    unique_rxn_ids = sorted(rxn_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_rxn_ids)

    n_val = int(len(unique_rxn_ids) * val_split)
    val_rxn_ids = set(unique_rxn_ids[:n_val])
    train_rxn_ids = set(unique_rxn_ids[n_val:])

    # Build train/val example lists
    train_examples = []
    val_examples = []
    for rxn_id, examples in rxn_groups.items():
        if rxn_id in val_rxn_ids:
            # For validation, use only the canonical (augment_idx=0) version
            for ex in examples:
                if ex.get("augment_idx", 0) == 0:
                    val_examples.append(ex)
                    break
            else:
                # If no augment_idx=0, use the first one
                val_examples.append(examples[0])
        else:
            train_examples.extend(examples)

    logger.info(
        f"Split by rxn_id: {len(train_rxn_ids)} train reactions ({len(train_examples)} examples), "
        f"{len(val_rxn_ids)} val reactions ({len(val_examples)} examples)"
    )

    # Write temporary split files
    import tempfile
    train_path = Path(data_path).parent / "augmented_train_split.jsonl"
    val_path = Path(data_path).parent / "augmented_val_split.jsonl"

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    train_dataset = RetroDatasetV2(
        data_path=train_path,
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        conditioning_dropout=conditioning_dropout,
        use_reaction_class=use_reaction_class,
    )

    val_dataset = RetroDatasetV2(
        data_path=val_path,
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        conditioning_dropout=0.0,  # No dropout at validation
        use_reaction_class=use_reaction_class,
    )

    return train_dataset, val_dataset
