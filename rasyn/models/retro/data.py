"""Dataset for RetroTransformer with SMILES augmentation and conditioning dropout.

Parses the existing edit_conditioned_train.jsonl to extract:
  - Product SMILES (from <PROD> ... <EDIT>)
  - Synthon SMILES (from <SYNTHONS> ... <LG_HINTS>)
  - Reactant SMILES (from completion)

Input format: product_smiles | synthon1 . synthon2
Output format: reactant1 . reactant2

With 20% conditioning dropout, the input becomes just: product_smiles
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _parse_prompt(prompt: str) -> tuple[str, str]:
    """Extract product and synthons from edit-conditioned prompt.

    Returns:
        (product_smiles, synthons_string)
    """
    # Extract product: between <PROD> and <EDIT>
    prod_match = re.search(r"<PROD>\s+(.+?)\s+<EDIT>", prompt)
    product = prod_match.group(1).strip() if prod_match else ""

    # Extract synthons: between <SYNTHONS> and <LG_HINTS>
    synth_match = re.search(r"<SYNTHONS>\s+(.+?)\s+<LG_HINTS>", prompt)
    synthons = synth_match.group(1).strip() if synth_match else ""

    return product, synthons


def randomize_smiles(smiles: str) -> str:
    """Generate a random (non-canonical) SMILES representation.

    Uses RDKit's doRandom=True to get a random atom ordering.
    Falls back to the original SMILES on any error.
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        pass
    return smiles


def randomize_multi_smiles(smiles_str: str, separator: str = " . ") -> str:
    """Randomize each component in a multi-component SMILES string."""
    parts = smiles_str.split(separator)
    randomized = []
    for part in parts:
        part = part.strip()
        if part:
            randomized.append(randomize_smiles(part))
    return separator.join(randomized)


class RetroDataset(Dataset):
    """Dataset for RetroTransformer training.

    Features:
      - On-the-fly SMILES augmentation (random atom ordering)
      - Conditioning dropout (20% — train without synthons)
      - Character-level tokenization
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_src_len: int = 512,
        max_tgt_len: int = 256,
        augment: bool = True,
        conditioning_dropout: float = 0.2,
    ):
        """Load and prepare the dataset.

        Args:
            data_path: Path to edit_conditioned_train.jsonl.
            tokenizer: CharSmilesTokenizer instance.
            max_src_len: Maximum encoder input length (chars).
            max_tgt_len: Maximum decoder output length (chars).
            augment: Enable SMILES randomization augmentation.
            conditioning_dropout: Probability of dropping synthon conditioning.
        """
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.augment = augment
        self.conditioning_dropout = conditioning_dropout

        # Load and parse examples
        self.examples: list[dict] = []
        skipped = 0

        with open(data_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                product, synthons = _parse_prompt(ex["prompt"])
                completion = ex["completion"]

                if not product or not completion:
                    skipped += 1
                    continue

                self.examples.append({
                    "product": product,
                    "synthons": synthons,
                    "reactants": completion,
                })

        logger.info(
            f"RetroDataset: {len(self.examples)} examples loaded from {data_path} "
            f"({skipped} skipped)"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        product = ex["product"]
        synthons = ex["synthons"]
        reactants = ex["reactants"]

        # SMILES augmentation: random non-canonical SMILES
        if self.augment:
            product = randomize_smiles(product)
            reactants = randomize_multi_smiles(reactants)
            if synthons:
                synthons = randomize_multi_smiles(synthons)

        # Conditioning dropout: sometimes drop synthon conditioning
        use_conditioning = synthons and (random.random() > self.conditioning_dropout)

        # Build encoder input
        if use_conditioning:
            src_text = f"{product}|{synthons}"
        else:
            src_text = product

        # Build decoder target
        tgt_text = reactants

        # Tokenize
        src_ids = self.tokenizer.encode(src_text, max_len=self.max_src_len)
        tgt_ids = self.tokenizer.encode(tgt_text, max_len=self.max_tgt_len)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }

    def get_raw_example(self, idx: int) -> dict:
        """Get raw (un-augmented) example for debugging."""
        return self.examples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Collate batch — tensors are already padded to max_len."""
    return {
        "src_ids": torch.stack([b["src_ids"] for b in batch]),
        "tgt_ids": torch.stack([b["tgt_ids"] for b in batch]),
    }


def load_retro_data(
    data_path: str | Path,
    tokenizer,
    val_split: float = 0.1,
    max_src_len: int = 512,
    max_tgt_len: int = 256,
    augment_train: bool = True,
    conditioning_dropout: float = 0.2,
    seed: int = 42,
) -> tuple[RetroDataset, RetroDataset]:
    """Load data and split into train/val.

    Returns:
        (train_dataset, val_dataset)
    """
    full_dataset = RetroDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        augment=False,  # Don't augment yet — we'll set it after split
        conditioning_dropout=conditioning_dropout,
    )

    # Split
    n = len(full_dataset)
    n_val = int(n * val_split)
    n_train = n - n_val

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create separate datasets for train and val
    train_dataset = RetroSubset(full_dataset, train_indices, augment=augment_train)
    val_dataset = RetroSubset(full_dataset, val_indices, augment=False)

    logger.info(f"Split: {n_train} train, {n_val} val")
    return train_dataset, val_dataset


class RetroSubset(Dataset):
    """Subset of RetroDataset with configurable augmentation."""

    def __init__(self, parent: RetroDataset, indices: list[int], augment: bool = True):
        self.parent = parent
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # Temporarily set augmentation
        orig_augment = self.parent.augment
        self.parent.augment = self.augment
        item = self.parent[real_idx]
        self.parent.augment = orig_augment
        return item

    def get_raw_example(self, idx):
        return self.parent.get_raw_example(self.indices[idx])
