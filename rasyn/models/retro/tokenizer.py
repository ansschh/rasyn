"""Character-level SMILES tokenizer for RetroTransformer.

Uses individual characters as tokens — no BPE, no multi-character tokens,
no tokenization artifacts. This is the same approach as our forward model
which achieved val_loss=0.2146.

Special tokens:
  0: <pad>  — padding
  1: <bos>  — beginning of sequence
  2: <eos>  — end of sequence
  3: <unk>  — unknown character
Separator:
  |         — separates product from synthon conditioning
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default SMILES character set (covers USPTO-50K + common organic chemistry)
# Includes ALL lowercase letters (for Cl, Br, Si, Se, As, Te, etc.)
DEFAULT_SMILES_CHARS = list(
    "CNOSPFIBHK"                 # Common uppercase atoms
    "abcdefghijklmnopqrstuvwxyz" # All lowercase (Cl, Br, Si, Se, aromatic, etc.)
    "()[]"                       # Grouping
    "=#@+-\\/.:"                 # Bonds and stereochemistry
    "0123456789"                 # Ring numbers, charges
    "%"                          # Multi-digit ring numbers
    "*"                          # Dummy atoms in synthons [1*]
    " "                          # Space in multi-component SMILES
    "|"                          # Conditioning separator
)

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class CharSmilesTokenizer:
    """Character-level SMILES tokenizer.

    Every SMILES character maps to exactly one token ID. No surprises.
    """

    def __init__(self, chars: list[str] | None = None):
        """Build tokenizer from a character list.

        Args:
            chars: List of unique characters. If None, uses DEFAULT_SMILES_CHARS.
        """
        if chars is None:
            chars = DEFAULT_SMILES_CHARS

        # Build vocabulary: special tokens first, then characters
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token2id[tok] = i
            self.id2token[i] = tok

        offset = len(SPECIAL_TOKENS)
        for i, ch in enumerate(chars):
            if ch not in self.token2id:
                idx = offset + len(self.token2id) - len(SPECIAL_TOKENS)
                self.token2id[ch] = idx
                self.id2token[idx] = ch

        self.pad_token_id = self.token2id["<pad>"]
        self.bos_token_id = self.token2id["<bos>"]
        self.eos_token_id = self.token2id["<eos>"]
        self.unk_token_id = self.token2id["<unk>"]

        logger.info(f"CharSmilesTokenizer: {len(self.token2id)} tokens")

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def encode(
        self,
        text: str,
        max_len: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Encode a SMILES string to token IDs.

        Args:
            text: SMILES string (or product|synthons formatted string).
            max_len: If set, pad/truncate to this length (including BOS/EOS).
            add_bos: Prepend BOS token.
            add_eos: Append EOS token.

        Returns:
            List of token IDs.
        """
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)

        for ch in text:
            ids.append(self.token2id.get(ch, self.unk_token_id))

        if add_eos:
            ids.append(self.eos_token_id)

        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len - 1] + [self.eos_token_id]  # Keep EOS
            else:
                ids = ids + [self.pad_token_id] * (max_len - len(ids))

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs back to a string.

        Args:
            ids: List of token IDs.
            skip_special: If True, skip pad/bos/eos/unk tokens.

        Returns:
            Decoded string.
        """
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        chars = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            if i == self.eos_token_id:
                break
            token = self.id2token.get(i, "")
            chars.append(token)
        return "".join(chars)

    def encode_batch(
        self,
        texts: list[str],
        max_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[list[int]]:
        """Encode a batch of strings."""
        return [
            self.encode(t, max_len=max_len, add_bos=add_bos, add_eos=add_eos)
            for t in texts
        ]

    def roundtrip_check(self, text: str) -> bool:
        """Verify that encode -> decode gives back the original text."""
        ids = self.encode(text, add_bos=True, add_eos=True)
        decoded = self.decode(ids, skip_special=True)
        return decoded == text

    def get_vocab_dict(self) -> dict:
        """Get serializable vocabulary."""
        return {
            "token2id": self.token2id,
            "id2token": {str(k): v for k, v in self.id2token.items()},
        }

    @classmethod
    def from_vocab_dict(cls, vocab_dict: dict) -> CharSmilesTokenizer:
        """Reconstruct tokenizer from saved vocabulary."""
        tok = cls.__new__(cls)
        tok.token2id = vocab_dict["token2id"]
        tok.id2token = {int(k): v for k, v in vocab_dict["id2token"].items()}
        tok.pad_token_id = tok.token2id["<pad>"]
        tok.bos_token_id = tok.token2id["<bos>"]
        tok.eos_token_id = tok.token2id["<eos>"]
        tok.unk_token_id = tok.token2id["<unk>"]
        return tok

    @classmethod
    def build_from_data(cls, smiles_list: list[str]) -> CharSmilesTokenizer:
        """Build tokenizer from actual data, including all observed characters."""
        all_chars = set()
        for smi in smiles_list:
            all_chars.update(smi)
        # Always include default chars + observed chars
        chars = sorted(set(DEFAULT_SMILES_CHARS) | all_chars)
        tok = cls(chars=chars)
        logger.info(f"Built tokenizer from {len(smiles_list)} strings: {tok.vocab_size} tokens")
        return tok
