"""Regex/atom-level SMILES tokenizer for RetroTransformer v2.

Replaces character-level tokenization (~50 tokens/molecule) with atom-level
tokenization (~20-25 tokens/molecule) using the standard Schwaller regex.

This roughly halves sequence lengths, making exact-match recovery exponentially
easier: at 95% per-token accuracy, 0.95^25 ≈ 28% vs 0.95^50 ≈ 7.7%.

Regex tokenizes:
  [NH], [C@@H], [nH]  → single bracket-atom tokens
  Br, Cl               → two-letter element tokens
  @@, @                → stereo tokens
  %10, %11             → multi-digit ring tokens
  C, N, O, etc.        → single uppercase atoms
  (, ), =, #, etc.     → individual punctuation

Special tokens:
  0: <pad>    — padding
  1: <bos>    — beginning of sequence
  2: <eos>    — end of sequence
  3: <unk>    — unknown token
  4-13: <RXN_1> through <RXN_10> — reaction class tokens
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Standard SMILES tokenization regex (Schwaller et al.)
# Order matters: longer patterns first to avoid partial matches
SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]"       # Bracket atoms: [NH], [C@@H], [nH], [1*], etc.
    r"|Br|Cl"            # Two-letter elements
    r"|@@|@"             # Stereo markers (check @@ before @)
    r"|%\d{2}"           # Multi-digit ring: %10, %11, etc.
    r"|[A-Z][a-z]?"      # Single-letter atoms: C, N, O, Si, Se, etc.
    r"|[^A-Za-z])"       # Everything else: (, ), =, #, ., +, -, digits, etc.
)

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
RXN_CLASS_TOKENS = [f"<RXN_{i}>" for i in range(1, 11)]  # <RXN_1> through <RXN_10>
SEPARATOR_TOKENS = ["|", " "]  # Conditioning separator and space


class RegexSmilesTokenizer:
    """Atom-level SMILES tokenizer using regex splitting.

    Same API as CharSmilesTokenizer for drop-in replacement.
    Roughly halves sequence lengths compared to character-level.
    """

    def __init__(self, tokens: list[str] | None = None):
        """Build tokenizer from a token list.

        Args:
            tokens: List of unique tokens. If None, built from defaults.
        """
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

        # Add special tokens first (fixed IDs)
        all_tokens = SPECIAL_TOKENS + RXN_CLASS_TOKENS
        if tokens is not None:
            all_tokens = all_tokens + [t for t in tokens if t not in all_tokens]

        for i, tok in enumerate(all_tokens):
            self.token2id[tok] = i
            self.id2token[i] = tok

        # Ensure separator tokens are present
        for sep in SEPARATOR_TOKENS:
            if sep not in self.token2id:
                idx = len(self.token2id)
                self.token2id[sep] = idx
                self.id2token[idx] = sep

        self.pad_token_id = self.token2id["<pad>"]
        self.bos_token_id = self.token2id["<bos>"]
        self.eos_token_id = self.token2id["<eos>"]
        self.unk_token_id = self.token2id["<unk>"]

        logger.info(f"RegexSmilesTokenizer: {len(self.token2id)} tokens")

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @staticmethod
    def tokenize_smiles(text: str) -> list[str]:
        """Split a SMILES string (or conditioned input) into atom-level tokens.

        Handles the conditioning separator '|' and spaces in multi-component SMILES.
        Also handles reaction class tokens like <RXN_3>.

        Args:
            text: SMILES string, possibly with '|' separator or reaction class prefix.

        Returns:
            List of string tokens.
        """
        tokens = []

        # Check for reaction class prefix: <RXN_N> at start
        rxn_match = re.match(r"(<RXN_\d+>)\s*", text)
        if rxn_match:
            tokens.append(rxn_match.group(1))
            text = text[rxn_match.end():]

        # Split by '|' separator (product|synthons)
        parts = text.split("|")
        for part_idx, part in enumerate(parts):
            if part_idx > 0:
                tokens.append("|")

            # Tokenize the SMILES part using regex
            # But first handle spaces (multi-component separator " . " or " ")
            # We keep spaces as explicit tokens
            subparts = part.split(" ")
            for sub_idx, sub in enumerate(subparts):
                if sub_idx > 0:
                    tokens.append(" ")
                if sub:
                    regex_tokens = SMILES_REGEX.findall(sub)
                    tokens.extend(regex_tokens)

        return tokens

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

        tokens = self.tokenize_smiles(text)
        for tok in tokens:
            ids.append(self.token2id.get(tok, self.unk_token_id))

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
            skip_special: If True, skip pad/bos/eos tokens.

        Returns:
            Decoded string.
        """
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens = []
        for i in ids:
            if i == self.eos_token_id:
                break
            if skip_special and i in special_ids:
                continue
            token = self.id2token.get(i, "")
            if token.startswith("<") and token.endswith(">") and skip_special:
                # Skip reaction class tokens and other special tokens in output
                if token.startswith("<RXN_"):
                    continue
            tokens.append(token)
        return "".join(tokens)

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
            "type": "regex",
        }

    @classmethod
    def from_vocab_dict(cls, vocab_dict: dict) -> RegexSmilesTokenizer:
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
    def build_from_data(cls, smiles_list: list[str]) -> RegexSmilesTokenizer:
        """Build tokenizer from actual data, including all observed tokens.

        Args:
            smiles_list: List of SMILES strings (can include conditioned inputs).

        Returns:
            RegexSmilesTokenizer with vocabulary covering all observed tokens.
        """
        all_tokens = set()
        for smi in smiles_list:
            tokens = cls.tokenize_smiles(smi)
            all_tokens.update(tokens)

        # Sort for deterministic ordering
        sorted_tokens = sorted(all_tokens)

        tok = cls(tokens=sorted_tokens)
        logger.info(
            f"Built RegexSmilesTokenizer from {len(smiles_list)} strings: "
            f"{tok.vocab_size} tokens ({len(all_tokens)} unique SMILES tokens)"
        )
        return tok

    def get_segment_ids(self, token_ids: list[int]) -> list[int]:
        """Compute segment IDs for product|synthon separation.

        Returns:
            List of segment IDs: 0 for product tokens, 1 for synthon tokens.
        """
        pipe_id = self.token2id.get("|", -1)
        segment_ids = []
        current_segment = 0
        for tid in token_ids:
            if tid == pipe_id:
                current_segment = 1
            segment_ids.append(current_segment)
        return segment_ids

    def get_rxn_class_token(self, rxn_class: int) -> str:
        """Get the reaction class token string for a given class number."""
        if 1 <= rxn_class <= 10:
            return f"<RXN_{rxn_class}>"
        return ""

    def get_rxn_class_id(self, rxn_class: int) -> int | None:
        """Get the token ID for a reaction class."""
        token = self.get_rxn_class_token(rxn_class)
        return self.token2id.get(token)
