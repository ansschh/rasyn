"""C-SMILES (Compositional SMILES) tokenizer for RetroTransformer v2.

Decomposes bracket atoms like [C@@H] into separate tokens: [, C, @@, H, ]
instead of treating the whole bracket atom as a single token. This reduces
vocabulary from ~150-200 to ~60-80 tokens at the cost of ~10% longer sequences.

Benefits:
  - Much better generalization for rare bracket atoms (e.g., [Se], [Te])
  - Charges (+, -, ++, --), chirality (@@, @), and hydrogen counts (H, H2)
    are shared across all bracket atoms instead of creating unique tokens
  - No OOV bracket atoms at inference time

Reference: C-SMILES (2025) — "Element-token decomposition for chemical SMILES"

Drop-in replacement for RegexSmilesTokenizer — same API.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Regex to tokenize contents inside brackets: atoms, charges, H counts, stereo
# Examples:
#   [C@@H] -> [, C, @@, H, ]
#   [nH]   -> [, n, H, ]
#   [NH2+] -> [, N, H, 2, +, ]
#   [1*]   -> [, 1, *, ]
#   [Si]   -> [, Si, ]
#   [Fe2+] -> [, Fe, 2, +, ]
BRACKET_CONTENT_REGEX = re.compile(
    r"(@@|@"               # Stereo markers
    r"|Br|Cl|Si|Se|Te|As"  # Two-letter elements (inside brackets)
    r"|[A-Z][a-z]?"        # Single atoms: C, N, Fe, etc.
    r"|[a-z]"              # Aromatic atoms: n, c, s, etc.
    r"|H\d?"               # Hydrogen: H, H2, H3
    r"|[+\-]{1,2}"         # Charges: +, -, ++, --
    r"|\d+"                # Isotope numbers, atom map numbers
    r"|[:\*]"              # Colon (atom map separator) and wildcard
    r"|.)"                 # Anything else (safety catch-all)
)

# Standard SMILES regex for non-bracket parts (same as RegexSmilesTokenizer,
# but WITHOUT the bracket atom pattern since we handle brackets separately)
SMILES_NON_BRACKET_REGEX = re.compile(
    r"(Br|Cl"             # Two-letter elements
    r"|@@|@"              # Stereo markers
    r"|%\d{2}"            # Multi-digit ring: %10, %11
    r"|[A-Z][a-z]?"       # Single-letter atoms
    r"|[a-z]"             # Aromatic atoms
    r"|[^A-Za-z\[\]])"    # Everything else except brackets
)

from rasyn.models.retro.tokenizer_v2 import (
    SPECIAL_TOKENS,
    RXN_CLASS_TOKENS,
    SEPARATOR_TOKENS,
)


class CSmilesTokenizer:
    """C-SMILES tokenizer: decomposes bracket atoms into element-level tokens.

    Drop-in replacement for RegexSmilesTokenizer with identical API.
    """

    def __init__(self, tokens: list[str] | None = None):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

        all_tokens = SPECIAL_TOKENS + RXN_CLASS_TOKENS
        if tokens is not None:
            all_tokens = all_tokens + [t for t in tokens if t not in all_tokens]

        for i, tok in enumerate(all_tokens):
            self.token2id[tok] = i
            self.id2token[i] = tok

        # Ensure separator and bracket tokens are present
        for sep in SEPARATOR_TOKENS + ["[", "]"]:
            if sep not in self.token2id:
                idx = len(self.token2id)
                self.token2id[sep] = idx
                self.id2token[idx] = sep

        self.pad_token_id = self.token2id["<pad>"]
        self.bos_token_id = self.token2id["<bos>"]
        self.eos_token_id = self.token2id["<eos>"]
        self.unk_token_id = self.token2id["<unk>"]

        logger.info(f"CSmilesTokenizer: {len(self.token2id)} tokens")

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @staticmethod
    def tokenize_smiles(text: str) -> list[str]:
        """Split SMILES with C-SMILES decomposition of bracket atoms.

        Bracket atoms like [C@@H] are decomposed into: [, C, @@, H, ]
        Non-bracket parts use the standard regex tokenization.
        """
        tokens = []

        # Check for reaction class prefix
        rxn_match = re.match(r"(<RXN_\d+>)\s*", text)
        if rxn_match:
            tokens.append(rxn_match.group(1))
            text = text[rxn_match.end():]

        # Split by '|' separator
        parts = text.split("|")
        for part_idx, part in enumerate(parts):
            if part_idx > 0:
                tokens.append("|")

            subparts = part.split(" ")
            for sub_idx, sub in enumerate(subparts):
                if sub_idx > 0:
                    tokens.append(" ")
                if sub:
                    tokens.extend(_tokenize_csmiles(sub))

        return tokens

    def encode(
        self,
        text: str,
        max_len: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
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
                ids = ids[:max_len - 1] + [self.eos_token_id]
            else:
                ids = ids + [self.pad_token_id] * (max_len - len(ids))

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens = []
        for i in ids:
            if i == self.eos_token_id:
                break
            if skip_special and i in special_ids:
                continue
            token = self.id2token.get(i, "")
            if token.startswith("<") and token.endswith(">") and skip_special:
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
        return [
            self.encode(t, max_len=max_len, add_bos=add_bos, add_eos=add_eos)
            for t in texts
        ]

    def roundtrip_check(self, text: str) -> bool:
        ids = self.encode(text, add_bos=True, add_eos=True)
        decoded = self.decode(ids, skip_special=True)
        return decoded == text

    def get_vocab_dict(self) -> dict:
        return {
            "token2id": self.token2id,
            "id2token": {str(k): v for k, v in self.id2token.items()},
            "type": "csmiles",
        }

    @classmethod
    def from_vocab_dict(cls, vocab_dict: dict) -> CSmilesTokenizer:
        tok = cls.__new__(cls)
        tok.token2id = vocab_dict["token2id"]
        tok.id2token = {int(k): v for k, v in vocab_dict["id2token"].items()}
        tok.pad_token_id = tok.token2id["<pad>"]
        tok.bos_token_id = tok.token2id["<bos>"]
        tok.eos_token_id = tok.token2id["<eos>"]
        tok.unk_token_id = tok.token2id["<unk>"]
        return tok

    @classmethod
    def build_from_data(cls, smiles_list: list[str]) -> CSmilesTokenizer:
        all_tokens = set()
        for smi in smiles_list:
            tokens = cls.tokenize_smiles(smi)
            all_tokens.update(tokens)

        sorted_tokens = sorted(all_tokens)
        tok = cls(tokens=sorted_tokens)
        logger.info(
            f"Built CSmilesTokenizer from {len(smiles_list)} strings: "
            f"{tok.vocab_size} tokens ({len(all_tokens)} unique C-SMILES tokens)"
        )
        return tok

    def get_segment_ids(self, token_ids: list[int]) -> list[int]:
        pipe_id = self.token2id.get("|", -1)
        segment_ids = []
        current_segment = 0
        for tid in token_ids:
            if tid == pipe_id:
                current_segment = 1
            segment_ids.append(current_segment)
        return segment_ids

    def get_rxn_class_token(self, rxn_class: int) -> str:
        if 1 <= rxn_class <= 10:
            return f"<RXN_{rxn_class}>"
        return ""

    def get_rxn_class_id(self, rxn_class: int) -> int | None:
        token = self.get_rxn_class_token(rxn_class)
        return self.token2id.get(token)


def _tokenize_csmiles(smiles: str) -> list[str]:
    """Tokenize a single SMILES string with C-SMILES bracket decomposition.

    Iterates through the string character by character, decomposing
    bracket atoms while using regex for non-bracket portions.
    """
    tokens = []
    i = 0
    n = len(smiles)

    while i < n:
        if smiles[i] == "[":
            # Find the matching closing bracket
            j = smiles.index("]", i) + 1
            bracket_content = smiles[i+1:j-1]  # Contents without [ and ]

            tokens.append("[")

            # Decompose bracket contents using regex
            content_tokens = BRACKET_CONTENT_REGEX.findall(bracket_content)
            tokens.extend(content_tokens)

            tokens.append("]")
            i = j
        else:
            # Find the next bracket (or end of string) for non-bracket portion
            j = smiles.find("[", i)
            if j == -1:
                j = n
            non_bracket = smiles[i:j]
            if non_bracket:
                tokens.extend(SMILES_NON_BRACKET_REGEX.findall(non_bracket))
            i = j

    return tokens
