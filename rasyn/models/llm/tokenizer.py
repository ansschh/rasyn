"""Edit Token Language and tokenizer wrapper for the LLM generator.

Defines the special tokens used in the Edit Token Language and provides
utilities for encoding/decoding edit-conditioned prompts.

Edit Token Language format:
  <PROD> {product_smiles}
  <EDIT> DISCONNECT {atom_i}-{atom_j} [DISCONNECT ...]
  <SYNTHONS> {synthon_1} . {synthon_2}
  <LG_HINTS> [{lg_1},{lg_2}] [{lg_3}]
  <CONSTRAINTS> {optional}
  <OUT> {reactant_1} . {reactant_2}
"""

from __future__ import annotations

import re
from typing import Any

# Special tokens for the Edit Token Language
SPECIAL_TOKENS = {
    "product_start": "<PROD>",
    "edit_start": "<EDIT>",
    "synthons_start": "<SYNTHONS>",
    "lg_hints_start": "<LG_HINTS>",
    "constraints_start": "<CONSTRAINTS>",
    "output_start": "<OUT>",
    "disconnect": "DISCONNECT",
    "no_disconnect": "NO_DISCONNECT",
    "pad": "<PAD>",
    "eos": "<EOS>",
    "bos": "<BOS>",
    "null": "<NULL>",
}

ALL_SPECIAL_TOKENS = list(SPECIAL_TOKENS.values())


def build_edit_prompt(
    product_smiles: str,
    edit_tokens: str | None = None,
    synthon_smiles: list[str] | None = None,
    lg_hints: list[list[str]] | None = None,
    constraints: list[str] | None = None,
) -> str:
    """Build an edit-conditioned prompt for the LLM.

    Args:
        product_smiles: Canonical product SMILES (no atom mapping).
        edit_tokens: Pre-formatted edit string (e.g. "DISCONNECT 3-7").
        synthon_smiles: List of synthon SMILES.
        lg_hints: Per-synthon LG candidates.
        constraints: Optional constraint strings.

    Returns:
        Formatted prompt string.
    """
    parts = [f"<PROD> {product_smiles}"]

    if edit_tokens:
        parts.append(f"<EDIT> {edit_tokens}")
    else:
        parts.append("<EDIT> NO_DISCONNECT")

    if synthon_smiles:
        parts.append(f"<SYNTHONS> {' . '.join(synthon_smiles)}")

    if lg_hints:
        lg_parts = []
        for options in lg_hints:
            lg_parts.append("[" + ",".join(options) + "]")
        parts.append(f"<LG_HINTS> {' '.join(lg_parts)}")

    if constraints:
        parts.append(f"<CONSTRAINTS> {' '.join(constraints)}")

    parts.append("<OUT>")

    return " ".join(parts)


def parse_edit_prompt(prompt: str) -> dict[str, Any]:
    """Parse an edit-conditioned prompt back into its components.

    Args:
        prompt: Formatted prompt string.

    Returns:
        Dict with keys: product, edit, synthons, lg_hints, constraints.
    """
    result = {
        "product": "",
        "edit": "",
        "synthons": [],
        "lg_hints": [],
        "constraints": [],
    }

    # Extract product
    prod_match = re.search(r"<PROD>\s*(.+?)(?=\s*<EDIT>)", prompt)
    if prod_match:
        result["product"] = prod_match.group(1).strip()

    # Extract edit
    edit_match = re.search(r"<EDIT>\s*(.+?)(?=\s*<(?:SYNTHONS|LG_HINTS|CONSTRAINTS|OUT)>)", prompt)
    if edit_match:
        result["edit"] = edit_match.group(1).strip()

    # Extract synthons
    synth_match = re.search(r"<SYNTHONS>\s*(.+?)(?=\s*<(?:LG_HINTS|CONSTRAINTS|OUT)>)", prompt)
    if synth_match:
        result["synthons"] = [s.strip() for s in synth_match.group(1).split(".")]

    # Extract LG hints
    lg_match = re.search(r"<LG_HINTS>\s*(.+?)(?=\s*<(?:CONSTRAINTS|OUT)>)", prompt)
    if lg_match:
        lg_text = lg_match.group(1)
        result["lg_hints"] = re.findall(r"\[([^\]]+)\]", lg_text)
        result["lg_hints"] = [
            [x.strip() for x in group.split(",")]
            for group in result["lg_hints"]
        ]

    # Extract constraints
    const_match = re.search(r"<CONSTRAINTS>\s*(.+?)(?=\s*<OUT>)", prompt)
    if const_match:
        result["constraints"] = const_match.group(1).strip().split()

    return result


def build_training_sequence(prompt: str, completion: str) -> str:
    """Build a full training sequence (prompt + completion) for SFT.

    Args:
        prompt: Edit-conditioned prompt (everything before <OUT>).
        completion: Target reactant SMILES.

    Returns:
        Full sequence: "{prompt} {completion} <EOS>"
    """
    # Ensure prompt ends with <OUT>
    if not prompt.strip().endswith("<OUT>"):
        prompt = prompt.strip() + " <OUT>"
    return f"{prompt} {completion} <EOS>"


def extract_completion(generated_text: str) -> str:
    """Extract the reactant SMILES from a generated sequence.

    Args:
        generated_text: Full generated text from the LLM.

    Returns:
        Extracted reactant SMILES string.
    """
    # Find text after <OUT> and before <EOS> (or end of string)
    match = re.search(r"<OUT>\s*(.+?)(?:\s*<EOS>|$)", generated_text)
    if match:
        return match.group(1).strip()

    # Fallback: return everything after the last known token
    for token in ["<OUT>", "<LG_HINTS>", "<SYNTHONS>", "<EDIT>", "<PROD>"]:
        if token in generated_text:
            return generated_text.split(token)[-1].strip()

    return generated_text.strip()
