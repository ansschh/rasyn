"""Build edit-conditioned training dataset for LLM fine-tuning.

For each reaction in the training set:
  1. Extract edit labels (changed bonds, synthons, LGs)
  2. Build a prompt in the Edit Token Language
  3. Target is the reactant SMILES

Output format: JSONL with fields (prompt, completion, metadata).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rasyn.preprocess.canonicalize import canonicalize_smiles, parse_reaction_smiles
from rasyn.preprocess.extract_edits import extract_edits_from_reaction

logger = logging.getLogger(__name__)


def build_training_example(rxn_smiles: str, rxn_id: str = "") -> dict | None:
    """Build a single training example from an atom-mapped reaction SMILES.

    Args:
        rxn_smiles: Atom-mapped reaction SMILES.
        rxn_id: Optional reaction identifier.

    Returns:
        Dict with 'prompt', 'completion', and 'metadata' fields, or None if fails.
    """
    labels = extract_edits_from_reaction(rxn_smiles)
    if labels is None:
        return None

    reactants_list, _, _ = parse_reaction_smiles(rxn_smiles)

    # Build completion: canonical reactant SMILES (no atom mapping)
    clean_reactants = []
    for r in reactants_list:
        canon = canonicalize_smiles(r, remove_mapping=True)
        if canon:
            clean_reactants.append(canon)

    if not clean_reactants:
        return None

    completion = " . ".join(sorted(clean_reactants, key=lambda s: (-len(s), s)))

    return {
        "prompt": labels.edit_tokens,
        "completion": completion,
        "metadata": {
            "rxn_id": rxn_id,
            "num_reactants": len(clean_reactants),
            "num_changed_bonds": len(labels.changed_bonds),
            "leaving_groups": labels.leaving_groups,
        },
    }


def build_edit_dataset(
    reactions: list[dict],
    output_path: str | Path,
    rxn_key: str = "rxn_smiles",
    id_key: str = "id",
) -> int:
    """Process a list of reactions and write edit-conditioned training data to JSONL.

    Args:
        reactions: List of dicts with reaction SMILES.
        output_path: Path to write JSONL output.
        rxn_key: Key for reaction SMILES in the input dicts.
        id_key: Key for reaction ID.

    Returns:
        Number of successfully processed examples.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    failed = 0

    with open(output_path, "w") as f:
        for rxn in reactions:
            rxn_smi = rxn.get(rxn_key, "")
            rxn_id = rxn.get(id_key, "")

            example = build_training_example(rxn_smi, rxn_id)
            if example is None:
                failed += 1
                continue

            f.write(json.dumps(example) + "\n")
            count += 1

    logger.info(
        f"Built {count} training examples ({failed} failed) -> {output_path}"
    )
    return count
