"""Build 20x augmented dataset for LLM fine-tuning (canonical SMILES, not R-SMILES).

Takes edit_conditioned_train.jsonl (prompt/completion format) and creates N
augmented copies with randomized product/synthon SMILES in the prompt,
keeping canonical completion (reactants) unchanged.

Usage:
    python -u scripts/build_augmented_llm_dataset.py --n-augments 20
    python -u scripts/build_augmented_llm_dataset.py --n-augments 10 --output data/processed/uspto50k/augmented_10x_llm_train.jsonl
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def randomize_smiles(smiles: str) -> str:
    """Generate a random SMILES representation."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        pass
    return smiles


def randomize_multi(smiles_str: str, shuffle_order: bool = True) -> str:
    """Randomize each component of a multi-component SMILES string."""
    parts = smiles_str.split(".")
    randomized = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        randomized.append(randomize_smiles(p))
    if shuffle_order and len(randomized) > 1:
        random.shuffle(randomized)
    return ".".join(randomized)


def augment_prompt(prompt: str) -> str:
    """Randomize the SMILES in an LLM prompt while keeping structure tags intact.

    Prompt format: <PROD> {product} <EDIT> {edit} <SYNTHONS> {synthons} <LG_HINTS> ...
    We randomize {product} and {synthons}, keep everything else.
    """
    # Extract and randomize product
    prod_match = re.search(r"(<PROD>\s+)(.+?)(\s+<EDIT>)", prompt)
    if prod_match:
        rand_prod = randomize_smiles(prod_match.group(2).strip())
        prompt = prompt[:prod_match.start()] + prod_match.group(1) + rand_prod + prod_match.group(3) + prompt[prod_match.end():]

    # Extract and randomize synthons
    synth_match = re.search(r"(<SYNTHONS>\s+)(.+?)(\s+<LG_HINTS>)", prompt)
    if synth_match:
        rand_synth = randomize_multi(synth_match.group(2).strip(), shuffle_order=True)
        prompt = prompt[:synth_match.start()] + synth_match.group(1) + rand_synth + synth_match.group(3) + prompt[synth_match.end():]

    return prompt


@click.command()
@click.option("--data", default="data/processed/uspto50k/edit_conditioned_train.jsonl",
              help="Path to edit-conditioned training data")
@click.option("--output", default="data/processed/uspto50k/augmented_20x_llm_train.jsonl")
@click.option("--n-augments", default=20, type=int,
              help="Number of augmented copies per example (source-only randomization)")
def main(data, output, n_augments):
    """Build augmented LLM dataset with randomized prompts."""
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)

    data_path = PROJECT_ROOT / data
    output_path = PROJECT_ROOT / output

    # Load examples
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            if "prompt" in ex and "completion" in ex:
                examples.append(ex)
    logger.info(f"Loaded {len(examples)} examples from {data_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with open(output_path, "w") as fout:
        for ex in examples:
            prompt = ex["prompt"]
            completion = ex["completion"]
            metadata = ex.get("metadata", {})

            # Write canonical (original) example
            out = {
                "prompt": prompt,
                "completion": completion,
                "metadata": {**metadata, "augment_idx": 0},
            }
            fout.write(json.dumps(out) + "\n")
            total_written += 1

            # Write N augmented copies
            for aug_idx in range(1, n_augments + 1):
                aug_prompt = augment_prompt(prompt)
                out = {
                    "prompt": aug_prompt,
                    "completion": completion,  # Always canonical
                    "metadata": {**metadata, "augment_idx": aug_idx},
                }
                fout.write(json.dumps(out) + "\n")
                total_written += 1

    logger.info(f"Augmented LLM dataset built:")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Total examples: {total_written}")
    logger.info(f"  Original: {len(examples)}")
    logger.info(f"  Augmentation: {n_augments}x + 1 canonical = {n_augments + 1}x")
    logger.info(f"  Expected: {len(examples) * (n_augments + 1)}")


if __name__ == "__main__":
    main()
