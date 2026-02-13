"""Build R-SMILES (root-aligned) augmented dataset in LLM prompt format.

Takes atom-mapped reactions and edit-conditioned data, produces training
examples where both the product (in the prompt) and reactants (in the
completion) are root-aligned at the reaction center atom.

Source randomization is applied to the prompt; the R-SMILES target stays
canonical across all augmented copies.

Usage:
    # Standard R-SMILES LLM dataset with 5x augmentation
    python -u scripts/build_r_smiles_llm_dataset.py

    # 10x augmentation for larger effective dataset
    python -u scripts/build_r_smiles_llm_dataset.py --n-augments 10

    # Custom output path
    python -u scripts/build_r_smiles_llm_dataset.py --output data/processed/uspto50k/r_smiles_llm_10x.jsonl
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import click
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def randomize_smiles(smiles: str) -> str:
    """Generate a random SMILES representation."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        pass
    return smiles


def randomize_multi(smiles_str: str, separator: str = " . ") -> str:
    """Randomize each component and shuffle order."""
    parts = smiles_str.split(separator)
    randomized = []
    for p in parts:
        p = p.strip()
        if p:
            randomized.append(randomize_smiles(p))
    if len(randomized) > 1:
        random.shuffle(randomized)
    return separator.join(randomized)


def load_edit_conditioned_data(path: Path) -> dict[str, dict]:
    """Load edit-conditioned training data keyed by rxn_id."""
    data = {}
    with open(path) as f:
        for line in f:
            ex = json.loads(line.strip())
            rxn_id = ex.get("metadata", {}).get("rxn_id", "")
            if rxn_id:
                data[rxn_id] = ex
    logger.info(f"Loaded {len(data)} edit-conditioned examples from {path}")
    return data


def load_reactions(path: Path) -> list[dict]:
    """Load reactions from reactions.jsonl."""
    records = []
    with open(path) as f:
        for line in f:
            record = json.loads(line.strip())
            if record.get("rxn_smiles"):
                records.append(record)
    logger.info(f"Loaded {len(records)} reactions from {path}")
    return records


def _augment_one(args):
    """Augment a single example N times. For multiprocessing pool."""
    example, n_augments = args
    prompt = example["prompt"]
    completion = example["completion"]
    metadata = example["metadata"]

    results = []

    # Write the canonical R-SMILES version (augment_idx=0)
    results.append({
        "prompt": prompt,
        "completion": completion,
        "metadata": {**metadata, "augment_idx": 0, "r_smiles": True},
    })

    # Parse the prompt to extract product and synthons for randomization
    prod_match = re.search(r"<PROD>\s+(.+?)\s+<EDIT>", prompt)
    synth_match = re.search(r"<SYNTHONS>\s+(.+?)(?=\s+<(?:LG_HINTS|CONSTRAINTS|OUT)>)", prompt)

    if not prod_match:
        return results

    product = prod_match.group(1).strip()
    synthons = synth_match.group(1).strip() if synth_match else ""

    for aug_idx in range(1, n_augments + 1):
        rand_product = randomize_smiles(product)

        # Rebuild prompt with randomized product (and synthons if present)
        aug_prompt = prompt
        aug_prompt = aug_prompt.replace(
            f"<PROD> {product}",
            f"<PROD> {rand_product}",
            1,
        )

        if synthons:
            rand_synthons = randomize_multi(synthons)
            aug_prompt = aug_prompt.replace(
                f"<SYNTHONS> {synthons}",
                f"<SYNTHONS> {rand_synthons}",
                1,
            )

        results.append({
            "prompt": aug_prompt,
            "completion": completion,  # Always canonical R-SMILES target
            "metadata": {**metadata, "augment_idx": aug_idx, "r_smiles": True},
        })

    return results


@click.command()
@click.option("--reactions", default="data/processed/uspto50k/reactions.jsonl",
              help="Path to reactions.jsonl with atom-mapped SMILES")
@click.option("--edit-data", default="data/processed/uspto50k/edit_conditioned_train.jsonl",
              help="Path to edit-conditioned data")
@click.option("--output", default="data/processed/uspto50k/r_smiles_llm_train.jsonl")
@click.option("--n-augments", default=5, type=int,
              help="Number of augmented copies per reaction (source-only randomization)")
@click.option("--workers", default=4, type=int)
@click.option("--seed", default=42, type=int)
def main(reactions, edit_data, output, n_augments, workers, seed):
    """Build R-SMILES augmented dataset in LLM prompt format."""
    from rasyn.preprocess.r_smiles import build_r_smiles_example
    from rasyn.preprocess.canonicalize import canonicalize_smiles
    from rasyn.models.llm.tokenizer import build_edit_prompt

    random.seed(seed)

    reactions_path = PROJECT_ROOT / reactions
    edit_data_path = PROJECT_ROOT / edit_data
    output_path = PROJECT_ROOT / output

    rxn_records = load_reactions(reactions_path)
    edit_info = load_edit_conditioned_data(edit_data_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build R-SMILES examples with LLM prompt format
    r_smiles_examples = []
    failed = 0

    for record in tqdm(rxn_records, desc="Building R-SMILES LLM examples"):
        rxn_smiles = record.get("rxn_smiles", "")
        rxn_id = record.get("id", "")
        rxn_class = record.get("reaction_class", 0)

        # Need both atom-mapped reaction AND edit-conditioned data
        edit_ex = edit_info.get(rxn_id)
        if edit_ex is None:
            failed += 1
            continue

        # Build R-SMILES aligned product and reactants
        r_example = build_r_smiles_example(rxn_smiles, rxn_id, rxn_class)
        if r_example is None:
            failed += 1
            continue

        r_product = r_example["product"]
        r_reactants = r_example["reactants"]

        # Parse edit info from existing prompt
        orig_prompt = edit_ex["prompt"]
        edit_match = re.search(r"<EDIT>\s+(.+?)(?=\s+<SYNTHONS>)", orig_prompt)
        synth_match = re.search(r"<SYNTHONS>\s+(.+?)(?=\s+<(?:LG_HINTS|CONSTRAINTS|OUT)>)", orig_prompt)
        lg_match = re.search(r"<LG_HINTS>\s+(.+?)(?=\s+<(?:CONSTRAINTS|OUT)>)", orig_prompt)

        edit_str = edit_match.group(1).strip() if edit_match else "NO_DISCONNECT"
        synthons = synth_match.group(1).strip() if synth_match else ""
        lg_str = lg_match.group(1).strip() if lg_match else ""

        # Build R-SMILES prompt: use R-SMILES product, keep edit info
        prompt_parts = [f"<PROD> {r_product}"]
        prompt_parts.append(f"<EDIT> {edit_str}")
        if synthons:
            prompt_parts.append(f"<SYNTHONS> {synthons}")
        if lg_str:
            prompt_parts.append(f"<LG_HINTS> {lg_str}")
        prompt_parts.append("<OUT>")

        prompt = " ".join(prompt_parts)

        # Completion is R-SMILES aligned reactants (dot-separated, sorted)
        completion = r_reactants

        r_smiles_examples.append({
            "prompt": prompt,
            "completion": completion,
            "metadata": {
                "rxn_id": rxn_id,
                "reaction_class": rxn_class,
                "root_atom_map": r_example["root_atom_map"],
            },
        })

    logger.info(f"Built {len(r_smiles_examples)} R-SMILES examples ({failed} failed)")

    # Augment with source randomization
    logger.info(f"Augmenting with N={n_augments} copies...")
    args = [(ex, n_augments) for ex in r_smiles_examples]

    if workers > 1:
        with Pool(workers) as pool:
            all_results = pool.map(_augment_one, args)
    else:
        all_results = [_augment_one(a) for a in tqdm(args, desc="Augmenting")]

    # Write output
    total_written = 0
    with open(output_path, "w") as fout:
        for result_list in all_results:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")
                total_written += 1

    logger.info(f"\nR-SMILES LLM dataset built:")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Total examples: {total_written}")
    logger.info(f"  R-SMILES reactions: {len(r_smiles_examples)}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Augmentation: {n_augments}x + 1 canonical = {n_augments + 1}x")
    expected = len(r_smiles_examples) * (n_augments + 1)
    logger.info(f"  Expected: {expected}")

    # Print sample
    logger.info("\nSample examples:")
    with open(output_path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            ex = json.loads(line.strip())
            logger.info(f"  [{i}] prompt: {ex['prompt'][:100]}...")
            logger.info(f"       completion: {ex['completion'][:80]}...")
            logger.info(f"       aug_idx: {ex['metadata'].get('augment_idx', '?')}")


if __name__ == "__main__":
    main()
