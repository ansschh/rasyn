"""Build offline-augmented dataset for RetroTransformer v2.

Key differences from v1 on-the-fly augmentation:
  - Source (product|synthons): N random SMILES variants per example (offline)
  - Target (reactants): ALWAYS canonical (NO target randomization)
  - Component order shuffling on source synthons only
  - Includes reaction_class from preprocessing
  - Output is a flat JSONL ready for training (no further augmentation needed)

Usage:
    python scripts/build_augmented_dataset.py --n-augments 5
    python scripts/build_augmented_dataset.py --n-augments 10 --workers 8
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path
from multiprocessing import Pool

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def randomize_smiles(smiles: str) -> str:
    """Generate a random (non-canonical) SMILES representation."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        pass
    return smiles


def canonicalize_smiles(smiles: str) -> str:
    """Get canonical SMILES."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return smiles


def canonicalize_multi(smiles_str: str, separator: str = " . ") -> str:
    """Canonicalize and sort multi-component SMILES (deterministic ordering)."""
    parts = smiles_str.split(separator)
    canon_parts = []
    for p in parts:
        p = p.strip()
        if p:
            c = canonicalize_smiles(p)
            if c:
                canon_parts.append(c)
    # Sort for deterministic ordering: longest first, then alphabetical
    canon_parts.sort(key=lambda s: (-len(s), s))
    return separator.join(canon_parts)


def augment_single_example(args):
    """Augment a single example N times. Used by multiprocessing pool."""
    example, n_augments, include_original = args

    product = example["product"]
    synthons = example.get("synthons", "")
    reactants = example["reactants"]
    rxn_class = example.get("reaction_class", 0)
    rxn_id = example.get("rxn_id", "")

    # Target is ALWAYS canonical (sorted, deterministic)
    tgt_text = canonicalize_multi(reactants)
    if not tgt_text:
        return []

    results = []

    # Optionally include the canonical (non-augmented) version
    if include_original:
        canon_product = canonicalize_smiles(product)
        if synthons:
            canon_synthons = canonicalize_multi(synthons)
            src_text = f"{canon_product}|{canon_synthons}"
        else:
            src_text = canon_product

        results.append({
            "src_text": src_text,
            "tgt_text": tgt_text,
            "reaction_class": rxn_class,
            "rxn_id": rxn_id,
            "augment_idx": 0,
        })

    # Generate N augmented copies (source-only randomization)
    for aug_idx in range(n_augments):
        # Randomize product SMILES
        aug_product = randomize_smiles(product)

        # Randomize and shuffle synthon components (source only)
        if synthons:
            synth_parts = synthons.split(" . ")
            aug_synth_parts = [randomize_smiles(s.strip()) for s in synth_parts if s.strip()]
            random.shuffle(aug_synth_parts)
            aug_synthons = " . ".join(aug_synth_parts)
            src_text = f"{aug_product}|{aug_synthons}"
        else:
            src_text = aug_product

        results.append({
            "src_text": src_text,
            "tgt_text": tgt_text,  # ALWAYS canonical
            "reaction_class": rxn_class,
            "rxn_id": rxn_id,
            "augment_idx": aug_idx + 1,
        })

    return results


def load_examples(
    edit_data_path: Path,
    reactions_path: Path | None = None,
) -> list[dict]:
    """Load examples from edit_conditioned_train.jsonl and optionally reactions.jsonl.

    Merges reaction_class from reactions.jsonl if available.
    """
    # Load reaction classes from reactions.jsonl if available
    rxn_classes = {}
    if reactions_path and reactions_path.exists():
        logger.info(f"Loading reaction classes from {reactions_path}...")
        with open(reactions_path) as f:
            for line in f:
                rxn = json.loads(line.strip())
                rxn_id = rxn.get("id", "")
                rxn_class = rxn.get("reaction_class")
                if rxn_id and rxn_class is not None:
                    rxn_classes[rxn_id] = int(rxn_class)
        logger.info(f"  Loaded {len(rxn_classes)} reaction classes")

    # Load examples from edit_conditioned_train.jsonl
    examples = []
    with open(edit_data_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            prompt = ex["prompt"]
            completion = ex["completion"]

            # Extract product and synthons from prompt
            prod_match = re.search(r"<PROD>\s+(.+?)\s+<EDIT>", prompt)
            synth_match = re.search(r"<SYNTHONS>\s+(.+?)\s+<LG_HINTS>", prompt)

            product = prod_match.group(1).strip() if prod_match else ""
            synthons = synth_match.group(1).strip() if synth_match else ""

            if not product or not completion:
                continue

            # Get reaction class
            rxn_id = ex.get("metadata", {}).get("rxn_id", "")
            rxn_class = rxn_classes.get(rxn_id, 0)

            examples.append({
                "product": product,
                "synthons": synthons,
                "reactants": completion,
                "reaction_class": rxn_class,
                "rxn_id": rxn_id,
            })

    return examples


@click.command()
@click.option("--edit-data", default="data/processed/uspto50k/edit_conditioned_train.jsonl")
@click.option("--reactions-data", default="data/processed/uspto50k/reactions.jsonl")
@click.option("--output", default="data/processed/uspto50k/augmented_train.jsonl")
@click.option("--n-augments", default=5, type=int, help="Number of augmented copies per example")
@click.option("--include-original/--no-original", default=True,
              help="Include canonical (non-augmented) version")
@click.option("--workers", default=4, type=int, help="Number of parallel workers")
@click.option("--seed", default=42, type=int)
def main(edit_data, reactions_data, output, n_augments, include_original, workers, seed):
    """Build offline-augmented dataset for RetroTransformer v2."""
    random.seed(seed)

    edit_data_path = PROJECT_ROOT / edit_data
    reactions_path = PROJECT_ROOT / reactions_data
    output_path = PROJECT_ROOT / output

    logger.info(f"Loading data from {edit_data_path}...")
    examples = load_examples(edit_data_path, reactions_path)
    logger.info(f"Loaded {len(examples)} examples")

    # Count reaction class distribution
    class_counts = {}
    for ex in examples:
        c = ex.get("reaction_class", 0)
        class_counts[c] = class_counts.get(c, 0) + 1
    logger.info(f"Reaction class distribution: {dict(sorted(class_counts.items()))}")

    # Augment
    logger.info(f"Augmenting with N={n_augments} copies (include_original={include_original})...")
    args = [(ex, n_augments, include_original) for ex in examples]

    if workers > 1:
        with Pool(workers) as pool:
            all_results = pool.map(augment_single_example, args)
    else:
        all_results = [augment_single_example(a) for a in args]

    # Flatten and write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_path, "w") as f:
        for result_list in all_results:
            for result in result_list:
                f.write(json.dumps(result) + "\n")
                total += 1

    expected = len(examples) * (n_augments + (1 if include_original else 0))
    logger.info(f"Written {total} augmented examples to {output_path}")
    logger.info(f"  Expected: {expected} ({len(examples)} x {n_augments + (1 if include_original else 0)})")

    # Verify a few examples
    logger.info("\nSample augmented examples:")
    with open(output_path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            ex = json.loads(line.strip())
            logger.info(f"  [{i}] src: {ex['src_text'][:80]}...")
            logger.info(f"       tgt: {ex['tgt_text'][:80]}...")
            logger.info(f"       class: {ex['reaction_class']}, aug_idx: {ex['augment_idx']}")

    # Token length statistics (with regex tokenizer)
    try:
        from rasyn.models.retro.tokenizer_v2 import RegexSmilesTokenizer
        tok = RegexSmilesTokenizer.build_from_data(
            [json.loads(line)["src_text"] for line in open(output_path)]
            + [json.loads(line)["tgt_text"] for line in open(output_path)]
        )

        src_lens = []
        tgt_lens = []
        with open(output_path) as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                ex = json.loads(line.strip())
                src_lens.append(len(tok.tokenize_smiles(ex["src_text"])))
                tgt_lens.append(len(tok.tokenize_smiles(ex["tgt_text"])))

        logger.info(f"\nRegex token length statistics (first 1000 examples):")
        logger.info(f"  Source: mean={sum(src_lens)/len(src_lens):.1f}, "
                     f"median={sorted(src_lens)[len(src_lens)//2]}, "
                     f"max={max(src_lens)}")
        logger.info(f"  Target: mean={sum(tgt_lens)/len(tgt_lens):.1f}, "
                     f"median={sorted(tgt_lens)[len(tgt_lens)//2]}, "
                     f"max={max(tgt_lens)}")
    except Exception as e:
        logger.info(f"\n(Could not compute token length stats: {e})")


if __name__ == "__main__":
    main()
