"""Direct LLM evaluation on edit-conditioned retrosynthesis.

Loads the fine-tuned model and evaluates on edit-conditioned prompts,
measuring Top-1/3/5 exact match accuracy.

Usage:
    python scripts/evaluate_llm.py --checkpoint checkpoints/llm/uspto50k/final
    python scripts/evaluate_llm.py --checkpoint checkpoints/llm/uspto50k/final --max-samples 100
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click
import torch
from tqdm import tqdm

from rasyn.models.llm.generate import tokenize_prompt_for_inference

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


def canonicalize_smiles(smi: str) -> str:
    """Canonicalize a SMILES string, return empty string on failure."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return ""


def normalize_reactants(smiles_str: str) -> str:
    """Canonicalize and sort reactant SMILES for comparison."""
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    canon_parts = []
    for p in parts:
        c = canonicalize_smiles(p.strip())
        if c:
            canon_parts.append(c)
    return ".".join(sorted(canon_parts))


@click.command()
@click.option("--checkpoint", default="checkpoints/llm/uspto50k/final",
              help="Path to fine-tuned model checkpoint directory")
@click.option("--data", default="data/processed/uspto50k/edit_conditioned_test.jsonl",
              help="Path to evaluation data (JSONL with prompt/completion)")
@click.option("--max-samples", default=500, type=int,
              help="Maximum number of samples to evaluate")
@click.option("--num-beams", default=10, type=int,
              help="Total beam size for generation")
@click.option("--num-beam-groups", default=5, type=int,
              help="Number of beam groups for diverse beam search (1=standard)")
@click.option("--diversity-penalty", default=1.0, type=float,
              help="Diversity penalty for diverse beam search (0.0=no penalty)")
@click.option("--max-new-tokens", default=256, type=int,
              help="Maximum new tokens to generate")
@click.option("--skip", default=0, type=int,
              help="Skip first N samples (to evaluate on unseen tail)")
@click.option("--device", default="auto")
def main(checkpoint, data, max_samples, num_beams, num_beam_groups, diversity_penalty, max_new_tokens, skip, device):
    """Evaluate fine-tuned LLM on edit-conditioned retrosynthesis."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = Path(checkpoint)
    data = Path(data)

    # Load model
    logger.info(f"Loading model from {checkpoint}...")
    from transformers import AutoTokenizer
    from peft import PeftModel
    from rasyn.models.llm.model import load_rsgpt_model

    # Check if checkpoint has adapter_config (LoRA checkpoint)
    if (checkpoint / "adapter_config.json").exists():
        logger.info("Detected LoRA checkpoint, loading base model + adapter")
        # Need to find weights_path for base model
        weights_path = PROJECT_ROOT / "weights" / "rsgpt" / "finetune_50k.pth"
        base_model, _ = load_rsgpt_model(
            weights_path=str(weights_path),
            use_lora=False,
        )
        model = PeftModel.from_pretrained(base_model, str(checkpoint))
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    else:
        logger.info("Loading full model from checkpoint")
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(str(checkpoint))
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))

    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    # Load evaluation data
    logger.info(f"Loading data from {data}...")
    examples = []
    with open(data) as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    # Skip and limit
    if skip > 0:
        examples = examples[skip:]
        logger.info(f"Skipped first {skip} examples")
    if max_samples and max_samples < len(examples):
        examples = examples[:max_samples]
    logger.info(f"Evaluating on {len(examples)} samples")

    # Evaluate
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    total = 0
    invalid_count = 0
    total_unique_preds = 0

    start_time = time.time()

    for i, ex in enumerate(tqdm(examples, desc="Evaluating")):
        prompt = ex["prompt"]
        gt_completion = ex["completion"]
        gt_normalized = normalize_reactants(gt_completion)

        if not gt_normalized:
            continue

        # Tokenize prompt — match training BPE at the <OUT> boundary
        inputs = tokenize_prompt_for_inference(
            prompt, tokenizer, max_length=512, device=device,
        )

        # Generate with diverse beam search
        with torch.no_grad():
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=min(num_beams, 10),
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Use diverse beam search when num_beam_groups > 1
            if num_beam_groups > 1:
                gen_kwargs["num_beam_groups"] = num_beam_groups
                gen_kwargs["diversity_penalty"] = diversity_penalty
            outputs = model.generate(**gen_kwargs)

        # Decode and extract completions
        predictions = []
        for output_ids in outputs:
            text = tokenizer.decode(output_ids, skip_special_tokens=False)
            # Extract text after <OUT>
            if "<OUT>" in text:
                completion = text.split("<OUT>")[-1]
                # Clean up: remove </s>, <EOS>, <unk>, etc.
                for stop in ["</s>", "<EOS>", "<pad>", "<PAD>"]:
                    completion = completion.split(stop)[0]
                # Strip <unk> tokens (BPE space artifacts)
                completion = completion.replace("<unk>", "")
                completion = completion.strip()
            else:
                completion = ""

            if completion:
                norm = normalize_reactants(completion)
                if norm:
                    predictions.append(norm)
                else:
                    invalid_count += 1
            else:
                invalid_count += 1

        # Deduplicate predictions while preserving order
        seen = set()
        unique_preds = []
        for p in predictions:
            if p not in seen:
                seen.add(p)
                unique_preds.append(p)

        total += 1
        total_unique_preds += len(unique_preds)

        # Check top-k accuracy
        if len(unique_preds) >= 1 and unique_preds[0] == gt_normalized:
            top1_correct += 1
        if gt_normalized in unique_preds[:3]:
            top3_correct += 1
        if gt_normalized in unique_preds[:5]:
            top5_correct += 1
        if gt_normalized in unique_preds[:10]:
            top10_correct += 1

        # Print periodic progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            avg_unique = total_unique_preds / total
            logger.info(
                f"Step {i+1}/{len(examples)} | "
                f"Top-1: {top1_correct/total:.3f} | "
                f"Top-3: {top3_correct/total:.3f} | "
                f"Top-5: {top5_correct/total:.3f} | "
                f"Top-10: {top10_correct/total:.3f} | "
                f"Diversity: {avg_unique:.1f} | "
                f"Rate: {rate:.1f} samples/s"
            )

    elapsed = time.time() - start_time
    avg_diversity = total_unique_preds / max(total, 1)

    # Print results
    print("\n" + "=" * 60)
    print("LLM EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Data: {data}")
    print(f"Samples evaluated: {total}")
    print(f"Beam config: {num_beams} beams, {num_beam_groups} groups, penalty={diversity_penalty}")
    print(f"Invalid generations: {invalid_count}")
    print(f"Avg unique predictions per sample: {avg_diversity:.1f}")
    print(f"Time: {elapsed:.1f}s ({total/elapsed:.1f} samples/s)")
    print()
    print(f"  Top-1 accuracy:  {top1_correct/max(total,1):.4f} ({top1_correct}/{total})")
    print(f"  Top-3 accuracy:  {top3_correct/max(total,1):.4f} ({top3_correct}/{total})")
    print(f"  Top-5 accuracy:  {top5_correct/max(total,1):.4f} ({top5_correct}/{total})")
    print(f"  Top-10 accuracy: {top10_correct/max(total,1):.4f} ({top10_correct}/{total})")
    print("=" * 60)

    # Save results
    results = {
        "checkpoint": str(checkpoint),
        "data": str(data),
        "total": total,
        "num_beams": num_beams,
        "num_beam_groups": num_beam_groups,
        "diversity_penalty": diversity_penalty,
        "top1": top1_correct / max(total, 1),
        "top3": top3_correct / max(total, 1),
        "top5": top5_correct / max(total, 1),
        "top10": top10_correct / max(total, 1),
        "avg_diversity": avg_diversity,
        "invalid_count": invalid_count,
        "elapsed_seconds": elapsed,
    }
    results_path = checkpoint / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
