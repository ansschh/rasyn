"""Inference: generate reactant candidates conditioned on product + edit.

Given a product SMILES and an edit hypothesis from the graph head,
generate candidate reactant sets using the fine-tuned RSGPT model.
"""

from __future__ import annotations

import logging

import torch
from rdkit import Chem

from rasyn.models.llm.tokenizer import (
    build_edit_prompt,
    extract_completion,
)
from rasyn.preprocess.canonicalize import canonicalize_smiles
from rasyn.schema import EditHypothesis, StepCandidate

logger = logging.getLogger(__name__)


def tokenize_prompt_for_inference(
    prompt: str,
    tokenizer,
    max_length: int = 512,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Tokenize a prompt for inference, matching training BPE tokenization.

    During training, the full text ``prompt + " " + completion + eos`` is
    tokenized as a single string.  The space between ``<OUT>`` and the first
    completion token becomes ``<unk>`` (token 3).  Tokenizing the prompt *alone*
    causes the tokenizer to auto-append ``</s>`` instead, which the model has
    never seen at that position — leading it to generate ``</s>`` immediately.

    Fix: tokenize ``prompt + " X"`` (with a dummy suffix so the space is not
    trailing) and keep only the tokens through ``<OUT>`` + the space token.
    """
    out_token_id = tokenizer.convert_tokens_to_ids("<OUT>")

    # Tokenize with a dummy completion so the space after <OUT> gets a real token
    dummy_text = f"{prompt} X"
    dummy_enc = tokenizer(
        dummy_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    dummy_ids = dummy_enc["input_ids"][0]

    # Find the last <OUT> token position
    out_positions = (dummy_ids == out_token_id).nonzero(as_tuple=True)[0]

    if len(out_positions) > 0:
        # Keep everything through <OUT> + 1 (the space token after it)
        prompt_end = out_positions[-1].item() + 2
    else:
        # Fallback: strip trailing EOS from regular prompt encoding
        prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        prompt_end = prompt_enc["input_ids"].shape[1]
        if prompt_end > 0 and prompt_enc["input_ids"][0, -1].item() == tokenizer.eos_token_id:
            prompt_end -= 1

    input_ids = dummy_ids[:prompt_end].unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def generate_reactants(
    model,
    tokenizer,
    product_smiles: str,
    edit_hypothesis: EditHypothesis | None = None,
    constraints: list[str] | None = None,
    num_candidates: int = 5,
    beam_size: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    do_sample: bool = False,
    device: str | None = None,
) -> list[StepCandidate]:
    """Generate candidate reactant sets for a product.

    Args:
        model: Fine-tuned RSGPT model.
        tokenizer: Tokenizer with edit-language special tokens.
        product_smiles: Canonical product SMILES.
        edit_hypothesis: Optional edit hypothesis from graph head.
        constraints: Optional constraint strings.
        num_candidates: Number of candidates to generate.
        beam_size: Beam size for beam search.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (only if do_sample=True).
        top_p: Nucleus sampling threshold.
        do_sample: Use sampling instead of beam search.
        device: Device to use.

    Returns:
        List of StepCandidate objects with generated reactants.
    """
    if device is None:
        device = next(model.parameters()).device

    # Build prompt
    if edit_hypothesis is not None:
        prompt = build_edit_prompt(
            product_smiles=product_smiles,
            edit_tokens=" ".join(
                f"DISCONNECT {a}-{b}"
                for a, b in edit_hypothesis.reaction_center_bonds
            ),
            synthon_smiles=edit_hypothesis.synthon_smiles or None,
            lg_hints=edit_hypothesis.leaving_group_options or None,
            constraints=constraints,
        )
    else:
        # Unconditioned generation (no edit)
        prompt = f"<PROD> {product_smiles} <OUT>"

    # Tokenize — use the inference-safe tokenizer that matches training BPE
    inputs = tokenize_prompt_for_inference(
        prompt, tokenizer, max_length=512, device=device,
    )

    # Generate
    with torch.no_grad():
        if do_sample:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_candidates,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=beam_size,
                num_return_sequences=min(num_candidates, beam_size),
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
            )

    # Decode and parse results
    candidates = []
    seen_reactants = set()

    for i, output_ids in enumerate(outputs):
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=False)
        # Strip <unk> tokens (BPE space artifacts from training tokenization)
        generated_text = generated_text.replace("<unk>", "")
        reactants_str = extract_completion(generated_text)

        # Parse individual reactant SMILES
        reactant_smiles = []
        for smi in reactants_str.replace(" . ", ".").split("."):
            smi = smi.strip()
            canon = canonicalize_smiles(smi)
            if canon:
                # Validate with RDKit
                mol = Chem.MolFromSmiles(canon)
                if mol is not None:
                    reactant_smiles.append(canon)

        if not reactant_smiles:
            continue

        # Deduplicate
        reactants_key = ".".join(sorted(reactant_smiles))
        if reactants_key in seen_reactants:
            continue
        seen_reactants.add(reactants_key)

        # Get generation score (log probability)
        score = 0.0
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            score = outputs.sequences_scores[i].item()

        candidates.append(StepCandidate(
            product=product_smiles,
            reactants=reactant_smiles,
            edit_hypothesis=edit_hypothesis,
            llm_score=score,
        ))

    return candidates


def generate_for_all_edits(
    model,
    tokenizer,
    product_smiles: str,
    edit_hypotheses: list[EditHypothesis],
    candidates_per_edit: int = 5,
    constraints: list[str] | None = None,
    device: str | None = None,
) -> list[StepCandidate]:
    """Generate reactant candidates for all edit hypotheses.

    This is the main entry point used by the pipeline: it takes the top-K
    edit hypotheses from the graph head and generates N candidates per edit.

    Args:
        model: Fine-tuned RSGPT model.
        tokenizer: Tokenizer.
        product_smiles: Canonical product SMILES.
        edit_hypotheses: List of EditHypothesis from graph head.
        candidates_per_edit: Candidates to generate per edit.
        constraints: Optional constraints.
        device: Device.

    Returns:
        Combined list of all StepCandidate objects.
    """
    all_candidates = []

    for edit in edit_hypotheses:
        candidates = generate_reactants(
            model=model,
            tokenizer=tokenizer,
            product_smiles=product_smiles,
            edit_hypothesis=edit,
            constraints=constraints,
            num_candidates=candidates_per_edit,
            device=device,
        )
        all_candidates.extend(candidates)

    logger.info(
        f"Generated {len(all_candidates)} candidates from "
        f"{len(edit_hypotheses)} edits for {product_smiles}"
    )

    return all_candidates
