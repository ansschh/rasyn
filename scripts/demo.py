"""Interactive demo: input a product SMILES, get retrosynthetic predictions.

Usage:
    python scripts/demo.py
    python scripts/demo.py --smiles "CC(=O)Oc1ccccc1C(O)=O"
    python scripts/demo.py --interactive
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent


def format_step(step, indent: int = 2) -> str:
    """Format a StepObject for display."""
    prefix = " " * indent
    lines = [
        f"{prefix}Product:    {step.product}",
        f"{prefix}Reactants:  {' + '.join(step.reactants)}",
    ]
    if step.reagents:
        lines.append(f"{prefix}Reagents:   {', '.join(step.reagents)}")
    if step.process_scores:
        ps = step.process_scores
        lines.append(
            f"{prefix}Scores:     "
            f"safety={ps.safety_score:.2f}  "
            f"scale={ps.scalability_score:.2f}  "
            f"green={ps.greenness_score:.2f}  "
            f"total={ps.total_score:.2f}"
        )
    if step.risk_tags:
        lines.append(f"{prefix}Risk tags:  {', '.join(step.risk_tags)}")
    if step.edit_explanation:
        lines.append(f"{prefix}Edit:       bonds={step.edit_explanation.highlighted_bonds}")
    if step.verifier_results:
        vr = step.verifier_results
        lines.append(
            f"{prefix}Verifier:   "
            f"valid={vr.rdkit_valid}  "
            f"confidence={vr.overall_confidence:.2f}"
        )
    return "\n".join(lines)


def format_route(route, route_idx: int = 1) -> str:
    """Format a Route for display."""
    lines = [
        f"\n{'='*60}",
        f"Route #{route_idx}  "
        f"(steps={len(route.steps)}, score={route.total_process_score:.3f}, "
        f"complete={route.all_starting_materials_available})",
        f"{'='*60}",
    ]
    for i, step in enumerate(route.steps):
        lines.append(f"\n  Step {i+1}:")
        lines.append(format_step(step, indent=4))
    if route.starting_materials:
        lines.append(f"\n  Starting materials: {', '.join(route.starting_materials)}")
    return "\n".join(lines)


def run_single_step(pipeline, smiles: str) -> None:
    """Run single-step retrosynthesis and display results."""
    print(f"\nTarget: {smiles}")
    print("-" * 60)

    results = pipeline.predict(smiles)

    if not results:
        print("  No results found.")
        return

    for i, step in enumerate(results):
        print(f"\n  Candidate #{i+1} (rank={step.rank}):")
        print(format_step(step, indent=4))

    print()


@click.command()
@click.option("--smiles", "-s", default=None, help="Product SMILES to analyze")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.option("--mode", type=click.Choice(["single", "multi"]), default="single")
@click.option("--graph-head-checkpoint", default=None)
@click.option("--llm-checkpoint", default=None)
@click.option("--device", default="auto")
def main(smiles, interactive, mode, graph_head_checkpoint, llm_checkpoint, device):
    """Rasyn retrosynthesis demo."""
    logging.basicConfig(level=logging.WARNING)

    # Build pipeline
    from rasyn.pipeline.single_step import SingleStepRetro

    # Load graph head if checkpoint available
    graph_head = None
    lg_vocab = None
    if graph_head_checkpoint and Path(graph_head_checkpoint).exists():
        import torch
        from rasyn.models.graph_head.model import GraphEditHead

        checkpoint = torch.load(graph_head_checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        graph_head = GraphEditHead(
            hidden_dim=config.get("hidden_dim", 32),
            lg_vocab_size=config.get("lg_vocab_size", 170),
        )
        graph_head.load_state_dict(checkpoint["model_state_dict"])
        graph_head.eval()
        print("Graph head loaded.")

        # Try loading LG vocab
        vocab_path = PROJECT_ROOT / "data" / "vocab" / "lg_vocab.json"
        if vocab_path.exists():
            with open(vocab_path) as f:
                lg_vocab = json.load(f)

    # Load LLM if checkpoint available
    llm_model = None
    llm_tokenizer = None
    if llm_checkpoint and Path(llm_checkpoint).exists():
        from rasyn.models.llm.model import load_trained_model
        llm_model, llm_tokenizer = load_trained_model(llm_checkpoint, device)
        print("LLM loaded.")

    pipeline = SingleStepRetro(
        graph_head=graph_head,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        lg_vocab=lg_vocab,
        device=device,
    )

    if smiles:
        run_single_step(pipeline, smiles)
    elif interactive:
        print("\nRasyn Retrosynthesis Demo")
        print("Enter a product SMILES (or 'quit' to exit):\n")
        while True:
            try:
                user_input = input("SMILES> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            run_single_step(pipeline, user_input)
    else:
        # Default: run on example molecules
        examples = [
            "CC(=O)Oc1ccccc1C(O)=O",  # Aspirin
            "CC(=O)Nc1ccc(O)cc1",      # Acetaminophen
            "c1ccc(-c2ccccc2)cc1",     # Biphenyl (simple Suzuki product)
        ]
        print("\nRasyn Retrosynthesis Demo (example molecules)")
        print("=" * 60)
        for smi in examples:
            run_single_step(pipeline, smi)


if __name__ == "__main__":
    main()
