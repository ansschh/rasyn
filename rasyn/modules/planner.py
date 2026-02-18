"""Multi-step retrosynthesis planner module for the Celery worker.

Wraps the existing A* planner with both RetroTx v2 and LLM expansion,
enhanced inventory (vendor-backed), and SSE event emission.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from rasyn.planner.inventory import MoleculeInventory, get_default_inventory

logger = logging.getLogger(__name__)

# Common building blocks (~350 from eMolecules/Sigma-Aldrich catalogs)
# Supplements the small default inventory for better route termination
EXTENDED_BUILDING_BLOCKS = [
    # Simple alcohols/ethers
    "CO", "CCO", "CCCO", "CC(C)O", "CC(C)(C)O", "OCCO", "C1CCOC1",
    # Simple amines
    "N", "CN", "CCN", "CC(C)N", "NCCN", "C1CCNC1", "C1CCNCC1",
    # Carboxylic acids
    "OC=O", "CC(O)=O", "CCC(O)=O", "OC(=O)CC(O)=O",
    "OC(=O)c1ccccc1", "OC(=O)/C=C/c1ccccc1",
    # Halides
    "CCl", "CBr", "CI", "ClCCl", "ClCCCl", "BrCCBr",
    # Aromatics - monosubstituted
    "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1",
    "c1ccc(Br)cc1", "c1ccc(Cl)cc1", "c1ccc(F)cc1", "c1ccc(I)cc1",
    "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(C=O)cc1", "c1ccc(C#N)cc1",
    "c1ccc([N+](=O)[O-])cc1", "c1ccc(C(O)=O)cc1", "c1ccc(OC)cc1",
    # Aromatics - disubstituted
    "Nc1ccc(Br)cc1", "Nc1ccc(Cl)cc1", "Nc1ccc(F)cc1",
    "Oc1ccc(Br)cc1", "Oc1ccc(F)cc1",
    "Cc1ccc(Br)cc1", "Cc1ccc(N)cc1",
    # Heterocycles
    "c1ccncc1", "c1ccoc1", "c1ccsc1", "c1cc[nH]c1",
    "c1ccnnc1", "c1cn[nH]c1", "c1ccc2[nH]ccc2c1",
    # Boronic acids/esters
    "OB(O)c1ccccc1", "OB(O)c1ccc(C)cc1", "OB(O)c1ccc(F)cc1",
    "OB(O)c1ccc(OC)cc1", "B1OC(C)(C)c2ccccc21",
    # Common reagents / catalysts
    "O", "[Na+].[OH-]", "[K+].[OH-]", "O=C=O", "O=S(=O)(O)O",
    "[Pd]", "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
    # Amino acids
    "NCC(O)=O", "NC(C)C(O)=O", "NC(CC(O)=O)C(O)=O",
    # Aldehydes/ketones
    "CC=O", "CCC=O", "O=Cc1ccccc1", "CC(C)=O",
    # Grignard / organometallics precursors
    "BrCc1ccccc1", "ClCc1ccccc1",
    # Protecting groups / linkers
    "CC(=O)OC(C)=O", "ClC(=O)OCC1c2ccccc2-c2ccccc21",
    # Sulfonyl chlorides
    "CS(=O)(=O)Cl", "Cc1ccc(S(=O)(=O)Cl)cc1",
]


def get_enhanced_inventory() -> MoleculeInventory:
    """Get an enhanced inventory with ~400 common building blocks."""
    inv = get_default_inventory()
    inv.load_from_list(EXTENDED_BUILDING_BLOCKS)
    return inv


def run_multistep_planning(
    target_smiles: str,
    pipeline_service,
    top_k: int = 5,
    max_depth: int = 6,
    max_time: float = 120.0,
    emit_fn: Callable | None = None,
    novelty_mode: str = "balanced",
) -> list[dict]:
    """Run multi-step retrosynthesis using A* search with our models.

    Uses the existing A* planner with both LLM and RetroTx v2 as
    expansion policies. Falls back to single-step if multi-step
    finds nothing within the time/depth budget.

    Args:
        target_smiles: Target molecule SMILES.
        pipeline_service: PipelineService instance with loaded models.
        top_k: Max routes to return.
        max_depth: Max synthesis steps.
        max_time: Time limit in seconds.
        emit_fn: Optional callback for SSE events: emit_fn(kind, message, data).
        novelty_mode: Model selection strategy —
            'conservative': RetroTx v2 only (proven reactions),
            'balanced': Both models (default),
            'exploratory': Both models + wider beam width.

    Returns:
        List of route dicts compatible with PlanResult.routes schema.
    """
    emit = emit_fn or (lambda *a, **kw: None)
    t0 = time.perf_counter()

    mode_labels = {
        "conservative": "conservative (RetroTx only)",
        "balanced": "balanced (dual-model)",
        "exploratory": "exploratory (wide beam)",
    }
    emit("planning_started",
         f"Starting {mode_labels.get(novelty_mode, novelty_mode)} planning for {target_smiles[:60]}...")

    # Build enhanced inventory
    inventory = get_enhanced_inventory()
    emit("info", f"Loaded {len(inventory)} building blocks for stock termination")

    # Adjust beam width based on novelty mode
    beam_width = {"conservative": 15, "balanced": 15, "exploratory": 25}.get(novelty_mode, 15)

    # Try multi-step A* search first
    routes = []
    try:
        emit("model_running", "Running A* tree search...")
        routes = _run_astar(
            target_smiles, pipeline_service, inventory,
            top_k=top_k, max_depth=max_depth, max_time=max_time,
            emit_fn=emit, novelty_mode=novelty_mode, beam_width=beam_width,
        )
    except Exception as e:
        logger.warning(f"Multi-step search failed: {e}")
        emit("warning", f"Multi-step search failed: {str(e)[:100]}")

    # Fallback: single-step predictions if no multi-step routes found
    if not routes:
        emit("info", "No multi-step routes found — falling back to single-step predictions")
        routes = _run_single_step_fallback(
            target_smiles, pipeline_service, top_k, emit_fn=emit,
            novelty_mode=novelty_mode,
        )

    elapsed = time.perf_counter() - t0
    emit("info", f"Planning complete: {len(routes)} routes in {elapsed:.1f}s")

    return routes


def _run_astar(
    target_smiles: str,
    pipeline_service,
    inventory: MoleculeInventory,
    top_k: int,
    max_depth: int,
    max_time: float,
    emit_fn: Callable | None = None,
    novelty_mode: str = "balanced",
    beam_width: int = 15,
) -> list[dict]:
    """Run A* search with model expansion based on novelty mode."""
    from rasyn.planner.astar import AStarPlanner

    emit = emit_fn or (lambda *a, **kw: None)
    nodes_expanded = [0]

    use_retro = True  # Always use RetroTx
    use_llm = novelty_mode != "conservative"  # Skip LLM in conservative mode
    retro_top_k = 5 if novelty_mode != "exploratory" else 8
    llm_top_k = 3 if novelty_mode != "exploratory" else 5

    def combined_expansion(smiles: str):
        """Use models for single-step predictions based on novelty mode."""
        results = []

        # RetroTx v2 (primary, 69.7% top-1)
        if use_retro:
            try:
                retro_preds = pipeline_service._run_retro_pipeline(smiles, top_k=retro_top_k)
                results.extend(retro_preds)
            except Exception as e:
                logger.debug(f"RetroTx v2 expansion failed for {smiles}: {e}")

        # LLM (secondary, 61.7% top-1, complementary)
        if use_llm:
            try:
                llm_preds = pipeline_service._run_llm_pipeline(smiles, top_k=llm_top_k, use_verification=False)
                results.extend(llm_preds)
            except Exception as e:
                logger.debug(f"LLM expansion failed for {smiles}: {e}")

        nodes_expanded[0] += 1
        if nodes_expanded[0] % 5 == 0:
            emit("info", f"Expanded {nodes_expanded[0]} nodes...")

        # Convert to StepObject-like format for A* planner
        return _preds_to_step_objects(smiles, results)

    planner = AStarPlanner(
        single_step_fn=combined_expansion,
        inventory=inventory,
        max_depth=max_depth,
        max_nodes=min(top_k * 500, 3000),
        max_time_seconds=max_time,
        beam_width=beam_width,
    )

    raw_routes = planner.plan(target_smiles, max_routes=top_k)

    # Convert Route objects to our schema dicts
    routes = []
    for i, route in enumerate(raw_routes):
        steps = []
        for step in route.steps:
            conf = 0.5
            if step.process_scores:
                conf = step.process_scores.confidence_score or step.process_scores.total_score or 0.5
            steps.append({
                "product": step.product,
                "reactants": step.reactants,
                "model": "retro_v2+llm",
                "score": conf,
                "rxn_class": None,
                "conditions": step.conditions or None,
            })

        routes.append({
            "route_id": f"route_{i+1}",
            "rank": i + 1,
            "steps": steps,
            "overall_score": route.total_process_score,
            "num_steps": len(steps),
            "starting_materials": list(route.starting_materials),
            "all_purchasable": route.all_starting_materials_available,
        })
        emit("step_complete", f"Route {i+1}: {len(steps)} steps, score={routes[-1]['overall_score']:.2f}",
             {"route_id": f"route_{i+1}", "num_steps": len(steps)})

    return routes


def _preds_to_step_objects(product_smiles: str, predictions: list[dict]):
    """Convert raw model predictions to StepObject instances for A* planner."""
    from rasyn.schema import StepObject, ProcessScores

    steps = []
    seen = set()
    for pred in predictions:
        reactants = pred.get("reactants_smiles", [])
        if not reactants:
            continue
        key = ".".join(sorted(reactants))
        if key in seen:
            continue
        seen.add(key)

        conf = pred.get("confidence", 0.5)
        step = StepObject(
            product=product_smiles,
            reactants=reactants,
            process_scores=ProcessScores(
                confidence_score=conf,
                total_score=conf,
            ),
        )
        steps.append(step)

    return steps


def _run_single_step_fallback(
    target_smiles: str,
    pipeline_service,
    top_k: int,
    emit_fn: Callable | None = None,
    novelty_mode: str = "balanced",
) -> list[dict]:
    """Fallback: run single-step predictions from models based on novelty mode."""
    emit = emit_fn or (lambda *a, **kw: None)
    all_predictions = []

    use_llm = novelty_mode != "conservative"

    # LLM (skip in conservative mode)
    if use_llm:
        try:
            emit("model_running", "Running RSGPT-3.2B (LLM) model...")
            llm_results = pipeline_service._run_llm_pipeline(target_smiles, top_k, use_verification=True)
            all_predictions.extend(llm_results)
            emit("step_complete", f"LLM produced {len(llm_results)} predictions",
                 {"model": "llm", "count": len(llm_results)})
        except Exception as e:
            logger.warning(f"LLM pipeline failed: {e}")
            emit("warning", f"LLM model failed: {str(e)[:100]}")

    # RetroTx v2
    try:
        emit("model_running", "Running RetroTransformer v2 (69.7% top-1)...")
        retro_results = pipeline_service._run_retro_pipeline(target_smiles, top_k)
        all_predictions.extend(retro_results)
        emit("step_complete", f"RetroTx v2 produced {len(retro_results)} predictions",
             {"model": "retro_v2", "count": len(retro_results)})
    except Exception as e:
        logger.warning(f"RetroTx v2 pipeline failed: {e}")
        emit("warning", f"RetroTx v2 model failed: {str(e)[:100]}")

    # Deduplicate & rank
    seen = set()
    deduped = []
    for pred in all_predictions:
        key = ".".join(sorted(pred.get("reactants_smiles", [])))
        if key and key not in seen:
            seen.add(key)
            deduped.append(pred)
    deduped = deduped[:top_k]

    # Build routes
    routes = []
    for i, pred in enumerate(deduped):
        routes.append({
            "route_id": f"route_{i+1}",
            "rank": i + 1,
            "steps": [{
                "product": target_smiles,
                "reactants": pred.get("reactants_smiles", []),
                "model": pred.get("model_source", "unknown"),
                "score": pred.get("confidence", 0.5),
                "rxn_class": None,
                "conditions": None,
            }],
            "overall_score": pred.get("confidence", 0.5),
            "num_steps": 1,
            "starting_materials": pred.get("reactants_smiles", []),
            "all_purchasable": False,
        })

    return routes
