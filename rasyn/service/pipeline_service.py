"""Pipeline orchestration service — wraps existing pipeline for API use."""

from __future__ import annotations

import asyncio
import logging
import time
from functools import partial

from rasyn.preprocess.canonicalize import canonicalize_smiles
from rasyn.service.model_manager import ModelManager

logger = logging.getLogger(__name__)


class PipelineService:
    """Async wrapper around SingleStepRetro and MultiStepPlanner.

    Blocking GPU operations are dispatched to a thread executor
    so the event loop stays responsive.
    """

    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        self._inference_config = model_manager.config.get("inference", {})

    # ------------------------------------------------------------------
    # Single-step retrosynthesis
    # ------------------------------------------------------------------

    async def single_step(
        self,
        product_smiles: str,
        model: str = "llm",
        top_k: int = 10,
        use_verification: bool = True,
    ) -> dict:
        """Run single-step retrosynthesis.

        Args:
            product_smiles: Product SMILES string.
            model: Which model to use — "llm", "retro", or "both".
            top_k: Max number of results.
            use_verification: Whether to run round-trip verification.

        Returns:
            Dict with product, predictions list, and compute_time_ms.
        """
        canon = canonicalize_smiles(product_smiles)
        if not canon:
            return {"error": "Invalid SMILES", "product": product_smiles, "predictions": []}

        t0 = time.perf_counter()

        results = []
        loop = asyncio.get_event_loop()

        if model in ("llm", "both"):
            async with self.mm.lock:
                llm_results = await loop.run_in_executor(
                    None, partial(self._run_llm_pipeline, canon, top_k, use_verification)
                )
            results.extend(llm_results)

        if model in ("retro", "both"):
            async with self.mm.lock:
                retro_results = await loop.run_in_executor(
                    None, partial(self._run_retro_pipeline, canon, top_k)
                )
            results.extend(retro_results)

        # Deduplicate by canonical reactant set
        seen = set()
        deduped = []
        for r in results:
            key = ".".join(sorted(r["reactants_smiles"]))
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        deduped = deduped[:top_k]

        # Re-rank
        for i, pred in enumerate(deduped):
            pred["rank"] = i + 1

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "product": canon,
            "predictions": deduped,
            "compute_time_ms": round(elapsed, 1),
        }

    def _run_llm_pipeline(
        self, product_smiles: str, top_k: int, use_verification: bool
    ) -> list[dict]:
        """Synchronous LLM pipeline (runs in executor thread)."""
        from rasyn.pipeline.single_step import SingleStepRetro
        from rasyn.verifier.ensemble import VerifierEnsemble

        llm_model, llm_tokenizer = self.mm.get_llm()
        if llm_model is None:
            return []

        graph_head = self.mm.get_graph_head()
        lg_vocab = self.mm.get_lg_vocab()

        verifier = None
        if use_verification:
            fwd_model, fwd_tok = self.mm.get_forward()
            if fwd_model is not None:
                verifier = VerifierEnsemble(
                    forward_model=fwd_model, forward_tokenizer=fwd_tok
                )

        ss_cfg = self._inference_config.get("single_step", {})
        pipeline = SingleStepRetro(
            graph_head=graph_head,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            verifier=verifier,
            lg_vocab=lg_vocab,
            top_k_bonds=ss_cfg.get("top_k_bonds", 10),
            top_m_lgs=ss_cfg.get("top_m_lgs", 3),
            candidates_per_edit=ss_cfg.get("candidates_per_edit", 5),
            top_n_results=top_k,
            device=self.mm.device,
        )

        steps = pipeline.predict(product_smiles)
        return [self._step_to_dict(s, "llm") for s in steps]

    def _run_retro_pipeline(self, product_smiles: str, top_k: int) -> list[dict]:
        """Synchronous RetroTx v2 pipeline (runs in executor thread)."""
        import torch
        from rasyn.preprocess.canonicalize import canonicalize_smiles as canon_fn

        retro_model, retro_tok = self.mm.get_retro()
        if retro_model is None:
            return []

        # Encode product
        src_ids = retro_tok.encode(product_smiles)
        src_tensor = torch.tensor([src_ids], device=self.mm.device)

        # Build segment IDs (all 0 = product)
        segment_ids = torch.zeros_like(src_tensor)

        # Beam search
        beam_results = retro_model.generate_beam(
            src_ids=src_tensor,
            bos_token_id=retro_tok.bos_token_id,
            eos_token_id=retro_tok.eos_token_id,
            beam_size=min(top_k, 10),
            max_len=128,
            segment_ids=segment_ids,
        )

        results = []
        seen = set()
        for beams in beam_results:
            for token_ids, score in beams:
                smiles = retro_tok.decode(token_ids)
                # Parse and canonicalize reactants
                reactants = []
                for part in smiles.replace(" . ", ".").split("."):
                    c = canon_fn(part.strip())
                    if c:
                        reactants.append(c)
                if not reactants:
                    continue
                key = ".".join(sorted(reactants))
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "rank": len(results) + 1,
                    "reactants_smiles": reactants,
                    "confidence": round(float(torch.sigmoid(torch.tensor(score)).item()), 4),
                    "model_source": "retro_v2",
                    "verification": None,
                    "edit_info": None,
                })
                if len(results) >= top_k:
                    break

        return results

    # ------------------------------------------------------------------
    # Multi-step route planning
    # ------------------------------------------------------------------

    async def multi_step(
        self,
        target_smiles: str,
        max_depth: int = 10,
        max_routes: int = 5,
    ) -> dict:
        """Run multi-step retrosynthesis route planning."""
        canon = canonicalize_smiles(target_smiles)
        if not canon:
            return {"error": "Invalid SMILES", "target": target_smiles, "routes": []}

        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()

        async with self.mm.lock:
            routes = await loop.run_in_executor(
                None, partial(self._run_multi_step, canon, max_depth, max_routes)
            )

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "target": canon,
            "routes": routes,
            "compute_time_ms": round(elapsed, 1),
        }

    def _run_multi_step(
        self, target_smiles: str, max_depth: int, max_routes: int
    ) -> list[dict]:
        """Synchronous multi-step planning."""
        from rasyn.pipeline.multi_step import MultiStepPlanner
        from rasyn.pipeline.single_step import SingleStepRetro

        llm_model, llm_tokenizer = self.mm.get_llm()
        if llm_model is None:
            return []

        graph_head = self.mm.get_graph_head()
        lg_vocab = self.mm.get_lg_vocab()

        ss_cfg = self._inference_config.get("single_step", {})
        single_step = SingleStepRetro(
            graph_head=graph_head,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            lg_vocab=lg_vocab,
            top_k_bonds=ss_cfg.get("top_k_bonds", 10),
            top_m_lgs=ss_cfg.get("top_m_lgs", 3),
            candidates_per_edit=ss_cfg.get("candidates_per_edit", 5),
            device=self.mm.device,
        )

        ms_cfg = self._inference_config.get("multi_step", {})
        planner = MultiStepPlanner(
            single_step=single_step,
            max_depth=min(max_depth, ms_cfg.get("max_depth", 10)),
            max_nodes=ms_cfg.get("max_nodes", 5000),
            max_time_seconds=ms_cfg.get("max_time_seconds", 300),
            beam_width=ms_cfg.get("beam_width", 20),
        )

        routes = planner.plan(target_smiles, max_routes=max_routes)
        return [self._route_to_dict(r) for r in routes]

    # ------------------------------------------------------------------
    # SMILES validation
    # ------------------------------------------------------------------

    async def validate_smiles(self, smiles: str) -> dict:
        """Validate and return info about a SMILES string."""
        from rasyn.service.molecule_service import MoleculeService
        svc = MoleculeService()
        return svc.validate_and_info(smiles)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _step_to_dict(step, model_source: str) -> dict:
        verification = None
        if step.verifier_results:
            vr = step.verifier_results
            verification = {
                "rdkit_valid": vr.rdkit_valid,
                "forward_match_score": round(vr.forward_match_score, 4),
                "overall_confidence": round(vr.overall_confidence, 4),
            }

        edit_info = None
        if step.edit_explanation:
            ee = step.edit_explanation
            edit_info = {
                "bonds": ee.highlighted_bonds,
                "synthons": ee.synthon_smiles,
                "leaving_groups": ee.leaving_groups_used,
            }

        return {
            "rank": step.rank,
            "reactants_smiles": step.reactants,
            "confidence": round(
                step.verifier_results.overall_confidence
                if step.verifier_results
                else 0.5,
                4,
            ),
            "model_source": model_source,
            "verification": verification,
            "edit_info": edit_info,
        }

    @staticmethod
    def _route_to_dict(route) -> dict:
        steps = []
        for s in route.steps:
            steps.append({
                "product": s.product,
                "reactants": s.reactants,
                "confidence": round(
                    s.verifier_results.overall_confidence
                    if s.verifier_results
                    else 0.5,
                    4,
                ),
            })
        return {
            "steps": steps,
            "total_score": round(route.total_process_score, 4),
            "num_steps": route.num_steps,
            "all_available": route.all_starting_materials_available,
            "starting_materials": route.starting_materials,
        }
