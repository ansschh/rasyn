"""Single-step retrosynthesis pipeline.

Orchestrates: graph_head -> llm -> verify -> rank

This is the core pipeline function that combines all components into
a single retrosynthetic step prediction.
"""

from __future__ import annotations

import logging

from rasyn.preprocess.canonicalize import canonicalize_smiles
from rasyn.preprocess.featurize import mol_to_pyg_data
from rasyn.schema import EditHypothesis, StepCandidate, StepObject
from rasyn.scoring.ranker import CandidateRanker
from rasyn.verifier.ensemble import VerifierEnsemble

logger = logging.getLogger(__name__)


class SingleStepRetro:
    """Full single-step retrosynthesis pipeline.

    graph_head(product) -> edits
    -> llm(product, edit) -> candidates
    -> verify(candidates)
    -> rank(candidates)
    -> StepObject list
    """

    def __init__(
        self,
        graph_head=None,
        llm_model=None,
        llm_tokenizer=None,
        verifier: VerifierEnsemble | None = None,
        ranker: CandidateRanker | None = None,
        lg_vocab: dict | None = None,
        # Inference params
        top_k_bonds: int = 10,
        top_m_lgs: int = 3,
        candidates_per_edit: int = 5,
        top_n_results: int = 10,
        device: str = "auto",
    ):
        self.graph_head = graph_head
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.verifier = verifier or VerifierEnsemble()
        self.ranker = ranker or CandidateRanker()
        self.lg_vocab = lg_vocab or {}

        self.top_k_bonds = top_k_bonds
        self.top_m_lgs = top_m_lgs
        self.candidates_per_edit = candidates_per_edit
        self.top_n_results = top_n_results
        self.device = device

    def predict(
        self,
        product_smiles: str,
        constraints: list[str] | None = None,
    ) -> list[StepObject]:
        """Run the full single-step retrosynthesis pipeline.

        Args:
            product_smiles: Canonical product SMILES.
            constraints: Optional process constraints.

        Returns:
            Ranked list of StepObject predictions.
        """
        product_smiles = canonicalize_smiles(product_smiles)
        if not product_smiles:
            logger.error("Invalid product SMILES")
            return []

        # Stage 1: Graph head -> edit hypotheses
        edit_hypotheses = self._get_edits(product_smiles)
        logger.info(f"Generated {len(edit_hypotheses)} edit hypotheses")

        # Stage 2: LLM -> candidate reactants per edit
        candidates = self._generate_candidates(product_smiles, edit_hypotheses, constraints)
        logger.info(f"Generated {len(candidates)} total candidates")

        if not candidates:
            return []

        # Stage 3: Verify candidates
        verified = self.verifier.verify_candidates(candidates)
        verified_candidates = [c for c, _ in verified]
        logger.info(f"{len(verified_candidates)} candidates passed verification")

        if not verified_candidates:
            # Fallback: return unverified candidates with low confidence
            verified_candidates = candidates[:self.top_n_results]

        # Stage 4: Score and rank
        results = self.ranker.rank(verified_candidates, top_n=self.top_n_results)
        logger.info(f"Returning {len(results)} ranked results")

        return results

    def _get_edits(self, product_smiles: str) -> list[EditHypothesis]:
        """Get edit hypotheses from the graph head (or fallback)."""
        if self.graph_head is not None:
            data = mol_to_pyg_data(product_smiles)
            if data is not None:
                import torch
                if self.device == "auto":
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    device = torch.device(self.device)
                data = data.to(device)
                return self.graph_head.predict(
                    data,
                    top_k_bonds=self.top_k_bonds,
                    top_m_lgs=self.top_m_lgs,
                    lg_vocab=self.lg_vocab,
                )

        # Fallback: return a single unconditioned edit
        return [EditHypothesis(
            reaction_center_bonds=[],
            synthon_smiles=[],
            leaving_group_options=[],
            confidence=0.5,
            edit_tokens="",
        )]

    def _generate_candidates(
        self,
        product_smiles: str,
        edit_hypotheses: list[EditHypothesis],
        constraints: list[str] | None = None,
    ) -> list[StepCandidate]:
        """Generate candidates from all edit hypotheses."""
        if self.llm_model is not None and self.llm_tokenizer is not None:
            from rasyn.models.llm.generate import generate_for_all_edits
            return generate_for_all_edits(
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                product_smiles=product_smiles,
                edit_hypotheses=edit_hypotheses,
                candidates_per_edit=self.candidates_per_edit,
                constraints=constraints,
            )

        # Fallback: return empty candidates (no LLM available)
        logger.warning("No LLM model loaded. No candidates generated.")
        return []

    def __call__(self, product_smiles: str) -> list[StepObject]:
        """Convenience callable interface for the planner."""
        return self.predict(product_smiles)
