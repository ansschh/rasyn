"""Candidate reranker interface.

Provides a clean interface for reranking step candidates using
process scores. This is the module that will later be upgraded
to a learned ranker (ProcessScore v1).
"""

from __future__ import annotations

import logging

from rasyn.schema import ProcessScores, StepCandidate, StepObject, EditExplanation
from rasyn.scoring.process_score import ProcessScorer

logger = logging.getLogger(__name__)


class CandidateRanker:
    """Rank and filter step candidates to produce final StepObjects."""

    def __init__(self, scorer: ProcessScorer | None = None):
        self.scorer = scorer or ProcessScorer()

    def rank(
        self,
        candidates: list[StepCandidate],
        top_n: int = 10,
    ) -> list[StepObject]:
        """Score, rank, and convert candidates to final StepObjects.

        Args:
            candidates: List of verified StepCandidate objects.
            top_n: Number of top candidates to return.

        Returns:
            List of StepObject (the pipeline's output unit).
        """
        # Score and rank
        ranked = self.scorer.score_and_rank(candidates)

        # Convert top-N to StepObjects
        results = []
        for rank_idx, candidate in enumerate(ranked[:top_n]):
            # Build edit explanation
            edit_explanation = None
            if candidate.edit_hypothesis is not None:
                edit_explanation = EditExplanation(
                    highlighted_bonds=candidate.edit_hypothesis.reaction_center_bonds,
                    synthon_smiles=candidate.edit_hypothesis.synthon_smiles,
                    leaving_groups_used=(
                        candidate.edit_hypothesis.leaving_group_options[0]
                        if candidate.edit_hypothesis.leaving_group_options
                        else []
                    ),
                )

            # Collect risk tags
            risk_tags = []
            if candidate.process_scores and candidate.process_scores.safety_details:
                risk_tags = candidate.process_scores.safety_details.get("risk_tags", [])

            step = StepObject(
                product=candidate.product,
                reactants=candidate.reactants,
                reagents=candidate.reagents,
                conditions=candidate.conditions,
                edit_explanation=edit_explanation,
                verifier_results=candidate.verifier_results,
                process_scores=candidate.process_scores,
                risk_tags=risk_tags,
                rank=rank_idx + 1,
            )
            results.append(step)

        return results
