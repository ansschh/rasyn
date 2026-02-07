"""ProcessScore v0: combined process-aware scoring.

Combines safety, scalability, and greenness scores into a single
weighted process score. Weights are configurable per user/organization.

Default weights:
  w_safety = 0.30
  w_scale  = 0.30
  w_green  = 0.15
  w_confidence = 0.25
"""

from __future__ import annotations

import logging

from rasyn.schema import ProcessScores, StepCandidate, VerifierResults
from rasyn.scoring.greenness import compute_greenness_score
from rasyn.scoring.safety import compute_safety_score
from rasyn.scoring.scalability import compute_scalability_score

logger = logging.getLogger(__name__)


class ProcessScorer:
    """Compute combined process scores for retrosynthetic step candidates."""

    def __init__(
        self,
        w_safety: float = 0.30,
        w_scale: float = 0.30,
        w_green: float = 0.15,
        w_confidence: float = 0.25,
    ):
        self.w_safety = w_safety
        self.w_scale = w_scale
        self.w_green = w_green
        self.w_confidence = w_confidence

    def score(
        self,
        candidate: StepCandidate,
        verifier_results: VerifierResults | None = None,
    ) -> ProcessScores:
        """Compute process scores for a single candidate.

        Args:
            candidate: StepCandidate with product, reactants, etc.
            verifier_results: Optional verifier results for confidence.

        Returns:
            ProcessScores dataclass with all scores.
        """
        # Safety
        safety_score, safety_details = compute_safety_score(
            candidate.reactants,
            candidate.reagents if candidate.reagents else None,
        )

        # Scalability
        scale_score, scale_details = compute_scalability_score(
            candidate.reactants,
            candidate.conditions if candidate.conditions else None,
            candidate.reagents if candidate.reagents else None,
        )

        # Greenness
        green_score, green_details = compute_greenness_score(
            candidate.product,
            candidate.reactants,
            candidate.reagents if candidate.reagents else None,
        )

        # Confidence from verifier
        confidence = 0.5  # Default
        if verifier_results is not None:
            confidence = verifier_results.overall_confidence
        elif candidate.verifier_results is not None:
            confidence = candidate.verifier_results.overall_confidence

        # Combined score
        total = (
            self.w_safety * safety_score
            + self.w_scale * scale_score
            + self.w_green * green_score
            + self.w_confidence * confidence
        )

        return ProcessScores(
            safety_score=safety_score,
            scalability_score=scale_score,
            greenness_score=green_score,
            confidence_score=confidence,
            total_score=total,
            safety_details=safety_details,
            scalability_details=scale_details,
            greenness_details=green_details,
        )

    def score_and_rank(
        self,
        candidates: list[StepCandidate],
    ) -> list[StepCandidate]:
        """Score all candidates and sort by total process score (descending).

        Updates each candidate's process_scores field in-place.

        Returns:
            Sorted list of candidates (highest process score first).
        """
        for candidate in candidates:
            candidate.process_scores = self.score(candidate, candidate.verifier_results)

        ranked = sorted(candidates, key=lambda c: c.process_scores.total_score, reverse=True)

        if ranked:
            logger.info(
                f"Ranked {len(ranked)} candidates. "
                f"Best score: {ranked[0].process_scores.total_score:.3f}, "
                f"Worst: {ranked[-1].process_scores.total_score:.3f}"
            )

        return ranked
