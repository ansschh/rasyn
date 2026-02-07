"""Verifier ensemble orchestrator.

Combines RDKit sanity checks, forward round-trip verification, and
RDChiral template checks into a single verification pipeline.

Rejection logic:
  - Hard reject: RDKit parse fails
  - Hard reject: Both forward model AND template check disagree
  - Soft penalty: Only one verifier disagrees (downrank but keep)
"""

from __future__ import annotations

import logging

from rasyn.schema import StepCandidate, VerifierResults
from rasyn.verifier.rdkit_check import rdkit_verify
from rasyn.verifier.roundtrip import RoundTripVerifier
from rasyn.verifier.template_check import rdchiral_verify

logger = logging.getLogger(__name__)


class VerifierEnsemble:
    """Orchestrates multiple verification methods."""

    def __init__(
        self,
        forward_model=None,
        forward_tokenizer=None,
        roundtrip_threshold: float = 0.95,
    ):
        self.roundtrip = RoundTripVerifier(
            forward_model=forward_model,
            forward_tokenizer=forward_tokenizer,
            threshold=roundtrip_threshold,
        )

    def verify(
        self,
        candidate: StepCandidate,
        template_smarts: str | None = None,
    ) -> VerifierResults:
        """Run all verifiers on a candidate and produce combined results.

        Args:
            candidate: StepCandidate with product and reactants.
            template_smarts: Optional reaction SMARTS template.

        Returns:
            VerifierResults with all check outcomes.
        """
        # 1. RDKit sanity checks
        rdkit_results = rdkit_verify(candidate.product, candidate.reactants)

        # 2. Round-trip verification
        roundtrip_results = self.roundtrip.verify(candidate.product, candidate.reactants)

        # 3. Template check
        template_results = rdchiral_verify(
            candidate.product, candidate.reactants, template_smarts,
        )

        # Combine results
        results = VerifierResults(
            rdkit_valid=rdkit_results["overall_pass"],
            forward_match_score=roundtrip_results["forward_score"],
            template_match=template_results["template_match"],
            template_id=template_smarts[:50] if template_smarts else None,
        )

        # Compute overall confidence
        if not results.rdkit_valid:
            # Hard reject
            results.overall_confidence = 0.0
        elif not roundtrip_results["pass"] and template_results["template_match"] is False:
            # Both verifiers disagree -> hard reject
            results.overall_confidence = 0.05
        elif not roundtrip_results["pass"] or template_results["template_match"] is False:
            # Only one disagrees -> soft penalty
            results.overall_confidence = 0.3 + 0.3 * results.forward_match_score
        else:
            # All pass
            results.overall_confidence = 0.5 + 0.5 * results.forward_match_score

        return results

    def verify_candidates(
        self,
        candidates: list[StepCandidate],
        template_smarts: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[tuple[StepCandidate, VerifierResults]]:
        """Verify a list of candidates, filtering and sorting by confidence.

        Args:
            candidates: List of StepCandidate objects.
            template_smarts: Optional template for all candidates.
            min_confidence: Minimum confidence to keep (0.0 = keep all).

        Returns:
            List of (candidate, results) tuples sorted by descending confidence.
        """
        verified = []
        rejected = 0

        for candidate in candidates:
            results = self.verify(candidate, template_smarts)
            candidate.verifier_results = results

            if results.overall_confidence >= min_confidence:
                verified.append((candidate, results))
            else:
                rejected += 1

        # Sort by confidence descending
        verified.sort(key=lambda x: x[1].overall_confidence, reverse=True)

        logger.info(
            f"Verified {len(candidates)} candidates: "
            f"{len(verified)} passed, {rejected} rejected"
        )

        return verified
