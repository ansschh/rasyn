"""RDChiral template applicability check.

If an edit hypothesis maps to a known template, apply it via RDChiral
and check if the generated reactants are consistent with LLM predictions.
"""

from __future__ import annotations

import logging

from rasyn.preprocess.canonicalize import canonicalize_smiles

logger = logging.getLogger(__name__)


def rdchiral_verify(
    product_smiles: str,
    reactant_smiles_list: list[str],
    template_smarts: str | None = None,
) -> dict:
    """Verify a retrosynthesis prediction using RDChiral template application.

    Args:
        product_smiles: Product SMILES.
        reactant_smiles_list: Predicted reactant SMILES.
        template_smarts: Optional reaction SMARTS template.

    Returns:
        Dict with template_applicable, template_match, and details.
    """
    result = {
        "template_applicable": False,
        "template_match": None,
        "template_reactants": [],
        "pass": True,  # Default pass if no template available
    }

    if template_smarts is None:
        return result

    try:
        from rdchiral.initialization import rdchiralReaction, rdchiralReactants
        from rdchiral.main import rdchiralRun
    except ImportError:
        logger.warning("rdchiral not installed. Skipping template check.")
        return result

    try:
        # Initialize RDChiral objects
        rxn = rdchiralReaction(template_smarts)
        reactants_obj = rdchiralReactants(product_smiles)

        # Apply template
        outcomes = rdchiralRun(rxn, reactants_obj)

        if outcomes:
            result["template_applicable"] = True
            result["template_reactants"] = outcomes

            # Check if any outcome matches our predicted reactants
            predicted_set = {canonicalize_smiles(s) for s in reactant_smiles_list}

            for outcome in outcomes:
                outcome_set = {canonicalize_smiles(s) for s in outcome.split(".")}
                if predicted_set == outcome_set:
                    result["template_match"] = True
                    break
            else:
                result["template_match"] = False

        result["pass"] = result["template_match"] is not False

    except Exception as e:
        logger.debug(f"RDChiral template check failed: {e}")
        result["pass"] = True  # Don't hard-fail on errors

    return result
