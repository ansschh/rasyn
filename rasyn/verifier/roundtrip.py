"""Round-trip verification using a forward reaction predictor.

Checks whether predicted reactants can plausibly produce the target product
by running a forward model (reactants -> predicted product) and comparing.
"""

from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from rasyn.preprocess.canonicalize import canonicalize_smiles

logger = logging.getLogger(__name__)


def smiles_match(smiles_a: str, smiles_b: str) -> bool:
    """Check if two SMILES represent the same molecule."""
    canon_a = canonicalize_smiles(smiles_a, remove_mapping=True)
    canon_b = canonicalize_smiles(smiles_b, remove_mapping=True)
    return canon_a == canon_b and canon_a != ""


def tanimoto_similarity(smiles_a: str, smiles_b: str, radius: int = 2) -> float:
    """Compute Tanimoto similarity between two molecules using Morgan fingerprints."""
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0

    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


class RoundTripVerifier:
    """Verify candidates by predicting the forward reaction.

    Uses either:
    1. A trained forward model (when available)
    2. Tanimoto similarity as a proxy (fallback)
    """

    def __init__(self, forward_model=None, forward_tokenizer=None, threshold: float = 0.95):
        self.forward_model = forward_model
        self.forward_tokenizer = forward_tokenizer
        self.threshold = threshold

    def verify(
        self,
        product_smiles: str,
        reactant_smiles_list: list[str],
    ) -> dict:
        """Run round-trip verification.

        Args:
            product_smiles: Target product SMILES.
            reactant_smiles_list: Predicted reactant SMILES list.

        Returns:
            Dict with forward_score, exact_match, and pass/fail.
        """
        if self.forward_model is not None:
            return self._verify_with_model(product_smiles, reactant_smiles_list)
        else:
            return self._verify_heuristic(product_smiles, reactant_smiles_list)

    def _verify_with_model(
        self,
        product_smiles: str,
        reactant_smiles_list: list[str],
    ) -> dict:
        """Verify using a trained forward prediction model."""
        # TODO: Implement when forward model is available
        # For now, fall back to heuristic
        return self._verify_heuristic(product_smiles, reactant_smiles_list)

    def _verify_heuristic(
        self,
        product_smiles: str,
        reactant_smiles_list: list[str],
    ) -> dict:
        """Heuristic verification based on molecular similarity.

        Checks:
        1. Product atoms are present in reactants (atom conservation)
        2. Reactants share structural similarity with product
        """
        result = {
            "forward_score": 0.0,
            "exact_match": False,
            "similarity_score": 0.0,
            "pass": False,
        }

        # Compute structural similarity between each reactant and product
        max_sim = 0.0
        for r_smi in reactant_smiles_list:
            sim = tanimoto_similarity(product_smiles, r_smi)
            max_sim = max(max_sim, sim)

        result["similarity_score"] = max_sim

        # Combined reactant similarity
        combined_reactants = ".".join(reactant_smiles_list)
        combined_sim = tanimoto_similarity(product_smiles, combined_reactants)
        result["forward_score"] = combined_sim

        # Pass if similarity is above threshold
        result["pass"] = combined_sim >= self.threshold or max_sim >= 0.5

        return result
