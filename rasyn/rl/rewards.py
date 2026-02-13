"""Chemical reward functions for RL fine-tuning of retrosynthesis models.

Computes multi-component rewards for predicted reactants:
  - validity_reward: Can RDKit parse all predicted SMILES? (0/1)
  - roundtrip_reward: Does forward model reproduce the product? (0-1)
  - tanimoto_reward: Fingerprint similarity of forward prediction (0-1)
  - exact_match_bonus: Exactly matches ground truth? (0/1)

Used by both RetroTx PPO trainer and LLM PPO trainer (via trl).
"""

from __future__ import annotations

import logging

import torch

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDKIT_AVAILABLE = False

from rasyn.preprocess.canonicalize import canonicalize_smiles

logger = logging.getLogger(__name__)


def is_valid_smiles(smi: str) -> bool:
    """Check if SMILES is valid."""
    if not RDKIT_AVAILABLE:
        return bool(smi and smi.strip())
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        return mol is not None
    except Exception:
        return False


def tanimoto_similarity(smi1: str, smi2: str, radius: int = 2) -> float:
    """Compute Tanimoto similarity between two SMILES."""
    if not RDKIT_AVAILABLE:
        return 0.0
    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def normalize_reactants(smiles_str: str) -> str:
    """Canonicalize and sort reactant components."""
    parts = smiles_str.replace(" . ", ".").replace(" .", ".").replace(". ", ".").split(".")
    canon = [canonicalize_smiles(p.strip(), remove_mapping=True) for p in parts if p.strip()]
    canon = [c for c in canon if c]
    return ".".join(sorted(canon))


class ChemicalRewardFunction:
    """Multi-component chemical reward for RL fine-tuning.

    Args:
        forward_model: Optional forward model (reactants -> product).
        forward_tokenizer: Tokenizer for the forward model.
        device: Torch device.
        w_validity: Weight for validity reward (default 0.3).
        w_roundtrip: Weight for round-trip reward (default 0.3).
        w_tanimoto: Weight for Tanimoto reward (default 0.2).
        w_exact: Weight for exact match bonus (default 0.2).
    """

    def __init__(
        self,
        forward_model=None,
        forward_tokenizer=None,
        device: str = "cpu",
        w_validity: float = 0.3,
        w_roundtrip: float = 0.3,
        w_tanimoto: float = 0.2,
        w_exact: float = 0.2,
    ):
        self.forward_model = forward_model
        self.forward_tokenizer = forward_tokenizer
        self.device = device
        self.w_validity = w_validity
        self.w_roundtrip = w_roundtrip
        self.w_tanimoto = w_tanimoto
        self.w_exact = w_exact

    def compute_reward(
        self,
        product: str,
        predicted_reactants: str,
        gt_reactants: str | None = None,
    ) -> dict[str, float]:
        """Compute chemical rewards for a prediction.

        Args:
            product: Target product SMILES.
            predicted_reactants: Predicted reactant SMILES (dot-separated).
            gt_reactants: Optional ground truth reactants for exact match bonus.

        Returns:
            Dict with individual rewards and combined 'total' reward.
        """
        rewards = {
            "validity": 0.0,
            "roundtrip": 0.0,
            "tanimoto": 0.0,
            "exact_match": 0.0,
            "total": 0.0,
        }

        if not predicted_reactants or not predicted_reactants.strip():
            return rewards

        # 1. Validity reward: all components parseable by RDKit
        parts = predicted_reactants.replace(" . ", ".").split(".")
        all_valid = all(is_valid_smiles(p.strip()) for p in parts if p.strip())
        rewards["validity"] = 1.0 if all_valid else 0.0

        if not all_valid:
            # Invalid prediction — early return with small penalty
            rewards["total"] = -0.5
            return rewards

        # 2. Exact match bonus (if ground truth provided)
        if gt_reactants:
            pred_norm = normalize_reactants(predicted_reactants)
            gt_norm = normalize_reactants(gt_reactants)
            if pred_norm and gt_norm and pred_norm == gt_norm:
                rewards["exact_match"] = 1.0

        # 3. Round-trip + Tanimoto (requires forward model)
        if self.forward_model is not None and self.forward_tokenizer is not None:
            fwd_product = self._forward_predict(predicted_reactants)
            if fwd_product:
                product_canon = canonicalize_smiles(product, remove_mapping=True)
                fwd_canon = canonicalize_smiles(fwd_product, remove_mapping=True)

                if product_canon and fwd_canon:
                    # Round-trip: exact match of forward prediction
                    if product_canon == fwd_canon:
                        rewards["roundtrip"] = 1.0
                    else:
                        rewards["roundtrip"] = 0.0

                    # Tanimoto: fingerprint similarity
                    rewards["tanimoto"] = tanimoto_similarity(product, fwd_product)
        else:
            # No forward model — use Tanimoto between predicted reactants and product as proxy
            # This is a weak signal but better than nothing
            rewards["tanimoto"] = 0.5 if all_valid else 0.0

        # Combined reward
        rewards["total"] = (
            self.w_validity * rewards["validity"]
            + self.w_roundtrip * rewards["roundtrip"]
            + self.w_tanimoto * rewards["tanimoto"]
            + self.w_exact * rewards["exact_match"]
        )

        return rewards

    def compute_batch_rewards(
        self,
        products: list[str],
        predictions: list[str],
        gt_reactants: list[str | None] | None = None,
    ) -> list[dict[str, float]]:
        """Compute rewards for a batch of predictions."""
        if gt_reactants is None:
            gt_reactants = [None] * len(products)

        return [
            self.compute_reward(prod, pred, gt)
            for prod, pred, gt in zip(products, predictions, gt_reactants)
        ]

    def _forward_predict(self, reactants_str: str) -> str | None:
        """Run forward model to predict product from reactants."""
        if self.forward_model is None:
            return None

        try:
            src_ids = torch.tensor(
                [self.forward_tokenizer.encode(reactants_str, max_len=256)],
                dtype=torch.long, device=self.device,
            )
            with torch.no_grad():
                pred_ids = self.forward_model.generate_greedy(
                    src_ids,
                    self.forward_tokenizer.bos_token_id,
                    self.forward_tokenizer.eos_token_id,
                    max_len=256,
                )
            return self.forward_tokenizer.decode(pred_ids[0])
        except Exception as e:
            logger.debug(f"Forward prediction failed: {e}")
            return None
