"""Evidence retrieval module — find literature precedents for predicted reactions.

Uses RDKit reaction fingerprints for similarity search.
Future: pgvector cosine search against pre-indexed USPTO DRFP embeddings.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _reaction_fingerprint(product: str, reactants: list[str]) -> list[float] | None:
    """Compute a simple reaction fingerprint using Morgan differences.

    This is a lightweight alternative to DRFP — computes the difference
    between product and reactant Morgan fingerprints.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import numpy as np

        product_mol = Chem.MolFromSmiles(product)
        if product_mol is None:
            return None

        product_fp = AllChem.GetMorganFingerprintAsBitVect(product_mol, 2, nBits=2048)
        product_arr = np.zeros(2048)
        for bit in product_fp.GetOnBits():
            product_arr[bit] = 1.0

        reactant_arr = np.zeros(2048)
        for smi in reactants:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            for bit in fp.GetOnBits():
                reactant_arr[bit] = 1.0

        # Difference fingerprint
        diff = product_arr - reactant_arr
        return diff.tolist()
    except Exception as e:
        logger.warning(f"Fingerprint computation failed: {e}")
        return None


def _tanimoto_similarity(fp1: list[float], fp2: list[float]) -> float:
    """Compute Tanimoto similarity between two fingerprints."""
    import numpy as np
    a = np.array(fp1)
    b = np.array(fp2)
    dot = np.dot(a, b)
    denom = np.dot(a, a) + np.dot(b, b) - dot
    if denom == 0:
        return 0.0
    return float(dot / denom)


def find_evidence(target_smiles: str, routes: list[dict], top_k: int = 10) -> list[dict]:
    """Find literature evidence for predicted retrosynthetic routes.

    Currently returns an empty list — will be populated when
    we index USPTO reactions into pgvector with DRFP embeddings.

    Args:
        target_smiles: Target molecule SMILES.
        routes: List of route dicts from the retrosynthesis pipeline.
        top_k: Max evidence hits to return.

    Returns:
        List of dicts matching EvidenceHit schema.
    """
    # TODO: Implement pgvector search once USPTO reactions are indexed
    # For now, return empty — the frontend handles empty evidence gracefully
    #
    # Future implementation:
    # 1. For each reaction in routes, compute DRFP fingerprint
    # 2. Query pgvector: SELECT * FROM reactions ORDER BY embedding <=> $fp LIMIT $k
    # 3. Return top-K similar reactions with metadata

    return []
