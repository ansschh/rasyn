"""Evidence retrieval module — find literature precedents for predicted reactions.

Uses Morgan difference fingerprints for Tanimoto similarity search against
a pre-indexed ReactionIndex table (USPTO-50K, 37K reactions).
"""

from __future__ import annotations

import logging
import struct
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache (loaded once per process, thread-safe)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cached_fps: np.ndarray | None = None       # shape (N, 2048), dtype float32
_cached_meta: list[dict] | None = None       # list of {id, reaction_smiles, product_smiles, reactants_smiles, rxn_class, source, year}
_cache_loaded = False


def _load_index() -> tuple[np.ndarray, list[dict]]:
    """Load all reaction fingerprints from the DB into a numpy array.

    Returns (fps_array, meta_list). Caches result for the process lifetime.
    """
    global _cached_fps, _cached_meta, _cache_loaded

    with _cache_lock:
        if _cache_loaded and _cached_fps is not None:
            return _cached_fps, _cached_meta  # type: ignore[return-value]

        from sqlalchemy.orm import Session
        from rasyn.db.engine import sync_engine
        from rasyn.db.models import ReactionIndex

        session = Session(sync_engine)
        try:
            rows = session.query(ReactionIndex).all()
            if not rows:
                logger.warning("ReactionIndex table is empty — no evidence will be found.")
                _cached_fps = np.zeros((0, 2048), dtype=np.float32)
                _cached_meta = []
                _cache_loaded = True
                return _cached_fps, _cached_meta

            fps = np.zeros((len(rows), 2048), dtype=np.float32)
            meta = []

            for i, row in enumerate(rows):
                # Unpack 256 bytes → 2048 bits
                fp_bits = _unpack_fingerprint(row.fingerprint)
                fps[i] = fp_bits
                meta.append({
                    "id": row.id,
                    "reaction_smiles": row.reaction_smiles,
                    "product_smiles": row.product_smiles,
                    "reactants_smiles": row.reactants_smiles,
                    "rxn_class": row.rxn_class,
                    "source": row.source or "USPTO-50K",
                    "year": row.year,
                })

            _cached_fps = fps
            _cached_meta = meta
            _cache_loaded = True
            logger.info(f"Loaded {len(rows)} reaction fingerprints into evidence cache.")
            return _cached_fps, _cached_meta
        finally:
            session.close()


def _pack_fingerprint(fp_array: np.ndarray) -> bytes:
    """Pack a 2048-element binary array into 256 bytes."""
    bits = fp_array.astype(np.uint8)
    # Pack 8 bits per byte
    packed = np.packbits(bits)
    return packed.tobytes()


def _unpack_fingerprint(fp_bytes: bytes) -> np.ndarray:
    """Unpack 256 bytes into a 2048-element float32 array."""
    packed = np.frombuffer(fp_bytes, dtype=np.uint8)
    bits = np.unpackbits(packed).astype(np.float32)
    return bits[:2048]


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

def compute_reaction_fp(product_smiles: str, reactants_smiles: list[str]) -> np.ndarray | None:
    """Compute Morgan difference fingerprint for a reaction.

    Returns a 2048-element float32 array, or None if SMILES are invalid.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        product_mol = Chem.MolFromSmiles(product_smiles)
        if product_mol is None:
            return None

        product_fp = AllChem.GetMorganFingerprintAsBitVect(product_mol, 2, nBits=2048)
        product_arr = np.zeros(2048, dtype=np.float32)
        for bit in product_fp.GetOnBits():
            product_arr[bit] = 1.0

        reactant_arr = np.zeros(2048, dtype=np.float32)
        for smi in reactants_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            for bit in fp.GetOnBits():
                reactant_arr[bit] = 1.0

        # Difference fingerprint: |product - reactants|
        diff = np.abs(product_arr - reactant_arr)
        return diff
    except Exception as e:
        logger.warning(f"Fingerprint computation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Vectorized Tanimoto similarity
# ---------------------------------------------------------------------------

def _batch_tanimoto(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Compute Tanimoto similarity between a query FP and all database FPs.

    Args:
        query: shape (2048,) float32
        database: shape (N, 2048) float32

    Returns:
        shape (N,) array of Tanimoto similarities.
    """
    if database.shape[0] == 0:
        return np.array([], dtype=np.float32)

    # dot(query, each row) = intersection count
    intersection = database @ query                         # (N,)
    query_bits = query.sum()                                # scalar
    db_bits = database.sum(axis=1)                          # (N,)
    union = query_bits + db_bits - intersection             # (N,)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    return sim.astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_evidence(
    target_smiles: str,
    routes: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """Find literature evidence for predicted retrosynthetic routes.

    For each reaction step in the routes, computes a Morgan difference
    fingerprint and searches the ReactionIndex for similar known reactions.

    Args:
        target_smiles: Target molecule SMILES (unused directly, kept for API compat).
        routes: List of route dicts from the retrosynthesis pipeline.
        top_k: Max evidence hits to return across all steps.

    Returns:
        List of dicts matching EvidenceHit schema:
        {rxn_smiles, similarity, source, year, title, doi}
    """
    try:
        all_fps, all_meta = _load_index()
    except Exception as e:
        logger.error(f"Failed to load reaction index: {e}")
        return []

    if all_fps.shape[0] == 0:
        return []

    # Collect all steps across all routes
    seen_rxn_ids: set[int] = set()
    all_hits: list[tuple[float, dict]] = []  # (similarity, meta_dict)

    for route in routes:
        for step in route.get("steps", []):
            product = step.get("product", "")
            reactants = step.get("reactants", [])
            if not product or not reactants:
                continue

            query_fp = compute_reaction_fp(product, reactants)
            if query_fp is None:
                continue

            similarities = _batch_tanimoto(query_fp, all_fps)

            # Get top matches for this step (more than top_k to allow dedup)
            n_candidates = min(top_k * 2, len(similarities))
            top_indices = np.argpartition(similarities, -n_candidates)[-n_candidates:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

            for idx in top_indices:
                meta = all_meta[idx]
                rxn_id = meta["id"]
                if rxn_id in seen_rxn_ids:
                    continue
                seen_rxn_ids.add(rxn_id)

                sim = float(similarities[idx])
                if sim < 0.1:
                    continue  # too dissimilar to be useful

                all_hits.append((sim, meta))

    # Sort by similarity descending, take top_k
    all_hits.sort(key=lambda x: x[0], reverse=True)
    top_hits = all_hits[:top_k]

    # Format as EvidenceHit dicts
    evidence = []
    for sim, meta in top_hits:
        evidence.append({
            "rxn_smiles": meta["reaction_smiles"],
            "similarity": round(sim, 4),
            "source": meta["source"],
            "year": meta.get("year"),
            "title": None,  # USPTO doesn't have titles
            "doi": None,
        })

    logger.info(f"Evidence search: {len(evidence)} hits (top similarity: "
                f"{evidence[0]['similarity']:.3f})" if evidence else "no hits")
    return evidence


def compute_precedent_score(evidence: list[dict]) -> float:
    """Compute a 0-1 precedent score from evidence hits.

    Formula: max_similarity * 0.7 + min(count / 5, 1.0) * 0.3

    Args:
        evidence: List of EvidenceHit dicts.

    Returns:
        Float between 0 and 1.
    """
    if not evidence:
        return 0.0

    max_sim = max(e.get("similarity", 0) for e in evidence)
    count = len(evidence)
    score = max_sim * 0.7 + min(count / 5.0, 1.0) * 0.3
    return round(min(score, 1.0), 3)
