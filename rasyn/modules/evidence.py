"""Evidence retrieval module — find literature precedents for predicted reactions.

Two-pronged approach:
1. **Local index** (fast): Vectorized Tanimoto similarity against pre-indexed
   ReactionIndex table (USPTO-FULL ~1.8M reactions).
   Uses packed binary fingerprints (256 bytes each) for memory efficiency.
2. **Live search** (broad): Real-time queries to Semantic Scholar and OpenAlex
   APIs to find published reactions/papers/patents matching each step.

Both sources are combined and deduplicated before returning.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Popcount lookup table for uint8 — used for packed Tanimoto
# ---------------------------------------------------------------------------
_POPCOUNT_U8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.float32)

# ---------------------------------------------------------------------------
# Module-level cache for local fingerprint index (loaded once per process)
# Uses packed uint8 arrays: 256 bytes per FP → ~460 MB for 1.8M reactions
# (vs ~14.7 GB for unpacked float32)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cached_fps_packed: np.ndarray | None = None   # shape (N, 256), dtype uint8
_cached_bitcounts: np.ndarray | None = None     # shape (N,), dtype float32 — precomputed popcount per FP
_cached_meta: list[dict] | None = None
_cache_loaded = False


def _load_index() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load all reaction fingerprints from the DB into packed numpy arrays.

    Returns (fps_packed, bitcounts, meta_list). Caches for process lifetime.
    Memory: ~460 MB for 1.8M reactions (packed uint8).
    """
    global _cached_fps_packed, _cached_bitcounts, _cached_meta, _cache_loaded

    with _cache_lock:
        if _cache_loaded and _cached_fps_packed is not None:
            return _cached_fps_packed, _cached_bitcounts, _cached_meta  # type: ignore[return-value]

        from sqlalchemy.orm import Session
        from rasyn.db.engine import sync_engine
        from rasyn.db.models import ReactionIndex

        session = Session(sync_engine)
        try:
            count = session.query(ReactionIndex).count()
            if count == 0:
                logger.warning("ReactionIndex table is empty — local evidence search disabled.")
                _cached_fps_packed = np.zeros((0, 256), dtype=np.uint8)
                _cached_bitcounts = np.zeros(0, dtype=np.float32)
                _cached_meta = []
                _cache_loaded = True
                return _cached_fps_packed, _cached_bitcounts, _cached_meta

            logger.info(f"Loading {count:,} reaction fingerprints (packed)...")

            # Load in batches to avoid OOM on very large tables
            BATCH = 100_000
            fps_list = []
            meta = []
            offset = 0

            while True:
                rows = (
                    session.query(ReactionIndex)
                    .order_by(ReactionIndex.id)
                    .offset(offset)
                    .limit(BATCH)
                    .all()
                )
                if not rows:
                    break

                batch_fps = np.zeros((len(rows), 256), dtype=np.uint8)
                for i, row in enumerate(rows):
                    packed = np.frombuffer(row.fingerprint, dtype=np.uint8)
                    batch_fps[i, :len(packed)] = packed[:256]
                    meta.append({
                        "id": row.id,
                        "reaction_smiles": row.reaction_smiles,
                        "product_smiles": row.product_smiles,
                        "reactants_smiles": row.reactants_smiles,
                        "rxn_class": row.rxn_class,
                        "source": row.source or "USPTO",
                        "year": row.year,
                    })

                fps_list.append(batch_fps)
                offset += len(rows)
                logger.info(f"  Loaded {offset:,}/{count:,} fingerprints...")

                if len(rows) < BATCH:
                    break

            fps_packed = np.vstack(fps_list) if fps_list else np.zeros((0, 256), dtype=np.uint8)

            # Precompute bitcounts for all FPs (used in Tanimoto denominator)
            bitcounts = _POPCOUNT_U8[fps_packed].sum(axis=1)

            _cached_fps_packed = fps_packed
            _cached_bitcounts = bitcounts
            _cached_meta = meta
            _cache_loaded = True

            mem_mb = fps_packed.nbytes / 1024 / 1024
            logger.info(f"Loaded {len(meta):,} fingerprints into evidence cache ({mem_mb:.0f} MB packed).")
            return _cached_fps_packed, _cached_bitcounts, _cached_meta
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Fingerprint pack/unpack
# ---------------------------------------------------------------------------

def _pack_fingerprint(fp_array: np.ndarray) -> bytes:
    """Pack a 2048-element binary array into 256 bytes."""
    bits = fp_array.astype(np.uint8)
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

        diff = np.abs(product_arr - reactant_arr)
        return diff
    except Exception as e:
        logger.warning(f"Fingerprint computation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Packed Tanimoto similarity (memory-efficient for 1.8M+ reactions)
# ---------------------------------------------------------------------------

def _batch_tanimoto_packed(query_packed: np.ndarray, db_packed: np.ndarray,
                           db_bitcounts: np.ndarray) -> np.ndarray:
    """Compute Tanimoto similarity using packed uint8 fingerprints.

    Uses popcount lookup table — no unpacking needed.

    Args:
        query_packed: shape (256,) uint8 — packed query fingerprint
        db_packed: shape (N, 256) uint8 — packed database fingerprints
        db_bitcounts: shape (N,) float32 — precomputed popcount for each DB FP

    Returns:
        shape (N,) float32 — Tanimoto similarities
    """
    if db_packed.shape[0] == 0:
        return np.array([], dtype=np.float32)

    # Intersection: popcount(query & db) for each row
    intersection = _POPCOUNT_U8[query_packed & db_packed].sum(axis=1)

    # Query bitcount
    query_bits = _POPCOUNT_U8[query_packed].sum()

    # Union = |A| + |B| - |A & B|
    union = query_bits + db_bitcounts - intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    return sim.astype(np.float32)


# ---------------------------------------------------------------------------
# Local fingerprint search (fast, against pre-indexed reactions)
# ---------------------------------------------------------------------------

def _search_local_index(routes: list[dict], top_k: int = 10) -> list[dict]:
    """Search the local ReactionIndex for similar reactions."""
    try:
        all_fps_packed, all_bitcounts, all_meta = _load_index()
    except Exception as e:
        logger.warning(f"Failed to load local reaction index: {e}")
        return []

    if all_fps_packed.shape[0] == 0:
        return []

    seen_rxn_ids: set[int] = set()
    all_hits: list[tuple[float, dict]] = []

    for route in routes:
        for step in route.get("steps", []):
            product = step.get("product", "")
            reactants = step.get("reactants", [])
            if not product or not reactants:
                continue

            query_fp = compute_reaction_fp(product, reactants)
            if query_fp is None:
                continue

            # Pack the query for packed Tanimoto
            query_packed = np.packbits(query_fp.astype(np.uint8))

            similarities = _batch_tanimoto_packed(query_packed, all_fps_packed, all_bitcounts)

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
                    continue

                all_hits.append((sim, meta))

    all_hits.sort(key=lambda x: x[0], reverse=True)
    top_hits = all_hits[:top_k]

    evidence = []
    for sim, meta in top_hits:
        source = meta["source"]
        # Extract patent number and generate Google Patents URL
        patent_url = None
        patent_number = None
        if "USPTO" in source:
            # Match "US08551963" or bare "05523424" (7-8 digit patent number)
            m = re.search(r'US(\d+)', source)
            if m:
                patent_number = f"US{m.group(1)}"
                patent_url = f"https://patents.google.com/patent/{patent_number}"
            else:
                m = re.search(r'\((\d{7,8})\)', source)
                if m:
                    patent_number = f"US{m.group(1)}"
                    patent_url = f"https://patents.google.com/patent/{patent_number}"

        evidence.append({
            "rxn_smiles": meta["reaction_smiles"],
            "similarity": round(sim, 4),
            "source": source,
            "year": meta.get("year"),
            "title": f"USPTO Patent {patent_number}" if patent_number else None,
            "doi": patent_url,  # Use doi field for the clickable URL
        })

    return evidence


# ---------------------------------------------------------------------------
# Live API search (broad, real-time against public databases)
# ---------------------------------------------------------------------------

def _search_live_apis(target_smiles: str, routes: list[dict], top_k: int = 5) -> list[dict]:
    """Search Semantic Scholar + OpenAlex for published reactions/papers.

    Uses product SMILES from each route step to find:
    - Semantic Scholar: Papers about synthesis of similar compounds
    - OpenAlex: Broader chemistry literature coverage
    """
    import httpx

    evidence = []
    seen_dois: set[str] = set()

    # Collect unique products from all route steps
    products = set()
    for route in routes:
        for step in route.get("steps", []):
            p = step.get("product", "")
            if p:
                products.add(p)
    products.add(target_smiles)

    # Search Semantic Scholar for synthesis papers
    try:
        for smiles in list(products)[:3]:  # limit to 3 queries to control latency
            query = f"synthesis {smiles[:40]} organic chemistry retrosynthesis"
            resp = httpx.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "fields": "title,year,citationCount,externalIds,url",
                    "limit": min(top_k, 5),
                },
                timeout=8.0,
            )
            if resp.status_code != 200:
                continue

            for paper in resp.json().get("data", []):
                ext = paper.get("externalIds", {}) or {}
                doi = ext.get("DOI")
                doi_url = f"https://doi.org/{doi}" if doi else None

                if doi_url and doi_url in seen_dois:
                    continue
                if doi_url:
                    seen_dois.add(doi_url)

                evidence.append({
                    "rxn_smiles": "",  # papers don't have reaction SMILES
                    "similarity": 0.0,  # not a fingerprint match
                    "source": "Semantic Scholar",
                    "year": paper.get("year"),
                    "title": paper.get("title", ""),
                    "doi": doi_url,
                })

            if len(evidence) >= top_k:
                break

    except Exception as e:
        logger.debug(f"Semantic Scholar evidence search failed: {e}")

    # Search OpenAlex for broader coverage
    try:
        query = f"retrosynthesis {target_smiles[:30]} reaction"
        resp = httpx.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "filter": "concepts.id:C185592680",  # Chemistry
                "per_page": min(top_k, 10),
                "sort": "relevance_score:desc",
                "mailto": "team@rasyn.ai",
            },
            timeout=8.0,
        )
        if resp.status_code == 200:
            for work in resp.json().get("results", []):
                doi = work.get("doi")
                if doi and doi in seen_dois:
                    continue
                if doi:
                    seen_dois.add(doi)

                evidence.append({
                    "rxn_smiles": "",
                    "similarity": 0.0,
                    "source": "OpenAlex",
                    "year": work.get("publication_year"),
                    "title": work.get("title", ""),
                    "doi": doi,
                })

                if len(evidence) >= top_k:
                    break

    except Exception as e:
        logger.debug(f"OpenAlex evidence search failed: {e}")

    return evidence[:top_k]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_evidence(
    target_smiles: str,
    routes: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """Find literature evidence for predicted retrosynthetic routes.

    Combines two sources:
    1. Local ReactionIndex (fast Tanimoto similarity) — up to top_k/2 hits
    2. Live API search (Semantic Scholar + OpenAlex) — up to top_k/2 hits

    Args:
        target_smiles: Target molecule SMILES.
        routes: List of route dicts from the retrosynthesis pipeline.
        top_k: Max evidence hits to return (split between local + live).

    Returns:
        List of dicts matching EvidenceHit schema:
        {rxn_smiles, similarity, source, year, title, doi}
    """
    local_k = (top_k + 1) // 2  # ceil division
    live_k = top_k // 2

    # Local fingerprint search
    local_hits = _search_local_index(routes, top_k=local_k)

    # Live API search (runs even if local index is empty — this is the broad coverage)
    live_hits = _search_live_apis(target_smiles, routes, top_k=live_k)

    # Combine: local hits first (they have similarity scores), then live hits
    combined = local_hits + live_hits

    # Deduplicate by reaction_smiles or doi
    seen = set()
    deduped = []
    for hit in combined:
        key = hit.get("doi") or hit.get("rxn_smiles") or hit.get("title", "")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(hit)

    result = deduped[:top_k]

    if result:
        n_local = sum(1 for h in result if h.get("similarity", 0) > 0)
        n_live = len(result) - n_local
        logger.info(f"Evidence search: {len(result)} hits ({n_local} local, {n_live} live)")
    else:
        logger.info("Evidence search: no hits")

    return result


def compute_precedent_score(evidence: list[dict]) -> float:
    """Compute a 0-1 precedent score from evidence hits.

    Weights local fingerprint matches more heavily than API paper hits.
    - Local hits (similarity > 0): max_similarity * 0.5 + count_factor * 0.2
    - Live hits (papers found): count_factor * 0.3

    Args:
        evidence: List of EvidenceHit dicts.

    Returns:
        Float between 0 and 1.
    """
    if not evidence:
        return 0.0

    local_hits = [e for e in evidence if e.get("similarity", 0) > 0]
    live_hits = [e for e in evidence if e.get("similarity", 0) == 0]

    score = 0.0

    if local_hits:
        max_sim = max(e["similarity"] for e in local_hits)
        local_count_factor = min(len(local_hits) / 5.0, 1.0)
        score += max_sim * 0.5 + local_count_factor * 0.2

    if live_hits:
        live_count_factor = min(len(live_hits) / 3.0, 1.0)
        score += live_count_factor * 0.3

    return round(min(score, 1.0), 3)
