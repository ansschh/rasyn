"""Literature discovery module — search PubChem, OpenAlex, Semantic Scholar.

Finds published precedents for reactions, provides evidence cards,
and clusters similar reactions using fingerprints.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
OPENALEX_BASE = "https://api.openalex.org"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"


async def search_literature(
    query: str,
    smiles: str | None = None,
    max_results: int = 20,
    timeout: float = 15.0,
) -> dict:
    """Run parallel literature search across multiple APIs.

    Args:
        query: Text search query (reaction description, keywords, etc.)
        smiles: Optional target molecule SMILES for structure-based search.
        max_results: Max results per source.
        timeout: HTTP timeout per request.

    Returns:
        Dict with papers, compounds, and summary.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _search_openalex(client, query, max_results),
            _search_semantic_scholar(client, query, max_results),
        ]
        if smiles:
            tasks.append(_search_pubchem_compound(client, smiles))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    openalex = results[0] if not isinstance(results[0], Exception) else []
    semscho = results[1] if not isinstance(results[1], Exception) else []
    pubchem = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else {}

    # Deduplicate by DOI
    seen_dois = set()
    papers = []
    for paper in openalex + semscho:
        doi = paper.get("doi")
        if doi and doi in seen_dois:
            continue
        if doi:
            seen_dois.add(doi)
        papers.append(paper)

    # Sort by relevance (citation count + recency)
    papers.sort(key=lambda p: _relevance_score(p), reverse=True)
    papers = papers[:max_results]

    return {
        "papers": papers,
        "compound_info": pubchem,
        "sources_queried": ["OpenAlex", "Semantic Scholar"] + (["PubChem"] if smiles else []),
        "total_results": len(papers),
    }


def search_literature_sync(
    query: str,
    smiles: str | None = None,
    max_results: int = 20,
    timeout: float = 15.0,
) -> dict:
    """Synchronous wrapper for literature search (for Celery worker)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    search_literature(query, smiles, max_results, timeout)
                )
                return future.result(timeout=timeout + 5)
        return asyncio.run(search_literature(query, smiles, max_results, timeout))
    except Exception as e:
        logger.warning(f"Sync literature search failed: {e}")
        return {"papers": [], "compound_info": {}, "sources_queried": [], "total_results": 0}


async def _search_openalex(client: httpx.AsyncClient, query: str, max_results: int) -> list[dict]:
    """Search OpenAlex for papers (free, 250M+ works)."""
    try:
        resp = await client.get(
            f"{OPENALEX_BASE}/works",
            params={
                "search": query,
                "filter": "concepts.id:C185592680",  # Chemistry concept
                "per_page": min(max_results, 25),
                "sort": "relevance_score:desc",
                "mailto": "team@rasyn.ai",
            },
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        papers = []
        for work in data.get("results", []):
            papers.append({
                "title": work.get("title", ""),
                "authors": ", ".join(
                    a.get("author", {}).get("display_name", "")
                    for a in (work.get("authorships", []))[:3]
                ),
                "year": work.get("publication_year"),
                "doi": work.get("doi"),
                "citation_count": work.get("cited_by_count", 0),
                "source": "OpenAlex",
                "journal": _extract_journal(work),
                "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
                "url": work.get("doi") or work.get("id"),
                "source_type": "paper",
            })
        return papers
    except Exception as e:
        logger.debug(f"OpenAlex search failed: {e}")
        return []


async def _search_semantic_scholar(client: httpx.AsyncClient, query: str, max_results: int) -> list[dict]:
    """Search Semantic Scholar (free, returns embeddings)."""
    try:
        resp = await client.get(
            f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
            params={
                "query": query,
                "fields": "title,abstract,authors,year,citationCount,externalIds,journal,url",
                "limit": min(max_results, 20),
            },
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        papers = []
        for paper in data.get("data", []):
            ext = paper.get("externalIds", {}) or {}
            doi = ext.get("DOI")
            if doi:
                doi = f"https://doi.org/{doi}"
            papers.append({
                "title": paper.get("title", ""),
                "authors": ", ".join(
                    a.get("name", "") for a in (paper.get("authors", []))[:3]
                ),
                "year": paper.get("year"),
                "doi": doi,
                "citation_count": paper.get("citationCount", 0),
                "source": "Semantic Scholar",
                "journal": (paper.get("journal") or {}).get("name"),
                "abstract": paper.get("abstract"),
                "url": paper.get("url") or doi,
                "source_type": "paper",
            })
        return papers
    except Exception as e:
        logger.debug(f"Semantic Scholar search failed: {e}")
        return []


async def _search_pubchem_compound(client: httpx.AsyncClient, smiles: str) -> dict:
    """Search PubChem for compound information."""
    try:
        resp = await client.get(
            f"{PUBCHEM_BASE}/compound/smiles/{smiles}/cids/JSON"
        )
        if resp.status_code != 200:
            return {}

        cids = resp.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return {}

        cid = cids[0]

        # Get properties
        resp2 = await client.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
            "MolecularFormula,MolecularWeight,IUPACName,InChIKey,XLogP,ExactMass/JSON"
        )
        props = {}
        if resp2.status_code == 200:
            compounds = resp2.json().get("PropertyTable", {}).get("Properties", [])
            if compounds:
                props = compounds[0]

        return {
            "cid": cid,
            "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "formula": props.get("MolecularFormula"),
            "mw": props.get("MolecularWeight"),
            "iupac_name": props.get("IUPACName"),
            "inchi_key": props.get("InChIKey"),
            "xlogp": props.get("XLogP"),
        }
    except Exception as e:
        logger.debug(f"PubChem compound search failed: {e}")
        return {}


def _relevance_score(paper: dict) -> float:
    """Compute a simple relevance score for ranking papers."""
    score = 0.0
    # Citation count (log scale)
    cites = paper.get("citation_count", 0)
    if cites > 0:
        import math
        score += min(math.log10(cites + 1) / 4, 1.0) * 0.4

    # Recency (newer papers score higher)
    year = paper.get("year")
    if year:
        recency = max(0, min(1, (year - 2000) / 25))
        score += recency * 0.3

    # Has abstract (more information = more useful)
    if paper.get("abstract"):
        score += 0.2

    # Has DOI (more reliable)
    if paper.get("doi"):
        score += 0.1

    return score


def _extract_journal(work: dict) -> str | None:
    """Extract journal name from OpenAlex work object."""
    source = work.get("primary_location", {})
    if source:
        source_obj = source.get("source")
        if source_obj:
            return source_obj.get("display_name")
    return None


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return None
    try:
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        return " ".join(word for _, word in word_positions)
    except Exception:
        return None
