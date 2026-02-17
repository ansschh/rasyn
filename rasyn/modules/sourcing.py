"""Sourcing module — vendor API clients for chemical procurement.

Searches ChemSpace, MolPort, and PubChem for commercial availability
and pricing of starting materials.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# API configuration
CHEMSPACE_API_KEY = os.environ.get("CHEMSPACE_API_KEY", "")
MOLPORT_API_KEY = os.environ.get("MOLPORT_API_KEY", "")
CHEMSPACE_BASE = "https://api.chem-space.com/v3"
MOLPORT_BASE = "https://api.molport.com/api"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Cache duration (1 hour)
CACHE_TTL_SECONDS = 3600


def search_vendors(smiles_list: list[str], timeout: float = 10.0) -> dict:
    """Search multiple vendors for availability and pricing.

    Args:
        smiles_list: List of SMILES strings to look up.
        timeout: HTTP request timeout per vendor.

    Returns:
        Dict with items (list of SourcingItem-like dicts) and summary.
    """
    all_items = []

    for smiles in smiles_list:
        # Try PubChem first (free, no key needed)
        items = _search_pubchem(smiles, timeout=timeout)

        # Try ChemSpace if key available
        if CHEMSPACE_API_KEY:
            items.extend(_search_chemspace(smiles, timeout=timeout))

        # Try MolPort if key available
        if MOLPORT_API_KEY:
            items.extend(_search_molport(smiles, timeout=timeout))

        # If no vendor results, add an "unknown" entry
        if not items:
            items.append({
                "smiles": smiles,
                "vendor": "not_found",
                "catalog_id": None,
                "price_per_gram": None,
                "lead_time_days": None,
                "in_stock": False,
                "url": None,
            })

        all_items.extend(items)

    # Compute summary
    total_cost = None
    in_stock_count = sum(1 for item in all_items if item.get("in_stock"))
    unique_smiles = set(item["smiles"] for item in all_items)
    available_count = sum(
        1 for smi in unique_smiles
        if any(i["in_stock"] for i in all_items if i["smiles"] == smi)
    )

    return {
        "items": all_items,
        "total_estimated_cost": total_cost,
        "summary": {
            "total_compounds": len(unique_smiles),
            "available": available_count,
            "not_available": len(unique_smiles) - available_count,
            "in_stock_offers": in_stock_count,
        },
    }


def _search_pubchem(smiles: str, timeout: float = 10.0) -> list[dict]:
    """Search PubChem for compound info (free, no API key)."""
    try:
        with httpx.Client(timeout=timeout) as client:
            # Look up CID
            resp = client.get(
                f"{PUBCHEM_BASE}/compound/smiles/{smiles}/cids/JSON"
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            if not cids:
                return []

            cid = cids[0]

            # Get compound properties
            resp2 = client.get(
                f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
                "MolecularFormula,MolecularWeight,IUPACName/JSON"
            )
            props = {}
            if resp2.status_code == 200:
                compounds = resp2.json().get("PropertyTable", {}).get("Properties", [])
                if compounds:
                    props = compounds[0]

            return [{
                "smiles": smiles,
                "vendor": "PubChem",
                "catalog_id": f"CID:{cid}",
                "price_per_gram": None,
                "lead_time_days": None,
                "in_stock": True,  # If PubChem has it, it's likely available somewhere
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                "properties": {
                    "formula": props.get("MolecularFormula"),
                    "mw": props.get("MolecularWeight"),
                    "iupac_name": props.get("IUPACName"),
                },
            }]
    except Exception as e:
        logger.debug(f"PubChem search failed for {smiles}: {e}")
        return []


def _search_chemspace(smiles: str, timeout: float = 10.0) -> list[dict]:
    """Search ChemSpace for pricing and availability."""
    if not CHEMSPACE_API_KEY:
        return []

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{CHEMSPACE_BASE}/search",
                headers={
                    "Authorization": f"Bearer {CHEMSPACE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "smiles": smiles,
                    "categories": ["BB", "SC"],  # Building blocks + screening compounds
                    "count": 5,
                },
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            items = []
            for hit in data.get("items", [])[:3]:
                items.append({
                    "smiles": smiles,
                    "vendor": f"ChemSpace/{hit.get('vendorName', 'unknown')}",
                    "catalog_id": hit.get("csId"),
                    "price_per_gram": _parse_price(hit.get("prices", [])),
                    "lead_time_days": _parse_lead_time(hit.get("deliveryDays")),
                    "in_stock": hit.get("inStock", False),
                    "url": hit.get("link"),
                })
            return items
    except Exception as e:
        logger.debug(f"ChemSpace search failed for {smiles}: {e}")
        return []


def _search_molport(smiles: str, timeout: float = 10.0) -> list[dict]:
    """Search MolPort for availability."""
    if not MOLPORT_API_KEY:
        return []

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{MOLPORT_BASE}/chemical-search/search",
                headers={
                    "Authorization": f"Bearer {MOLPORT_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "searchType": "EXACT",
                    "smiles": smiles,
                    "maxResults": 5,
                },
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            items = []
            for hit in data.get("data", {}).get("molecules", [])[:3]:
                items.append({
                    "smiles": smiles,
                    "vendor": f"MolPort/{hit.get('supplierName', 'unknown')}",
                    "catalog_id": hit.get("molportId"),
                    "price_per_gram": _parse_molport_price(hit),
                    "lead_time_days": hit.get("deliveryDays"),
                    "in_stock": hit.get("availability", "") == "In stock",
                    "url": hit.get("url"),
                })
            return items
    except Exception as e:
        logger.debug(f"MolPort search failed for {smiles}: {e}")
        return []


def _parse_price(prices: list) -> float | None:
    """Extract price per gram from ChemSpace price list."""
    for p in prices:
        if p.get("packSize") and "g" in str(p.get("packSize", "")):
            try:
                amount = float(str(p["packSize"]).replace("g", "").strip())
                price = float(p.get("price", 0))
                if amount > 0:
                    return round(price / amount, 2)
            except (ValueError, TypeError):
                continue
    return None


def _parse_lead_time(days) -> int | None:
    """Parse delivery days to integer."""
    if days is None:
        return None
    try:
        return int(days)
    except (ValueError, TypeError):
        return None


def _parse_molport_price(hit: dict) -> float | None:
    """Extract price per gram from MolPort result."""
    try:
        prices = hit.get("prices", [])
        for p in prices:
            amount = p.get("amount", 0)
            unit = p.get("measure", "")
            price = p.get("price", 0)
            if "g" in unit and amount > 0:
                return round(price / amount, 2)
    except (ValueError, TypeError):
        pass
    return None


def find_alternates(smiles: str, top_k: int = 5) -> list[dict]:
    """Find structurally similar alternate building blocks.

    Useful when a specific starting material is unavailable —
    suggests similar compounds that might serve the same role.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from rasyn.planner.inventory import COMMON_BUILDING_BLOCKS
        from rasyn.modules.planner import EXTENDED_BUILDING_BLOCKS

        target_mol = Chem.MolFromSmiles(smiles)
        if target_mol is None:
            return []

        target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)

        scored = []
        for bb_smi in set(COMMON_BUILDING_BLOCKS + EXTENDED_BUILDING_BLOCKS):
            bb_mol = Chem.MolFromSmiles(bb_smi)
            if bb_mol is None:
                continue
            bb_fp = AllChem.GetMorganFingerprintAsBitVect(bb_mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(target_fp, bb_fp)
            if sim > 0.3:  # Minimum similarity threshold
                scored.append((sim, bb_smi))

        scored.sort(reverse=True)
        return [
            {"smiles": smi, "similarity": round(sim, 3), "source": "building_blocks"}
            for sim, smi in scored[:top_k]
            if smi != smiles
        ]
    except Exception as e:
        logger.warning(f"Alternate search failed: {e}")
        return []
