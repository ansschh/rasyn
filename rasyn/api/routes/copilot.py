"""Co-pilot chat endpoint — Claude-powered chemistry assistant.

Context-aware: pulls full pipeline state (safety, evidence, sourcing, green chem)
from the DB when a job_id is provided, and does evidence RAG to include similar
known reactions when the user asks about specific chemistry.

Requires ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["copilot"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CopilotMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class CopilotChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    context: dict | None = Field(None, description="Current route/molecule context")
    history: list[CopilotMessage] = Field(default_factory=list, description="Previous messages")


class CopilotChatResponse(BaseModel):
    reply: str
    model: str


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Rasyn Chemistry Co-Pilot — an expert synthetic chemistry assistant embedded in a retrosynthesis planning platform called Rasyn.

## About Rasyn
Rasyn is an AI-powered retrosynthesis platform that predicts synthetic routes to target molecules. It uses:
- **RetroTransformer v2**: A Transformer-based model with copy mechanism, trained on USPTO-50K (69.7% top-1 accuracy)
- **LLM (RSGPT)**: A 3.2B parameter language model fine-tuned for retrosynthesis (61.7% top-1 accuracy)
- **Multi-step A* search**: Finds complete routes to purchasable starting materials
- **Composite scoring**: Routes scored on confidence, safety, green chemistry, precedent, availability, and efficiency
- **Evidence search**: Morgan fingerprint similarity against 37K known USPTO reactions

## Your Capabilities
You have access to the user's FULL pipeline context, including:
- Target molecule SMILES and properties
- All predicted retrosynthetic routes with step details
- Safety screening results (PAINS/BRENK structural alerts, drug-likeness)
- Green chemistry metrics (atom economy, E-factor, solvent sustainability)
- Literature evidence (similar known reactions from USPTO with Tanimoto similarity scores)
- Sourcing data (vendor availability and pricing for starting materials)
- Score breakdowns (confidence, safety, green, precedent, availability, efficiency)

## Your Role
- Help chemists understand and evaluate predicted retrosynthetic routes
- Explain reaction mechanisms, selectivity, and feasibility for specific steps
- Identify potential issues: competing reactions, selectivity problems, safety hazards, scalability concerns
- Suggest modifications to improve routes (better reagents, milder conditions, higher-yielding alternatives)
- Answer questions about reagents, solvents, catalysts, workup, and purification
- Interpret safety alerts and explain their significance
- Help choose between alternative routes based on practical considerations
- Provide practical lab guidance for executing predicted syntheses

## Guidelines
- Be concise and precise — chemists value accuracy over verbosity
- Use standard chemistry nomenclature and IUPAC names where helpful
- Reference specific SMILES, step numbers, and scores from the context
- If you're uncertain about something, say so — never fabricate reaction conditions, yields, or references
- When discussing a reaction step, mention its model source and confidence score
- Format chemical equations and lists clearly using markdown
- Reference well-known named reactions when applicable (Suzuki, Buchwald-Hartwig, etc.)
- When safety alerts are present, always mention them proactively
- Keep responses focused and actionable"""


# ---------------------------------------------------------------------------
# Context building helpers
# ---------------------------------------------------------------------------

def _build_context_from_job(job_id: str) -> str:
    """Pull full PlanResult from DB and format as structured context."""
    try:
        from sqlalchemy.orm import Session
        from rasyn.db.engine import sync_engine
        from rasyn.db.models import Job

        session = Session(sync_engine)
        try:
            job = session.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
            if not job or not job.result:
                return ""
            result = job.result
        finally:
            session.close()

        parts = []
        parts.append(f"## Pipeline Results (Job {job_id[:8]}...)")
        parts.append(f"Target SMILES: {result.get('smiles', 'N/A')}")
        parts.append(f"Status: {result.get('status', 'N/A')}")
        parts.append(f"Compute time: {result.get('compute_time_ms', 0):.0f}ms")

        # Routes
        routes = result.get("routes", [])
        if routes:
            parts.append(f"\n### Routes ({len(routes)} found)")
            for route in routes:
                rank = route.get("rank", "?")
                score = route.get("overall_score", 0)
                n_steps = route.get("num_steps", len(route.get("steps", [])))
                purchasable = route.get("all_purchasable", False)
                parts.append(f"\n**Route #{rank}** — Score: {score:.0%}, Steps: {n_steps}, "
                           f"All purchasable: {'Yes' if purchasable else 'No'}")

                # Score breakdown
                sb = route.get("score_breakdown", {})
                if sb:
                    parts.append(f"  Score breakdown: confidence={sb.get('roundtrip_confidence', 0):.0%}, "
                               f"safety={sb.get('safety', 0):.0%}, green={sb.get('green_chemistry', 0):.0%}, "
                               f"precedent={sb.get('precedent', 0):.0%}, availability={sb.get('availability', 0):.0%}, "
                               f"efficiency={sb.get('step_efficiency', 0):.0%}")

                # Steps
                for i, step in enumerate(route.get("steps", [])):
                    product = step.get("product", "?")
                    reactants = step.get("reactants", [])
                    model = step.get("model", "?")
                    conf = step.get("score", 0)
                    rxn_class = step.get("rxn_class", "")
                    parts.append(f"  Step {i+1}: {' + '.join(reactants)} → {product}")
                    parts.append(f"    Model: {model}, Confidence: {conf:.0%}"
                               + (f", Class: {rxn_class}" if rxn_class else ""))

                    # Conditions
                    conditions = step.get("conditions")
                    if conditions and isinstance(conditions, dict):
                        cond_str = ", ".join(f"{k}={v}" for k, v in conditions.items() if v)
                        if cond_str:
                            parts.append(f"    Conditions: {cond_str}")

                # Starting materials
                sm = route.get("starting_materials", [])
                if sm:
                    parts.append(f"  Starting materials: {', '.join(sm)}")

        # Safety
        safety = result.get("safety")
        if safety:
            parts.append("\n### Safety Screening")
            alerts = safety.get("alerts", [])
            if alerts:
                for a in alerts:
                    parts.append(f"  ⚠ {a.get('name', '?')}: {a.get('description', '')} "
                               f"(severity: {a.get('severity', '?')})")
            else:
                parts.append("  No structural alerts detected.")

            dl = safety.get("druglikeness", {})
            if dl:
                parts.append(f"  Drug-likeness: MW={dl.get('mw', 0):.1f}, LogP={dl.get('logp', 0):.2f}, "
                           f"HBD={dl.get('hbd', 0)}, HBA={dl.get('hba', 0)}, "
                           f"Lipinski: {'Pass' if dl.get('passes_lipinski') else 'Fail'}")
                violations = dl.get("violations", [])
                if violations:
                    parts.append(f"  Violations: {', '.join(violations)}")

            tox = safety.get("tox_flags", [])
            if tox:
                parts.append(f"  Toxicity flags: {', '.join(tox)}")

        # Evidence
        evidence = result.get("evidence", [])
        if evidence:
            parts.append(f"\n### Literature Evidence ({len(evidence)} similar reactions)")
            for e in evidence[:5]:
                parts.append(f"  - {e.get('rxn_smiles', '?')} "
                           f"(similarity: {e.get('similarity', 0):.1%}, "
                           f"source: {e.get('source', '?')}"
                           + (f", year: {e['year']}" if e.get('year') else "") + ")")

        # Green chemistry
        green = result.get("green_chem")
        if green:
            parts.append("\n### Green Chemistry")
            ae = green.get("atom_economy")
            if ae is not None:
                parts.append(f"  Atom economy: {ae:.1f}%")
            ef = green.get("e_factor")
            if ef is not None:
                parts.append(f"  E-factor: {ef:.2f}")
            sol = green.get("solvent_score")
            if sol is not None:
                parts.append(f"  Solvent sustainability: {sol:.0%}")

        # Sourcing
        sourcing = result.get("sourcing")
        if sourcing:
            items = sourcing.get("items", [])
            if items:
                in_stock = sum(1 for it in items if it.get("in_stock"))
                parts.append(f"\n### Sourcing ({in_stock}/{len(items)} in stock)")
                total_cost = sourcing.get("total_estimated_cost")
                if total_cost:
                    parts.append(f"  Estimated total cost: ${total_cost:.2f}")
                for it in items:
                    status = "In stock" if it.get("in_stock") else "Out of stock"
                    vendor = it.get("vendor", "unknown")
                    price = it.get("price_per_gram")
                    parts.append(f"  - {it.get('smiles', '?')}: {status} ({vendor}"
                               + (f", ${price:.2f}/g" if price else "") + ")")

        return "\n".join(parts)

    except Exception as e:
        logger.warning(f"Failed to load job context for copilot: {e}")
        return ""


def _build_context_from_request(ctx: dict) -> str:
    """Build context string from the frontend-provided context dict."""
    parts = []

    smiles = ctx.get("smiles")
    if smiles:
        parts.append(f"Target molecule SMILES: {smiles}")

    route = ctx.get("route")
    if route and isinstance(route, dict):
        parts.append(f"\nSelected route (rank #{route.get('rank', '?')}):")
        parts.append(f"Overall score: {route.get('overall_score', 0):.0%}")
        purchasable = route.get("all_purchasable", False)
        parts.append(f"All purchasable: {'Yes' if purchasable else 'No'}")

        sb = route.get("score_breakdown", {})
        if sb:
            parts.append(f"Score breakdown: confidence={sb.get('roundtrip_confidence', 0):.0%}, "
                       f"safety={sb.get('safety', 0):.0%}, green={sb.get('green_chemistry', 0):.0%}, "
                       f"precedent={sb.get('precedent', 0):.0%}")

        steps = route.get("steps", [])
        for i, step in enumerate(steps):
            product = step.get("product", "?")
            reactants = step.get("reactants", [])
            model = step.get("model", "?")
            score = step.get("score", 0)
            rxn_class = step.get("rxn_class", "")
            parts.append(f"  Step {i + 1}: {' + '.join(reactants)} → {product} "
                       f"(model: {model}, confidence: {score:.0%}"
                       + (f", class: {rxn_class}" if rxn_class else "") + ")")

        sm = route.get("starting_materials", [])
        if sm:
            parts.append(f"Starting materials: {', '.join(sm)}")

    constraints = ctx.get("constraints")
    if constraints and isinstance(constraints, list):
        parts.append(f"Active constraints: {', '.join(constraints)}")

    return "\n".join(parts)


def _do_evidence_rag(message: str, smiles: str | None) -> str:
    """If the user's message mentions reactions or chemistry, search evidence index.

    Returns a context snippet with similar known reactions, or empty string.
    """
    if not smiles:
        return ""

    # Only do RAG if the message seems chemistry-related
    chemistry_keywords = [
        "reaction", "mechanism", "step", "route", "reactant", "product",
        "yield", "selectivity", "conditions", "catalyst", "reagent",
        "how", "why", "what", "which", "suggest", "alternative", "better",
        "feasib", "practic", "scalab", "precedent", "literature", "known",
    ]
    msg_lower = message.lower()
    if not any(kw in msg_lower for kw in chemistry_keywords):
        return ""

    try:
        from rasyn.modules.evidence import compute_reaction_fp, _batch_tanimoto, _load_index
        import numpy as np

        # Search for reactions similar to the target molecule itself
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""

        # Use product-only fingerprint as query
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        query = np.zeros(2048, dtype=np.float32)
        for bit in fp.GetOnBits():
            query[bit] = 1.0

        all_fps, all_meta = _load_index()
        if all_fps.shape[0] == 0:
            return ""

        similarities = _batch_tanimoto(query, all_fps)
        top_k = min(5, len(similarities))
        if top_k == 0:
            return ""

        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        hits = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < 0.15:
                continue
            meta = all_meta[idx]
            hits.append(f"  - {meta['reaction_smiles']} (similarity: {sim:.1%}, "
                      f"class: {meta.get('rxn_class', 'N/A')}, source: {meta['source']})")

        if not hits:
            return ""

        return "\n### Relevant Known Reactions (from evidence RAG)\n" + "\n".join(hits)

    except Exception as e:
        logger.debug(f"Evidence RAG failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/copilot/chat", response_model=CopilotChatResponse)
async def copilot_chat(req: CopilotChatRequest):
    """Chat with the AI co-pilot about chemistry and routes.

    Context hierarchy:
    1. If job_id provided → pull full PlanResult from DB (richest context)
    2. Else use frontend-provided route/smiles context
    3. Additionally, do evidence RAG if the question is chemistry-related
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Co-pilot unavailable: ANTHROPIC_API_KEY not configured on server",
        )

    # Build context — prefer DB context if job_id available
    context_text = ""
    smiles = None

    if req.context:
        smiles = req.context.get("smiles")
        job_id = req.context.get("job_id")

        if job_id:
            # Rich context from DB
            context_text = _build_context_from_job(job_id)

        if not context_text:
            # Fallback to frontend-provided context
            context_text = _build_context_from_request(req.context)

        # Evidence RAG — append similar known reactions
        rag_context = _do_evidence_rag(req.message, smiles)
        if rag_context:
            context_text += "\n" + rag_context

    # Build messages for Claude
    messages = []

    # Add conversation history (limit to last 20 messages to control cost)
    for msg in req.history[-20:]:
        messages.append({"role": msg.role, "content": msg.content})

    # Add current user message with context
    if context_text:
        user_content = (
            f"<context>\n{context_text}\n</context>\n\n"
            f"{req.message}"
        )
    else:
        user_content = req.message

    messages.append({"role": "user", "content": user_content})

    # Call Anthropic API
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        reply = response.content[0].text if response.content else "No response generated."
        model_used = response.model

        return CopilotChatResponse(reply=reply, model=model_used)

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Co-pilot unavailable: 'anthropic' package not installed. Run: pip install anthropic",
        )
    except Exception as e:
        logger.error(f"Copilot API error: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Co-pilot error: {str(e)[:200]}",
        )
