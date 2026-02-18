"""Execute module — protocol generation, stoichiometry, sample tracking, PDF export.

Slice 7: Generate lab-ready protocols from planned retrosynthetic routes.

Protocol generation uses Claude Haiku (via Anthropic SDK) when available,
with a template-based fallback for when no API key is configured.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stoichiometry calculator (RDKit, pure math)
# ---------------------------------------------------------------------------

def calculate_stoichiometry(
    reactants: list[dict],
    scale_mmol: float = 0.5,
) -> list[dict]:
    """Calculate masses/volumes for each reactant at the given scale.

    Each entry in `reactants` should have:
        smiles: str
        name: str (optional)
        role: str (substrate, reagent, catalyst, solvent, …)
        equivalents: float (1.0 for substrate, etc.)
        density: float | None (g/mL for liquids)

    Returns enriched list with mw, mmol, mass_mg, volume_uL.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        logger.warning("RDKit not available for stoichiometry — returning raw list")
        return reactants

    enriched = []
    for r in reactants:
        entry = dict(r)
        smiles = r.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles) if smiles else None

        if mol is not None:
            mw = round(Descriptors.ExactMolWt(mol), 2)
        else:
            mw = r.get("mw", 0)

        equiv = r.get("equivalents", 1.0)
        mmol = round(scale_mmol * equiv, 4)
        mass_mg = round(mmol * mw, 2)

        entry["mw"] = mw
        entry["mmol"] = mmol
        entry["mass_mg"] = mass_mg

        # Volume for liquids
        density = r.get("density")
        if density and density > 0:
            entry["volume_uL"] = round(mass_mg / density, 1)  # mg / (g/mL) → µL

        # Human-readable amount string
        if mass_mg >= 1000:
            entry["amount"] = f"{mass_mg / 1000:.2f} g"
        else:
            entry["amount"] = f"{mass_mg:.1f} mg"

        enriched.append(entry)

    return enriched


# ---------------------------------------------------------------------------
# Sample ID generation
# ---------------------------------------------------------------------------

def create_sample(
    experiment_id: str,
    sample_type: str = "crude",
    label: str = "",
    planned_analysis: list[str] | None = None,
) -> dict:
    """Create a tracked sample entry with unique ID."""
    date_code = datetime.now(timezone.utc).strftime("%y%m%d")
    short_id = uuid.uuid4().hex[:4].upper()
    sample_id = f"RSN-{date_code}-{short_id}"

    return {
        "id": sample_id,
        "experiment_id": experiment_id,
        "label": label or f"{sample_type.replace('_', ' ').title()} sample",
        "type": sample_type,
        "plannedAnalysis": planned_analysis or ["LCMS", "HPLC"],
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_samples(experiment_id: str) -> list[dict]:
    """Generate standard set of samples for an experiment."""
    return [
        create_sample(experiment_id, "reaction_mixture", "Reaction mixture aliquot",
                       ["TLC", "HPLC"]),
        create_sample(experiment_id, "crude", "Crude product",
                       ["LCMS", "HPLC", "1H NMR"]),
        create_sample(experiment_id, "purified", "Purified product",
                       ["HPLC", "1H NMR", "13C NMR", "LCMS"]),
    ]


# ---------------------------------------------------------------------------
# Protocol generation — LLM (Claude Haiku) with template fallback
# ---------------------------------------------------------------------------

def _build_protocol_prompt(
    product_smiles: str,
    reactants: list[str],
    conditions: dict | None = None,
    scale: str = "0.5 mmol",
    step_number: int = 1,
    reaction_name: str = "",
) -> str:
    """Build prompt for LLM protocol generation."""
    cond_str = json.dumps(conditions) if conditions else "Not specified — use best practices"

    return f"""You are an expert synthetic organic chemist. Generate a detailed lab protocol.

Reaction: {' + '.join(reactants)} → {product_smiles}
Reaction type: {reaction_name or 'General organic transformation'}
Literature conditions: {cond_str}
Scale: {scale}

Provide your response as JSON with exactly these fields:
{{
  "protocol": ["Step 1: ...", "Step 2: ...", ...],
  "reagents": [
    {{"name": "...", "role": "Substrate|Reagent|Catalyst|Solvent|Base|Ligand", "equivalents": 1.0, "amount": "X mg", "mw": 123.4}},
    ...
  ],
  "workup_checklist": ["Quench reaction with ...", "Extract with ...", ...],
  "safety_notes": ["Wear appropriate PPE", ...],
  "estimated_time": "4 hours",
  "tlc_checkpoints": ["After 2h: TLC (EtOAc/hex 1:3)", ...]
}}

Be specific about temperatures, times, solvents, and concentrations. Include TLC/LCMS checkpoints."""


async def generate_protocol_llm(
    product_smiles: str,
    reactants: list[str],
    conditions: dict | None = None,
    scale: str = "0.5 mmol",
    step_number: int = 1,
    reaction_name: str = "",
) -> dict:
    """Generate protocol using Claude Haiku."""
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed — pip install anthropic")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    prompt = _build_protocol_prompt(
        product_smiles, reactants, conditions, scale, step_number, reaction_name
    )

    message = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse JSON from response
    text = message.content[0].text
    # Try to extract JSON from the response
    try:
        # Find JSON block
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        result = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        # Return raw text as protocol steps
        result = {
            "protocol": [line.strip() for line in text.split("\n") if line.strip()],
            "reagents": [],
            "workup_checklist": [],
        }

    return result


def generate_protocol_template(
    product_smiles: str,
    reactants: list[str],
    conditions: dict | None = None,
    scale: str = "0.5 mmol",
    step_number: int = 1,
    reaction_name: str = "",
) -> dict:
    """Generate a template-based protocol (no LLM needed)."""
    cond = conditions or {}
    solvent = cond.get("solvent", "DCM")
    temp = cond.get("temperature", "room temperature")
    time_h = cond.get("time", "4 hours")
    catalyst = cond.get("catalyst", "")
    base = cond.get("base", "")

    reagent_names = [r.split(".")[-1] if "." in r else r for r in reactants]

    protocol = [
        f"Charge a round-bottom flask with {reagent_names[0]} ({scale}).",
        f"Dissolve in {solvent} (5 mL/mmol) under nitrogen atmosphere.",
    ]

    if len(reagent_names) > 1:
        for rname in reagent_names[1:]:
            protocol.append(f"Add {rname} (1.2 equiv) portion-wise at {temp}.")

    if catalyst:
        protocol.append(f"Add {catalyst} (5 mol%) and stir briefly to ensure homogeneity.")

    if base:
        protocol.append(f"Add {base} (2.0 equiv) slowly, monitoring internal temperature.")

    protocol.extend([
        f"Stir at {temp} for {time_h}. Monitor by TLC ({solvent}/hexanes).",
        "Check reaction progress by TLC at 1h, 2h, and 4h intervals.",
        f"Upon completion, concentrate under reduced pressure.",
        f"Purify by column chromatography (SiO2, EtOAc/hexanes gradient).",
        "Collect product fractions and concentrate to obtain the desired product.",
        "Record yield and characterize by 1H NMR, LCMS.",
    ])

    # Build reagent table
    reagents = []
    for i, smi in enumerate(reactants):
        reagents.append({
            "name": reagent_names[i] if i < len(reagent_names) else f"Reagent {i+1}",
            "smiles": smi,
            "role": "Substrate" if i == 0 else "Reagent",
            "equivalents": 1.0 if i == 0 else 1.2,
            "amount": scale if i == 0 else f"{float(scale.split()[0]) * 1.2:.1f} {scale.split()[1] if len(scale.split()) > 1 else 'mmol'}",
            "mw": 0,
        })

    if catalyst:
        reagents.append({"name": catalyst, "role": "Catalyst", "equivalents": 0.05,
                         "amount": "5 mol%", "mw": 0})
    if base:
        reagents.append({"name": base, "role": "Base", "equivalents": 2.0,
                         "amount": f"{float(scale.split()[0]) * 2:.1f} {scale.split()[1] if len(scale.split()) > 1 else 'mmol'}",
                         "mw": 0})

    reagents.append({"name": solvent, "role": "Solvent", "equivalents": 0,
                     "amount": "5 mL/mmol", "mw": 0})

    workup = [
        f"Quench reaction with saturated NH4Cl (aq) (5 mL)",
        f"Transfer to separating funnel and extract with {solvent} (3 × 10 mL)",
        "Wash combined organic layers with brine (10 mL)",
        "Dry over anhydrous Na2SO4, filter",
        "Concentrate under reduced pressure on rotary evaporator",
        "Purify crude product by flash column chromatography",
        "Collect product fractions, concentrate, and dry under high vacuum",
    ]

    return {
        "protocol": protocol,
        "reagents": reagents,
        "workup_checklist": workup,
        "safety_notes": [
            "Wear appropriate PPE (lab coat, safety glasses, gloves)",
            f"Handle {solvent} in a well-ventilated fume hood",
            "Dispose of waste in appropriate chemical waste containers",
        ],
        "estimated_time": time_h if isinstance(time_h, str) else f"{time_h} hours",
        "tlc_checkpoints": [
            f"After 1h: TLC (EtOAc/hex 1:4) — check for product formation",
            f"After 2h: TLC — check for SM consumption",
            f"After {time_h}: TLC — confirm reaction complete",
        ],
    }


def generate_protocol(
    product_smiles: str,
    reactants: list[str],
    conditions: dict | None = None,
    scale: str = "0.5 mmol",
    step_number: int = 1,
    reaction_name: str = "",
) -> dict:
    """Generate a protocol — uses LLM if available, else template."""
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(generate_protocol_llm(
                product_smiles, reactants, conditions, scale, step_number, reaction_name
            ))
            loop.close()
            return result
        except Exception as e:
            logger.warning(f"LLM protocol generation failed, using template: {e}")

    return generate_protocol_template(
        product_smiles, reactants, conditions, scale, step_number, reaction_name
    )


# ---------------------------------------------------------------------------
# Full experiment generation from a route
# ---------------------------------------------------------------------------

def generate_experiment(
    route: dict,
    step_index: int = 0,
    scale: str = "0.5 mmol",
) -> dict:
    """Generate a complete experiment template from a route step.

    Args:
        route: Route dict from PlanResult (with steps, route_id, etc.)
        step_index: Which step in the route to generate for (0 = first step)
        scale: Reaction scale (e.g., "0.5 mmol", "5 mmol")

    Returns:
        ExperimentTemplate-compatible dict
    """
    steps = route.get("steps", [])
    if not steps or step_index >= len(steps):
        raise ValueError(f"No step at index {step_index} in route")

    step = steps[step_index]
    product = step.get("product", "")
    reactants = step.get("reactants", [])
    conditions = step.get("conditions")
    rxn_class = step.get("rxn_class", "")

    # Generate experiment ID
    date_code = datetime.now(timezone.utc).strftime("%y%m%d")
    exp_id = f"EXP-{date_code}-{uuid.uuid4().hex[:4].upper()}"

    # Generate protocol
    protocol_data = generate_protocol(
        product, reactants, conditions, scale,
        step_number=step_index + 1,
        reaction_name=rxn_class or "",
    )

    # Calculate stoichiometry for reagents
    raw_reagents = protocol_data.get("reagents", [])
    for r in raw_reagents:
        if "smiles" not in r:
            # Try to match to reactant SMILES
            idx = raw_reagents.index(r)
            if idx < len(reactants):
                r["smiles"] = reactants[idx]
            else:
                r["smiles"] = ""

    reagents = calculate_stoichiometry(raw_reagents, scale_mmol=float(scale.split()[0]))

    # Generate samples
    samples = generate_samples(exp_id)

    return {
        "id": exp_id,
        "stepNumber": step_index + 1,
        "reactionName": rxn_class or f"Step {step_index + 1}",
        "product_smiles": product,
        "reactant_smiles": reactants,
        "protocol": protocol_data.get("protocol", []),
        "reagents": [
            {
                "name": r.get("name", "Unknown"),
                "role": r.get("role", "Reagent"),
                "equivalents": r.get("equivalents", 0),
                "amount": r.get("amount", "—"),
                "mw": r.get("mw", 0),
            }
            for r in reagents
        ],
        "workupChecklist": protocol_data.get("workup_checklist", []),
        "samples": samples,
        "elnExportReady": True,
        "safety_notes": protocol_data.get("safety_notes", []),
        "estimated_time": protocol_data.get("estimated_time", ""),
        "tlc_checkpoints": protocol_data.get("tlc_checkpoints", []),
        "scale": scale,
        "route_id": route.get("route_id", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

def export_protocol_pdf(experiment: dict) -> bytes:
    """Export an experiment protocol to PDF bytes.

    Uses reportlab if available, otherwise returns a simple text-based PDF.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        import io

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle("Title", parent=styles["Title"],
                                      fontSize=16, spaceAfter=12)
        story.append(Paragraph(f"Experiment Protocol: {experiment['id']}", title_style))
        story.append(Spacer(1, 6))

        # Reaction info
        story.append(Paragraph(
            f"<b>Step {experiment.get('stepNumber', 1)}:</b> {experiment.get('reactionName', 'Synthesis')}",
            styles["Normal"]
        ))
        story.append(Paragraph(
            f"<b>Scale:</b> {experiment.get('scale', '0.5 mmol')} | "
            f"<b>Date:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))

        # Protocol steps
        story.append(Paragraph("<b>Procedure:</b>", styles["Heading2"]))
        for i, step in enumerate(experiment.get("protocol", []), 1):
            story.append(Paragraph(f"{i}. {step}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Reagent table
        reagents = experiment.get("reagents", [])
        if reagents:
            story.append(Paragraph("<b>Reagent Table:</b>", styles["Heading2"]))
            table_data = [["Reagent", "Role", "Equiv.", "Amount", "MW"]]
            for r in reagents:
                table_data.append([
                    r.get("name", ""),
                    r.get("role", ""),
                    str(r.get("equivalents", "")),
                    r.get("amount", ""),
                    f"{r.get('mw', 0):.1f}" if r.get("mw") else "—",
                ])
            t = Table(table_data, colWidths=[2*inch, 1.2*inch, 0.7*inch, 1.2*inch, 0.8*inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (2, 0), (-1, -1), "CENTER"),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

        # Workup
        workup = experiment.get("workupChecklist", [])
        if workup:
            story.append(Paragraph("<b>Workup & Purification:</b>", styles["Heading2"]))
            for item in workup:
                story.append(Paragraph(f"☐ {item}", styles["Normal"]))
            story.append(Spacer(1, 12))

        # Samples
        samples = experiment.get("samples", [])
        if samples:
            story.append(Paragraph("<b>Sample Tracking:</b>", styles["Heading2"]))
            for s in samples:
                story.append(Paragraph(
                    f"<b>{s['id']}</b> — {s.get('label', '')} "
                    f"({', '.join(s.get('plannedAnalysis', []))})",
                    styles["Normal"]
                ))

        # Footer
        story.append(Spacer(1, 24))
        story.append(Paragraph(
            f"Generated by Rasyn Chemistry OS — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
        ))

        doc.build(story)
        return buffer.getvalue()

    except ImportError:
        # Fallback: simple text PDF
        logger.warning("reportlab not available — generating text-only PDF")
        lines = [f"Experiment Protocol: {experiment['id']}", ""]
        lines.append(f"Step {experiment.get('stepNumber', 1)}: {experiment.get('reactionName', '')}")
        lines.append(f"Scale: {experiment.get('scale', '0.5 mmol')}")
        lines.append("")
        lines.append("PROCEDURE:")
        for i, step in enumerate(experiment.get("protocol", []), 1):
            lines.append(f"  {i}. {step}")
        lines.append("")
        lines.append("REAGENTS:")
        for r in experiment.get("reagents", []):
            lines.append(f"  {r.get('name', '')} | {r.get('role', '')} | {r.get('amount', '')}")

        text = "\n".join(lines)
        return text.encode("utf-8")


# ---------------------------------------------------------------------------
# Multi-format export (JSON, CSV, SDF, webhook)
# ---------------------------------------------------------------------------

def export_experiment_json(experiment: dict) -> bytes:
    """Export experiment as pretty-printed JSON."""
    return json.dumps(experiment, indent=2, default=str).encode("utf-8")


def export_experiment_csv(experiment: dict) -> bytes:
    """Export experiment reagent table + protocol as CSV."""
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header info
    writer.writerow(["Experiment ID", experiment.get("id", "")])
    writer.writerow(["Reaction", experiment.get("reactionName", "")])
    writer.writerow(["Scale", experiment.get("scale", "")])
    writer.writerow(["Product SMILES", experiment.get("product_smiles", "")])
    writer.writerow([])

    # Reagent table
    writer.writerow(["Reagent", "Role", "Equivalents", "Amount", "MW (g/mol)"])
    for r in experiment.get("reagents", []):
        writer.writerow([
            r.get("name", ""),
            r.get("role", ""),
            r.get("equivalents", ""),
            r.get("amount", ""),
            r.get("mw", ""),
        ])
    writer.writerow([])

    # Protocol steps
    writer.writerow(["Protocol Steps"])
    for i, step in enumerate(experiment.get("protocol", []), 1):
        writer.writerow([f"{i}. {step}"])
    writer.writerow([])

    # Workup
    writer.writerow(["Workup Checklist"])
    for item in experiment.get("workupChecklist", []):
        writer.writerow([item])
    writer.writerow([])

    # Samples
    writer.writerow(["Sample ID", "Label", "Type", "Planned Analysis", "Status"])
    for s in experiment.get("samples", []):
        writer.writerow([
            s.get("id", ""),
            s.get("label", ""),
            s.get("type", ""),
            ", ".join(s.get("plannedAnalysis", [])),
            s.get("status", ""),
        ])

    return buf.getvalue().encode("utf-8")


def export_experiment_sdf(experiment: dict) -> bytes:
    """Export product + reactant structures as SDF file."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        writer = Chem.SDWriter("/dev/null")  # dummy to get the class
        writer.close()

        import io
        sdf_buf = io.StringIO()

        all_smiles = []
        product = experiment.get("product_smiles", "")
        if product:
            all_smiles.append(("Product", product))
        for smi in experiment.get("reactant_smiles", []):
            all_smiles.append(("Reactant", smi))

        for role, smi in all_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
            mol = Chem.RemoveHs(mol)
            mol.SetProp("_Name", f"{role}: {smi}")
            mol.SetProp("Role", role)
            mol.SetProp("SMILES", smi)
            mol.SetProp("Experiment_ID", experiment.get("id", ""))
            sdf_buf.write(Chem.MolToMolBlock(mol))
            sdf_buf.write("\n$$$$\n")

        return sdf_buf.getvalue().encode("utf-8")

    except ImportError:
        logger.warning("RDKit not available for SDF export")
        return b""
    except Exception as e:
        logger.warning(f"SDF export failed: {e}")
        return b""


def export_webhook(experiment: dict, webhook_url: str | None = None) -> dict:
    """Push experiment to ELN webhook endpoint.

    Returns dict with status and response info.
    """
    import os
    import urllib.request
    import urllib.error

    url = webhook_url or os.environ.get("RASYN_ELN_WEBHOOK_URL")
    if not url:
        return {"status": "error", "message": "No ELN webhook URL configured"}

    payload = json.dumps({
        "source": "rasyn",
        "version": "2.0",
        "experiment": experiment,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {
                "status": "success",
                "http_status": resp.status,
                "message": f"Pushed to ELN ({resp.status})",
            }
    except urllib.error.HTTPError as e:
        return {"status": "error", "http_status": e.code, "message": f"ELN webhook error: {e.code}"}
    except Exception as e:
        return {"status": "error", "message": f"ELN webhook failed: {str(e)[:200]}"}
