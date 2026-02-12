"""Gradio demo interface for Rasyn retrosynthesis.

Provides an interactive web UI with tabs for:
  1. Single-step retrosynthesis
  2. Multi-step route planning
  3. Molecule validation / preview
"""

from __future__ import annotations

import base64
import logging
import time

import gradio as gr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Example molecules for the demo
# ---------------------------------------------------------------------------
EXAMPLES = [
    ["CC(=O)Oc1ccccc1C(O)=O", "Aspirin"],
    ["CC(=O)Nc1ccc(O)cc1", "Acetaminophen (Paracetamol)"],
    ["c1ccc(-c2ccccc2)cc1", "Biphenyl (Suzuki product)"],
    ["CC(C)Cc1ccc(C(C)C(O)=O)cc1", "Ibuprofen"],
    ["OC(=O)c1ccccc1O", "Salicylic acid"],
]


def _svg_to_html(svg_b64: str | None, width: int = 300) -> str:
    """Convert base64 SVG to an inline HTML img tag."""
    if not svg_b64:
        return ""
    return f'<img src="data:image/svg+xml;base64,{svg_b64}" width="{width}"/>'


def _png_to_html(png_b64: str | None, width: int = 500) -> str:
    """Convert base64 PNG to an inline HTML img tag."""
    if not png_b64:
        return ""
    return f'<img src="data:image/png;base64,{png_b64}" width="{width}"/>'


# ---------------------------------------------------------------------------
# Backend functions (called by Gradio)
# ---------------------------------------------------------------------------

def _get_pipeline():
    """Get the PipelineService. Works both standalone and mounted."""
    import rasyn.gradio_app as _self
    if hasattr(_self, "_pipeline_service") and _self._pipeline_service is not None:
        return _self._pipeline_service
    # Standalone mode — create a fresh one
    from rasyn.service.model_manager import ModelManager
    from rasyn.service.pipeline_service import PipelineService
    from rasyn.api.app import load_config
    config = load_config()
    mm = ModelManager(config)
    ps = PipelineService(mm)
    _self._pipeline_service = ps
    return ps


def run_single_step(smiles: str, model: str, top_k: int):
    """Gradio callback for single-step retrosynthesis."""
    import asyncio
    from rasyn.service.molecule_service import MoleculeService

    if not smiles.strip():
        return "", "Please enter a SMILES string.", []

    mol_svc = MoleculeService()
    pipeline = _get_pipeline()

    # Validate and draw product
    info = mol_svc.validate_and_info(smiles.strip())
    if not info["valid"]:
        return "", "Invalid SMILES. Please check your input.", []

    product_html = _svg_to_html(info["svg"])
    product_html += f"<p><b>{info['canonical']}</b><br>"
    product_html += f"{info['formula']} &middot; MW {info['mol_weight']}</p>"

    # Run retrosynthesis
    model_map = {"LLM (RSGPT)": "llm", "RetroTransformer v2": "retro", "Both": "both"}
    model_key = model_map.get(model, "llm")

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(
        pipeline.single_step(info["canonical"], model=model_key, top_k=top_k)
    )
    loop.close()

    predictions = result.get("predictions", [])
    elapsed = result.get("compute_time_ms", 0)

    if not predictions:
        return product_html, f"No predictions found ({elapsed:.0f} ms).", []

    # Build results table
    table_data = []
    for pred in predictions:
        reactants_str = " + ".join(pred["reactants_smiles"])
        confidence = pred.get("confidence", 0)
        source = pred.get("model_source", "unknown")
        table_data.append([
            pred["rank"],
            reactants_str,
            f"{confidence:.2%}",
            source,
        ])

    status = f"Found {len(predictions)} predictions in {elapsed:.0f} ms"
    return product_html, status, table_data


def run_multi_step(smiles: str, max_depth: int, max_routes: int):
    """Gradio callback for multi-step route planning."""
    import asyncio
    from rasyn.service.molecule_service import MoleculeService

    if not smiles.strip():
        return "", "Please enter a SMILES string."

    mol_svc = MoleculeService()
    info = mol_svc.validate_and_info(smiles.strip())
    if not info["valid"]:
        return "", "Invalid SMILES. Please check your input."

    pipeline = _get_pipeline()

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(
        pipeline.multi_step(info["canonical"], max_depth=max_depth, max_routes=max_routes)
    )
    loop.close()

    routes = result.get("routes", [])
    elapsed = result.get("compute_time_ms", 0)

    if not routes:
        return "", f"No routes found ({elapsed:.0f} ms)."

    # Format routes as readable text
    lines = [f"Found {len(routes)} route(s) in {elapsed:.0f} ms\n"]
    for i, route in enumerate(routes):
        lines.append(f"{'='*60}")
        lines.append(f"Route #{i+1} ({route['num_steps']} steps, score={route['total_score']:.3f})")
        lines.append(f"Complete: {'Yes' if route['all_available'] else 'No'}")
        lines.append(f"{'='*60}")
        for j, step in enumerate(route["steps"]):
            lines.append(f"  Step {j+1}: {step['product']}")
            lines.append(f"    -> {' + '.join(step['reactants'])}")
            lines.append(f"    Confidence: {step['confidence']:.2%}")
        if route["starting_materials"]:
            lines.append(f"  Starting materials: {', '.join(route['starting_materials'])}")
        lines.append("")

    return _svg_to_html(info["svg"]), "\n".join(lines)


def run_validate(smiles: str):
    """Gradio callback for molecule validation."""
    from rasyn.service.molecule_service import MoleculeService

    if not smiles.strip():
        return "", "Please enter a SMILES string."

    mol_svc = MoleculeService()
    info = mol_svc.validate_and_info(smiles.strip())

    if not info["valid"]:
        return "", "Invalid SMILES. RDKit could not parse this molecule."

    html = _svg_to_html(info["svg"], width=400)
    details = (
        f"Valid: Yes\n"
        f"Canonical SMILES: {info['canonical']}\n"
        f"Formula: {info['formula']}\n"
        f"Molecular Weight: {info['mol_weight']}"
    )
    return html, details


# ---------------------------------------------------------------------------
# Gradio App Builder
# ---------------------------------------------------------------------------

# Module-level cache for standalone pipeline
_pipeline_service = None


def create_gradio_app(auth: tuple[str, str] | None = None) -> gr.Blocks:
    """Create the Gradio Blocks application.

    Args:
        auth: Optional (username, password) tuple for login protection.
    """
    with gr.Blocks(
        title="Rasyn Retrosynthesis",
    ) as demo:
        gr.Markdown(
            "# Rasyn Retrosynthesis\n"
            "AI-powered retrosynthetic analysis. Enter a product SMILES to predict "
            "how it can be synthesized from simpler building blocks."
        )

        with gr.Tabs():
            # ---------------------------------------------------------------
            # Tab 1: Single-Step
            # ---------------------------------------------------------------
            with gr.TabItem("Single-Step Retrosynthesis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        smiles_input = gr.Textbox(
                            label="Product SMILES",
                            placeholder="e.g. CC(=O)Oc1ccccc1C(O)=O (Aspirin)",
                            lines=1,
                        )
                        with gr.Row():
                            model_choice = gr.Dropdown(
                                choices=["LLM (RSGPT)", "RetroTransformer v2", "Both"],
                                value="LLM (RSGPT)",
                                label="Model",
                            )
                            topk_slider = gr.Slider(
                                minimum=1, maximum=20, value=10, step=1,
                                label="Top-K Results",
                            )
                        run_btn = gr.Button("Run Retrosynthesis", variant="primary")
                    with gr.Column(scale=1):
                        product_display = gr.HTML(label="Product")

                status_text = gr.Textbox(label="Status", interactive=False)
                results_table = gr.Dataframe(
                    headers=["Rank", "Reactants", "Confidence", "Model"],
                    interactive=False,
                    wrap=True,
                )

                gr.Examples(
                    examples=[[e[0]] for e in EXAMPLES],
                    inputs=[smiles_input],
                    label="Example Molecules",
                )

                run_btn.click(
                    fn=run_single_step,
                    inputs=[smiles_input, model_choice, topk_slider],
                    outputs=[product_display, status_text, results_table],
                )

            # ---------------------------------------------------------------
            # Tab 2: Multi-Step
            # ---------------------------------------------------------------
            with gr.TabItem("Multi-Step Planning"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ms_smiles = gr.Textbox(
                            label="Target SMILES",
                            placeholder="Enter target molecule SMILES",
                            lines=1,
                        )
                        with gr.Row():
                            ms_depth = gr.Slider(
                                minimum=1, maximum=15, value=10, step=1,
                                label="Max Depth",
                            )
                            ms_routes = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="Max Routes",
                            )
                        ms_btn = gr.Button("Plan Routes", variant="primary")
                    with gr.Column(scale=1):
                        ms_product = gr.HTML(label="Target")

                ms_output = gr.Textbox(
                    label="Routes",
                    interactive=False,
                    lines=20,
                )

                ms_btn.click(
                    fn=run_multi_step,
                    inputs=[ms_smiles, ms_depth, ms_routes],
                    outputs=[ms_product, ms_output],
                )

            # ---------------------------------------------------------------
            # Tab 3: Molecule Validator
            # ---------------------------------------------------------------
            with gr.TabItem("Molecule Validator"):
                with gr.Row():
                    val_input = gr.Textbox(
                        label="SMILES",
                        placeholder="Enter a SMILES string to validate",
                        lines=1,
                    )
                    val_btn = gr.Button("Validate", variant="primary")

                with gr.Row():
                    val_image = gr.HTML(label="Structure")
                    val_info = gr.Textbox(
                        label="Details",
                        interactive=False,
                        lines=5,
                    )

                val_btn.click(
                    fn=run_validate,
                    inputs=[val_input],
                    outputs=[val_image, val_info],
                )

    if auth:
        demo.auth = [auth]
        demo.auth_message = "Login to access the Rasyn demo"

    return demo
