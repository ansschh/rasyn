"""Molecule utilities — validation, SVG rendering, molecular info."""

from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger(__name__)


class MoleculeService:
    """SMILES validation, molecule images, and chemical info."""

    def validate_and_info(self, smiles: str) -> dict:
        """Validate SMILES and return molecular information.

        Returns:
            Dict with valid, canonical, formula, mol_weight, svg.
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "valid": False,
                "canonical": None,
                "formula": None,
                "mol_weight": None,
                "svg": None,
            }

        canonical = Chem.MolToSmiles(mol)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        mol_weight = round(Descriptors.MolWt(mol), 2)
        svg = self.draw_molecule_svg(canonical)

        return {
            "valid": True,
            "canonical": canonical,
            "formula": formula,
            "mol_weight": mol_weight,
            "svg": svg,
        }

    @staticmethod
    def draw_molecule_svg(smiles: str, width: int = 300, height: int = 200) -> str | None:
        """Render a molecule as an SVG string.

        Returns:
            Base64-encoded SVG string, or None if rendering fails.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem.Draw import rdMolDraw2D

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg_text = drawer.GetDrawingText()
            return base64.b64encode(svg_text.encode()).decode()
        except Exception:
            logger.exception(f"Failed to draw molecule: {smiles}")
            return None

    @staticmethod
    def draw_reaction_svg(
        product: str, reactants: list[str], width: int = 600, height: int = 200
    ) -> str | None:
        """Render a reaction (reactants → product) as SVG.

        Returns:
            Base64-encoded SVG string, or None on failure.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Draw

            rxn_smiles = ".".join(reactants) + ">>" + product
            rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
            if rxn is None:
                return None

            img = Draw.ReactionToImage(rxn, subImgSize=(200, 200))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            logger.exception(f"Failed to draw reaction: {product}")
            return None
