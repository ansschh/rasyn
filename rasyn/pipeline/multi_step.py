"""Multi-step retrosynthesis route planning.

Combines the single-step model with A* search to find complete
retrosynthetic routes from target to purchasable starting materials.
"""

from __future__ import annotations

import logging

from rasyn.pipeline.single_step import SingleStepRetro
from rasyn.planner.astar import AStarPlanner
from rasyn.planner.inventory import MoleculeInventory, get_default_inventory
from rasyn.schema import Route

logger = logging.getLogger(__name__)


class MultiStepPlanner:
    """End-to-end multi-step retrosynthesis planning."""

    def __init__(
        self,
        single_step: SingleStepRetro,
        inventory: MoleculeInventory | None = None,
        max_depth: int = 10,
        max_nodes: int = 5000,
        max_time_seconds: float = 300.0,
        beam_width: int = 20,
    ):
        self.single_step = single_step
        self.inventory = inventory or get_default_inventory()

        self.planner = AStarPlanner(
            single_step_fn=single_step,
            inventory=self.inventory,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_time_seconds=max_time_seconds,
            beam_width=beam_width,
        )

    def plan(
        self,
        target_smiles: str,
        max_routes: int = 5,
    ) -> list[Route]:
        """Plan retrosynthetic routes for a target molecule.

        Args:
            target_smiles: Target product SMILES.
            max_routes: Maximum number of routes to return.

        Returns:
            List of Route objects, ranked by process score.
        """
        logger.info(f"Planning routes for: {target_smiles}")
        routes = self.planner.plan(target_smiles, max_routes=max_routes)
        logger.info(f"Found {len(routes)} routes")
        return routes

    def plan_single_step_only(
        self,
        target_smiles: str,
        top_n: int = 10,
    ) -> list:
        """Run just single-step retrosynthesis (no route search).

        Useful for testing and evaluation.
        """
        return self.single_step.predict(target_smiles)
