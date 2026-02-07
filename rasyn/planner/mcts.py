"""MCTS route search for multi-step retrosynthesis.

Stub for future implementation. A* search in astar.py is the
primary planner for now.
"""

from __future__ import annotations

import logging

from rasyn.schema import Route

logger = logging.getLogger(__name__)


class MCTSPlanner:
    """Monte Carlo Tree Search planner (stub).

    Will be implemented after A* planner is validated.
    MCTS offers better exploration of the route space and
    can incorporate learned value/policy networks.
    """

    def __init__(self, single_step_fn, inventory, **kwargs):
        self.single_step = single_step_fn
        self.inventory = inventory
        logger.warning("MCTSPlanner is a stub. Use AStarPlanner instead.")

    def plan(self, target_smiles: str, max_routes: int = 5) -> list[Route]:
        """Plan routes using MCTS (not yet implemented)."""
        raise NotImplementedError(
            "MCTSPlanner is not yet implemented. Use AStarPlanner."
        )
