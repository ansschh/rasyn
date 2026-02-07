"""A* / best-first route search for multi-step retrosynthesis.

State: set of molecules remaining to synthesize
Action: pick a molecule, run single-step model -> candidate precursor sets
Terminal: all molecules purchasable/in-stock
Cost: cumulative process_score penalties (lower is better)
Heuristic: number of non-purchasable molecules remaining
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field

from rasyn.planner.inventory import MoleculeInventory
from rasyn.schema import Route, StepObject

logger = logging.getLogger(__name__)


@dataclass(order=True)
class SearchNode:
    """A node in the A* search tree."""

    priority: float  # f(n) = g(n) + h(n), lower is better
    cost: float = field(compare=False)       # g(n) = cumulative cost so far
    molecules: tuple = field(compare=False)   # Remaining molecules to synthesize
    steps: list = field(compare=False, default_factory=list)  # Steps taken so far
    depth: int = field(compare=False, default=0)


class AStarPlanner:
    """A* search for multi-step retrosynthetic route planning.

    Optimizes for process score (safety + scalability + greenness)
    rather than model likelihood.
    """

    def __init__(
        self,
        single_step_fn,
        inventory: MoleculeInventory,
        max_depth: int = 10,
        max_nodes: int = 5000,
        max_time_seconds: float = 300.0,
        beam_width: int = 50,
    ):
        """
        Args:
            single_step_fn: Callable(smiles) -> list[StepObject]
                The single-step retrosynthesis model.
            inventory: Purchasable molecule lookup.
            max_depth: Maximum route depth.
            max_nodes: Maximum nodes to expand.
            max_time_seconds: Time limit.
            beam_width: Max candidates to keep per expansion.
        """
        self.single_step = single_step_fn
        self.inventory = inventory
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_time_seconds = max_time_seconds
        self.beam_width = beam_width

    def _heuristic(self, molecules: tuple[str, ...]) -> float:
        """Admissible heuristic: count of non-purchasable molecules.

        This underestimates the true cost since each non-purchasable
        molecule needs at least one step (cost >= 0).
        """
        return sum(
            1.0 for m in molecules
            if not self.inventory.is_purchasable(m)
        )

    def _step_cost(self, step: StepObject) -> float:
        """Compute the cost of a single step (lower = better).

        We invert the process score: high process score = low cost.
        """
        if step.process_scores is not None:
            return 1.0 - step.process_scores.total_score
        return 0.5  # Default cost if no score available

    def plan(
        self,
        target_smiles: str,
        max_routes: int = 5,
    ) -> list[Route]:
        """Find retrosynthetic routes using A* search.

        Args:
            target_smiles: Target product SMILES.
            max_routes: Maximum number of routes to return.

        Returns:
            List of Route objects, sorted by total process score.
        """
        start_time = time.time()
        found_routes: list[Route] = []
        nodes_expanded = 0

        # Initialize
        initial_node = SearchNode(
            priority=self._heuristic((target_smiles,)),
            cost=0.0,
            molecules=(target_smiles,),
            steps=[],
            depth=0,
        )

        open_set: list[SearchNode] = [initial_node]
        visited: set[tuple] = set()

        while open_set and nodes_expanded < self.max_nodes:
            # Time check
            if time.time() - start_time > self.max_time_seconds:
                logger.info(f"Time limit reached ({self.max_time_seconds}s)")
                break

            # Pop best node
            node = heapq.heappop(open_set)

            # Check if already visited
            state_key = tuple(sorted(node.molecules))
            if state_key in visited:
                continue
            visited.add(state_key)
            nodes_expanded += 1

            # Check terminal condition: all molecules purchasable
            remaining = [m for m in node.molecules if not self.inventory.is_purchasable(m)]
            if not remaining:
                # Found a complete route!
                route = Route(
                    target=target_smiles,
                    steps=node.steps,
                    total_process_score=1.0 - node.cost if node.steps else 1.0,
                    all_starting_materials_available=True,
                    starting_materials=list(node.molecules),
                )
                found_routes.append(route)
                logger.info(
                    f"Found route #{len(found_routes)} "
                    f"({len(node.steps)} steps, score={route.total_process_score:.3f})"
                )
                if len(found_routes) >= max_routes:
                    break
                continue

            # Depth limit
            if node.depth >= self.max_depth:
                continue

            # Expand: try decomposing each non-purchasable molecule
            for mol_to_expand in remaining:
                # Get single-step candidates
                try:
                    candidates = self.single_step(mol_to_expand)
                except Exception as e:
                    logger.debug(f"Single-step failed for {mol_to_expand}: {e}")
                    continue

                # Limit beam width
                for step in candidates[:self.beam_width]:
                    # Build new molecule set: replace expanded mol with its reactants
                    new_molecules = list(node.molecules)
                    new_molecules.remove(mol_to_expand)
                    new_molecules.extend(step.reactants)
                    new_molecules = tuple(sorted(set(new_molecules)))

                    # Compute new cost
                    step_cost = self._step_cost(step)
                    new_cost = node.cost + step_cost
                    new_priority = new_cost + self._heuristic(new_molecules)

                    new_node = SearchNode(
                        priority=new_priority,
                        cost=new_cost,
                        molecules=new_molecules,
                        steps=node.steps + [step],
                        depth=node.depth + 1,
                    )

                    heapq.heappush(open_set, new_node)

        elapsed = time.time() - start_time
        logger.info(
            f"A* search complete: {nodes_expanded} nodes expanded, "
            f"{len(found_routes)} routes found in {elapsed:.1f}s"
        )

        # Sort routes by total process score (descending)
        found_routes.sort(key=lambda r: r.total_process_score, reverse=True)
        return found_routes
