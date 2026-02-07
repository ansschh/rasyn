"""Unified data schema for the Rasyn hybrid retrosynthesis pipeline.

All components (preprocessing, models, verifiers, scorers, planner) operate
on these shared dataclasses, ensuring a stable contract across the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Preprocessing / training data
# ---------------------------------------------------------------------------

@dataclass
class EditLabels:
    """Labels extracted from an atom-mapped reaction for training the graph head."""

    changed_bonds: list[tuple[int, int]]
    """Atom-index pairs in the product whose bond order changed or was broken."""

    synthon_smiles: list[str]
    """SMILES of product fragments after breaking reaction-center bonds."""

    leaving_groups: list[str]
    """Leaving-group SMILES for each synthon attachment point (e.g. 'Cl', 'H', 'OH')."""

    edit_tokens: str
    """Serialised edit string for LLM conditioning, e.g.
    '<EDIT> DISCONNECT 3-7 <SYNTHONS> C=O.CCN <LG_HINTS> [Cl] [H]'
    """


@dataclass
class ProcessLabels:
    """Weak / proxy labels for process-aware scoring (nullable early on)."""

    hazard_tags: list[str] = field(default_factory=list)
    safety_proxy_score: Optional[float] = None
    scale_proxy_score: Optional[float] = None


@dataclass
class ReactionRecord:
    """A single reaction with all extracted information.

    This is the row-level unit stored in the preprocessed dataset.
    """

    id: str
    product_smiles: str
    reactants_smiles: list[str]
    reagents_smiles: list[str] = field(default_factory=list)
    solvents_smiles: list[str] = field(default_factory=list)
    conditions: dict = field(default_factory=dict)
    atom_mapped_rxn_smiles: str = ""
    reaction_class: Optional[int] = None
    labels: Optional[EditLabels] = None
    process_labels: ProcessLabels = field(default_factory=ProcessLabels)


# ---------------------------------------------------------------------------
# Inference-time objects
# ---------------------------------------------------------------------------

@dataclass
class EditHypothesis:
    """A single edit hypothesis produced by the graph head."""

    reaction_center_bonds: list[tuple[int, int]]
    """Atom-index pairs in the product predicted as the reaction center."""

    synthon_smiles: list[str]
    leaving_group_options: list[list[str]]
    """Per-synthon ranked LG candidates, e.g. [['Cl','Br','OTf'], ['H','OH']]."""

    confidence: float
    edit_tokens: str
    """Pre-formatted string for LLM conditioning."""


@dataclass
class EditExplanation:
    """Human-readable explanation of the proposed disconnection."""

    highlighted_bonds: list[tuple[int, int]]
    synthon_smiles: list[str]
    leaving_groups_used: list[str]
    edit_description: str = ""


@dataclass
class VerifierResults:
    """Outcomes from the verifier ensemble for a single candidate."""

    rdkit_valid: bool = False
    forward_match_score: float = 0.0
    template_match: Optional[bool] = None
    template_id: Optional[str] = None
    overall_confidence: float = 0.0


@dataclass
class ProcessScores:
    """Process-aware scores for a single reaction step."""

    safety_score: float = 0.0
    scalability_score: float = 0.0
    greenness_score: float = 0.0
    confidence_score: float = 0.0
    total_score: float = 0.0

    safety_details: dict = field(default_factory=dict)
    scalability_details: dict = field(default_factory=dict)
    greenness_details: dict = field(default_factory=dict)


@dataclass
class StepCandidate:
    """A single candidate retrosynthetic step before final ranking."""

    product: str
    reactants: list[str]
    reagents: list[str] = field(default_factory=list)
    conditions: dict = field(default_factory=dict)
    edit_hypothesis: Optional[EditHypothesis] = None
    llm_score: float = 0.0
    verifier_results: Optional[VerifierResults] = None
    process_scores: Optional[ProcessScores] = None


@dataclass
class StepObject:
    """A fully scored and verified retrosynthetic step (the pipeline's output unit)."""

    product: str
    reactants: list[str]
    reagents: list[str] = field(default_factory=list)
    conditions: dict = field(default_factory=dict)
    edit_explanation: Optional[EditExplanation] = None
    verifier_results: Optional[VerifierResults] = None
    process_scores: Optional[ProcessScores] = None
    risk_tags: list[str] = field(default_factory=list)
    rank: int = 0


@dataclass
class Route:
    """A complete multi-step retrosynthetic route."""

    target: str
    """The original target product SMILES."""

    steps: list[StepObject] = field(default_factory=list)
    total_process_score: float = 0.0
    all_starting_materials_available: bool = False
    starting_materials: list[str] = field(default_factory=list)
    num_steps: int = 0

    def __post_init__(self):
        self.num_steps = len(self.steps)
