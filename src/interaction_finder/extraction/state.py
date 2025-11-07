"""Pipeline state management.

The State dataclass holds all mutable data shared across pipeline nodes.
Fields are precisely typed and organized by pipeline stage for clarity.
"""

from dataclasses import dataclass, field

from interaction_finder.extraction.models import (
    EntityAssessment,
    EntityMention,
    FinalJudgment,
    PairAssessment,
    PairKey,
    PairMention,
)
from interaction_finder.resources import ResourceId


@dataclass
class State:
    """Pipeline state with precise, upfront field definitions.

    Fields are organized by pipeline stage:
    1. Extraction: entities_by_resource, pairs_by_resource
    2. Entity Assessment: entity_assessments
    3. Pair Assessment: pair_assessments
    4. Final Judgment: final_pair_judgments
    """

    # === Required at initialization ===
    topic: str
    target_entity_types: list[str]

    # === Stage 1: Extraction ===
    # Entity mentions per resource: resource_id → {canonical_name → EntityMention}
    entities_by_resource: dict[ResourceId, dict[str, EntityMention]] = field(
        default_factory=dict
    )

    # Pair mentions per resource: resource_id → [PairMention, ...]
    pairs_by_resource: dict[ResourceId, list[PairMention]] = field(default_factory=dict)

    # === Stage 2: Entity Assessment ===
    # Assessments for each entity across all resources: entity_name → [EntityAssessment, ...]
    entity_assessments: dict[str, list[EntityAssessment]] = field(default_factory=dict)

    # === Stage 3: Pair Assessment ===
    # Assessments for each pair across all resources: PairKey → [PairAssessment, ...]
    pair_assessments: dict[PairKey, list[PairAssessment]] = field(default_factory=dict)

    # === Stage 4: Final Judgment ===
    # Which pairs are accepted after cross-document synthesis: PairKey → FinalJudgment
    final_pair_judgments: dict[PairKey, FinalJudgment] = field(default_factory=dict)

    # === Metrics (populated throughout) ===
    quotes_validated: int = 0
    quotes_failed: int = 0
