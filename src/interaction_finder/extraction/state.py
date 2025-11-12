"""Pipeline state management.

The State dataclass holds all mutable data shared across pipeline nodes.
Fields are precisely typed and organized by pipeline stage for clarity.

Pipeline flow:
1. ExtractEntitiesNode → entities_by_resource
2. ValidateEntitiesNode → validated_entities_by_resource, entities_merged
3. IdentifyProximalSetsNode → proximal_sets_by_resource
4. ExtractPairsFromProximalSetsNode + AssessPairsNode → pair_assessments_by_resource
5. JudgeCrossDocumentNode → pair_judgments
6. FinalizeNode → ExtractionResult
"""

from dataclasses import dataclass, field

from interaction_finder.extraction.models import (
    EntityMention,
    EntityPairKey,
    PairAssessment,
    PairJudgment,
    ProximalEntitySet,
)
from interaction_finder.resources import ResourceId


@dataclass
class State:
    """Pipeline state with precise, upfront field definitions.

    Fields are organized by pipeline stage to clearly show data flow.
    All fields are documented with their purpose and the stage that populates them.
    """

    # === Required at initialization ===
    topic: str
    target_entity_types: list[str]

    # === Stage 1: Entity Extraction ===
    # Raw entities per resource (before validation/merging)
    entities_by_resource: dict[ResourceId, dict[str, EntityMention]] = field(
        default_factory=dict
    )

    # === Stage 2: Entity Validation ===
    # Entities after kind validation and substring merging
    validated_entities_by_resource: dict[ResourceId, dict[str, EntityMention]] = field(
        default_factory=dict
    )
    # Count of entities merged (for metadata)
    entities_merged: int = 0

    # === Stage 3: Proximal Set Identification ===
    # Groups of entities found in close proximity per resource
    proximal_sets_by_resource: dict[ResourceId, list[ProximalEntitySet]] = field(
        default_factory=dict
    )

    # === Stage 4: Pair Extraction and Assessment ===
    # Final assessed pairs per resource (after deduplication and judgment)
    pair_assessments_by_resource: dict[ResourceId, list[PairAssessment]] = field(
        default_factory=dict
    )

    # === Stage 5: Cross-Document Judgment ===
    # Final judgments for each unique entity pair across all documents
    pair_judgments: dict[EntityPairKey, PairJudgment] = field(default_factory=dict)

    # === Metrics (populated throughout) ===
    quotes_validated: int = 0
    quotes_failed: int = 0
