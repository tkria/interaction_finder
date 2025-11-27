"""Pipeline state management.

The State dataclass holds all mutable data shared across pipeline nodes.
Fields are precisely typed and organized by pipeline stage for clarity.

Pipeline flow:
1. ProcessDocumentsNode (per-document, concurrent):
   - Extract entities → entities_by_resource
   - Validate kinds → validated_entities_by_resource
   - Identify proximal sets → proximal_sets_by_resource
   - Extract & assess pairs → pair_assessments_by_resource
2. ConsolidateEntitiesNode → entities_merged, canonical_name_variants (global merging + pair reference updates)
3. ConsolidateRelationshipsNode → relationship_mappings, relationship_polarities
4. SweepCoMentionsNode → co_mention_sweep_stats (additional assessments added to pair_assessments_by_resource)
5. ConsolidateNewRelationshipsNode → extends relationship_polarities for new labels
6. JudgeCrossDocumentNode → pair_judgments
7. FinalizeNode → ExtractionResult
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interaction_finder.extraction.sweep_co_mentions import CoMentionSweepStats

from interaction_finder.extraction.models import (
    EntityMention,
    EntityPairKey,
    EntityRef,
    PairAssessment,
    PairJudgment,
    ProximalEntitySet,
)
from interaction_finder.resources import ResourceId


@dataclass
class QuoteValidationFailure:
    """Record of a quote validation failure with complete information.

    Captures all details needed for debugging and reporting without truncation.
    """

    entity_name: str
    resource_id: ResourceId
    quote_text: str
    error_type: str  # "paraphrased", "missing", "fuzzy_match_failed"
    similarity: float | None = None
    details: str = ""


@dataclass
class State:
    """Pipeline state with precise, upfront field definitions.

    Fields are organized by pipeline stage to clearly show data flow.
    All fields are documented with their purpose and the stage that populates them.
    """

    # === Required at initialization ===
    topic: str
    target_entity_types: list[str]
    # Map of which entity kinds can pair with which (derived from target_entity_types)
    # A kind must appear twice in target_entity_types to allow self-pairs
    permitted_pairs: dict[str, set[str]]

    # === Stage 1: Entity Extraction ===
    # Raw entities per resource (before validation/merging)
    entities_by_resource: dict[ResourceId, dict[str, EntityMention]] = field(
        default_factory=dict
    )

    # === Stage 2: Entity Validation ===
    # Entities after kind validation and substring merging (stored as EntityRefs)
    validated_entities_by_resource: dict[ResourceId, dict[str, EntityRef]] = field(
        default_factory=dict
    )
    # Count of entities merged (for metadata)
    entities_merged: int = 0
    # Memoization cache for consolidation decisions: (norm_parent, norm_child, kind) → target | False
    # Stores target canonical name for merge/rename, or False for skip decisions
    merge_decision_cache: dict[tuple[str, str, str], str | bool] = field(
        default_factory=dict
    )
    # Track all canonical name variants for each normalized entity name
    # Maps (normalized_name, kind) → set of all canonical (un-normalized) variants seen
    # Used to apply merge decisions correctly across documents with different capitalizations
    canonical_name_variants: dict[tuple[str, str], set[str]] = field(
        default_factory=dict
    )
    # Cache hits/misses for metrics
    merge_cache_hits: int = 0
    merge_cache_misses: int = 0
    # Consolidation rules for provenance: (normalized_name, kind) → (target_canonical, reasoning)
    # Reasoning is "auto:cap", "auto:fuzzy", or full LLM reasoning string
    consolidation_rules: dict[tuple[str, str], tuple[str, str]] = field(
        default_factory=dict
    )

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

    # === Stage 5: Relationship Consolidation + Polarity Classification ===
    # Mapping from normalized old label to normalized new label
    relationship_mappings: dict[str, str] = field(default_factory=dict)
    # Mapping from relationship label to polarity (supporting/refuting/neutral/irrelevant)
    relationship_polarities: dict[str, str] = field(default_factory=dict)
    # Count of assessments with updated relationship labels
    relationships_merged: int = 0

    # === Stage 5b: Co-mention Sweep ===
    # Statistics from the co-mention sweep (None if sweep disabled or not yet run)
    co_mention_sweep_stats: "CoMentionSweepStats | None" = None

    # === Stage 6: Cross-Document Judgment ===
    # Final judgments for each unique entity pair across all documents
    pair_judgments: dict[EntityPairKey, PairJudgment] = field(default_factory=dict)

    # === Metrics (populated throughout) ===
    quotes_validated: int = 0
    quotes_failed: int = 0
    quote_failures: list[QuoteValidationFailure] = field(default_factory=list)
