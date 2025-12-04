"""Pipeline state management.

The State dataclass holds all mutable data shared across pipeline nodes.
Fields are precisely typed and organized by pipeline stage for clarity.

Pipeline flow:
1. ProcessDocumentsNode (per-document, concurrent):
   - Extract entities → entities_by_resource
   - Validate kinds → validated_entities_by_resource
   - Identify proximal sets → proximal_sets_by_resource
   - Extract & assess pairs → pair_assessments_by_resource
2. ConsolidateEntitiesNode → entities_merged, consolidation_rules (global merging + pair reference updates)
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
    from interaction_finder.resources import ResourcePool

from interaction_finder.extraction.models import (
    EntityMention,
    EntityPairKey,
    EntityRef,
    PairAssessment,
    PairJudgment,
    ProximalEntitySet,
)
from interaction_finder.resources import ResourceId

# Registry of special serialization handlers for State fields
# Maps field_name → (handler_type, inner_type_for_validation)
_SERIALIZATION_REGISTRY: dict[str, tuple[str, type | None]] = {
    "entities_by_resource": ("by_resource", dict[str, EntityMention]),
    "validated_entities_by_resource": ("by_resource", dict[str, EntityRef]),
    "proximal_sets_by_resource": ("by_resource", list[ProximalEntitySet]),
    "pair_assessments_by_resource": ("by_resource", list[PairAssessment]),
    "pair_judgments": ("pair_key", PairJudgment),
    "consolidation_rules": ("tuple_key", None),
    "agent_merge_cache": ("tuple_key", None),
    "relationship_oppositions": ("set_value", None),
}


def _rehydrate_entity_quotes(
    entities_data: dict, resource_pool: "ResourcePool"
) -> None:
    """Rehydrate resource_url → resource in EntityMention/EntityRef quotes.

    Modifies entities_data in-place.
    """
    for resource_entities in entities_data.values():
        for entity_data in resource_entities.values():
            # Get quotes list (either direct or nested in mentions)
            quotes_lists = []
            if isinstance(entity_data, dict):
                if "quotes" in entity_data:  # EntityMention
                    quotes_lists.append(entity_data["quotes"])
                if "mentions" in entity_data:  # EntityRef
                    quotes_lists.extend(
                        m["quotes"] for m in entity_data["mentions"] if "quotes" in m
                    )

            # Rehydrate all quotes
            for quotes in quotes_lists:
                for quote in quotes:
                    if isinstance(quote, dict) and "resource_url" in quote:
                        quote["resource"] = resource_pool.get(quote["resource_url"])
                        del quote["resource_url"]


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
    # Clustering metadata by entity kind (for debugging/analysis)
    clustering_metadata: dict[str, dict] = field(default_factory=dict)
    # Consolidation rules for provenance: (normalized_name, kind) → (target_canonical, reasoning)
    # Reasoning format: "auto:<speculation>:<source>:<match_kind>" or full LLM reasoning string
    # Examples: "auto:0:original:exact", "auto:1:before_paren:exact", "auto:3:original:fuzzy"
    consolidation_rules: dict[tuple[str, str], tuple[str, str]] = field(
        default_factory=dict
    )
    # Within-run cache for LLM merge decisions: (child_canonical, parent_canonical, kind) → (target, reasoning)
    # target is None for "skip", parent for "merge", or custom name for "rename"
    # Prevents redundant LLM calls when same pair appears across multiple documents
    agent_merge_cache: dict[tuple[str, str, str], tuple[str | None, str]] = field(
        default_factory=dict
    )
    # Cache statistics
    merge_cache_hits: int = 0
    merge_cache_misses: int = 0

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
    # Mapping from relationship label to polarity (positive/negative/neutral/irrelevant)
    relationship_polarities: dict[str, str] = field(default_factory=dict)
    # Mapping from normalized relationship to set of normalized opposing relationships
    relationship_oppositions: dict[str, set[str]] = field(default_factory=dict)
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

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict using handler registry.

        Uses Pydantic's JSON mode for automatic dehydration:
        - ResourceQuote.resource → resource_url
        - EntityRef → canonical string
        """
        from dataclasses import fields
        from pydantic import TypeAdapter

        def key_str(k):
            """Convert complex keys to strings."""
            return k.url if hasattr(k, "url") else str(k)

        result = {}

        for fld in fields(self):
            value = getattr(self, fld.name)

            if fld.name not in _SERIALIZATION_REGISTRY:
                # Simple field - direct assignment
                result[fld.name] = value
                continue

            handler_type, inner_type = _SERIALIZATION_REGISTRY[fld.name]

            if handler_type == "by_resource":
                # dict[ResourceId, X] → dict[str, X_serialized]
                result[fld.name] = {
                    key_str(k): TypeAdapter(inner_type).dump_python(v, mode="json")
                    for k, v in value.items()
                }
            elif handler_type == "pair_key":
                # dict[EntityPairKey, X] → dict[str, X_serialized]
                result[fld.name] = {
                    str(k): TypeAdapter(inner_type).dump_python(v, mode="json")
                    for k, v in value.items()
                }
            elif handler_type == "tuple_key":
                # dict[tuple, X] → dict[str, X]
                result[fld.name] = {
                    "|".join(str(x) for x in k): v for k, v in value.items()
                }
            elif handler_type == "set_value":
                # dict[K, set[V]] → dict[K, list[V]]
                result[fld.name] = {k: list(v) for k, v in value.items()}

        return result

    @classmethod
    def from_dict(
        cls,
        data: dict,
        topic: str,
        target_entity_types: list[str],
        permitted_pairs: dict[str, set[str]],
        resource_pool: "ResourcePool",
    ) -> "State":
        """Deserialize from dict using handler registry.

        Uses existing _rehydrate_judgments_quotes for automatic rehydration:
        - resource_url → ResourceQuote.resource
        - canonical string → EntityRef
        """
        from dataclasses import fields
        from pydantic import TypeAdapter
        from interaction_finder.checkpoint import _rehydrate_judgments_quotes

        state = cls(
            topic=topic,
            target_entity_types=target_entity_types,
            permitted_pairs=permitted_pairs,
        )

        def get_rid(url: str) -> "ResourceId":
            """Get ResourceId from URL string."""
            resource = resource_pool.get(url)
            if resource is None:
                raise ValueError(f"Resource {url} not found in pool")
            return resource.id

        for fld in fields(state):
            # Skip constructor args or missing fields
            if fld.name in ("topic", "target_entity_types", "permitted_pairs"):
                continue
            if fld.name not in data:
                continue

            value_data = data[fld.name]

            if fld.name not in _SERIALIZATION_REGISTRY:
                # Simple field - direct assignment
                setattr(state, fld.name, value_data)
                continue

            handler_type, inner_type = _SERIALIZATION_REGISTRY[fld.name]

            if handler_type == "by_resource":
                # dict[str, X_serialized] → dict[ResourceId, X]
                # Rehydrate quotes in entity fields before validation
                if fld.name in (
                    "entities_by_resource",
                    "validated_entities_by_resource",
                ):
                    _rehydrate_entity_quotes(value_data, resource_pool)

                value = {
                    get_rid(url): TypeAdapter(inner_type).validate_python(v)
                    for url, v in value_data.items()
                }
                setattr(state, fld.name, value)

            elif handler_type == "pair_key":
                # dict[str, X_serialized] → dict[EntityPairKey, X]
                # Rehydrate quotes first using existing helper
                judgment_list = list(value_data.values())
                _rehydrate_judgments_quotes(judgment_list, resource_pool, entities=None)

                # Then reconstruct with validated models
                value = {
                    EntityPairKey(*k.split("|")): inner_type.model_validate(v)
                    for k, v in zip(value_data.keys(), judgment_list)
                }
                setattr(state, fld.name, value)

            elif handler_type == "tuple_key":
                # dict[str, X] → dict[tuple, X]
                value = {tuple(k.split("|")): v for k, v in value_data.items()}
                setattr(state, fld.name, value)

            elif handler_type == "set_value":
                # dict[K, list[V]] → dict[K, set[V]]
                value = {k: set(v) for k, v in value_data.items()}
                setattr(state, fld.name, value)

        return state
