"""Pipeline state management.

The State dataclass holds all mutable data shared across pipeline nodes.
Fields are precisely typed and organized by pipeline stage for clarity.

Pipeline flow:
1. ProcessDocumentsNode (per-document, concurrent):
   - Extract entities → entities_by_resource
   - Validate kinds → validated_entities_by_resource
   - Identify proximal sets → proximal_sets_by_resource
   - Extract & assess pairs → pair_assessments_by_resource
2. ConsolidateEntitiesNode → entities_merged, consolidated.entities (global merging + pair reference updates)
3. ConsolidateRelationshipsNode → relationship_mappings, relationship_polarities, consolidated.relationships
4. SweepCoMentionsNode → co_mention_sweep_stats (additional assessments added to pair_assessments_by_resource)
5. ConsolidateNewRelationshipsNode → extends relationship_polarities and consolidated.relationships
6. JudgeCrossDocumentNode → pair_judgments
7. FinalizeNode → ExtractionResult
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interaction_finder.extraction.sweep_co_mentions import CoMentionSweepStats
    from interaction_finder.resources import ResourcePool

from interaction_finder.extraction.models import (
    ConsolidatedData,
    EntityMention,
    EntityPairKey,
    EntityRef,
    PairAssessment,
    PairJudgment,
    PaperQualityAssessment,
    ProximalEntitySet,
)
from interaction_finder.resources import ResourceId, ResourceQuote


@dataclass
class MergeCacheForKind:
    """Per-kind cache for entity merge decisions.

    Isolates cache state by entity kind to enable concurrent processing.
    """

    # (child_canonical, parent_canonical) → (target, reasoning)
    # target is None for "skip", parent for "merge", or custom name for "rename"
    cache: dict[tuple[str, str], tuple[str | None, str]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0


# Registry of special serialization handlers for State fields
# Maps field_name → (handler_type, inner_type_for_validation)
_SERIALIZATION_REGISTRY: dict[str, tuple[str, type | None]] = {
    "entities_by_resource": ("by_resource", dict[str, EntityMention]),
    "validated_entities_by_resource": ("entity_ref_by_resource", dict[str, EntityRef]),
    "global_entities": ("global_entity_ref", dict[str, EntityRef]),
    "proximal_sets_by_resource": ("by_resource", list[ProximalEntitySet]),
    "pair_assessments_by_resource": ("by_resource", list[PairAssessment]),
    "pair_judgments": ("pair_key", PairJudgment),
    "merge_cache_by_kind": ("merge_cache_by_kind", None),
    "consolidated": ("pydantic_model", ConsolidatedData),
    "paper_quality": ("by_resource", PaperQualityAssessment),
}


def _rehydrate_quotes_list(quotes: list[dict], resource_pool: "ResourcePool") -> None:
    """Rehydrate resource_url → Resource and convert dicts back to ResourceQuote."""
    for idx, quote in enumerate(quotes):
        if isinstance(quote, dict):
            if "resource_url" in quote:
                quote["resource"] = resource_pool.get(quote["resource_url"])
                del quote["resource_url"]
            # Replace the dict with a ResourceQuote model
            quotes[idx] = ResourceQuote.model_validate(quote)


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
                _rehydrate_quotes_list(quotes, resource_pool)


def _rehydrate_proximal_sets(sets_data: dict, resource_pool: "ResourcePool") -> None:
    """Rehydrate resource_url → resource in ProximalEntitySet quotes.

    Modifies sets_data in-place. Structure is:
    {resource_url: [ProximalEntitySet_dict, ...]}
    where each set has entity_quotes: {entity_name: [quote_dict, ...]}
    """
    for proximal_sets in sets_data.values():
        for pset in proximal_sets:
            if "entity_quotes" in pset:
                for quotes in pset["entity_quotes"].values():
                    _rehydrate_quotes_list(quotes, resource_pool)


def _rehydrate_pair_assessments(
    assessments_data: dict,
    resource_pool: "ResourcePool",
    validated_entities: dict["ResourceId", dict[str, EntityRef]] | None,
) -> None:
    """Rehydrate resource_url → resource in PairAssessment quotes.

    Modifies assessments_data in-place. Structure is:
    {resource_url: [PairAssessment_dict, ...]}
    where each assessment has quotes: [quote_dict, ...]

    Note: entity1/entity2 fields serialize to canonical strings (not dicts
    with mentions), so they don't need rehydration here.
    """
    for resource_url, assessments in assessments_data.items():
        # Find validated entities for this resource (if available)
        entities_for_resource: dict[str, EntityRef] = {}
        if validated_entities:
            resource = resource_pool.get(resource_url)
            if resource is not None:
                entities_for_resource = validated_entities.get(resource.id, {}) or {}

        for assessment in assessments:
            if "quotes" in assessment:
                _rehydrate_quotes_list(assessment["quotes"], resource_pool)
            # Replace entity string references with validated EntityRef objects
            for field in ("entity1", "entity2"):
                entity = assessment.get(field)
                if isinstance(entity, str) and entity in entities_for_resource:
                    assessment[field] = entities_for_resource[entity]


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
    # Unified consolidation provenance - built incrementally during pipeline
    consolidated: ConsolidatedData = field(default_factory=ConsolidatedData)
    # Global entity index: canonical → EntityRef with aggregated mentions from all resources
    # Built at end of ConsolidateEntitiesNode, used for PairJudgment alias lookup
    global_entities: dict[str, EntityRef] = field(default_factory=dict)
    # Per-kind cache for LLM merge decisions, enabling concurrent kind processing
    # Each kind has its own cache and hit/miss counters
    merge_cache_by_kind: dict[str, MergeCacheForKind] = field(default_factory=dict)

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
    # Count of assessments with updated relationship labels
    relationships_merged: int = 0

    # === Stage 5b: Co-mention Sweep ===
    # Statistics from the co-mention sweep (None if sweep disabled or not yet run)
    co_mention_sweep_stats: "CoMentionSweepStats | None" = None

    # === Stage 6: Cross-Document Judgment ===
    # Final judgments for each unique entity pair across all documents
    pair_judgments: dict[EntityPairKey, PairJudgment] = field(default_factory=dict)

    # === Paper Quality Assessment (populated during entity extraction) ===
    # Quality assessment for each processed resource, keyed by ResourceId
    paper_quality: dict[ResourceId, PaperQualityAssessment] = field(
        default_factory=dict
    )

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
            elif handler_type == "entity_ref_by_resource":
                # dict[ResourceId, dict[str, EntityRef]] → preserve full EntityRef data
                # Bypass EntityRef's JSON serializer that returns only canonical string
                from pydantic.json_schema import to_jsonable_python

                result[fld.name] = {
                    key_str(k): {
                        name: {
                            "canonical": ref.canonical,
                            "mentions": to_jsonable_python(
                                ref.mentions,
                                fallback=lambda x: x.model_dump(mode="json"),
                            ),
                        }
                        for name, ref in entities.items()
                    }
                    for k, entities in value.items()
                }
            elif handler_type == "global_entity_ref":
                # dict[str, EntityRef] → preserve full EntityRef data (not nested by resource)
                from pydantic.json_schema import to_jsonable_python

                result[fld.name] = {
                    canonical: {
                        "canonical": ref.canonical,
                        "mentions": to_jsonable_python(
                            ref.mentions,
                            fallback=lambda x: x.model_dump(mode="json"),
                        ),
                    }
                    for canonical, ref in value.items()
                }
            elif handler_type == "pair_key":
                # dict[EntityPairKey, X] → dict[str, X_serialized]
                result[fld.name] = {
                    str(k): TypeAdapter(inner_type).dump_python(v, mode="json")
                    for k, v in value.items()
                }
            elif handler_type == "merge_cache_by_kind":
                # dict[str, MergeCacheForKind] → dict[str, dict]
                result[fld.name] = {
                    kind: {
                        "cache": {"|".join(k): v for k, v in cache.cache.items()},
                        "hits": cache.hits,
                        "misses": cache.misses,
                    }
                    for kind, cache in value.items()
                }
            elif handler_type == "pydantic_model":
                # Pydantic model → JSON dict
                result[fld.name] = value.model_dump(mode="json")

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
                # Rehydrate quotes before validation based on field type
                if fld.name == "entities_by_resource":
                    _rehydrate_entity_quotes(value_data, resource_pool)
                elif fld.name == "proximal_sets_by_resource":
                    _rehydrate_proximal_sets(value_data, resource_pool)
                elif fld.name == "pair_assessments_by_resource":
                    _rehydrate_pair_assessments(
                        value_data, resource_pool, state.validated_entities_by_resource
                    )
                value = {
                    get_rid(url): TypeAdapter(inner_type).validate_python(v)
                    for url, v in value_data.items()
                }
                setattr(state, fld.name, value)

            elif handler_type == "entity_ref_by_resource":
                # dict[str, dict[str, EntityRef_data]] → dict[ResourceId, dict[str, EntityRef]]
                _rehydrate_entity_quotes(value_data, resource_pool)
                value = {
                    get_rid(url): TypeAdapter(inner_type).validate_python(v)
                    for url, v in value_data.items()
                }
                setattr(state, fld.name, value)

            elif handler_type == "global_entity_ref":
                # dict[str, EntityRef_data] → dict[str, EntityRef]
                # Rehydrate quotes in mentions
                for entity_data in value_data.values():
                    if "mentions" in entity_data:
                        for mention in entity_data["mentions"]:
                            if "quotes" in mention:
                                _rehydrate_quotes_list(mention["quotes"], resource_pool)
                value = TypeAdapter(inner_type).validate_python(value_data)
                setattr(state, fld.name, value)

            elif handler_type == "pair_key":
                # dict[str, X_serialized] → dict[EntityPairKey, X]
                # Rehydrate quotes (and entity refs) using existing helper
                judgment_list = list(value_data.values())
                entities_lookup: dict[str, EntityRef] = {}
                if state.global_entities:
                    entities_lookup = state.global_entities
                elif state.validated_entities_by_resource:
                    # Flatten validated entities as fallback
                    entities_lookup = {
                        canonical: ref
                        for entities in state.validated_entities_by_resource.values()
                        for canonical, ref in entities.items()
                    }
                _rehydrate_judgments_quotes(
                    judgment_list, resource_pool, entities=entities_lookup or None
                )

                # Then reconstruct with validated models
                def parse_pair_key(key_str: str) -> EntityPairKey:
                    """Support legacy EntityPairKey string formats."""
                    import ast
                    import re

                    if "|" in key_str and not key_str.startswith("EntityPairKey("):
                        parts = key_str.split("|")
                        if len(parts) == 2:
                            return EntityPairKey(parts[0], parts[1])
                    try:
                        parsed = ast.literal_eval(key_str)
                        if isinstance(parsed, tuple) and len(parsed) == 2:
                            return EntityPairKey(*parsed)
                    except Exception:
                        pass
                    match = re.match(
                        r"EntityPairKey\(entity1_name='([^']+)', entity2_name='([^']+)'\)",
                        key_str,
                    )
                    if match:
                        return EntityPairKey(match.group(1), match.group(2))
                    raise ValueError(f"Cannot parse EntityPairKey from '{key_str}'")

                value = {
                    parse_pair_key(k): inner_type.model_validate(v)
                    for k, v in zip(value_data.keys(), judgment_list)
                }
                setattr(state, fld.name, value)

            elif handler_type == "merge_cache_by_kind":
                # dict[str, dict] → dict[str, MergeCacheForKind]
                value = {
                    kind: MergeCacheForKind(
                        cache={
                            tuple(k.split("|")): v for k, v in data["cache"].items()
                        },
                        hits=data["hits"],
                        misses=data["misses"],
                    )
                    for kind, data in value_data.items()
                }
                setattr(state, fld.name, value)
            elif handler_type == "pydantic_model":
                # JSON dict → Pydantic model
                setattr(state, fld.name, inner_type.model_validate(value_data))

        return state
