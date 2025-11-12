"""Data models for the extraction pipeline.

This module defines:
1. Core data structures (dataclasses for State-internal, Pydantic for serialization)
2. Agent output models (Pydantic) returned by LLM agents
3. Final pipeline output models (Pydantic)

The separation ensures clean boundaries: agents return Pydantic models with
raw string quotes, which are immediately converted to ResourceQuote objects
for storage in State. Models that appear in serialized output use Pydantic BaseModel
to enable proper JSON serialization via model_dump().
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interaction_finder.resources import ResourceId, ResourcePool, ResourceQuote

# =============================================================================
# Core data structures (used in State)
# =============================================================================


@dataclass
class EntityMention:
    """Entity found in a single resource (after quote validation).

    This is the validated, resource-quote-enriched version of EntityInfo.
    Multiple EntityInfo instances with the same name are merged during conversion.

    Attributes:
        kind: Entity type (e.g., "gene", "disease", "protein")
        name: Canonical entity name (e.g., "BRCA1")
        aliases: Names as they appear in text (e.g., ["BRCA-1", "BRCA1"])
        quotes: Supporting quotes from the resource (validated ResourceQuote objects)
        reasoning: Explanation of extraction choices (merged if duplicates combined)
    """

    kind: str
    name: str
    aliases: list[str]
    quotes: list[ResourceQuote]
    reasoning: str


@dataclass
class ProximalEntitySet:
    """Group of entities found in close proximity within a document.

    Used to identify regions where multiple entities co-occur, which may
    indicate potential associations worth investigating.

    Attributes:
        entities: Set of canonical entity names found in proximity
        chunk_range: (start_chunk_idx, end_chunk_idx) spanning all quotes
        entity_quotes: Mapping from canonical name to quotes for that entity
    """

    entities: set[str]
    chunk_range: tuple[int, int]
    entity_quotes: dict[str, list[ResourceQuote]]


class EntityPairKey(NamedTuple):
    """Identifier for an entity pair across documents.

    Entity names are ordered lexicographically by their kinds to ensure
    consistent pairing (e.g., always gene-disease, not disease-gene).

    Attributes:
        entity1_name: First entity canonical name
        entity2_name: Second entity canonical name
    """

    entity1_name: str
    entity2_name: str


# =============================================================================
# Agent output models (Pydantic - what LLMs return)
# =============================================================================


class EntityInfo(BaseModel):
    """LLM output for a single entity.

    This is converted to EntityMention after quote validation.
    """

    model_config = ConfigDict(populate_by_name=True)

    kind: str = Field(
        alias="type", description='Entity type (e.g., "gene", "disease", "protein")'
    )
    name: str = Field(description="Canonical name of the entity")
    aliases: list[str] = Field(
        description="Names as they appear in the text, verbatim", min_length=1
    )
    quotes: list[str] = Field(
        description="Direct quotes from text supporting this entity", min_length=1
    )
    reasoning: str = Field(
        min_length=20, description="Brief explanation of why this entity was extracted"
    )


class EntityExtractionOut(BaseModel):
    """LLM output: all entities found in document."""

    entities: list[EntityInfo] = Field(description="Entities found in document")


class EntityMergeDecision(BaseModel):
    """LLM decision on whether to merge two entities.

    Used when one entity name is a substring of another (e.g., "BRCA" vs "BRCA1").
    """

    parent_entity: str = Field(description="Entity to keep (canonical name)")
    child_entity: str = Field(description="Entity to merge into parent")
    should_merge: bool = Field(description="Whether these entities should be merged")
    reasoning: str = Field(min_length=20, description="Explanation of merge decision")


class EntityMergeDecisions(BaseModel):
    """LLM output: batch of merge decisions."""

    decisions: list[EntityMergeDecision] = Field(
        description="Merge decisions for all candidate pairs"
    )


class ProximalPairInfo(BaseModel):
    """LLM output for a single pair extracted from a proximal region."""

    entity1: str = Field(description="First entity (canonical name)")
    entity2: str = Field(description="Second entity (canonical name)")
    relationship_types: list[str] = Field(
        description="Possible relationship types (may suggest multiple)",
        min_length=1,
    )
    supporting_quotes: list[str] = Field(
        description="Direct quotes supporting this association", min_length=1
    )


class ProximalPairExtraction(BaseModel):
    """LLM output: all pairs found in a proximal region."""

    pairs: list[ProximalPairInfo] = Field(
        description="Associations found in this text region"
    )
    reasoning: str = Field(
        min_length=20, description="Brief explanation of extraction choices"
    )


class PairEvidenceJudgment(BaseModel):
    """LLM assessment of pair evidence in a single document.

    Evaluates the strength of evidence and selects the most appropriate
    relationship type from candidates.
    """

    relationship: str = Field(
        description="Selected relationship type (from candidates or new)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this pair's validity based on evidence"
    )
    reasoning: str = Field(
        min_length=30, description="Explanation of confidence and relationship choice"
    )
    supporting_quote_ids: list[int] = Field(
        description="Indices (0-based) of quotes that support this assessment"
    )


class CrossDocumentJudgment(BaseModel):
    """LLM final judgment synthesizing evidence across all documents."""

    accepted: bool = Field(
        description="Whether to accept this pair as a valid association"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in this judgment"
    )
    reasoning: str = Field(
        min_length=50, description="Detailed explanation of the decision"
    )


# =============================================================================
# Final pipeline output models
# =============================================================================


class SimpleEntity(BaseModel):
    """Lightweight entity representation for final output.

    Contains only the essential information needed to identify an entity
    without the full provenance chain.
    """

    name: str = Field(description="Canonical entity name")
    kind: str = Field(description='Entity type (e.g., "gene", "disease")')
    aliases: list[str] = Field(description="Alternative names found in text")


class PairAssessment(BaseModel):
    """Evidence assessment for one pair in one document.

    This is the core result of per-document pair analysis, containing
    the selected relationship, confidence level, and supporting evidence.

    Attributes:
        resource_id: Which document this assessment is from
        entity1: First entity with full information
        entity2: Second entity with full information
        relationship: Selected relationship type
        quotes: Supporting quotes from this document
        confidence: Qualitative confidence in this association
        reasoning: Explanation of confidence level
    """

    resource_id: ResourceId
    entity1: EntityMention
    entity2: EntityMention
    relationship: str
    quotes: list[ResourceQuote]
    confidence: Literal["high", "medium", "low"]
    reasoning: str


class PairJudgment(BaseModel):
    """Final cross-document judgment on an entity pair.

    This is the primary output of the extraction pipeline, containing
    all per-document assessments and the final accept/reject decision.
    """

    entity1: SimpleEntity = Field(description="First entity")
    entity2: SimpleEntity = Field(description="Second entity")
    relationship: str = Field(description="Final relationship type")
    assessments: list[PairAssessment] = Field(
        description="All per-document assessments"
    )
    accepted: bool = Field(description="Whether this pair is accepted")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in final judgment"
    )
    reasoning: str = Field(description="Explanation of final decision")


class ExtractionMetadata(BaseModel):
    """Summary statistics for extraction run."""

    topic: str
    resource_count: int
    total_entities_found: int
    entities_after_validation: int
    entities_merged: int
    merge_cache_hits: int
    merge_cache_misses: int
    proximal_sets_found: int
    total_pairs_found: int
    pairs_accepted: int
    pairs_rejected: int
    quotes_validated: int
    quotes_failed: int


class ExtractionResult(BaseModel):
    """Final pipeline output.

    Contains accepted pair judgments with full provenance and shared resource pool.
    Provides efficient serialization by storing resources once and referencing
    by ID in quotes.
    """

    resources: ResourcePool = Field(
        description="Shared pool of all resources referenced by quotes"
    )
    judgments: list[PairJudgment] = Field(
        description="All pair judgments (accepted and rejected)"
    )
    metadata: ExtractionMetadata = Field(description="Extraction statistics")

    @model_validator(mode="before")
    @classmethod
    def _rehydrate_quotes(cls, data):
        """Restore Resource objects in quotes from resource_id references.

        During deserialization, ResourceQuotes contain resource_id instead of
        full Resource. This validator looks up resources from the pool and
        injects them into quote dicts before Pydantic validates the structure.
        """
        if not isinstance(data, dict) or "resources" not in data:
            return data

        # Check if this is serialized data (judgments are dicts with resource_id in quotes)
        judgments = data.get("judgments", [])
        if not judgments or not isinstance(judgments[0], dict):
            return data

        # Check first assessment for resource_id (indicates serialized data)
        first_judgment = judgments[0]
        if not first_judgment.get("assessments"):
            return data

        first_assessment = first_judgment["assessments"][0]
        if not first_assessment.get("quotes"):
            return data

        first_quote = first_assessment["quotes"][0]
        if not isinstance(first_quote, dict) or "resource_id" not in first_quote:
            return data

        # Deserialize pool and inject resources into all quotes
        pool = ResourcePool.model_validate(data["resources"])

        def inject_resource(quote_dict: dict) -> None:
            """Replace resource_id with actual Resource from pool."""
            rid_data = quote_dict["resource_id"]
            counter = int(rid_data["id"].split("_")[0])
            rid = ResourceId(url=rid_data["url"], counter=counter)

            resource = pool.get(rid)
            if resource is None:
                raise ValueError(f"Resource {rid.id} not found in pool")

            quote_dict["resource"] = resource
            del quote_dict["resource_id"]

        # Process all quotes in all assessments in all judgments
        for judgment in judgments:
            for assessment in judgment["assessments"]:
                # Inject resources into quotes
                for quote_dict in assessment["quotes"]:
                    inject_resource(quote_dict)

                # Inject resources into entity quotes (entity1 and entity2)
                for entity_key in ["entity1", "entity2"]:
                    entity = assessment[entity_key]
                    for quote_dict in entity["quotes"]:
                        inject_resource(quote_dict)

        return data
