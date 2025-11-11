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
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from interaction_finder.resources import ResourceId, ResourceQuote

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
class PairMention:
    """Association found in a single resource.

    Attributes:
        entity1: First entity (canonical name)
        entity2: Second entity (canonical name)
        relationship_type: Type of relationship (e.g., "associated_with", "regulates")
        quotes: Supporting quotes from the resource
    """

    entity1: str
    entity2: str
    relationship_type: str
    quotes: list[ResourceQuote]


# Type alias for pair identification across resources
PairKey = tuple[str, str, str]  # (entity1, entity2, relationship_type)


class EntityAssessment(BaseModel):
    """Evidence assessment for one entity in one resource.

    Attributes:
        resource_id: Which resource this assessment is from
        strength: Evidence strength rating
        rationale: Why this rating was assigned
        quotes: Subset of entity's quotes supporting this assessment
    """

    resource_id: ResourceId
    strength: Literal["none", "weak", "strong"]
    rationale: str
    quotes: list[ResourceQuote]


class PairAssessment(BaseModel):
    """Evidence assessment for one pair in one resource.

    Attributes:
        resource_id: Which resource this assessment is from
        strength: Evidence strength rating
        rationale: Why this rating was assigned
        quotes: Supporting quotes for this assessment
    """

    resource_id: ResourceId
    strength: Literal["none", "weak", "strong"]
    rationale: str
    quotes: list[ResourceQuote]


class FinalJudgment(BaseModel):
    """Cross-document judgment on whether a pair is valid.

    Attributes:
        accepted: Whether the pair is accepted as valid
        confidence: Confidence level in this judgment
        rationale: Explanation of the decision
    """

    accepted: bool
    confidence: Literal["high", "medium", "low"]
    rationale: str


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


class PairInfo(BaseModel):
    """LLM output for a single pair."""

    entity1: str = Field(description="First entity (canonical name)")
    entity2: str = Field(description="Second entity (canonical name)")
    relationship_type: str = Field(
        description='Relationship type (e.g., "associated_with", "regulates")'
    )
    supporting_quotes: list[str] = Field(
        description="Direct quotes supporting this association", min_length=1
    )


class PairExtractionOut(BaseModel):
    """LLM output: all pairs found in document."""

    pairs: list[PairInfo] = Field(description="Associations found in document")
    reasoning: str = Field(
        min_length=20, description="Brief explanation of extraction choices"
    )


class EntityEvidenceAssessment(BaseModel):
    """LLM assessment of entity's relevance in one document."""

    strength: Literal["none", "weak", "strong"] = Field(
        description="Evidence strength for this entity's relevance to the topic"
    )
    rationale: str = Field(
        min_length=30, description="Explanation for the strength rating"
    )
    supporting_quote_ids: list[int] = Field(
        description="Indices (0-based) of quotes that support this assessment"
    )


class PairEvidenceAssessment(BaseModel):
    """LLM assessment of pair's validity in one document."""

    strength: Literal["none", "weak", "strong"] = Field(
        description="Evidence strength for this pair's validity"
    )
    rationale: str = Field(
        min_length=30, description="Explanation for the strength rating"
    )
    supporting_quote_ids: list[int] = Field(
        description="Indices (0-based) of quotes that support this assessment"
    )


class FinalJudgmentOut(BaseModel):
    """LLM final judgment across all documents."""

    accepted: bool = Field(description="Whether to accept this pair as valid")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in this judgment"
    )
    rationale: str = Field(
        min_length=50, description="Detailed explanation of the decision"
    )


# =============================================================================
# Final pipeline output
# =============================================================================


class PairWithProvenance(BaseModel):
    """Accepted pair with complete provenance chain.

    This is the primary output of the extraction pipeline.
    """

    entity1: str = Field(description="First entity (canonical name)")
    entity2: str = Field(description="Second entity (canonical name)")
    relationship_type: str = Field(description="Type of relationship")
    entity1_type: str = Field(description='Entity type (e.g., "gene")')
    entity2_type: str = Field(description='Entity type (e.g., "disease")')
    all_quotes: list[ResourceQuote] = Field(
        description="All quotes from all resources supporting this pair"
    )
    assessments: list[PairAssessment] = Field(
        description="Per-resource evidence assessments"
    )
    final_judgment: FinalJudgment = Field(
        description="Cross-document judgment on validity"
    )


class ExtractionMetadata(BaseModel):
    """Summary statistics for extraction run."""

    topic: str
    resource_count: int
    total_entities_found: int
    total_pairs_found: int
    pairs_accepted: int
    pairs_rejected: int
    quotes_validated: int
    quotes_failed: int


class ExtractionResult(BaseModel):
    """Final pipeline output.

    Contains only accepted pairs with full provenance.
    """

    accepted_pairs: list[PairWithProvenance] = Field(
        description="Pairs accepted after evidence assessment"
    )
    metadata: ExtractionMetadata = Field(description="Extraction statistics")
