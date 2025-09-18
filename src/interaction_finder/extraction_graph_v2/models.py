"""
Pydantic models for extraction graph V2 with ResourceQuote integration.

These models maintain the essential pattern of individual entity assessment
while ensuring every entity and claim has verifiable quotes with exact positions.
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field

# Import with TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..resources import ResourceQuote


class EntityWithQuotes(BaseModel):
    """
    Entity with ALL its occurrences via ResourceQuotes.

    This model preserves the complete provenance chain - every mention
    of the entity can be traced to exact character positions.
    """

    name: str = Field(description="Entity name (e.g., 'BRCA1', 'breast cancer')")
    kind: str = Field(description="Entity type from task configuration")
    quotes: List["ResourceQuote"] = Field(
        description="All occurrences with exact positions", min_length=1
    )
    confidence: float = Field(
        ge=0.0, le=1.0, default=1.0, description="Confidence in entity extraction"
    )

    def validate(self) -> bool:
        """Ensure all quotes are valid and non-empty."""
        if not self.quotes:
            return False
        return all(quote.count > 0 for quote in self.quotes)

    @property
    def all_contexts(self) -> List[str]:
        """Get contexts for all occurrences of this entity."""
        contexts = []
        for quote in self.quotes:
            contexts.extend([quote.get_context(i + 1, 200) for i in range(quote.count)])
        return contexts

    @property
    def total_occurrences(self) -> int:
        """Total number of times this entity appears in all documents."""
        return sum(quote.count for quote in self.quotes)

    def get_all_source_urls(self) -> List[str]:
        """Get all unique source URLs where this entity appears."""
        urls = []
        for quote in self.quotes:
            url = quote.resource.id.url
            if url not in urls:
                urls.append(url)
        return urls


class IndividualAssessment(BaseModel):
    """
    Assessment of ONE entity's relationship potential with evidence quotes.

    This captures the key insight: assess each entity individually based on
    its specific context, with verifiable evidence quotes.
    """

    entity: EntityWithQuotes = Field(description="The assessed entity")
    relationship_potential: Literal["high", "medium", "low", "none"] = Field(
        description="Assessed potential for relationships"
    )
    related_entities: List[str] = Field(
        description="Names of potentially related entities found", default_factory=list
    )
    evidence_quotes: List["ResourceQuote"] = Field(
        description="Verified evidence supporting the assessment", default_factory=list
    )
    reasoning: str = Field(description="Detailed assessment reasoning")
    confidence: float = Field(
        ge=0.0, le=1.0, default=1.0, description="Confidence in this assessment"
    )

    def validate_evidence(self) -> bool:
        """Ensure evidence quotes contain the entity name."""
        if not self.evidence_quotes:
            return self.relationship_potential == "none"

        entity_name_lower = self.entity.name.lower()
        for quote in self.evidence_quotes:
            texts = quote.get_all_quote_texts()
            if any(entity_name_lower in text.lower() for text in texts):
                return True

        return False

    def get_all_evidence_texts(self) -> List[str]:
        """Get all evidence texts from resource quotes."""
        all_evidence = []
        for quote in self.evidence_quotes:
            all_evidence.extend(quote.get_all_quote_texts())
        return all_evidence

    def get_evidence_contexts(self) -> List[str]:
        """Get context around evidence quotes."""
        contexts = []
        for quote in self.evidence_quotes:
            contexts.extend([quote.get_context(i + 1, 150) for i in range(quote.count)])
        return contexts


class EntityPairOut(BaseModel):
    """
    Final entity pair with complete ResourceQuote provenance chain.

    Every pair maintains the full provenance from source documents
    through individual assessments to final output.
    """

    entity_a: EntityWithQuotes = Field(description="First entity")
    entity_b: EntityWithQuotes = Field(description="Second entity")
    relationship: str = Field(
        description="Relationship type (from config)", default="interaction"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Overall confidence in the relationship"
    )
    evidence_quotes: List["ResourceQuote"] = Field(
        description="Verified evidence supporting the relationship", min_length=1
    )
    reasoning: str = Field(description="Explanation of the relationship", default="")

    def to_output_format(self) -> Dict[str, Any]:
        """
        Convert to required pairs.json format with quote details.

        Returns:
            Dictionary with resources containing actual quotes and positions
        """
        # Build resources dictionary with quote details
        resources = {}
        for quote in self.evidence_quotes:
            resource_id = quote.resource.id.id
            if resource_id not in resources:
                resources[resource_id] = []

            # Add each occurrence as a separate entry
            for i in range(quote.count):
                quote_text = quote.get_quote_text(i + 1)  # 1-indexed
                span_start, span_end = quote.spans[i]

                resources[resource_id].append(
                    {"quote": quote_text, "span": [span_start, span_end]}
                )

        # Build reasoning from evidence if not provided
        reasoning = self.reasoning
        if not reasoning and self.evidence_quotes:
            evidence_texts = []
            for quote in self.evidence_quotes:
                evidence_texts.extend(quote.get_all_quote_texts())
            reasoning = ". ".join(evidence_texts[:3])  # First 3 pieces of evidence

        return {
            "first": self.entity_a.name,
            "first_kind": self.entity_a.kind,
            "second": self.entity_b.name,
            "second_kind": self.entity_b.kind,
            "reasoning": reasoning,
            "resources": resources,
            "confidence": self.confidence,
        }

    def validate_provenance(self) -> bool:
        """Validate that provenance chain is complete."""
        # Must have evidence
        if not self.evidence_quotes:
            return False

        # Both entities must have quotes
        if not (self.entity_a.validate() and self.entity_b.validate()):
            return False

        # Evidence should mention at least one entity
        entity_names = [self.entity_a.name.lower(), self.entity_b.name.lower()]
        for quote in self.evidence_quotes:
            texts = [text.lower() for text in quote.get_all_quote_texts()]
            if any(name in text for name in entity_names for text in texts):
                return True

        return False

    def get_all_source_urls(self) -> List[str]:
        """Get all unique source URLs involved in this pair."""
        urls = []

        # From entities
        urls.extend(self.entity_a.get_all_source_urls())
        urls.extend(self.entity_b.get_all_source_urls())

        # From evidence
        for quote in self.evidence_quotes:
            url = quote.resource.id.url
            if url not in urls:
                urls.append(url)

        return urls


# Output models for agents
class EntityQuoteOut(BaseModel):
    """Schema for individual entity quote with source attribution."""

    text: str = Field(description="Exact quote text from the document")
    source: str = Field(description="Resource identifier (e.g., 'Resource abc123')")


class EntityOut(BaseModel):
    """Schema for individual entity with quotes and metadata."""

    name: str = Field(description="Exact entity name as it appears in text")
    kind: str = Field(description="Entity kind (gene, disease, etc.)")
    quotes: List[EntityQuoteOut] = Field(
        description="Supporting quotes with text and source attribution",
        min_length=1,  # Require at least one quote per entity
    )


class EntityListOut(BaseModel):
    """Output from entity extraction agent."""

    entities: List[EntityOut] = Field(
        description="Extracted entities with name, kind, and supporting quotes"
    )
    entity_kinds: List[str] = Field(description="Entity kinds that were searched for")
    reasoning: str = Field(description="Explanation of extraction process", default="")


class AssessmentOut(BaseModel):
    """Output from individual assessment agent."""

    potential: Literal["high", "medium", "low", "none"] = Field(
        description="Relationship potential for this entity"
    )
    related: List[str] = Field(
        description="Names of potentially related entities", default_factory=list
    )
    evidence: List[str] = Field(
        description="Exact quotes supporting the assessment", default_factory=list
    )
    reasoning: str = Field(description="Detailed reasoning for the assessment")


class BatchExtractionResultV2(BaseModel):
    """Result of V2 extraction pipeline, compatible with CLI expectations."""

    total_groups: int = Field(
        description="Number of document groups processed", default=1
    )
    successful_groups: int = Field(
        description="Number successfully processed", default=1
    )
    total_entities: Dict[str, int] = Field(
        description="Total entities extracted by kind", default_factory=dict
    )
    total_pairs: int = Field(description="Total entity pairs extracted", default=0)
    entity_pairs: List[EntityPairOut] = Field(
        default_factory=list, description="All extracted entity pairs"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Errors encountered during processing"
    )

    @classmethod
    def from_pairs(cls, pairs: List[EntityPairOut]) -> "BatchExtractionResultV2":
        """Create result from list of entity pairs."""
        entity_counts = {}
        for pair in pairs:
            entity_counts[pair.entity_a.kind] = (
                entity_counts.get(pair.entity_a.kind, 0) + 1
            )
            entity_counts[pair.entity_b.kind] = (
                entity_counts.get(pair.entity_b.kind, 0) + 1
            )

        return cls(
            total_pairs=len(pairs),
            entity_pairs=pairs,
            total_entities=entity_counts,
        )


# Rebuild models to resolve forward references
def _rebuild_models():
    """Rebuild models to resolve ResourceQuote forward references."""
    try:
        # Import needed to resolve forward references during model rebuild
        from ..resources import ResourceQuote  # noqa: F401

        EntityWithQuotes.model_rebuild()
        IndividualAssessment.model_rebuild()
        EntityPairOut.model_rebuild()
        BatchExtractionResultV2.model_rebuild()
    except ImportError:
        # ResourceQuote not available, models will work but without validation
        pass


# Attempt to rebuild on import
_rebuild_models()
