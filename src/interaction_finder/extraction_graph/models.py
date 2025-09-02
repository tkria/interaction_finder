"""
Pydantic models for the extraction graph pipeline.

These models define the typed outputs for each agent in the graph,
following the pydantic-graph pattern of using BaseModel outputs.
"""

from typing import List, Dict, Any, Optional, Literal, TYPE_CHECKING, Union
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..resources import ResourceQuote


class EntityInfo(BaseModel):
    """
    Information about a single extracted entity with full attribution.

    This model tracks not just the entity name and type, but also
    exactly where it was found for complete provenance.
    """

    name: str = Field(
        description="Entity name (e.g., 'BRCA1', 'lung cancer', 'T cell')"
    )
    kind: str = Field(description="Entity type from task configuration")
    source_url: str = Field(description="URL of document where entity was found")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in extraction (0-1)"
    )
    context: Optional[str] = Field(None, description="Surrounding text context")
    synonyms: List[str] = Field(
        default_factory=list, description="Alternative names found"
    )
    # Resource tracking fields (optional for backward compatibility)
    resource_quotes: List["ResourceQuote"] = Field(
        default_factory=list, description="Resource quotes supporting this entity"
    )

    def get_document_text(self, document_group) -> str:
        """Get text content from the document where this entity was found."""
        try:
            if self.source_url in document_group.chunks:
                chunks = document_group.chunks[self.source_url]
                return "\n\n".join([chunk.get("text", "") for chunk in chunks])
            return ""
        except (AttributeError, KeyError):
            return ""

    def get_all_source_urls(self) -> List[str]:
        """Get all source URLs including those from resource quotes."""
        urls = [self.source_url]
        for quote in self.resource_quotes:
            if quote.resource.id.url not in urls:
                urls.append(quote.resource.id.url)
        return urls


class RouteDecision(BaseModel):
    """Decision from the initial routing agent."""

    classification: Literal["scientific", "non-scientific", "unclear"] = Field(
        description="Classification of document group content"
    )
    reasoning: str = Field(description="Explanation of classification decision")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in classification"
    )


class EntityListOut(BaseModel):
    """Output from entity extraction agent."""

    entities: List[EntityInfo] = Field(
        description="Extracted entities with attribution", min_length=0
    )
    kinds_searched: List[str] = Field(description="Entity kinds that were searched for")
    processing_notes: Optional[str] = Field(
        None, description="Notes about extraction process"
    )


class EntityAssessmentOut(BaseModel):
    """Assessment of relationship between two entities."""

    entity_a: EntityInfo = Field(description="First entity in relationship")
    entity_b: EntityInfo = Field(description="Second entity in relationship")
    relationship_type: str = Field(description="Type of relationship (from config)")
    confidence: Literal["high", "moderate", "low"] = Field(
        description="Confidence in relationship assessment"
    )
    evidence: List[str] = Field(
        description="Text evidence supporting the relationship", min_length=1
    )
    source_urls: List[str] = Field(
        description="URLs where relationship evidence was found"
    )
    reasoning: str = Field(description="Explanation of assessment")
    # Resource tracking fields (optional for backward compatibility)
    evidence_quotes: List["ResourceQuote"] = Field(
        default_factory=list, description="Resource quotes supporting the evidence"
    )

    def get_all_evidence_texts(self) -> List[str]:
        """Get all evidence texts from both string evidence and resource quotes."""
        all_evidence = list(self.evidence)  # Copy existing string evidence
        for quote in self.evidence_quotes:
            for quote_text in quote.get_all_quote_texts():
                if quote_text not in all_evidence:
                    all_evidence.append(quote_text)
        return all_evidence

    def get_all_source_urls(self) -> List[str]:
        """Get all source URLs including those from evidence quotes."""
        urls = list(self.source_urls)  # Copy existing URLs
        for quote in self.evidence_quotes:
            if quote.resource.id.url not in urls:
                urls.append(quote.resource.id.url)
        return urls


class EntityPairOut(BaseModel):
    """Final validated entity pair with complete provenance."""

    entity_a: EntityInfo = Field(description="First entity")
    entity_b: EntityInfo = Field(description="Second entity")
    relationship: str = Field(description="Relationship type")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence score (0-1)"
    )
    evidence: List[str] = Field(description="Supporting evidence texts", min_length=1)
    source_documents: List[str] = Field(description="URLs of source documents")
    assessment_details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed assessment information"
    )
    # Resource tracking fields (optional for backward compatibility)
    evidence_quotes: List["ResourceQuote"] = Field(
        default_factory=list, description="Resource quotes supporting the evidence"
    )

    def get_all_evidence_texts(self) -> List[str]:
        """Get all evidence texts from both string evidence and resource quotes."""
        all_evidence = list(self.evidence)  # Copy existing string evidence
        for quote in self.evidence_quotes:
            for quote_text in quote.get_all_quote_texts():
                if quote_text not in all_evidence:
                    all_evidence.append(quote_text)
        return all_evidence

    def get_all_source_urls(self) -> List[str]:
        """Get all source URLs including those from source documents and evidence quotes."""
        urls = list(self.source_documents)  # Copy existing URLs
        for quote in self.evidence_quotes:
            if quote.resource.id.url not in urls:
                urls.append(quote.resource.id.url)
        return urls


class ValidationOut(BaseModel):
    """Output from quality validation agent."""

    verdict: Literal["approve", "revise", "reject"] = Field(
        description="Validation decision"
    )
    feedback: str = Field(description="Feedback for improvement (empty if approved)")
    quality_score: float = Field(
        ge=0.0, le=1.0, description="Quality assessment score (0-1)"
    )
    issues_found: List[str] = Field(
        default_factory=list, description="Specific issues identified"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Specific improvement suggestions"
    )


class IndividualEntityContext(BaseModel):
    """Context extracted for a single entity with full attribution."""

    entity: EntityInfo = Field(description="The entity being analyzed")
    mentions: List[str] = Field(
        description="Direct text mentions of this entity", min_length=0
    )
    claims: List[str] = Field(description="Claims made about this entity", min_length=0)
    evidence: List[str] = Field(
        description="Supporting evidence for the entity", min_length=0
    )
    chunk_contexts: List[str] = Field(
        description="Relevant chunk excerpts containing the entity", min_length=0
    )
    context_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in context extraction"
    )


class IndividualEntityAssessment(BaseModel):
    """Assessment of single entity's relationship potential."""

    entity: EntityInfo = Field(description="The assessed entity")
    context: IndividualEntityContext = Field(description="Entity-specific context")
    relationship_potential: Literal["high", "moderate", "low", "none"] = Field(
        description="Assessed potential for relationships"
    )
    target_entity_hints: List[str] = Field(
        description="Names of potential related entities found", min_length=0
    )
    reasoning: str = Field(description="Detailed assessment reasoning")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in assessment")
    processing_notes: Optional[str] = Field(
        None, description="Additional processing notes"
    )


class EntityAggregationOut(BaseModel):
    """Result of aggregating individual assessments into pairs."""

    high_confidence_pairs: List[EntityPairOut] = Field(
        description="Entity pairs with high confidence", min_length=0
    )
    moderate_confidence_pairs: List[EntityPairOut] = Field(
        description="Entity pairs with moderate confidence", min_length=0
    )
    unmatched_entities: List[EntityInfo] = Field(
        description="Entities without clear pair matches", min_length=0
    )
    aggregation_reasoning: str = Field(description="Explanation of aggregation process")
    total_individuals_processed: int = Field(
        description="Number of individual entities processed"
    )
    aggregation_confidence: float = Field(
        ge=0.0, le=1.0, description="Overall aggregation confidence"
    )


class ExtractionSummary(BaseModel):
    """Summary of complete extraction process for a document group."""

    document_group_urls: List[str] = Field(description="URLs in processed group")
    total_chunks: int = Field(description="Total chunks processed")
    entity_counts: Dict[str, int] = Field(description="Count of entities by kind")
    pairs_extracted: int = Field(description="Number of entity pairs found")
    pairs_validated: int = Field(description="Number of pairs that passed validation")
    individual_entities_processed: int = Field(
        0, description="Number of entities processed individually"
    )
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )
    node_path: List[str] = Field(description="Path through graph nodes")
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Quality metrics for the extraction"
    )


class BatchExtractionResult(BaseModel):
    """Result of processing multiple document groups."""

    total_groups: int = Field(description="Number of document groups processed")
    successful_groups: int = Field(description="Number successfully processed")
    total_entities: Dict[str, int] = Field(
        description="Total entities extracted by kind"
    )
    total_pairs: int = Field(description="Total entity pairs extracted")
    entity_pairs: List[EntityPairOut] = Field(
        default_factory=list, description="All extracted entity pairs"
    )
    group_summaries: List[ExtractionSummary] = Field(
        description="Summary for each processed group"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Errors encountered during processing"
    )
    processing_config: Dict[str, Any] = Field(
        description="Configuration used for extraction"
    )


def convert_pairs_to_final_format(
    all_pairs: List[EntityPairOut],
) -> List[Dict[str, Any]]:
    """
    Convert EntityPairOut objects to final aggregated format.

    Groups pairs by entity names and aggregates evidence from multiple sources.

    Args:
        all_pairs: List of EntityPairOut objects from processing

    Returns:
        List of aggregated pairs in the final output format
    """
    # Group pairs by entity names (order-normalized)
    pair_groups = {}

    for pair in all_pairs:
        # Filter out same-type pairs (e.g., disease-disease, gene-gene)
        if pair.entity_a.kind == pair.entity_b.kind:
            continue  # Skip same-type pairs

        # Create normalized key for grouping (alphabetical order)
        entities = sorted(
            [
                (pair.entity_a.name, pair.entity_a.kind),
                (pair.entity_b.name, pair.entity_b.kind),
            ]
        )
        key = f"{entities[0][0]}:{entities[0][1]}|{entities[1][0]}:{entities[1][1]}"

        if key not in pair_groups:
            pair_groups[key] = []
        pair_groups[key].append(pair)

    # Aggregate each group
    aggregated_pairs = []
    for key, group in pair_groups.items():
        if not group:
            continue

        # Use the first pair as the base
        base_pair = group[0]

        # Collect all evidence and resources
        all_evidence = []
        all_resources = set()
        confidence_scores = []

        for pair in group:
            # Collect evidence
            all_evidence.extend(pair.evidence)
            # Collect source documents
            all_resources.update(pair.source_documents)
            # Collect confidence scores
            confidence_scores.append(pair.confidence)

        # Determine aggregate confidence
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            if avg_confidence >= 0.8:
                final_confidence = "high"
            elif avg_confidence >= 0.6:
                final_confidence = "medium"
            else:
                final_confidence = "low"
        else:
            final_confidence = "low"

        # Combine evidence into reasoning
        unique_evidence = []
        seen_evidence = set()
        for evidence in all_evidence:
            if evidence and evidence.strip() not in seen_evidence:
                unique_evidence.append(evidence.strip())
                seen_evidence.add(evidence.strip())

        combined_reasoning = (
            ". ".join(unique_evidence)
            if unique_evidence
            else "Evidence from multiple sources supports this relationship."
        )

        # Determine entity order (ensure consistent ordering)
        entities = sorted(
            [
                (base_pair.entity_a.name, base_pair.entity_a.kind),
                (base_pair.entity_b.name, base_pair.entity_b.kind),
            ]
        )

        # Create aggregated pair
        aggregated_pair = {
            "first": entities[0][0],
            "first_kind": entities[0][1],
            "second": entities[1][0],
            "second_kind": entities[1][1],
            "reasoning": combined_reasoning,
            "resources": sorted(list(all_resources)),
            "confidence": final_confidence,
        }

        aggregated_pairs.append(aggregated_pair)

    return aggregated_pairs


# Rebuild models to resolve forward references
def _rebuild_models():
    """Rebuild models to resolve ResourceQuote forward references."""
    try:
        from ..resources import ResourceQuote  # Import at runtime

        EntityInfo.model_rebuild()
        EntityAssessmentOut.model_rebuild()
        EntityPairOut.model_rebuild()
    except ImportError:
        # ResourceQuote not available, models will work but without validation
        pass


# Attempt to rebuild on import
_rebuild_models()
