"""
Pydantic models for extraction graph V3 with semantic caching and pair candidates.

Extends V2 models with support for:
- Semantic caching of extraction and assessment results
- Explicit pair candidate tracking before evaluation
- Enhanced metadata and metrics tracking
- Checkpoint-based resumability
"""

from typing import List, Dict, Any, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# Import V2 models for reuse
from ..extraction_graph_v2.models import (
    EntityWithQuotes,
    EntityPairOut,
    QuoteErrorRecord,
)

# Import with TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class PairCandidate(BaseModel):
    """
    Candidate entity pair generated before LLM evaluation.

    Separates candidate generation (co-occurrence analysis) from evaluation
    (LLM assessment), enabling caching and parallelism optimization.
    """

    entity_a: "EntityWithQuotes" = Field(description="First entity")
    entity_b: "EntityWithQuotes" = Field(description="Second entity")
    co_occurrence_count: int = Field(description="Number of co-occurrences found", ge=0)
    shared_resources: List[str] = Field(
        description="Resource IDs where entities co-occur", default_factory=list
    )
    generation_strategy: Literal[
        "same_chunk",
        "adjacent_chunks",
        "document_level",
        "assessment_suggested",
        "both",
    ] = Field(description="How this candidate was generated")

    def get_pair_key(self) -> tuple[str, str]:
        """Get normalized pair key for deduplication."""
        # Always return in sorted order for consistent lookup
        sorted_names = sorted([self.entity_a.name, self.entity_b.name])
        return (sorted_names[0], sorted_names[1])


class PairEvaluationOut(BaseModel):
    """
    Output from pair evaluation agent.

    LLM assessment of whether a candidate pair represents a genuine relationship.
    """

    relationship_exists: bool = Field(
        description="Whether a relationship exists between the entities"
    )
    relationship_type: str = Field(
        description="Type of relationship (from task config)", default="interaction"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this evaluation"
    )
    evidence: List[str] = Field(
        description="Exact quotes supporting the relationship", default_factory=list
    )
    reasoning: str = Field(description="Detailed explanation of the evaluation")


class ExtractionMetadata(BaseModel):
    """
    Metadata about the extraction run for provenance and reproducibility.

    Tracks pipeline version, model configuration, and timing information.
    """

    timestamp: datetime = Field(
        default_factory=datetime.now, description="When extraction started"
    )
    model: str = Field(description="LLM model used (e.g., 'openai:gpt-4o')")
    pipeline_version: str = Field(
        description="Pipeline version identifier", default="v3"
    )
    prompt_version: str = Field(description="Prompt template version", default="v3.0")
    entity_kinds: List[str] = Field(
        description="Entity kinds requested", default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "pipeline_version": self.pipeline_version,
            "prompt_version": self.prompt_version,
            "entity_kinds": self.entity_kinds,
        }


class BatchExtractionResultV3(BaseModel):
    """
    Result of V3 extraction pipeline with enhanced metadata and caching stats.

    Extends V2 result format with cache statistics, stage-specific metrics,
    and checkpoint path for resumability.
    """

    # Core results (V2 compatibility)
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
    quote_errors: List[QuoteErrorRecord] = Field(
        default_factory=list, description="Quote validation errors and corrections"
    )

    # V3 enhancements
    metadata: Optional[ExtractionMetadata] = Field(
        description="Extraction run metadata", default=None
    )
    cache_stats: Dict[str, int] = Field(
        description="Cache hit/miss statistics", default_factory=dict
    )
    stage_metrics: Dict[str, Dict[str, Any]] = Field(
        description="Per-stage performance metrics", default_factory=dict
    )
    checkpoint_path: Optional[str] = Field(
        description="Path to checkpoint file for resumability", default=None
    )

    @classmethod
    def from_pairs(
        cls,
        pairs: List[EntityPairOut],
        metadata: Optional[ExtractionMetadata] = None,
        cache_stats: Optional[Dict[str, int]] = None,
        stage_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        checkpoint_path: Optional[str] = None,
    ) -> "BatchExtractionResultV3":
        """Create result from list of entity pairs with optional V3 enhancements."""
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
            metadata=metadata,
            cache_stats=cache_stats or {},
            stage_metrics=stage_metrics or {},
            checkpoint_path=checkpoint_path,
        )


# Rebuild models to resolve forward references
def _rebuild_models():
    """Rebuild models to resolve ResourceQuote forward references."""
    try:
        # Import needed to resolve forward references during model rebuild
        from ..resources import ResourceQuote  # noqa: F401

        PairCandidate.model_rebuild()
        PairEvaluationOut.model_rebuild()
        ExtractionMetadata.model_rebuild()
        BatchExtractionResultV3.model_rebuild()
    except ImportError:
        # ResourceQuote not available, models will work but without validation
        pass


# Attempt to rebuild on import
_rebuild_models()
