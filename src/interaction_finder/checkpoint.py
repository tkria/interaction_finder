"""Unified checkpoint model for multi-stage pipeline.

This module provides a single PipelineCheckpoint model that supersedes the
legacy stage-specific models (BridgingTermsOut, WidesearchCheckpoint, ExtractionResult).

The checkpoint follows a superset design where each stage adds optional nested data
while preserving all data from previous stages. A single top-level ResourcePool
accumulates resources across all stages.
"""

from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchResult

if TYPE_CHECKING:
    from interaction_finder.extraction.models import (
        ExtractionMetadata,
        PairJudgment,
    )


def _needs_rehydration(judgments: list[dict]) -> bool:
    """Check if judgments need quote rehydration (have resource_url instead of resource)."""
    if not judgments or not isinstance(judgments[0], dict):
        return False
    first = judgments[0]
    # Check both structures: old (assessments list) and new (spread dict with categories)
    for key in ["assessments", "spread"]:
        container = first.get(key)
        if not container:
            continue
        # Extract first assessment from either structure
        assessments = container if isinstance(container, list) else []
        if not assessments and isinstance(container, dict):
            for cat in ("supporting", "refuting", "neutral", "irrelevant"):
                if container.get(cat):
                    assessments = container[cat]
                    break
        if assessments and isinstance(assessments[0], dict):
            quotes = assessments[0].get("quotes", [])
            if quotes and isinstance(quotes[0], dict):
                return "resource_url" in quotes[0]
    return False


def _rehydrate_judgments_quotes(judgments: list[dict], pool: ResourcePool) -> None:
    """Rehydrate ResourceQuote objects by replacing resource_url with Resource.

    Modifies judgments in-place. Shared by PipelineCheckpoint and ExtractionResult.
    """

    def iter_assessments(judgment: dict):
        """Yield assessment dicts from assessments list or spread categories."""
        for a in judgment.get("assessments", []):
            if isinstance(a, dict):
                yield a
        spread = judgment.get("spread", {})
        for cat in ("supporting", "refuting", "neutral", "irrelevant"):
            for a in spread.get(cat, []):
                if isinstance(a, dict):
                    yield a

    def inject(quote: dict):
        """Replace resource_url with Resource from pool."""
        resource = pool.get(quote["resource_url"])
        if resource is None:
            raise ValueError(f"Resource {quote['resource_url']} not in pool")
        quote["resource"] = resource
        del quote["resource_url"]

    for judgment in judgments:
        for assessment in iter_assessments(judgment):
            for quote in assessment.get("quotes", []):
                inject(quote)
            for entity in [assessment.get(k) for k in ["entity1", "entity2"]]:
                if isinstance(entity, dict):
                    for quote in entity.get("quotes", []):
                        inject(quote)


class KeywordsStageData(BaseModel):
    """Keyword extraction stage results.

    Contains bridging terms discovered from review articles along with
    metadata about the extraction process.
    """

    terms: list[str] = Field(description="Bridging terms discovered")
    scores: list[float] = Field(
        description="Relevance scores (0-1, higher is more relevant)"
    )
    total_documents_processed: int = Field(
        ge=0, description="Number of documents analyzed"
    )
    rounds_completed: int = Field(ge=1, description="Number of search rounds completed")
    coverage_assessment: str = Field(
        min_length=50, description="Assessment of coverage and what was discovered"
    )
    resource_urls: list[str] = Field(
        description="URLs of resources added during keyword extraction (review articles)"
    )


class SearchStageData(BaseModel):
    """Wide search stage results.

    Contains search results, queries executed, and query-to-URL mappings.
    The resource_urls property infers URLs from results.
    """

    results: list[SearchResult] = Field(
        description="Search results selected during widesearch"
    )
    queries: list[str] = Field(description="All queries executed (cumulative)")
    query_results: dict[str, list[str]] = Field(
        description="Mapping from each query to selected URLs"
    )
    keyphrases: list[str] = Field(
        description="Keyphrases incorporated into query generation"
    )
    rounds_completed: int = Field(ge=0, description="Number of search rounds completed")

    @property
    def resource_urls(self) -> list[str]:
        """Infer resource URLs from search results.

        Returns URLs of all selected search results. This avoids storing
        redundant data since results already contain URLs.
        """
        return [result.url for result in self.results]


class ExtractionStageData(BaseModel):
    """Entity extraction stage results.

    Contains entity pair judgments with full provenance and extraction metadata.
    Does not have resource_urls since extraction doesn't add new resources.
    """

    target_entity_types: list[str] = Field(
        description="Entity types that were extracted"
    )
    permitted_pairs: dict[str, list[str]] = Field(
        description="Permitted entity kind pairs for filtering"
    )
    judgments: list["PairJudgment"] = Field(
        description="All pair judgments (accepted and rejected)"
    )
    metadata: "ExtractionMetadata" = Field(description="Extraction statistics")


class PipelineCheckpoint(BaseModel):
    """Unified checkpoint for entire pipeline.

    Progressive structure where each stage adds optional nested data while
    preserving all data from previous stages. The single top-level ResourcePool
    accumulates resources across all stages.

    Stage progression:
    - After keywords: has topic, resources, keywords
    - After search: has topic, resources, keywords, search
    - After extraction: has topic, resources, keywords, search, extraction

    The checkpoint is a true superset - later stages contain all data from
    earlier stages, enabling complete provenance tracking.
    """

    # Core fields (always present)
    topic: str = Field(description="Research topic being investigated")
    resources: ResourcePool = Field(
        description="Unified resource pool accumulating across all stages"
    )

    # Optional stage-specific data (added progressively)
    keywords: Optional[KeywordsStageData] = Field(
        None, description="Keyword extraction results (Stage 1)"
    )
    search: Optional[SearchStageData] = Field(
        None, description="Wide search results (Stage 2)"
    )
    extraction: Optional[ExtractionStageData] = Field(
        None, description="Entity extraction results (Stage 3)"
    )

    @property
    def stage(self) -> Literal["keywords", "search", "extraction"]:
        """Determine latest completed stage.

        Returns the most advanced stage present in the checkpoint.

        Raises:
            ValueError: If checkpoint has no stage data
        """
        if self.extraction is not None:
            return "extraction"
        elif self.search is not None:
            return "search"
        elif self.keywords is not None:
            return "keywords"
        else:
            raise ValueError("Invalid checkpoint: no stage data present")

    @model_validator(mode="before")
    @classmethod
    def _rehydrate_extraction_quotes(cls, data):
        """Rehydrate ResourceQuote objects in extraction stage if present."""
        if not isinstance(data, dict):
            return data
        extraction = data.get("extraction")
        if not isinstance(extraction, dict):
            return data
        judgments = extraction.get("judgments", [])
        if not _needs_rehydration(judgments):
            return data
        if "resources" not in data:
            raise ValueError("Cannot rehydrate quotes: resources pool missing")
        pool = ResourcePool.model_validate(data["resources"])
        _rehydrate_judgments_quotes(judgments, pool)
        return data


def _rebuild_models():
    """Rebuild models to resolve forward references.

    This function is called after all extraction models are loaded to resolve
    forward references used during TYPE_CHECKING.
    """
    from interaction_finder.extraction.models import (
        ExtractionMetadata,
        PairJudgment,
    )

    ExtractionStageData.model_rebuild()
