"""Unified checkpoint model for multi-stage pipeline.

This module provides a single PipelineCheckpoint model that supersedes the
legacy stage-specific models (BridgingTermsOut, WidesearchCheckpoint, ExtractionResult).

The checkpoint follows a superset design where each stage adds optional nested data
while preserving all data from previous stages. A single top-level ResourcePool
accumulates resources across all stages.
"""

from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field, model_serializer, model_validator
from pydantic_core import to_jsonable_python

from interaction_finder.resources import ResourcePool, ResourceQuote
from interaction_finder.search.models import SearchResult
from interaction_finder.version import get_version_string

if TYPE_CHECKING:
    from interaction_finder.extraction.models import (
        ConsolidatedData,
        EntityRef,
        ExtractionMetadata,
        PairJudgment,
    )


def _get_consolidated_data_default():
    """Deferred import to avoid circular dependency."""
    from interaction_finder.extraction.models import ConsolidatedData

    return ConsolidatedData()


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
            for cat in ("positive", "negative", "neutral", "irrelevant"):
                if container.get(cat):
                    assessments = container[cat]
                    break
        if assessments and isinstance(assessments[0], dict):
            quotes = assessments[0].get("quotes", [])
            if quotes and isinstance(quotes[0], dict):
                return "resource_url" in quotes[0]
    return False


def _rehydrate_judgments_quotes(
    judgments: list[dict],
    pool: ResourcePool,
    entities: dict[str, dict] | None = None,
) -> None:
    """Rehydrate ResourceQuote and EntityRef objects.

    Replaces resource_url strings with Resource objects from pool, and
    canonical entity name strings with full EntityRef dicts from entities dict.

    Modifies judgments in-place. Shared by PipelineCheckpoint and ExtractionResult.

    Parameters:
        judgments: List of judgment dicts to modify in-place
        pool: ResourcePool to resolve resource_url references
        entities: Optional dict mapping canonical names to EntityRef dicts
    """

    def iter_assessments(judgment: dict):
        """Yield assessment dicts from assessments list or spread categories."""
        for a in judgment.get("assessments", []):
            if isinstance(a, dict):
                yield a
        spread = judgment.get("spread", {})
        for cat in ("positive", "negative", "neutral", "irrelevant"):
            for a in spread.get(cat, []):
                if isinstance(a, dict):
                    yield a

    def inject_quote(quote: dict):
        """Replace resource_url with Resource from pool (if needed)."""
        # Skip if already has resource object (not yet serialized)
        if "resource" in quote:
            return
        # Otherwise rehydrate from resource_url
        if "resource_url" not in quote:
            return  # Nothing to do
        resource = pool.get(quote["resource_url"])
        if resource is None:
            raise ValueError(f"Resource {quote['resource_url']} not in pool")
        quote["resource"] = resource
        del quote["resource_url"]

    def inject_entity(entity_ref: str | dict | list) -> dict | list:
        """Replace canonical string with full EntityRef dict, or rehydrate quotes in existing dict."""
        # String reference - look up full entity from entities dict
        if isinstance(entity_ref, str):
            if entities is None:
                raise ValueError(
                    f"Entity string reference '{entity_ref}' found but no entities dict provided"
                )
            if entity_ref not in entities:
                raise ValueError(f"Entity '{entity_ref}' not found in entities dict")
            # Return the full entity dict, but need to add canonical field
            entity_dict = entities[entity_ref].copy()
            entity_dict["canonical"] = entity_ref
            # Rehydrate quotes in mentions within this entity
            for mention in entity_dict.get("mentions", []):
                if isinstance(mention, dict):
                    quotes_list = mention.get("quotes", [])
                    # Replace quote dicts with ResourceQuote objects
                    for i, quote in enumerate(quotes_list):
                        if isinstance(quote, dict):
                            inject_quote(quote)
                            # Construct ResourceQuote from dict
                            quotes_list[i] = ResourceQuote.model_construct(**quote)
            return entity_dict
        # Already a dict or list - rehydrate quotes within it
        return entity_ref

    for judgment in judgments:
        # Rehydrate top-level judgment entities (if present as strings)
        for entity_field in ["entity1", "entity2"]:
            if entity_field in judgment:
                judgment[entity_field] = inject_entity(judgment[entity_field])

        for assessment in iter_assessments(judgment):
            # Rehydrate assessment-level quotes
            for quote in assessment.get("quotes", []):
                inject_quote(quote)

            # Rehydrate entity structures
            for entity_field in ["entity1", "entity2"]:
                entity = assessment.get(entity_field)
                if entity is None:
                    continue

                # String reference (new hoisted format) - replace with full EntityRef dict
                if isinstance(entity, str):
                    assessment[entity_field] = inject_entity(entity)
                    entity = assessment[entity_field]

                # EntityRef dict structure: {canonical, mentions}
                if isinstance(entity, dict) and "mentions" in entity:
                    for mention in entity.get("mentions", []):
                        if isinstance(mention, dict):
                            quotes_list = mention.get("quotes", [])
                            # Replace quote dicts with ResourceQuote objects
                            for i, quote in enumerate(quotes_list):
                                if isinstance(quote, dict):
                                    inject_quote(quote)
                                    # Construct ResourceQuote from dict
                                    quotes_list[i] = ResourceQuote.model_construct(
                                        **quote
                                    )


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

    EntityRef objects in judgments are dehydrated to canonical name strings during
    serialization, with full entity data stored in the entities dict (similar to
    how ResourceQuote references ResourcePool).
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
    consolidated: "ConsolidatedData" = Field(
        default_factory=_get_consolidated_data_default,
        description="Unified consolidation provenance for entities and relationships",
    )

    @model_serializer(mode="wrap")
    def _serialize(self, serializer, info):
        """Collect unique EntityRefs and serialize with entities dict."""
        if info.mode != "json":
            return serializer(self)

        # Collect unique EntityRefs before serialization
        entities: dict[str, "EntityRef"] = {}
        for judgment in self.judgments:
            for assessment in judgment.iter_assessments():
                if assessment.entity1.canonical not in entities:
                    entities[assessment.entity1.canonical] = assessment.entity1
                if assessment.entity2.canonical not in entities:
                    entities[assessment.entity2.canonical] = assessment.entity2

        # Do default serialization (EntityRefs become strings via their serializer)
        data = serializer(self)

        # Manually serialize entities dict with full EntityRef data
        # We serialize mentions in JSON mode (to convert ResourceQuote.resource to resource_url)
        # but keep the top-level EntityRef structure (canonical + mentions)
        data["entities"] = {
            canonical: {
                "canonical": ref.canonical,
                "mentions": to_jsonable_python(
                    ref.mentions, fallback=lambda x: x.model_dump(mode="json")
                ),
            }
            for canonical, ref in entities.items()
        }

        return data


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
    created_by: Optional[str] = Field(
        default=None,
        description="Version of interaction-finder that created this checkpoint",
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
        """Rehydrate ResourceQuote and EntityRef objects in extraction stage if present."""
        if not isinstance(data, dict):
            return data
        extraction = data.get("extraction")
        if not isinstance(extraction, dict):
            return data
        judgments = extraction.get("judgments", [])
        if not judgments:
            return data
        # Always rehydrate to handle both quote rehydration and entity tuple→dict conversion
        if "resources" not in data:
            raise ValueError("Cannot rehydrate quotes: resources pool missing")
        pool = ResourcePool.model_validate(data["resources"])
        entities = extraction.get("entities")
        _rehydrate_judgments_quotes(judgments, pool, entities)
        return data


def _rebuild_models():
    """Rebuild models to resolve forward references.

    This function is called after all extraction models are loaded to resolve
    forward references used during TYPE_CHECKING.
    """
    from interaction_finder.extraction.models import (
        ConsolidatedData,
        EntityRef,
        ExtractionMetadata,
        PairJudgment,
    )

    ExtractionStageData.model_rebuild()
