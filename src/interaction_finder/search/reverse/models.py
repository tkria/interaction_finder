"""
Core data models and types for reverse search functionality.

This module defines Pydantic models for reverse search, including resource
representations, matching results, session tracking, configuration, and errors.
All models provide comprehensive validation and type safety.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

from interaction_finder.search.base import SearchResult, SearchResults
from interaction_finder.search.reverse.utils import normalize_url


# Core resource models


class KnownResource(BaseModel):
    """
    A target resource to find via reverse search.

    Represents a paper or document we want to locate through search queries.
    Each resource has a URL and optionally a PMID for PubMed papers.
    The canonical_url provides a normalized identifier for deduplication.

    Fields:
        url: str - Direct URL to the resource
        pmid: Optional[str] - PubMed ID if this is a PubMed paper
        hint_fields: Dict[str, str] - Domain-specific metadata (celltype, marker, etc.)
        canonical_url: str - Computed normalized URL for deduplication

    Example:
        >>> resource = KnownResource(
        ...     pmid="12345678",
        ...     url="https://example.com/paper",
        ...     hint_fields={"celltype": "CD8+ T cell"}
        ... )
        >>> resource.canonical_url
        'https://pubmed.ncbi.nlm.nih.gov/12345678/'
    """

    url: str = Field(description="Direct URL to the resource")
    pmid: Optional[str] = Field(None, description="PubMed ID if applicable")
    hint_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Domain-specific metadata (celltype, marker, etc.)",
    )
    canonical_url: str = Field(
        default="", description="Computed normalized URL for deduplication"
    )

    @model_validator(mode="after")
    def compute_canonical_url(self) -> "KnownResource":
        """
        Compute canonical URL for deduplication.

        If PMID exists, use PubMed URL format.
        Otherwise, normalize the provided URL (lowercase, strip trailing slash,
        remove tracking parameters).

        Returns:
            Self with canonical_url set
        """
        if self.pmid:
            # PMID takes precedence - use canonical PubMed URL
            self.canonical_url = f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid.strip()}/"
        else:
            # Use comprehensive normalization: lowercase, strip trailing slash,
            # remove tracking parameters, remove fragments
            self.canonical_url = normalize_url(self.url)

        return self

    def __hash__(self) -> int:
        """Hash based on canonical_url for set operations."""
        return hash(self.canonical_url)

    def __eq__(self, other: object) -> bool:
        """Equality based on canonical_url."""
        if not isinstance(other, KnownResource):
            return False
        return self.canonical_url == other.canonical_url


# Matching models


class ResourceMatch(BaseModel):
    """
    Records how a resource was matched to a search result.

    Captures the relationship between a target resource and a search result,
    including the matching method used and confidence level.

    Fields:
        resource: KnownResource - The target resource that was matched
        search_result: SearchResult - The search result that matched
        match_method: Literal - How the match was determined (pmid/url/doi/title_similarity)
        confidence: float - Match confidence (0.0-1.0)
        query_index: int - Which query found this resource

    Example:
        >>> match = ResourceMatch(
        ...     resource=resource,
        ...     search_result=result,
        ...     match_method="pmid",
        ...     confidence=1.0,
        ...     query_index=0
        ... )
    """

    resource: KnownResource = Field(description="The target resource that was matched")
    search_result: SearchResult = Field(description="The search result that matched")
    match_method: Literal["pmid", "url", "doi", "title_similarity"] = Field(
        description="How the match was determined"
    )
    confidence: float = Field(description="Match confidence score", ge=0.0, le=1.0)
    query_index: int = Field(description="Which query found this resource")


# Result models


class ReverseSearchResult(BaseModel):
    """
    Result of a single query execution.

    Captures the outcome of executing one search query, including what was found,
    how many new resources were discovered, and execution metrics.

    Fields:
        query: str - The search query string
        query_index: int - Sequential index of this query
        search_results: SearchResults - Raw search results from backend
        resources_found: List[KnownResource] - Target resources found in these results
        new_finds: int - How many new resources this query discovered
        cumulative_coverage: float - Coverage achieved after this query (0.0-1.0)
        search_time: float - Time taken for search in seconds
        backend: str - Search backend used

    Example:
        >>> result = ReverseSearchResult(
        ...     query="BRCA1 breast cancer",
        ...     query_index=0,
        ...     search_results=search_results,
        ...     resources_found=[resource1, resource2],
        ...     new_finds=2,
        ...     cumulative_coverage=0.4,
        ...     search_time=1.5,
        ...     backend="pubmed"
        ... )
    """

    query: str = Field(description="The search query string")
    query_index: int = Field(description="Sequential index of this query")
    search_results: SearchResults = Field(description="Raw search results from backend")
    resources_found: List[KnownResource] = Field(
        description="Target resources found in these results"
    )
    new_finds: int = Field(description="How many new resources this query discovered")
    cumulative_coverage: float = Field(
        description="Coverage achieved after this query", ge=0.0, le=1.0
    )
    search_time: float = Field(description="Time taken for search in seconds")
    backend: str = Field(description="Search backend used")


class ReverseSearchSession(BaseModel):
    """
    Complete session with all queries and final metrics.

    Aggregates all query results and provides overall session statistics,
    including coverage metrics, timing, and stopping reason.

    Fields:
        target_resources: List[KnownResource] - Resources we tried to find
        query_results: List[ReverseSearchResult] - Results from each query
        matches: List[ResourceMatch] - All matches found
        total_queries: int - Total number of queries executed
        final_coverage: float - Final coverage achieved (0.0-1.0)
        found_count: int - Number of resources successfully found
        unfound_resources: List[KnownResource] - Resources not found
        total_time: float - Total session time in seconds
        stopping_reason: Literal - Why the session ended

    Properties:
        coverage_pct: float - Final coverage as percentage (0.0-100.0)

    Example:
        >>> session = ReverseSearchSession(
        ...     target_resources=[res1, res2, res3],
        ...     query_results=[result1, result2],
        ...     matches=[match1, match2],
        ...     total_queries=2,
        ...     final_coverage=0.67,
        ...     found_count=2,
        ...     unfound_resources=[res3],
        ...     total_time=5.3,
        ...     stopping_reason="coverage_achieved"
        ... )
        >>> session.coverage_pct
        67.0
    """

    target_resources: List[KnownResource] = Field(
        description="Resources we tried to find"
    )
    query_results: List[ReverseSearchResult] = Field(
        description="Results from each query"
    )
    matches: List[ResourceMatch] = Field(description="All matches found")
    total_queries: int = Field(description="Total number of queries executed")
    final_coverage: float = Field(description="Final coverage achieved", ge=0.0, le=1.0)
    found_count: int = Field(description="Number of resources successfully found")
    unfound_resources: List[KnownResource] = Field(description="Resources not found")
    total_time: float = Field(description="Total session time in seconds")
    stopping_reason: Literal[
        "coverage_achieved", "consecutive_zero_finds", "max_queries"
    ] = Field(description="Why the session ended")

    @property
    def coverage_pct(self) -> float:
        """Final coverage as percentage (0.0-100.0)."""
        return self.final_coverage * 100.0


# Configuration model


class ReverseSearchConfig(BaseModel):
    """
    Configuration for reverse search with validated constraints.

    Defines all parameters for reverse search execution, including stopping
    criteria, query generation, clustering, matching, performance tuning,
    and backend settings. All numeric parameters have validated bounds.

    Stopping criteria:
        coverage_target: float - Stop when this coverage fraction is reached (0.5-1.0)
        consecutive_zero_limit: int - Stop after this many queries with no finds (1-10)
        max_queries: int - Hard limit on total queries (5-200)

    Query generation:
        keyword_extractor: Literal - Algorithm for keyword extraction (yake/rake/tfidf/llm)
        keywords_per_query: int - Number of keywords per query (3-15)
        use_hint_fields: bool - Include hint_fields in query generation
        sort_by: Literal - Sort order for search results (relevance/date/date_desc)
        llm_query_config: Dict[str, Any] - Configuration for LLM-based query generation

    Clustering:
        enable_clustering: bool - Whether to cluster resources before query generation
        min_cluster_size: int - Minimum resources per cluster (>=2)
        target_clusters: int - Target number of clusters (2-10)

    Matching:
        title_similarity_threshold: float - Min similarity for title matching (0.5-1.0)

    Performance:
        batch_pmid_fetch_size: int - Batch size for PMID metadata fetching (1-200)
        cache_ttl_days: int - Cache TTL for search results (1-365)
        pmid_metadata_cache_days: int - Cache TTL for PMID metadata (1-365)

    Backend:
        search_backend: str - Search backend to use (default: "pubmed")
        max_results_per_query: int - Max results per query (10-500)

    Example:
        >>> config = ReverseSearchConfig(
        ...     coverage_target=0.95,
        ...     consecutive_zero_limit=3,
        ...     max_queries=50,
        ...     keyword_extractor="yake",
        ...     keywords_per_query=7
        ... )
        >>> # LLM-based query generation
        >>> config_llm = ReverseSearchConfig(
        ...     keyword_extractor="llm",
        ...     sort_by="date_desc",
        ...     llm_query_config={"model": "anthropic:claude-3-sonnet", "temperature": 0.5}
        ... )
    """

    # Stopping criteria
    coverage_target: float = Field(
        0.95, ge=0.5, le=1.0, description="Stop when this coverage fraction is reached"
    )
    consecutive_zero_limit: int = Field(
        3, ge=1, le=10, description="Stop after this many queries with no finds"
    )
    max_queries: int = Field(
        100, ge=5, le=200, description="Hard limit on total queries"
    )

    # Query generation
    keyword_extractor: Literal["yake", "rake", "tfidf", "llm"] = Field(
        "yake", description="Algorithm for keyword extraction"
    )
    keywords_per_query: int = Field(
        7, ge=3, le=15, description="Number of keywords per query"
    )
    use_hint_fields: bool = Field(
        True, description="Include hint_fields in query generation"
    )
    sort_by: Literal["relevance", "date", "date_desc"] = Field(
        "relevance", description="Sort order for search results"
    )
    llm_query_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "openai:gpt-4o-mini",
            "temperature": 0.7,
            "max_queries_per_resource": 1,
            "max_queries_per_cluster": 1,
            "backend_specific_syntax": True,
        },
        description="Configuration for LLM-based query generation",
    )

    # Clustering
    enable_clustering: bool = Field(
        True, description="Whether to cluster resources before query generation"
    )
    min_cluster_size: int = Field(3, ge=2, description="Minimum resources per cluster")
    target_clusters: int = Field(
        4, ge=2, le=10, description="Target number of clusters"
    )

    # Matching
    title_similarity_threshold: float = Field(
        0.8, ge=0.5, le=1.0, description="Min similarity for title matching"
    )

    # Performance
    batch_pmid_fetch_size: int = Field(
        200, ge=1, le=200, description="Batch size for PMID metadata fetching"
    )
    cache_ttl_days: int = Field(
        7, ge=1, le=365, description="Cache TTL for search results"
    )
    pmid_metadata_cache_days: int = Field(
        30, ge=1, le=365, description="Cache TTL for PMID metadata"
    )

    # Backend
    search_backend: str = Field("pubmed", description="Search backend to use")
    max_results_per_query: int = Field(
        100, ge=10, le=500, description="Max results per query"
    )


# Error hierarchy


class ReverseSearchError(Exception):
    """
    Base exception for reverse search errors.

    Provides structured error reporting with message and optional context.
    All reverse search errors inherit from this base class.

    Args:
        message: str - Human-readable error description
        context: Optional[Dict[str, Any]] - Additional error context

    Example:
        >>> raise ReverseSearchError(
        ...     "Query generation failed",
        ...     context={"resources": 5, "extractor": "yake"}
        ... )
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Format error with context."""
        parts = [self.message]
        if self.context:
            parts.append(f"Context: {self.context}")
        return " | ".join(parts)


class ResourceParseError(ReverseSearchError):
    """
    Failed to parse JSONL input.

    Raised when input resources cannot be parsed from JSONL format,
    typically due to malformed JSON, missing required fields, or
    validation errors.

    Example:
        >>> raise ResourceParseError(
        ...     "Invalid JSON on line 5",
        ...     context={"line": 5, "content": "..."}
        ... )
    """

    pass


class QueryGenerationError(ReverseSearchError):
    """
    Failed to generate queries.

    Raised when query generation fails, typically due to insufficient
    text for keyword extraction or extractor errors.

    Example:
        >>> raise QueryGenerationError(
        ...     "No keywords extracted",
        ...     context={"resources": 3, "extractor": "yake"}
        ... )
    """

    pass


class MatchingError(ReverseSearchError):
    """
    Failed to match results.

    Raised when matching resources to search results fails, typically
    due to missing metadata or comparison errors.

    Example:
        >>> raise MatchingError(
        ...     "PMID extraction failed",
        ...     context={"result_url": "https://...", "backend": "pubmed"}
        ... )
    """

    pass
