"""
Investigation logging for reverse search pipeline debugging and analysis.

Provides comprehensive structured logging of reverse search execution, capturing
resource processing, clustering, query generation, search execution, matching,
and errors throughout the pipeline. All log entries follow JSON Lines format
for incremental processing.

Usage:
    async with InvestigationLogger("session.jsonl") as logger:
        await logger.log_session_start(resources, config, "pubmed")
        await logger.log_content_fetch(resources, contents)
        # ... pipeline execution
        await logger.log_session_end(session)

All entries share a common session_id for traceability. The logger provides
thread-safe concurrent writes via asyncio.Lock and automatic file lifecycle
management via async context manager protocol.
"""

import asyncio
import traceback as tb
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import aiofiles
from pydantic import BaseModel, Field

from interaction_finder.search.base import SearchResults
from interaction_finder.search.reverse.models import (
    KnownResource,
    ResourceMatch,
    ReverseSearchConfig,
    ReverseSearchSession,
)


# Helper function for compact resource ID formatting


def _format_resource_ref(resource: KnownResource) -> str:
    """
    Format resource as compact identifier.

    Uses PMID:X format (with prefix) if PMID available, otherwise raw URL (no prefix).
    This compact format improves log readability while maintaining sufficient information
    for analysis and cross-referencing.

    Args:
        resource: KnownResource to format

    Returns:
        Compact identifier string (PMID:X or URL)
    """
    return f"PMID:{resource.pmid}" if resource.pmid else resource.url


# Detail models for structured per-item logging


class ResourceFetchDetail(BaseModel):
    """Per-resource content fetch details."""

    resource_id: str = Field(
        description="Compact identifier: PMID:X (with prefix) or raw URL (no prefix)"
    )
    source: Optional[Literal["metadata", "content"]] = Field(
        None, description="Content source type (None if fetch failed)"
    )
    title: Optional[str] = Field(
        None, description="Title excerpt (first 100 chars, None if fetch failed)"
    )
    content_length: int = Field(description="Content length in characters")
    success: bool = Field(description="Whether fetch succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class KeywordDetail(BaseModel):
    """Keyword with score from extractor."""

    keyword: str = Field(description="Keyword text")
    score: Optional[float] = Field(
        None, description="Score from extractor (null for LLM)"
    )


class SearchResultDetail(BaseModel):
    """Search result with position tracking."""

    result_index: int = Field(description="0-based position in result list")
    resource_id: str = Field(
        description="Compact identifier: PMID:X (with prefix) or raw URL (no prefix)"
    )
    title: str = Field(description="Result title")
    url: str = Field(description="Result URL")


class MatchDetail(BaseModel):
    """Match attempt for a single search result."""

    result_index: int = Field(description="0-based position in search results")
    result_id: str = Field(description="Compact result identifier")
    matched: bool = Field(description="Whether a match was found")
    matched_resource: Optional[str] = Field(
        None, description="Compact ID of matched target resource"
    )
    match_method: Optional[str] = Field(
        None, description="Method used (pmid/url/doi/title)"
    )
    confidence: float = Field(description="Match confidence (0.0-1.0)")


class QueryResultsSummary(BaseModel):
    """
    Summary of search results and matched resources for a query.

    Consolidates search results and match outcomes to enable correlation analysis
    between queries, results, and target resources. Uses 0-based indexing consistent
    with SearchResultDetail and MatchDetail.

    Fields:
        results: List[str] - Compact identifiers for all search results (PMID:X or URL)
        found_resources: List[Dict[str, Any]] - Matched resources with indices and methods
    """

    results: List[str] = Field(
        description="Compact identifiers: PMID:X (with prefix) or raw URL (no prefix) in 0-based order"
    )
    found_resources: List[Dict[str, Any]] = Field(
        description="Matched resources: {resource: str, index: int, method: str, confidence: float}"
    )


# Base log entry model


class InvestigationLogEntry(BaseModel):
    """
    Base model for all investigation log entries.

    All log entries include timestamp, pipeline stage identifier, and session ID
    for correlation. Specific entry types inherit from this base and add their
    own fields.

    Fields:
        timestamp: str - ISO 8601 timestamp with timezone (UTC)
        stage: Literal - Pipeline stage identifier
        session_id: str - Unique session identifier (UUID4)
    """

    timestamp: str = Field(description="ISO 8601 timestamp with timezone (UTC)")
    stage: Literal[
        "session_start",
        "content_fetch",
        "clustering",
        "keyword_extraction",
        "query_construction",
        "query_generation",  # Deprecated: legacy stage for backward compatibility
        "search_execution",
        "matching",
        "session_end",
        "error",
    ] = Field(description="Pipeline stage identifier")
    session_id: str = Field(description="Unique session identifier (UUID4)")


# Specific entry models


class SessionStartEntry(InvestigationLogEntry):
    """
    Logged at session initialization.

    Captures target resources (as compact IDs), configuration, and backend selection
    for the entire reverse search session.

    Fields:
        stage: Literal["session_start"] - Always "session_start"
        target_count: int - Number of target resources to find
        target_resources: List[str] - Compact target resource IDs
        config: Dict[str, Any] - Serialized reverse search configuration
        backend: str - Search backend name (e.g., "pubmed")
    """

    stage: Literal["session_start"] = "session_start"
    target_count: int = Field(description="Number of target resources to find")
    target_resources: List[str] = Field(
        description="Compact target resource IDs: PMID:X (with prefix) or raw URL (no prefix)"
    )
    config: Dict[str, Any] = Field(
        description="Serialized reverse search configuration"
    )
    backend: str = Field(description="Search backend name")


class ContentFetchEntry(InvestigationLogEntry):
    """
    Logged after fetching content for target resources.

    Captures per-resource fetch details including source type, title, content length,
    success status, and any errors. Source counts and failed count provide summary.

    Fields:
        stage: Literal["content_fetch"] - Always "content_fetch"
        resources: List[ResourceFetchDetail] - Per-resource fetch details
        source_counts: Dict[str, int] - Distribution of content types
        failed_count: int - Number of resources that failed to fetch
    """

    stage: Literal["content_fetch"] = "content_fetch"
    resources: List[ResourceFetchDetail] = Field(
        description="Per-resource fetch details"
    )
    source_counts: Dict[str, int] = Field(description="Distribution of content types")
    failed_count: int = Field(description="Number of resources that failed to fetch")


class ClusteringEntry(InvestigationLogEntry):
    """
    Logged after clustering resources (or skipping clustering).

    Captures clustering configuration, resource assignments, and cluster
    statistics for query generation.

    Fields:
        stage: Literal["clustering"] - Always "clustering"
        enabled: bool - Whether clustering was performed
        resource_count: int - Total number of resources processed
        clusters: List[Dict[str, Any]] - Cluster details with statistics
    """

    stage: Literal["clustering"] = "clustering"
    enabled: bool = Field(description="Whether clustering was performed")
    resource_count: int = Field(description="Total number of resources processed")
    clusters: List[Dict[str, Any]] = Field(
        description="Cluster details with statistics"
    )


class KeywordExtractionEntry(InvestigationLogEntry):
    """
    Logged after keyword extraction stage.

    Captures extraction details including extractor type, configuration, extracted
    keywords with scores, extraction timing, and input resources.

    Fields:
        stage: Literal["keyword_extraction"] - Always "keyword_extraction"
        query_index: int - Sequential index of this query
        extractor_type: str - Keyword extractor used (yake/rake/tfidf/llm)
        extractor_config: Dict[str, Any] - Extractor configuration
        keywords: List[KeywordDetail] - Extracted keywords with scores
        extraction_time: float - Time taken for extraction in seconds
        input_resources: List[str] - Compact resource IDs contributing to extraction
        content_source: str - Content source type ("metadata", "content")
    """

    stage: Literal["keyword_extraction"] = "keyword_extraction"
    query_index: int = Field(description="Sequential index of this query")
    extractor_type: str = Field(description="Keyword extractor used")
    extractor_config: Dict[str, Any] = Field(description="Extractor configuration")
    keywords: List[KeywordDetail] = Field(description="Extracted keywords with scores")
    extraction_time: float = Field(description="Time taken for extraction in seconds")
    input_resources: List[str] = Field(
        description="Compact resource IDs: PMID:X (with prefix) or raw URL (no prefix)"
    )
    content_source: str = Field(description="Content source type")


class QueryConstructionEntry(InvestigationLogEntry):
    """
    Logged after query construction stage.

    Captures construction details including constructor type, configuration, input
    keywords, final query, and whether fallback was used.

    Fields:
        stage: Literal["query_construction"] - Always "query_construction"
        query_index: int - Sequential index of this query
        constructor_type: str - Query constructor used (direct/llm)
        constructor_config: Dict[str, Any] - Constructor configuration
        input_keywords: List[str] - Keywords used for construction
        keyword_scores: List[Optional[float]] - Scores for corresponding keywords
        final_query: str - Final query text sent to backend
        construction_time: float - Time taken for construction in seconds
        fallback_used: bool - Whether fallback constructor was used
        backend: str - Target search backend
        cumulative_coverage: float - Coverage achieved after this query (0.0-1.0)
        query_results: Optional[QueryResultsSummary] - Search results and matches (populated after matching)
    """

    stage: Literal["query_construction"] = "query_construction"
    query_index: int = Field(description="Sequential index of this query")
    constructor_type: str = Field(description="Query constructor used")
    constructor_config: Dict[str, Any] = Field(description="Constructor configuration")
    input_keywords: List[str] = Field(description="Keywords used for construction")
    keyword_scores: List[Optional[float]] = Field(
        description="Scores for corresponding keywords"
    )
    final_query: str = Field(description="Final query text sent to backend")
    construction_time: float = Field(
        description="Time taken for construction in seconds"
    )
    fallback_used: bool = Field(description="Whether fallback constructor was used")
    backend: str = Field(description="Target search backend")
    cumulative_coverage: float = Field(
        description="Cumulative coverage after this query (0.0-1.0)", ge=0.0, le=1.0
    )
    query_results: Optional[QueryResultsSummary] = Field(
        None,
        description="Search results and matches (populated after matching completes)",
    )


class QueryGenerationEntry(InvestigationLogEntry):
    """
    Logged for each generated query.

    Captures query generation including input resources, keywords with scores from
    extractor, and cumulative coverage after this query.

    Fields:
        stage: Literal["query_generation"] - Always "query_generation"
        query_index: int - Sequential index of this query
        query_type: str - Query generation strategy (e.g., "cluster", "resource")
        extractor_type: str - Keyword extractor used (yake/rake/tfidf/llm)
        input_resources: List[str] - Compact resource IDs contributing to query
        keywords: List[KeywordDetail] - Keywords with scores from extractor
        final_query: str - Final query text sent to backend
        cluster_id: Optional[int] - Cluster ID if applicable
        cumulative_coverage: float - Coverage achieved after this query (0.0-1.0)
    """

    stage: Literal["query_generation"] = "query_generation"
    query_index: int = Field(description="Sequential index of this query")
    query_type: str = Field(description="Query generation strategy")
    extractor_type: str = Field(description="Keyword extractor used")
    input_resources: List[str] = Field(
        description="Compact resource IDs: PMID:X (with prefix) or raw URL (no prefix)"
    )
    keywords: List[KeywordDetail] = Field(
        description="Keywords with scores from extractor"
    )
    final_query: str = Field(description="Final query text sent to backend")
    cluster_id: Optional[int] = Field(None, description="Cluster ID if applicable")
    cumulative_coverage: float = Field(
        description="Cumulative coverage after this query (0.0-1.0)", ge=0.0, le=1.0
    )


class SearchExecutionEntry(InvestigationLogEntry):
    """
    Logged after executing a search query.

    Captures search results with 0-based position indices, enabling analysis of
    where targets were found in result lists.

    Fields:
        stage: Literal["search_execution"] - Always "search_execution"
        query_index: int - Sequential index of this query
        query_text: str - The query string executed
        backend: str - Search backend used
        cache_hit: bool - Whether results came from cache
        results: List[SearchResultDetail] - Search results with positions
        result_count: int - Number of results returned
        search_time: float - Time taken for search in seconds
    """

    stage: Literal["search_execution"] = "search_execution"
    query_index: int = Field(description="Sequential index of this query")
    query_text: str = Field(description="The query string executed")
    backend: str = Field(description="Search backend used")
    cache_hit: bool = Field(description="Whether results came from cache")
    results: List[SearchResultDetail] = Field(
        description="Search results with positions"
    )
    result_count: int = Field(description="Number of results returned")
    search_time: float = Field(description="Time taken for search in seconds")


class MatchingEntry(InvestigationLogEntry):
    """
    Logged after matching search results to target resources.

    Captures per-result match attempts with 0-based position indices, enabling
    analysis of which results matched which targets.

    Fields:
        stage: Literal["matching"] - Always "matching"
        query_index: int - Sequential index of this query
        result_count: int - Number of results processed
        matches_found: int - Number of new matches found
        match_details: List[MatchDetail] - Per-result match details with positions
        strategies_attempted: List[str] - Matching strategies tried
        cumulative_coverage: float - Coverage achieved after this query (0.0-1.0)
    """

    stage: Literal["matching"] = "matching"
    query_index: int = Field(description="Sequential index of this query")
    result_count: int = Field(description="Number of results processed")
    matches_found: int = Field(description="Number of new matches found")
    match_details: List[MatchDetail] = Field(
        description="Per-result match details with position tracking"
    )
    strategies_attempted: List[str] = Field(description="Matching strategies tried")
    cumulative_coverage: float = Field(
        description="Coverage achieved after this query", ge=0.0, le=1.0
    )


class SessionEndEntry(InvestigationLogEntry):
    """
    Logged at session completion.

    Captures final session statistics including unfound resources (as compact IDs),
    coverage, stopping reason, and query progression.

    Fields:
        stage: Literal["session_end"] - Always "session_end"
        total_queries: int - Total number of queries executed
        final_coverage: float - Final coverage achieved (0.0-1.0)
        found_count: int - Number of resources successfully found
        unfound_count: int - Number of resources not found
        unfound_resources: List[str] - Compact IDs of unfound resources
        stopping_reason: Literal - Why the session ended
        query_progression: List[Dict[str, Any]] - Coverage progression per query
    """

    stage: Literal["session_end"] = "session_end"
    total_queries: int = Field(description="Total number of queries executed")
    final_coverage: float = Field(description="Final coverage achieved", ge=0.0, le=1.0)
    found_count: int = Field(description="Number of resources successfully found")
    unfound_count: int = Field(description="Number of resources not found")
    unfound_resources: List[str] = Field(
        description="Compact IDs of resources not found: PMID:X (with prefix) or raw URL (no prefix)"
    )
    stopping_reason: Literal[
        "coverage_achieved", "consecutive_zero_finds", "max_queries"
    ] = Field(description="Why the session ended")
    query_progression: List[Dict[str, Any]] = Field(
        description="Coverage progression per query"
    )


class ErrorEntry(InvestigationLogEntry):
    """
    Logged when an error occurs during pipeline execution.

    Captures error details including stage, type, message, recovery action,
    and optional traceback for debugging.

    Fields:
        stage: Literal["error"] - Always "error"
        error_stage: str - Which pipeline stage the error occurred in
        error_type: str - Exception class name
        error_message: str - Exception message
        affected_resources: List[str] - Resource URLs affected by error
        recovery_action: str - What action was taken to recover
        traceback: Optional[str] - Full exception traceback if available
    """

    stage: Literal["error"] = "error"
    error_stage: str = Field(description="Which pipeline stage the error occurred in")
    error_type: str = Field(description="Exception class name")
    error_message: str = Field(description="Exception message")
    affected_resources: List[str] = Field(description="Resource URLs affected by error")
    recovery_action: str = Field(description="What action was taken to recover")
    traceback: Optional[str] = Field(None, description="Full exception traceback")


# Logger class


class InvestigationLogger:
    """
    Async investigation logger with JSON Lines output.

    Provides structured logging for reverse search pipeline with thread-safe
    concurrent writes, automatic file lifecycle management, and comprehensive
    entry types for all pipeline stages.

    All entries are written in JSON Lines format (one JSON object per line)
    with automatic flushing for incrementality. The logger uses asyncio.Lock
    for thread-safe concurrent writes.

    Usage:
        async with InvestigationLogger("session.jsonl") as logger:
            await logger.log_session_start(resources, config, "pubmed")
            # ... pipeline execution
            await logger.log_session_end(session)

    The logger generates a unique session_id (UUID4) for all entries, allowing
    correlation across the entire pipeline execution. An explicit session_id
    can be provided if needed.

    Args:
        output_path: Path - Output file path for JSON Lines log
        session_id: Optional[str] - Explicit session ID (generates UUID4 if None)
    """

    def __init__(self, output_path: Path, session_id: Optional[str] = None):
        """
        Initialize investigation logger.

        Args:
            output_path: Path to output JSON Lines file
            session_id: Optional explicit session ID (generates UUID4 if None)
        """
        self.output_path = output_path
        self.session_id = session_id or self._generate_session_id()
        self._lock = asyncio.Lock()
        self._file = None

    async def __aenter__(self) -> "InvestigationLogger":
        """Open log file for writing."""
        self._file = await aiofiles.open(self.output_path, mode="w")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close log file."""
        if self._file:
            await self._file.close()

    async def _write_entry(self, entry: InvestigationLogEntry) -> None:
        """
        Write log entry as JSON Line with thread-safe locking.

        Serializes entry to JSON with 2-space indentation for readability,
        writes entry with newlines, and flushes immediately for incremental
        processing.

        Args:
            entry: Log entry to write
        """
        async with self._lock:
            # Pretty-print with 2-space indent for readability
            line = entry.model_dump_json(indent=2) + "\n"
            await self._file.write(line)
            await self._file.flush()

    @staticmethod
    def _now() -> str:
        """Generate ISO 8601 timestamp with timezone (UTC)."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session identifier (UUID4)."""
        return str(uuid4())

    async def log_session_start(
        self,
        resources: List[KnownResource],
        config: ReverseSearchConfig,
        backend: str,
    ) -> None:
        """
        Log session initialization.

        Args:
            resources: Target resources to find
            config: Reverse search configuration
            backend: Search backend name
        """
        entry = SessionStartEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            target_count=len(resources),
            target_resources=[_format_resource_ref(r) for r in resources],
            config=config.model_dump(),
            backend=backend,
        )
        await self._write_entry(entry)

    async def log_content_fetch(
        self,
        resources: List[KnownResource],
        contents: Dict[str, Any],
    ) -> None:
        """
        Log content fetching results.

        Args:
            resources: Resources processed
            contents: Content fetching results with per-resource details
                      Format: {url: {source, title, content_length, success, error, ...}}
        """
        # Build per-resource details
        resource_details = []
        source_counts: Dict[str, int] = {}
        failed_count = 0

        for resource in resources:
            content_info = contents.get(resource.url, {})
            # Extract details
            source = content_info.get("source")
            title_raw = content_info.get("title")
            # Handle None title (failed fetch)
            title = title_raw[:100] if title_raw else None
            content_length = content_info.get("content_length", 0)
            success = not content_info.get("failed", False)
            error = content_info.get("error")
            # Create detail entry
            detail = ResourceFetchDetail(
                resource_id=_format_resource_ref(resource),
                source=source,
                title=title,
                content_length=content_length,
                success=success,
                error=str(error) if error else None,
            )
            resource_details.append(detail)
            # Update counts (only count successful fetches with valid sources)
            if success and source:
                source_counts[source] = source_counts.get(source, 0) + 1
            else:
                failed_count += 1

        entry = ContentFetchEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            resources=resource_details,
            source_counts=source_counts,
            failed_count=failed_count,
        )
        await self._write_entry(entry)

    async def log_clustering(
        self,
        enabled: bool,
        resources: List[KnownResource],
        clusters: Optional[List[List[int]]] = None,
        representatives: Optional[List[int]] = None,
        embeddings: Optional[Any] = None,
        centroids: Optional[Any] = None,
    ) -> None:
        """
        Log clustering results.

        Args:
            enabled: Whether clustering was performed
            resources: Resources that were clustered
            clusters: List of clusters (each cluster is list of resource indices)
            representatives: Representative resource index for each cluster
            embeddings: Resource embeddings (not logged, only for statistics)
            centroids: Cluster centroids (not logged, only for statistics)
        """
        cluster_details = []

        if enabled and clusters:
            for cluster_idx, cluster_members in enumerate(clusters):
                cluster_info = {
                    "cluster_id": cluster_idx,
                    "size": len(cluster_members),
                    "member_indices": cluster_members,
                    "representative_idx": (
                        representatives[cluster_idx] if representatives else None
                    ),
                }
                cluster_details.append(cluster_info)
        else:
            # No clustering - all resources in single "cluster"
            cluster_details.append(
                {
                    "cluster_id": 0,
                    "size": len(resources),
                    "member_indices": list(range(len(resources))),
                    "representative_idx": None,
                }
            )

        entry = ClusteringEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            enabled=enabled,
            resource_count=len(resources),
            clusters=cluster_details,
        )
        await self._write_entry(entry)

    async def log_keyword_extraction(
        self,
        query_index: int,
        extractor_type: str,
        extractor_config: Dict[str, Any],
        keywords: List[Dict[str, Any]],
        extraction_time: float,
        input_resources: List[str],
        content_source: str,
    ) -> None:
        """
        Log keyword extraction stage.

        Args:
            query_index: Sequential query index
            extractor_type: Keyword extractor used (yake/rake/tfidf/llm)
            extractor_config: Extractor configuration
            keywords: Extracted keywords with scores (list of dicts with keyword/score)
            extraction_time: Time taken for extraction in seconds
            input_resources: Compact resource IDs contributing to extraction
            content_source: Content source type ("metadata", "content")
        """
        # Convert keywords to KeywordDetail objects
        keyword_details = []
        for kw in keywords:
            if isinstance(kw, dict):
                keyword_details.append(
                    KeywordDetail(
                        keyword=kw.get("keyword", str(kw)),
                        score=kw.get("score"),
                    )
                )
            else:
                # Fallback for unexpected format
                keyword_details.append(KeywordDetail(keyword=str(kw), score=None))

        entry = KeywordExtractionEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            query_index=query_index,
            extractor_type=extractor_type,
            extractor_config=extractor_config,
            keywords=keyword_details,
            extraction_time=extraction_time,
            input_resources=input_resources,
            content_source=content_source,
        )
        await self._write_entry(entry)

    async def log_query_construction(
        self,
        query_index: int,
        constructor_type: str,
        constructor_config: Dict[str, Any],
        input_keywords: List[str],
        keyword_scores: List[Optional[float]],
        final_query: str,
        construction_time: float,
        fallback_used: bool,
        backend: str,
        cumulative_coverage: float,
    ) -> None:
        """
        Log query construction stage.

        Args:
            query_index: Sequential query index
            constructor_type: Query constructor used (direct/llm)
            constructor_config: Constructor configuration
            input_keywords: Keywords used for construction
            keyword_scores: Scores for corresponding keywords
            final_query: Final query text sent to backend
            construction_time: Time taken for construction in seconds
            fallback_used: Whether fallback constructor was used
            backend: Target search backend
            cumulative_coverage: Coverage achieved after this query (0.0-1.0)
        """
        entry = QueryConstructionEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            query_index=query_index,
            constructor_type=constructor_type,
            constructor_config=constructor_config,
            input_keywords=input_keywords,
            keyword_scores=keyword_scores,
            final_query=final_query,
            construction_time=construction_time,
            fallback_used=fallback_used,
            backend=backend,
            cumulative_coverage=cumulative_coverage,
        )
        await self._write_entry(entry)

    async def log_query_generation(
        self,
        query_index: int,
        query_type: str,
        extractor_type: str,
        keywords: List[Any],  # Can be List[str] (old) or List[dict] (new)
        final_query: str,
        cluster_id: Optional[int] = None,
        resource_count: int = 1,
        input_resources: Optional[List[str]] = None,
        cumulative_coverage: float = 0.0,
    ) -> None:
        """
        Log query generation (legacy single-stage format).

        DEPRECATED: This method logs the old single-stage query_generation format.
        New code should use log_keyword_extraction() and log_query_construction()
        for the two-stage format, which provides better traceability and enables
        reusability analysis.

        Args:
            query_index: Sequential query index
            query_type: Query generation strategy (e.g., "cluster", "resource")
            extractor_type: Keyword extractor used
            keywords: Keywords extracted (List[str] for backward compat, or List[dict] with score)
            final_query: Final query text
            cluster_id: Cluster ID if applicable
            resource_count: Number of resources contributing to query
            input_resources: Compact resource IDs contributing to query
            cumulative_coverage: Coverage achieved after this query (0.0-1.0)
        """
        # Handle backward compatibility: keywords might be List[str] or List[dict]
        keyword_details = []
        if keywords:
            for kw in keywords:
                if isinstance(kw, str):
                    # Old format: just string
                    keyword_details.append(KeywordDetail(keyword=kw, score=None))
                elif isinstance(kw, dict):
                    # New format: dict with keyword and score
                    keyword_details.append(
                        KeywordDetail(
                            keyword=kw.get("keyword", str(kw)),
                            score=kw.get("score"),
                        )
                    )
                else:
                    # Unknown format: convert to string
                    keyword_details.append(KeywordDetail(keyword=str(kw), score=None))

        entry = QueryGenerationEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            query_index=query_index,
            query_type=query_type,
            extractor_type=extractor_type,
            input_resources=input_resources or [],
            keywords=keyword_details,
            final_query=final_query,
            cluster_id=cluster_id,
            cumulative_coverage=cumulative_coverage,
        )
        await self._write_entry(entry)

    async def log_search_execution(
        self,
        query_index: int,
        query_text: str,
        backend: str,
        cache_hit: bool,
        results: SearchResults,
    ) -> None:
        """
        Log search execution.

        Args:
            query_index: Sequential query index
            query_text: Query string executed
            backend: Search backend used
            cache_hit: Whether results came from cache
            results: Search results from backend
        """
        # Build result details with indices
        result_details = []
        for result_index, result in enumerate(results.results):
            # Extract compact resource ID from result
            result_pmid = result.metadata.get("pmid") if result.metadata else None
            result_id = f"PMID:{result_pmid}" if result_pmid else result.url

            detail = SearchResultDetail(
                result_index=result_index,
                resource_id=result_id,
                title=result.title or "",
                url=result.url,
            )
            result_details.append(detail)

        entry = SearchExecutionEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            query_index=query_index,
            query_text=query_text,
            backend=backend,
            cache_hit=cache_hit,
            results=result_details,
            result_count=len(result_details),
            search_time=results.search_time or 0.0,
        )
        await self._write_entry(entry)

    async def log_matching(
        self,
        query_index: int,
        results: SearchResults,
        targets: List[KnownResource],
        matches: List[ResourceMatch],
        details: Dict[str, Any],
        coverage: float,
    ) -> None:
        """
        Log matching results.

        Args:
            query_index: Sequential query index
            results: Search results that were matched
            targets: Target resources to match against
            matches: Matches found
            details: Matching details with match_details list (if provided by matcher)
            coverage: Cumulative coverage after matching
        """
        # Check if enhanced match_details provided by matcher (Task 03)
        # If not, build from matches (backward compatibility)
        if "match_details" in details and details["match_details"]:
            match_details_list = details["match_details"]
        else:
            # Backward compatibility: build from matches
            match_details_list = []
            for m in matches:
                # Extract compact IDs
                result_pmid = (
                    m.search_result.metadata.get("pmid")
                    if m.search_result.metadata
                    else None
                )
                result_id = (
                    f"PMID:{result_pmid}" if result_pmid else m.search_result.url
                )
                matched_resource_id = _format_resource_ref(m.resource)

                match_details_list.append(
                    {
                        "result_index": 0,  # Unknown without matcher providing it
                        "result_id": result_id,
                        "matched": True,
                        "matched_resource": matched_resource_id,
                        "match_method": m.match_method,
                        "confidence": m.confidence,
                    }
                )

        strategies_attempted = details.get("strategies_attempted", [])

        entry = MatchingEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            query_index=query_index,
            result_count=len(results.results),
            matches_found=len(matches),
            match_details=match_details_list,
            strategies_attempted=strategies_attempted,
            cumulative_coverage=coverage,
        )
        await self._write_entry(entry)

    async def log_session_end(self, session: ReverseSearchSession) -> None:
        """
        Log session completion.

        Args:
            session: Complete reverse search session
        """
        # Build query progression
        query_progression = [
            {
                "query_index": r.query_index,
                "query": r.query,
                "new_finds": r.new_finds,
                "cumulative_coverage": r.cumulative_coverage,
            }
            for r in session.query_results
        ]

        # Format unfound resources as compact IDs
        unfound_ids = [_format_resource_ref(r) for r in session.unfound_resources]

        entry = SessionEndEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            total_queries=session.total_queries,
            final_coverage=session.final_coverage,
            found_count=session.found_count,
            unfound_count=len(session.unfound_resources),
            unfound_resources=unfound_ids,
            stopping_reason=session.stopping_reason,
            query_progression=query_progression,
        )
        await self._write_entry(entry)

    async def log_error(
        self,
        error_stage: str,
        error: Exception,
        affected_resources: List[KnownResource],
        recovery_action: str,
    ) -> None:
        """
        Log error occurrence.

        Args:
            error_stage: Pipeline stage where error occurred
            error: Exception that was raised
            affected_resources: Resources affected by the error
            recovery_action: What action was taken to recover
        """
        entry = ErrorEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            error_stage=error_stage,
            error_type=type(error).__name__,
            error_message=str(error),
            affected_resources=[r.url for r in affected_resources],
            recovery_action=recovery_action,
            traceback=tb.format_exc() if error.__traceback__ else None,
        )
        await self._write_entry(entry)
