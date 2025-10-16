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
import logging
import os
import secrets
import traceback as tb
from dataclasses import dataclass
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

    Uses PMID:X format (with prefix) if PMID available, otherwise raw URL
    (no prefix). This compact format improves log readability while
    maintaining sufficient information for analysis and cross-referencing.

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
        results: List[str] - Compact identifiers for all search results
                              (PMID:X or URL)
        found_resources: List[Dict[str, Any]] - Matched resources with
                                                 indices and methods
    """

    results: List[str] = Field(
        description=(
            "Compact identifiers: PMID:X (with prefix) or raw URL "
            "(no prefix) in 0-based order"
        )
    )
    found_resources: List[Dict[str, Any]] = Field(
        description=(
            "Matched resources: {resource: str, index: int, method: str, "
            "confidence: float}"
        )
    )


# Helper function for match string formatting


def _format_match_string(
    match_method: Optional[str], confidence: float
) -> Optional[str]:
    """
    Format match method and confidence as compact string.

    Produces deterministic, parseable match strings for compact logging.
    Exact matches (pmid, url, doi) produce single-token strings; fuzzy
    matches include confidence score.

    Args:
        match_method: Match method used (pmid/url/doi/title_similarity)
        confidence: Match confidence (0.0-1.0)

    Returns:
        Compact match string, or None if no match:
        - "exact" for pmid/url matches
        - "doi" for DOI matches
        - "title:{conf}" for title similarity (e.g., "title:0.85")
        - None if match_method is None

    Examples:
        >>> _format_match_string("pmid", 1.0)
        'exact'
        >>> _format_match_string("url", 1.0)
        'exact'
        >>> _format_match_string("doi", 1.0)
        'doi'
        >>> _format_match_string("title_similarity", 0.85)
        'title:0.85'
        >>> _format_match_string(None, 0.0)
        None
    """
    if match_method is None:
        return None
    if match_method in ("pmid", "url"):
        return "exact"
    if match_method == "doi":
        return "doi"
    if match_method == "title_similarity":
        return f"title:{confidence:.2f}"
    # Fallback for unknown methods
    return match_method


# Consolidated query entry models


@dataclass
class QueryConstructionDetails:
    """
    Query construction information passed from QueryGenerator to ReverseSearcher.

    This dataclass encapsulates all information needed to log a complete query
    entry, enabling QueryGenerator to return construction details without
    directly writing to the investigation log. ReverseSearcher combines these
    details with search results and matches to write a single consolidated entry.

    Fields:
        keywords: List[dict] - Keywords with scores (format: {keyword: str, score: float|None})
        keyword_scores: List[float|None] - Scores for corresponding keywords (parallel array)
        extractor_type: str - Keyword extractor used (yake/rake/tfidf/llm)
        extractor_config: dict - Extractor configuration
        constructor_type: str - Query constructor used (direct/llm)
        constructor_config: dict - Constructor configuration
        fallback_used: bool - Whether fallback constructor was used
        input_resources: List[str] - Compact resource IDs contributing to query
        cluster_id: int|None - Cluster ID if applicable
        construction_time: float - Time taken for construction in seconds
        extraction_time: float - Time taken for extraction in seconds
    """

    keywords: List[Dict[str, Any]]
    keyword_scores: List[Optional[float]]
    extractor_type: str
    extractor_config: Dict[str, Any]
    constructor_type: str
    constructor_config: Dict[str, Any]
    fallback_used: bool
    input_resources: List[str]
    cluster_id: Optional[int]
    construction_time: float
    extraction_time: float


class QueryResultWithMatch(BaseModel):
    """
    Single search result with embedded match information.

    Compact representation for query results array, combining result identity,
    position, and match outcome in a single structure.

    Fields:
        resource: str - Compact result identifier (PMID:X or URL)
        index: int - 0-based position in result list
        match: str|None - Compact match string ("exact", "doi", "title:0.85", or None)
    """

    resource: str = Field(
        description="Compact result identifier: PMID:X (with prefix) or raw URL (no prefix)"
    )
    index: int = Field(description="0-based position in result list")
    match: Optional[str] = Field(
        None,
        description="Compact match string: 'exact', 'doi', 'title:{conf}', or null",
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
        "query",  # Consolidated query entry (new format)
        "keyword_extraction",  # Deprecated: legacy stage for backward compatibility
        "query_construction",  # Deprecated: legacy stage for backward compatibility
        "query_generation",  # Deprecated: legacy stage for backward compatibility
        "search_execution",  # Deprecated: legacy stage for backward compatibility
        "matching",  # Deprecated: legacy stage for backward compatibility
        "session_end",
        "error",
    ] = Field(description="Pipeline stage identifier")
    session_id: str = Field(description="Unique session identifier (UUID4)")


# Specific entry models


class QueryEntry(InvestigationLogEntry):
    """
    Consolidated query entry combining keyword extraction, query construction,
    search execution, and matching into a single log entry.

    This replaces the separate keyword_extraction, query_construction,
    search_execution, and matching entries with a unified entry that captures
    the complete query lifecycle. Provides atomic query logging with full
    provenance and enables easier analysis of query-to-results relationships.

    Fields:
        stage: Literal["query"] - Always "query"
        query_index: int - Sequential index of this query

        # Keyword extraction
        keywords: List[dict] - Keywords with scores from extractor
        keyword_extractor: str - Extractor type (yake/rake/tfidf/llm)
        extractor_config: dict - Extractor configuration
        extraction_time: float - Extraction time in seconds

        # Query construction
        query_constructor: str - Constructor type (direct/llm)
        constructor_config: dict - Constructor configuration
        fallback_used: bool - Whether fallback constructor was used
        input_resources: List[str] - Compact resource IDs contributing to query
        cluster_id: int|None - Cluster ID if applicable
        query_text: str - Final query text sent to backend
        construction_time: float - Construction time in seconds

        # Search execution
        backend: str - Search backend used
        cache_hit: bool - Whether results came from cache
        search_time: float - Search execution time in seconds

        # Results and matching
        results: List[QueryResultWithMatch] - Results with embedded match info
        result_count: int - Number of results returned
        new_finds: int - Number of new matches found in this query
        cumulative_coverage: float - Coverage after this query (0.0-1.0)

        # Timing
        total_time: float - Total time for complete query lifecycle
    """

    stage: Literal["query"] = "query"
    query_index: int = Field(description="Sequential index of this query")
    # Keyword extraction fields
    keywords: List[Dict[str, Any]] = Field(
        description="Keywords with scores from extractor"
    )
    keyword_extractor: str = Field(description="Keyword extractor type")
    extractor_config: Dict[str, Any] = Field(description="Extractor configuration")
    extraction_time: float = Field(description="Extraction time in seconds")
    # Query construction fields
    query_constructor: str = Field(description="Query constructor type")
    constructor_config: Dict[str, Any] = Field(description="Constructor configuration")
    fallback_used: bool = Field(description="Whether fallback constructor was used")
    input_resources: List[str] = Field(
        description="Compact resource IDs contributing to query"
    )
    cluster_id: Optional[int] = Field(None, description="Cluster ID if applicable")
    query_text: str = Field(description="Final query text sent to backend")
    construction_time: float = Field(description="Construction time in seconds")
    # Search execution fields
    backend: str = Field(description="Search backend used")
    cache_hit: bool = Field(description="Whether results came from cache")
    search_time: float = Field(description="Search execution time in seconds")
    # Results and matching fields
    results: List[QueryResultWithMatch] = Field(
        description="Results with embedded match information"
    )
    result_count: int = Field(description="Number of results returned")
    new_finds: int = Field(description="Number of new matches found")
    cumulative_coverage: float = Field(
        description="Cumulative coverage after this query", ge=0.0, le=1.0
    )
    # Timing fields
    total_time: float = Field(description="Total time for complete query lifecycle")


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
        description=(
            "Compact target resource IDs: PMID:X (with prefix) or raw URL (no prefix)"
        )
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

    DEPRECATED: Use QueryEntry for new code. Kept for backward compatibility.

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

    DEPRECATED: Use QueryEntry for new code. Kept for backward compatibility.

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
        cumulative_coverage: float - Coverage achieved after this query
                                      (0.0-1.0)
        query_results: Optional[QueryResultsSummary] - Search results and
                       matches (populated after matching)
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

    DEPRECATED: Use QueryEntry for new code. Kept for backward compatibility.

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

    DEPRECATED: Use QueryEntry for new code. Kept for backward compatibility.

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

    DEPRECATED: Use QueryEntry for new code. Kept for backward compatibility.

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
        description=(
            "Compact IDs of resources not found: PMID:X (with prefix) or "
            "raw URL (no prefix)"
        )
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
        self._entries: List[InvestigationLogEntry] = []

    async def __aenter__(self) -> "InvestigationLogger":
        """Initialize log file by creating empty file to claim path."""
        # Create empty file to claim path, but don't keep file handle open
        # This allows external tools to read the file during execution
        async with aiofiles.open(self.output_path, mode="w") as _:
            pass  # Just create the file
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit (no cleanup needed with rewrite strategy)."""
        pass  # No file handle to close with atomic rewrite strategy

    async def _rewrite_log(self) -> None:
        """
        Atomically rewrite entire log file with all entries.

        Uses temp file + os.replace() for atomic writes. This ensures readers
        see either the complete old file or the complete new file, never partial
        writes. Errors during rewrite are non-fatal: caught, logged as warnings,
        but do not raise exceptions.

        The temp file uses a random suffix to handle concurrent writes safely.
        """
        try:
            # Generate temp file path with random suffix
            suffix = secrets.token_hex(8)
            temp_path = Path(f"{self.output_path}.tmp.{suffix}")
            # Write all entries to temp file
            async with aiofiles.open(temp_path, mode="w") as f:
                for entry in self._entries:
                    # Pretty-print with 2-space indent for readability
                    line = entry.model_dump_json(indent=2) + "\n"
                    await f.write(line)
                # Explicit flush before closing
                await f.flush()
            # Atomically replace original file (works on both Windows and POSIX)
            os.replace(str(temp_path), str(self.output_path))
        except Exception as e:
            # Non-fatal error: log warning but don't raise
            logging.warning(
                f"Failed to rewrite investigation log at {self.output_path}: "
                f"{type(e).__name__}: {e}"
            )

    async def _append_entry(self, entry: InvestigationLogEntry) -> None:
        """
        Append entry to in-memory list and trigger atomic file rewrite.

        This method is the primary interface for adding log entries. It appends
        to the in-memory entry list and then rewrites the entire file atomically.
        Errors are non-fatal and logged as warnings.

        Args:
            entry: Log entry to append
        """
        try:
            async with self._lock:
                self._entries.append(entry)
                await self._rewrite_log()
        except Exception as e:
            # Non-fatal error: log warning but don't raise
            logging.warning(
                f"Failed to append entry to investigation log: {type(e).__name__}: {e}"
            )

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
        try:
            entry = SessionStartEntry(
                timestamp=self._now(),
                session_id=self.session_id,
                target_count=len(resources),
                target_resources=[_format_resource_ref(r) for r in resources],
                config=config.model_dump(),
                backend=backend,
            )
            await self._append_entry(entry)
        except Exception as e:
            logging.warning(f"Failed to log session start: {type(e).__name__}: {e}")

    async def log_content_fetch(
        self,
        resources: List[KnownResource],
        contents: Dict[str, Any],
    ) -> None:
        """
        Log content fetching results.

        Args:
            resources: Resources processed
            contents: Content fetching results with per-resource details.
                      Format: {url: {source, title, content_length, success,
                      error, ...}}
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

        try:
            entry = ContentFetchEntry(
                timestamp=self._now(),
                session_id=self.session_id,
                resources=resource_details,
                source_counts=source_counts,
                failed_count=failed_count,
            )
            await self._append_entry(entry)
        except Exception as e:
            logging.warning(f"Failed to log content fetch: {type(e).__name__}: {e}")

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

        try:
            entry = ClusteringEntry(
                timestamp=self._now(),
                session_id=self.session_id,
                enabled=enabled,
                resource_count=len(resources),
                clusters=cluster_details,
            )
            await self._append_entry(entry)
        except Exception as e:
            logging.warning(f"Failed to log clustering: {type(e).__name__}: {e}")

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

        DEPRECATED: Use log_query() for new code. This method is kept for
        backward compatibility but logs to the legacy keyword_extraction stage.

        Args:
            query_index: Sequential query index
            extractor_type: Keyword extractor used (yake/rake/tfidf/llm)
            extractor_config: Extractor configuration
            keywords: Extracted keywords with scores (list of dicts with keyword/score)
            extraction_time: Time taken for extraction in seconds
            input_resources: Compact resource IDs contributing to extraction
            content_source: Content source type ("metadata", "content")
        """
        try:
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
            await self._append_entry(entry)
        except Exception as e:
            logging.warning(
                f"Failed to log keyword extraction: {type(e).__name__}: {e}"
            )

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
        search_results: Optional[SearchResults] = None,
        matches: Optional[List[ResourceMatch]] = None,
    ) -> None:
        """
        Log query construction stage.

        DEPRECATED: Use log_query() for new code. This method is kept for
        backward compatibility but logs to the legacy query_construction stage.

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
            search_results: Optional search results for populating query_results field
            matches: Optional resource matches for populating query_results field
        """
        try:
            # Build query_results summary if both search_results and matches are provided
            query_results: Optional[QueryResultsSummary] = None
            if search_results is not None and matches is not None:
                # Build compact results list (PMID:X for PMID results, raw URL otherwise)
                results_list = []
                for result in search_results.results:
                    # Extract PMID using same logic as log_search_execution()
                    result_pmid = (
                        result.metadata.get("pmid") if result.metadata else None
                    )
                    result_id = f"PMID:{result_pmid}" if result_pmid else result.url
                    results_list.append(result_id)

                # Build found_resources list with 0-based position indices
                # Create URL-to-index mapping for lookup
                result_url_to_index = {
                    result.url: idx for idx, result in enumerate(search_results.results)
                }

                found_resources_list = []
                for match in matches:
                    # Format resource using existing helper
                    resource_id = _format_resource_ref(match.resource)

                    # Look up result index
                    result_index = result_url_to_index.get(match.search_result.url)
                    if result_index is None:
                        # Match without corresponding result indicates matcher bug
                        # Log warning but continue (non-blocking)
                        logging.warning(
                            f"Match for resource {resource_id} references "
                            f"search result URL {match.search_result.url} which "
                            f"is not in results list. This indicates a matcher "
                            f"bug. Skipping this match."
                        )
                        continue

                    found_resources_list.append(
                        {
                            "resource": resource_id,
                            "index": result_index,
                        }
                    )

                # Create QueryResultsSummary
                query_results = QueryResultsSummary(
                    results=results_list,
                    found_resources=found_resources_list,
                )

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
                query_results=query_results,
            )
            await self._append_entry(entry)
        except Exception as e:
            logging.warning(
                f"Failed to log query construction: {type(e).__name__}: {e}"
            )

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
            keywords: Keywords extracted (List[str] for backward compat, or
                      List[dict] with score)
            final_query: Final query text
            cluster_id: Cluster ID if applicable
            resource_count: Number of resources contributing to query
            input_resources: Compact resource IDs contributing to query
            cumulative_coverage: Coverage achieved after this query (0.0-1.0)
        """
        try:
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
                        keyword_details.append(
                            KeywordDetail(keyword=str(kw), score=None)
                        )

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
            await self._append_entry(entry)
        except Exception as e:
            logging.warning(f"Failed to log query generation: {type(e).__name__}: {e}")

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

        DEPRECATED: Use log_query() for new code. This method is kept for
        backward compatibility but logs to the legacy search_execution stage.

        Args:
            query_index: Sequential query index
            query_text: Query string executed
            backend: Search backend used
            cache_hit: Whether results came from cache
            results: Search results from backend
        """
        try:
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
            await self._append_entry(entry)
        except Exception as e:
            logging.warning(f"Failed to log search execution: {type(e).__name__}: {e}")

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

        DEPRECATED: Use log_query() for new code. This method is kept for
        backward compatibility but logs to the legacy matching stage.

        Args:
            query_index: Sequential query index
            results: Search results that were matched
            targets: Target resources to match against
            matches: Matches found
            details: Matching details with match_details list (if provided by matcher)
            coverage: Cumulative coverage after matching
        """
        try:
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
            await self._append_entry(entry)
        except Exception as e:
            logging.warning(f"Failed to log matching: {type(e).__name__}: {e}")

    async def log_session_end(self, session: ReverseSearchSession) -> None:
        """
        Log session completion.

        Args:
            session: Complete reverse search session
        """
        try:
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
            await self._append_entry(entry)
        except Exception as e:
            logging.warning(f"Failed to log session end: {type(e).__name__}: {e}")

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
        try:
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
            await self._append_entry(entry)
        except Exception as e:
            logging.warning(f"Failed to log error: {type(e).__name__}: {e}")

    async def log_query(
        self,
        query_index: int,
        construction_details: QueryConstructionDetails,
        query_text: str,
        backend: str,
        cache_hit: bool,
        search_time: float,
        results: SearchResults,
        matches: List[ResourceMatch],
        cumulative_coverage: float,
    ) -> None:
        """
        Log consolidated query entry combining extraction, construction, search, and matching.

        This is the primary method for logging queries in the new consolidated format.
        It combines all query lifecycle information into a single atomic entry.

        Args:
            query_index: Sequential query index
            construction_details: Query construction details from QueryGenerator
            query_text: Final query text sent to backend
            backend: Search backend used
            cache_hit: Whether results came from cache
            search_time: Search execution time in seconds
            results: Search results from backend
            matches: Resource matches found
            cumulative_coverage: Coverage achieved after this query (0.0-1.0)
        """
        try:
            # Build results array with embedded match info
            results_with_matches = []
            # Create URL-to-match mapping for efficient lookup
            match_by_url = {match.search_result.url: match for match in matches}

            for result_index, result in enumerate(results.results):
                # Extract compact resource ID (PMID:X or URL)
                result_pmid = result.metadata.get("pmid") if result.metadata else None
                resource_id = f"PMID:{result_pmid}" if result_pmid else result.url

                # Look up match for this result
                match = match_by_url.get(result.url)
                match_str = None
                if match:
                    match_str = _format_match_string(
                        match.match_method, match.confidence
                    )

                results_with_matches.append(
                    QueryResultWithMatch(
                        resource=resource_id,
                        index=result_index,
                        match=match_str,
                    )
                )

            # Calculate total time (extraction + construction + search)
            total_time = (
                construction_details.extraction_time
                + construction_details.construction_time
                + search_time
            )

            # Create consolidated query entry
            entry = QueryEntry(
                timestamp=self._now(),
                session_id=self.session_id,
                query_index=query_index,
                # Keyword extraction fields
                keywords=construction_details.keywords,
                keyword_extractor=construction_details.extractor_type,
                extractor_config=construction_details.extractor_config,
                extraction_time=construction_details.extraction_time,
                # Query construction fields
                query_constructor=construction_details.constructor_type,
                constructor_config=construction_details.constructor_config,
                fallback_used=construction_details.fallback_used,
                input_resources=construction_details.input_resources,
                cluster_id=construction_details.cluster_id,
                query_text=query_text,
                construction_time=construction_details.construction_time,
                # Search execution fields
                backend=backend,
                cache_hit=cache_hit,
                search_time=search_time,
                # Results and matching fields
                results=results_with_matches,
                result_count=len(results.results),
                new_finds=len(matches),
                cumulative_coverage=cumulative_coverage,
                # Timing
                total_time=total_time,
            )
            await self._append_entry(entry)
        except Exception as e:
            logging.warning(
                f"Failed to log consolidated query: {type(e).__name__}: {e}"
            )
