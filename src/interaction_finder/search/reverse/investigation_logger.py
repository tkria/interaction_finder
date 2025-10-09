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
import json
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
        "query_generation",
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

    Captures target resources, configuration, and backend selection for the
    entire reverse search session.

    Fields:
        stage: Literal["session_start"] - Always "session_start"
        target_count: int - Number of target resources to find
        target_resources: List[Dict[str, Any]] - Serialized target resources
        config: Dict[str, Any] - Serialized reverse search configuration
        backend: str - Search backend name (e.g., "pubmed")
    """

    stage: Literal["session_start"] = "session_start"
    target_count: int = Field(description="Number of target resources to find")
    target_resources: List[Dict[str, Any]] = Field(
        description="Serialized target resources"
    )
    config: Dict[str, Any] = Field(
        description="Serialized reverse search configuration"
    )
    backend: str = Field(description="Search backend name")


class ContentFetchEntry(InvestigationLogEntry):
    """
    Logged after fetching content for target resources.

    Captures which resources had content fetched, content type distribution,
    and any failures during fetching.

    Fields:
        stage: Literal["content_fetch"] - Always "content_fetch"
        resources: List[Dict[str, Any]] - Serialized resources processed
        source_counts: Dict[str, int] - Distribution of content types
        failed_count: int - Number of resources that failed to fetch
    """

    stage: Literal["content_fetch"] = "content_fetch"
    resources: List[Dict[str, Any]] = Field(
        description="Serialized resources processed"
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


class QueryGenerationEntry(InvestigationLogEntry):
    """
    Logged for each generated query.

    Captures the query generation process including extractor type, keywords
    extracted, and final query text.

    Fields:
        stage: Literal["query_generation"] - Always "query_generation"
        query_index: int - Sequential index of this query
        query_type: str - Query generation strategy (e.g., "cluster", "resource")
        extractor_type: str - Keyword extractor used (yake/rake/tfidf/llm)
        keywords: List[str] - Keywords extracted for query
        final_query: str - Final query text sent to backend
        cluster_id: Optional[int] - Cluster ID if applicable
        resource_count: int - Number of resources contributing to query
    """

    stage: Literal["query_generation"] = "query_generation"
    query_index: int = Field(description="Sequential index of this query")
    query_type: str = Field(description="Query generation strategy")
    extractor_type: str = Field(description="Keyword extractor used")
    keywords: List[str] = Field(description="Keywords extracted for query")
    final_query: str = Field(description="Final query text sent to backend")
    cluster_id: Optional[int] = Field(None, description="Cluster ID if applicable")
    resource_count: int = Field(description="Number of resources contributing to query")


class SearchExecutionEntry(InvestigationLogEntry):
    """
    Logged after executing a search query.

    Captures the search execution including query text, backend, caching,
    results retrieved, and timing.

    Fields:
        stage: Literal["search_execution"] - Always "search_execution"
        query_index: int - Sequential index of this query
        query_text: str - The query string executed
        backend: str - Search backend used
        cache_hit: bool - Whether results came from cache
        results: List[Dict[str, Any]] - Serialized search results
        result_count: int - Number of results returned
        search_time: float - Time taken for search in seconds
    """

    stage: Literal["search_execution"] = "search_execution"
    query_index: int = Field(description="Sequential index of this query")
    query_text: str = Field(description="The query string executed")
    backend: str = Field(description="Search backend used")
    cache_hit: bool = Field(description="Whether results came from cache")
    results: List[Dict[str, Any]] = Field(description="Serialized search results")
    result_count: int = Field(description="Number of results returned")
    search_time: float = Field(description="Time taken for search in seconds")


class MatchingEntry(InvestigationLogEntry):
    """
    Logged after matching search results to target resources.

    Captures the matching process including strategies attempted, matches found,
    and coverage achieved.

    Fields:
        stage: Literal["matching"] - Always "matching"
        query_index: int - Sequential index of this query
        result_count: int - Number of results processed
        matches_found: int - Number of new matches found
        match_details: List[Dict[str, Any]] - Details of each match
        strategies_attempted: List[str] - Matching strategies tried
        cumulative_coverage: float - Coverage achieved after this query (0.0-1.0)
    """

    stage: Literal["matching"] = "matching"
    query_index: int = Field(description="Sequential index of this query")
    result_count: int = Field(description="Number of results processed")
    matches_found: int = Field(description="Number of new matches found")
    match_details: List[Dict[str, Any]] = Field(description="Details of each match")
    strategies_attempted: List[str] = Field(description="Matching strategies tried")
    cumulative_coverage: float = Field(
        description="Coverage achieved after this query", ge=0.0, le=1.0
    )


class SessionEndEntry(InvestigationLogEntry):
    """
    Logged at session completion.

    Captures final session statistics including total queries, coverage achieved,
    unfound resources, stopping reason, and query progression.

    Fields:
        stage: Literal["session_end"] - Always "session_end"
        total_queries: int - Total number of queries executed
        final_coverage: float - Final coverage achieved (0.0-1.0)
        found_count: int - Number of resources successfully found
        unfound_resources: List[Dict[str, Any]] - Resources not found
        stopping_reason: Literal - Why the session ended
        query_progression: List[Dict[str, Any]] - Coverage progression per query
    """

    stage: Literal["session_end"] = "session_end"
    total_queries: int = Field(description="Total number of queries executed")
    final_coverage: float = Field(description="Final coverage achieved", ge=0.0, le=1.0)
    found_count: int = Field(description="Number of resources successfully found")
    unfound_resources: List[Dict[str, Any]] = Field(description="Resources not found")
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

        Serializes entry to JSON, writes as single line, and flushes immediately
        for incremental processing.

        Args:
            entry: Log entry to write
        """
        async with self._lock:
            line = entry.model_dump_json() + "\n"
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
            target_resources=[r.model_dump() for r in resources],
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
            contents: Content fetching results with source types
        """
        # Compute source counts from contents
        source_counts: Dict[str, int] = {}
        failed_count = 0

        # Contents is a dict mapping resource URL -> content info
        for resource_url, content_info in contents.items():
            if content_info.get("failed", False):
                failed_count += 1
            else:
                source_type = content_info.get("source_type", "unknown")
                source_counts[source_type] = source_counts.get(source_type, 0) + 1

        entry = ContentFetchEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            resources=[r.model_dump() for r in resources],
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

    async def log_query_generation(
        self,
        query_index: int,
        query_type: str,
        extractor_type: str,
        keywords: List[str],
        final_query: str,
        cluster_id: Optional[int] = None,
        resource_count: int = 1,
    ) -> None:
        """
        Log query generation.

        Args:
            query_index: Sequential query index
            query_type: Query generation strategy (e.g., "cluster", "resource")
            extractor_type: Keyword extractor used
            keywords: Keywords extracted
            final_query: Final query text
            cluster_id: Cluster ID if applicable
            resource_count: Number of resources contributing to query
        """
        entry = QueryGenerationEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            query_index=query_index,
            query_type=query_type,
            extractor_type=extractor_type,
            keywords=keywords,
            final_query=final_query,
            cluster_id=cluster_id,
            resource_count=resource_count,
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
        entry = SearchExecutionEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            query_index=query_index,
            query_text=query_text,
            backend=backend,
            cache_hit=cache_hit,
            results=[r.model_dump() for r in results.results],
            result_count=len(results.results),
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
            details: Matching details with strategies attempted
            coverage: Cumulative coverage after matching
        """
        match_details = [
            {
                "resource_url": m.resource.url,
                "result_url": m.search_result.url,
                "method": m.match_method,
                "confidence": m.confidence,
            }
            for m in matches
        ]

        strategies_attempted = details.get("strategies_attempted", [])

        entry = MatchingEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            query_index=query_index,
            result_count=len(results.results),
            matches_found=len(matches),
            match_details=match_details,
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

        entry = SessionEndEntry(
            timestamp=self._now(),
            session_id=self.session_id,
            total_queries=session.total_queries,
            final_coverage=session.final_coverage,
            found_count=session.found_count,
            unfound_resources=[r.model_dump() for r in session.unfound_resources],
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
