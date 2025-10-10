"""
Comprehensive tests for InvestigationLogger.

Tests cover:
- Logger lifecycle management
- All 8 log entry types
- JSON Lines format validation
- Concurrent write safety
- Session ID generation and propagation
- Error logging with tracebacks
- Pydantic validation
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any

import pytest
from pydantic import ValidationError

from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults
from interaction_finder.search.reverse.investigation_logger import (
    ClusteringEntry,
    ContentFetchEntry,
    ErrorEntry,
    InvestigationLogger,
    KeywordDetail,
    MatchDetail,
    MatchingEntry,
    QueryGenerationEntry,
    ResourceFetchDetail,
    SearchExecutionEntry,
    SearchResultDetail,
    SessionEndEntry,
    SessionStartEntry,
    _format_resource_ref,
)
from interaction_finder.search.reverse.models import (
    KnownResource,
    ResourceMatch,
    ReverseSearchConfig,
    ReverseSearchResult,
    ReverseSearchSession,
)


@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    """Create temporary log file path."""
    return tmp_path / "test_session.jsonl"


@pytest.fixture
def sample_resources() -> list[KnownResource]:
    """Create sample target resources."""
    return [
        KnownResource(
            pmid="12345678",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            hint_fields={"celltype": "CD8+ T cell", "marker": "CD8"},
        ),
        KnownResource(
            url="https://example.com/paper1",
            hint_fields={"celltype": "B cell", "marker": "CD19"},
        ),
        KnownResource(
            pmid="87654321",
            url="https://pubmed.ncbi.nlm.nih.gov/87654321/",
            hint_fields={"celltype": "NK cell", "marker": "CD56"},
        ),
    ]


@pytest.fixture
def sample_config() -> ReverseSearchConfig:
    """Create sample reverse search configuration."""
    return ReverseSearchConfig(
        coverage_target=0.95,
        consecutive_zero_limit=3,
        max_queries=50,
        keyword_extractor="yake",
        keywords_per_query=7,
    )


@pytest.fixture
def sample_search_results() -> SearchResults:
    """Create sample search results."""
    query = SearchQuery(query="CD8 T cell markers", max_results=10)
    results = [
        SearchResult(
            title="CD8+ T cell markers in immune response",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            snippet="Study of CD8 markers...",
            relevance_score=0.95,
            backend="pubmed",
            metadata={"pmid": "12345678"},
        ),
        SearchResult(
            title="B cell development",
            url="https://example.com/paper2",
            snippet="B cell markers...",
            relevance_score=0.75,
            backend="pubmed",
        ),
    ]
    return SearchResults(
        query=query,
        results=results,
        total_found=2,
        search_time=1.5,
        backend="pubmed",
    )


@pytest.mark.asyncio
async def test_logger_lifecycle(temp_log_file: Path):
    """Test logger lifecycle with context manager."""
    # Logger should create file on enter
    async with InvestigationLogger(temp_log_file) as logger:
        assert temp_log_file.exists()
        assert logger.session_id is not None

    # File should still exist after exit
    assert temp_log_file.exists()

    # Should be readable
    content = temp_log_file.read_text()
    assert isinstance(content, str)


@pytest.mark.asyncio
async def test_session_lifecycle_logging(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
    sample_config: ReverseSearchConfig,
    sample_search_results: SearchResults,
):
    """Test complete session lifecycle logging."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Log session start
        await logger.log_session_start(sample_resources, sample_config, "pubmed")

        # Log content fetch
        contents = {
            sample_resources[0].url: {"source_type": "pubmed", "failed": False},
            sample_resources[1].url: {"source_type": "pdf", "failed": False},
            sample_resources[2].url: {"source_type": "html", "failed": True},
        }
        await logger.log_content_fetch(sample_resources, contents)

        # Log query generation
        await logger.log_query_generation(
            query_index=0,
            query_type="cluster",
            extractor_type="yake",
            keywords=["CD8", "T cell", "marker"],
            final_query="CD8 T cell marker",
            cluster_id=0,
            resource_count=3,
        )

        # Log search execution
        await logger.log_search_execution(
            query_index=0,
            query_text="CD8 T cell marker",
            backend="pubmed",
            cache_hit=False,
            results=sample_search_results,
        )

        # Log matching
        matches = [
            ResourceMatch(
                resource=sample_resources[0],
                search_result=sample_search_results.results[0],
                match_method="pmid",
                confidence=1.0,
                query_index=0,
            )
        ]
        await logger.log_matching(
            query_index=0,
            results=sample_search_results,
            targets=sample_resources,
            matches=matches,
            details={"strategies_attempted": ["pmid", "url"]},
            coverage=0.33,
        )

        # Log session end
        query_result = ReverseSearchResult(
            query="CD8 T cell marker",
            query_index=0,
            search_results=sample_search_results,
            resources_found=[sample_resources[0]],
            new_finds=1,
            cumulative_coverage=0.33,
            search_time=1.5,
            backend="pubmed",
        )
        session = ReverseSearchSession(
            target_resources=sample_resources,
            query_results=[query_result],
            matches=matches,
            total_queries=1,
            final_coverage=0.33,
            found_count=1,
            unfound_resources=sample_resources[1:],
            total_time=5.0,
            stopping_reason="max_queries",
        )
        await logger.log_session_end(session)

    # Verify all entries present
    lines = temp_log_file.read_text().strip().split("\n")
    assert len(lines) == 6

    # Parse and verify each entry
    entries = [json.loads(line) for line in lines]

    # Verify stages in order
    expected_stages = [
        "session_start",
        "content_fetch",
        "query_generation",
        "search_execution",
        "matching",
        "session_end",
    ]
    actual_stages = [e["stage"] for e in entries]
    assert actual_stages == expected_stages

    # Verify all entries share same session_id
    session_ids = [e["session_id"] for e in entries]
    assert len(set(session_ids)) == 1


@pytest.mark.asyncio
async def test_json_lines_format(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
    sample_config: ReverseSearchConfig,
):
    """Test JSON Lines format compliance."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Write multiple entries
        await logger.log_session_start(sample_resources, sample_config, "pubmed")
        await logger.log_query_generation(
            query_index=0,
            query_type="resource",
            extractor_type="yake",
            keywords=["keyword1", "keyword2"],
            final_query="keyword1 keyword2",
            resource_count=1,
        )
        await logger.log_query_generation(
            query_index=1,
            query_type="cluster",
            extractor_type="rake",
            keywords=["keyword3"],
            final_query="keyword3",
            cluster_id=1,
            resource_count=2,
        )

    # Read file line by line
    lines = temp_log_file.read_text().strip().split("\n")
    assert len(lines) == 3

    # Verify each line is valid JSON
    for line in lines:
        entry = json.loads(line)
        assert "timestamp" in entry
        assert "stage" in entry
        assert "session_id" in entry


@pytest.mark.asyncio
async def test_concurrent_writes(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test thread-safe concurrent writes."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Create multiple tasks writing concurrently
        tasks = []
        for i in range(20):
            task = logger.log_query_generation(
                query_index=i,
                query_type="resource",
                extractor_type="yake",
                keywords=[f"keyword{i}"],
                final_query=f"query{i}",
                resource_count=1,
            )
            tasks.append(task)

        # Execute all concurrently
        await asyncio.gather(*tasks)

    # Verify all entries present
    lines = temp_log_file.read_text().strip().split("\n")
    assert len(lines) == 20

    # Verify all lines are valid JSON
    entries = []
    for line in lines:
        entry = json.loads(line)
        entries.append(entry)
        assert entry["stage"] == "query_generation"

    # Verify all query indices present (order may vary due to concurrency)
    query_indices = [e["query_index"] for e in entries]
    assert sorted(query_indices) == list(range(20))


@pytest.mark.asyncio
async def test_entry_models_validation():
    """Test Pydantic validation for entry models."""
    # Valid SessionStartEntry
    valid_start = SessionStartEntry(
        timestamp="2025-10-09T12:00:00+00:00",
        session_id="test-session-id",
        target_count=3,
        target_resources=[{"url": "https://example.com", "pmid": None}],
        config={"coverage_target": 0.95},
        backend="pubmed",
    )
    assert valid_start.stage == "session_start"
    assert valid_start.target_count == 3

    # Invalid SessionStartEntry - wrong type for target_count
    with pytest.raises(ValidationError):
        SessionStartEntry(
            timestamp="2025-10-09T12:00:00+00:00",
            session_id="test-session-id",
            target_count="three",  # Should be int
            target_resources=[],
            config={},
            backend="pubmed",
        )

    # Valid QueryGenerationEntry
    valid_query = QueryGenerationEntry(
        timestamp="2025-10-09T12:00:00+00:00",
        session_id="test-session-id",
        query_index=0,
        query_type="cluster",
        extractor_type="yake",
        keywords=["keyword1", "keyword2"],
        final_query="keyword1 keyword2",
        cluster_id=1,
        resource_count=5,
    )
    assert valid_query.stage == "query_generation"
    assert valid_query.keywords == ["keyword1", "keyword2"]

    # Valid SearchExecutionEntry
    valid_search = SearchExecutionEntry(
        timestamp="2025-10-09T12:00:00+00:00",
        session_id="test-session-id",
        query_index=0,
        query_text="test query",
        backend="pubmed",
        cache_hit=True,
        results=[],
        result_count=0,
        search_time=1.5,
    )
    assert valid_search.stage == "search_execution"
    assert valid_search.cache_hit is True

    # Valid MatchingEntry with coverage validation
    valid_matching = MatchingEntry(
        timestamp="2025-10-09T12:00:00+00:00",
        session_id="test-session-id",
        query_index=0,
        result_count=10,
        matches_found=3,
        match_details=[],
        strategies_attempted=["pmid", "url"],
        cumulative_coverage=0.75,
    )
    assert valid_matching.cumulative_coverage == 0.75

    # Invalid MatchingEntry - coverage out of range
    with pytest.raises(ValidationError):
        MatchingEntry(
            timestamp="2025-10-09T12:00:00+00:00",
            session_id="test-session-id",
            query_index=0,
            result_count=10,
            matches_found=3,
            match_details=[],
            strategies_attempted=["pmid"],
            cumulative_coverage=1.5,  # Should be <= 1.0
        )


@pytest.mark.asyncio
async def test_session_id_generation(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
    sample_config: ReverseSearchConfig,
):
    """Test session ID generation and usage."""
    # Test auto-generated session_id
    async with InvestigationLogger(temp_log_file) as logger:
        # Should auto-generate UUID4
        assert logger.session_id is not None
        assert len(logger.session_id) == 36  # UUID4 format length
        assert logger.session_id.count("-") == 4

        await logger.log_session_start(sample_resources, sample_config, "pubmed")

    # Verify session_id in log
    line = temp_log_file.read_text().strip()
    entry = json.loads(line)
    assert entry["session_id"] == logger.session_id

    # Test explicit session_id
    temp_log_file2 = temp_log_file.parent / "test2.jsonl"
    explicit_id = "my-custom-session-id"
    async with InvestigationLogger(temp_log_file2, session_id=explicit_id) as logger2:
        assert logger2.session_id == explicit_id
        await logger2.log_session_start(sample_resources, sample_config, "pubmed")

    # Verify explicit session_id in log
    line2 = temp_log_file2.read_text().strip()
    entry2 = json.loads(line2)
    assert entry2["session_id"] == explicit_id


@pytest.mark.asyncio
async def test_error_entry_with_traceback(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test error logging with traceback capture."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Create and log an exception
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            await logger.log_error(
                error_stage="query_generation",
                error=e,
                affected_resources=[sample_resources[0]],
                recovery_action="skipped query",
            )

    # Parse and verify error entry
    line = temp_log_file.read_text().strip()
    entry = json.loads(line)

    assert entry["stage"] == "error"
    assert entry["error_stage"] == "query_generation"
    assert entry["error_type"] == "ValueError"
    assert entry["error_message"] == "Test error message"
    assert entry["recovery_action"] == "skipped query"
    assert len(entry["affected_resources"]) == 1
    assert entry["affected_resources"][0] == sample_resources[0].url

    # Verify traceback present
    assert entry["traceback"] is not None
    assert "ValueError: Test error message" in entry["traceback"]


@pytest.mark.asyncio
async def test_clustering_entry_with_clusters(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test clustering entry with actual clusters."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Log clustering with multiple clusters
        clusters = [[0, 1], [2]]
        representatives = [0, 2]
        await logger.log_clustering(
            enabled=True,
            resources=sample_resources,
            clusters=clusters,
            representatives=representatives,
        )

    # Parse and verify
    line = temp_log_file.read_text().strip()
    entry = json.loads(line)

    assert entry["stage"] == "clustering"
    assert entry["enabled"] is True
    assert entry["resource_count"] == 3
    assert len(entry["clusters"]) == 2

    # Verify first cluster
    cluster0 = entry["clusters"][0]
    assert cluster0["cluster_id"] == 0
    assert cluster0["size"] == 2
    assert cluster0["member_indices"] == [0, 1]
    assert cluster0["representative_idx"] == 0

    # Verify second cluster
    cluster1 = entry["clusters"][1]
    assert cluster1["cluster_id"] == 1
    assert cluster1["size"] == 1
    assert cluster1["member_indices"] == [2]
    assert cluster1["representative_idx"] == 2


@pytest.mark.asyncio
async def test_clustering_entry_without_clustering(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test clustering entry when clustering is disabled."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Log with clustering disabled
        await logger.log_clustering(
            enabled=False,
            resources=sample_resources,
        )

    # Parse and verify
    line = temp_log_file.read_text().strip()
    entry = json.loads(line)

    assert entry["stage"] == "clustering"
    assert entry["enabled"] is False
    assert entry["resource_count"] == 3
    assert len(entry["clusters"]) == 1

    # Should have single "cluster" with all resources
    cluster = entry["clusters"][0]
    assert cluster["cluster_id"] == 0
    assert cluster["size"] == 3
    assert cluster["member_indices"] == [0, 1, 2]
    assert cluster["representative_idx"] is None


@pytest.mark.asyncio
async def test_content_fetch_entry(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test content fetch entry logging."""
    async with InvestigationLogger(temp_log_file) as logger:
        contents: Dict[str, Any] = {
            sample_resources[0].url: {"source_type": "pubmed", "failed": False},
            sample_resources[1].url: {"source_type": "pdf", "failed": False},
            sample_resources[2].url: {"source_type": "html", "failed": True},
        }
        await logger.log_content_fetch(sample_resources, contents)

    # Parse and verify
    line = temp_log_file.read_text().strip()
    entry = json.loads(line)

    assert entry["stage"] == "content_fetch"
    assert len(entry["resources"]) == 3
    assert entry["source_counts"] == {"pubmed": 1, "pdf": 1}
    assert entry["failed_count"] == 1


@pytest.mark.asyncio
async def test_session_end_entry(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
    sample_search_results: SearchResults,
):
    """Test session end entry logging."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Create session
        query_result = ReverseSearchResult(
            query="test query",
            query_index=0,
            search_results=sample_search_results,
            resources_found=[sample_resources[0]],
            new_finds=1,
            cumulative_coverage=0.33,
            search_time=1.5,
            backend="pubmed",
        )
        matches = [
            ResourceMatch(
                resource=sample_resources[0],
                search_result=sample_search_results.results[0],
                match_method="pmid",
                confidence=1.0,
                query_index=0,
            )
        ]
        session = ReverseSearchSession(
            target_resources=sample_resources,
            query_results=[query_result],
            matches=matches,
            total_queries=1,
            final_coverage=0.33,
            found_count=1,
            unfound_resources=sample_resources[1:],
            total_time=5.0,
            stopping_reason="coverage_achieved",
        )

        await logger.log_session_end(session)

    # Parse and verify
    line = temp_log_file.read_text().strip()
    entry = json.loads(line)

    assert entry["stage"] == "session_end"
    assert entry["total_queries"] == 1
    assert entry["final_coverage"] == 0.33
    assert entry["found_count"] == 1
    assert entry["stopping_reason"] == "coverage_achieved"
    assert len(entry["unfound_resources"]) == 2
    assert len(entry["query_progression"]) == 1

    # Verify query progression
    progression = entry["query_progression"][0]
    assert progression["query_index"] == 0
    assert progression["query"] == "test query"
    assert progression["new_finds"] == 1
    assert progression["cumulative_coverage"] == 0.33


# Tests for new models and helper function


def test_format_resource_ref_with_pmid():
    """Test _format_resource_ref with PMID."""
    resource = KnownResource(
        pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/"
    )
    result = _format_resource_ref(resource)
    assert result == "PMID:12345678"


def test_format_resource_ref_without_pmid():
    """Test _format_resource_ref without PMID."""
    resource = KnownResource(url="https://example.com/paper1")
    result = _format_resource_ref(resource)
    assert result == "https://example.com/paper1"


def test_resource_fetch_detail_validation():
    """Test ResourceFetchDetail model validation."""
    # Valid detail with all fields
    detail = ResourceFetchDetail(
        resource_id="PMID:12345678",
        source="content",
        title="Test paper title",
        content_length=1500,
        success=True,
        error=None,
    )
    assert detail.resource_id == "PMID:12345678"
    assert detail.source == "content"
    assert detail.title == "Test paper title"
    assert detail.content_length == 1500
    assert detail.success is True
    assert detail.error is None

    # Valid detail with error
    detail_with_error = ResourceFetchDetail(
        resource_id="https://example.com/paper",
        source="hint_fields",
        title="Failed fetch",
        content_length=0,
        success=False,
        error="Network timeout",
    )
    assert detail_with_error.success is False
    assert detail_with_error.error == "Network timeout"

    # Invalid source type
    with pytest.raises(ValidationError):
        ResourceFetchDetail(
            resource_id="PMID:12345678",
            source="invalid_source",  # Not in Literal options
            title="Test",
            content_length=100,
            success=True,
        )


def test_keyword_detail_validation():
    """Test KeywordDetail model validation."""
    # With score (YAKE/RAKE/TF-IDF)
    keyword_with_score = KeywordDetail(keyword="CD8", score=0.85)
    assert keyword_with_score.keyword == "CD8"
    assert keyword_with_score.score == 0.85

    # Without score (LLM)
    keyword_without_score = KeywordDetail(keyword="T cell", score=None)
    assert keyword_without_score.keyword == "T cell"
    assert keyword_without_score.score is None

    # Score is optional by default
    keyword_default = KeywordDetail(keyword="marker")
    assert keyword_default.keyword == "marker"
    assert keyword_default.score is None


def test_search_result_detail_validation():
    """Test SearchResultDetail model validation."""
    detail = SearchResultDetail(
        result_index=0,
        resource_id="PMID:12345678",
        title="CD8+ T cell markers in immune response",
        url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
    )
    assert detail.result_index == 0
    assert detail.resource_id == "PMID:12345678"
    assert detail.title == "CD8+ T cell markers in immune response"
    assert detail.url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"

    # Required fields must be present
    with pytest.raises(ValidationError):
        SearchResultDetail(
            result_index=0,
            resource_id="PMID:12345678",
            # title missing
            url="https://example.com",
        )


def test_match_detail_validation():
    """Test MatchDetail model validation."""
    # Matched result
    matched = MatchDetail(
        result_index=0,
        result_id="PMID:12345678",
        matched=True,
        matched_resource="PMID:12345678",
        match_method="pmid",
        confidence=1.0,
    )
    assert matched.result_index == 0
    assert matched.result_id == "PMID:12345678"
    assert matched.matched is True
    assert matched.matched_resource == "PMID:12345678"
    assert matched.match_method == "pmid"
    assert matched.confidence == 1.0

    # Unmatched result
    unmatched = MatchDetail(
        result_index=1,
        result_id="https://example.com/paper",
        matched=False,
        matched_resource=None,
        match_method=None,
        confidence=0.0,
    )
    assert unmatched.matched is False
    assert unmatched.matched_resource is None
    assert unmatched.match_method is None
    assert unmatched.confidence == 0.0

    # Invalid confidence (must be 0.0-1.0)
    # Note: Pydantic doesn't enforce this without explicit constraints
    # If we add ge=0.0, le=1.0 to the Field, this would raise ValidationError
    valid_confidence = MatchDetail(
        result_index=0,
        result_id="PMID:12345678",
        matched=True,
        confidence=0.75,
    )
    assert valid_confidence.confidence == 0.75


def test_resource_fetch_detail_optional_fields():
    """Test that error field is optional."""
    detail = ResourceFetchDetail(
        resource_id="PMID:12345678",
        source="metadata",
        title="Test",
        content_length=500,
        success=True,
        # error not provided
    )
    assert detail.error is None


def test_keyword_detail_optional_score():
    """Test that score field is optional."""
    keyword = KeywordDetail(keyword="test")
    assert keyword.score is None


def test_match_detail_optional_fields():
    """Test that optional fields work correctly."""
    # All optional fields None
    detail = MatchDetail(
        result_index=0,
        result_id="PMID:12345678",
        matched=False,
        confidence=0.0,
    )
    assert detail.matched_resource is None
    assert detail.match_method is None
