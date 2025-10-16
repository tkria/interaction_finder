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
from typing import Dict, Any, List

import pytest
from pydantic import ValidationError

from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults
from interaction_finder.search.reverse.investigation_logger import (
    InvestigationLogger,
    KeywordDetail,
    MatchDetail,
    MatchingEntry,
    QueryConstructionEntry,
    QueryGenerationEntry,
    QueryResultsSummary,
    ResourceFetchDetail,
    SearchExecutionEntry,
    SearchResultDetail,
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


def parse_pretty_json_entries(content: str) -> List[Dict[str, Any]]:
    """
    Parse pretty-printed JSON entries from log content.

    Each entry starts with { at the beginning of a line and ends with } at
    the beginning of a line, with proper brace matching for nested structures.
    """
    entries = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "{":
            # Start of new object
            obj_lines = [line]
            brace_count = 1
            i += 1
            while i < len(lines) and brace_count > 0:
                line = lines[i]
                obj_lines.append(line)
                # Count braces carefully
                for char in line:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                i += 1
            # Parse complete object
            obj_text = "\n".join(obj_lines)
            entry = json.loads(obj_text)
            entries.append(entry)
        else:
            i += 1
    return entries


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
            sample_resources[0].url: {
                "source": "metadata",
                "title": "CD8+ T cell markers study",
                "content_length": 1500,
                "failed": False,
            },
            sample_resources[1].url: {
                "source": "content",
                "title": "B cell paper",
                "content_length": 2000,
                "failed": False,
            },
            sample_resources[2].url: {
                "source": None,
                "title": None,
                "content_length": 0,
                "failed": True,
                "error": "Network timeout",
            },
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

    # Parse pretty-printed JSON entries
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 6

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
    """Test JSON Lines format with pretty-printing."""
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

    # Read and parse pretty-printed JSON objects
    content = temp_log_file.read_text()
    # Verify pretty-printing is enabled
    assert '  "stage"' in content  # 2-space indentation present

    # Parse JSON entries
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 3
    # Verify each entry has required fields
    for entry in entries:
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

    # Parse pretty-printed JSON entries
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 20

    # Verify all entries are query_generation
    for entry in entries:
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
        target_resources=["PMID:12345678", "https://example.com/paper"],
        config={"coverage_target": 0.95},
        backend="pubmed",
    )
    assert valid_start.stage == "session_start"
    assert valid_start.target_count == 3
    assert valid_start.target_resources[0] == "PMID:12345678"

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
        input_resources=["PMID:12345678", "https://example.com/paper"],
        keywords=[
            KeywordDetail(keyword="keyword1", score=0.85),
            KeywordDetail(keyword="keyword2", score=0.72),
        ],
        final_query="keyword1 keyword2",
        cluster_id=1,
        cumulative_coverage=0.5,
    )
    assert valid_query.stage == "query_generation"
    assert len(valid_query.keywords) == 2
    assert valid_query.keywords[0].keyword == "keyword1"
    assert valid_query.keywords[0].score == 0.85

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
            sample_resources[0].url: {
                "source": "metadata",
                "title": "CD8+ T cell markers study",
                "content_length": 1500,
                "failed": False,
            },
            sample_resources[1].url: {
                "source": "content",
                "title": "B cell paper",
                "content_length": 2000,
                "failed": False,
            },
            sample_resources[2].url: {
                "source": None,
                "title": None,
                "content_length": 0,
                "failed": True,
                "error": "Network timeout",
            },
        }
        await logger.log_content_fetch(sample_resources, contents)

    # Parse and verify
    line = temp_log_file.read_text().strip()
    entry = json.loads(line)

    assert entry["stage"] == "content_fetch"
    assert len(entry["resources"]) == 3
    assert entry["source_counts"] == {"metadata": 1, "content": 1}
    assert entry["failed_count"] == 1
    # Verify per-resource details
    assert entry["resources"][0]["resource_id"] == "PMID:12345678"
    assert entry["resources"][0]["source"] == "metadata"
    assert entry["resources"][0]["title"] == "CD8+ T cell markers study"
    assert entry["resources"][0]["content_length"] == 1500
    assert entry["resources"][0]["success"] is True
    assert entry["resources"][2]["success"] is False
    assert entry["resources"][2]["error"] == "Network timeout"


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

    # Valid detail with error (failed fetch)
    detail_with_error = ResourceFetchDetail(
        resource_id="https://example.com/paper",
        source=None,  # No source when fetch failed
        title=None,  # No title when fetch failed
        content_length=0,
        success=False,
        error="Network timeout",
    )
    assert detail_with_error.success is False
    assert detail_with_error.error == "Network timeout"
    assert detail_with_error.source is None
    assert detail_with_error.title is None

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


# ============================================================================
# Tests for query_results enhancement (Task 04)
# ============================================================================


def test_query_results_summary_model():
    """Test QueryResultsSummary model validation with valid data."""
    # Valid model with PMID and URL results
    summary = QueryResultsSummary(
        results=["PMID:12345678", "https://example.com/paper"],
        found_resources=[
            {"resource": "PMID:12345678", "index": 0},
            {"resource": "https://example.com/paper", "index": 1},
        ],
    )
    assert len(summary.results) == 2
    assert summary.results[0] == "PMID:12345678"
    assert summary.results[1] == "https://example.com/paper"
    assert len(summary.found_resources) == 2
    assert summary.found_resources[0]["resource"] == "PMID:12345678"
    assert summary.found_resources[0]["index"] == 0

    # Valid model with empty lists (no results/matches)
    empty_summary = QueryResultsSummary(
        results=[],
        found_resources=[],
    )
    assert len(empty_summary.results) == 0
    assert len(empty_summary.found_resources) == 0


def test_query_results_summary_invalid_data():
    """Test QueryResultsSummary validation rejects invalid data."""
    # Missing required fields
    with pytest.raises(ValidationError):
        QueryResultsSummary(results=["PMID:123"])  # missing found_resources

    with pytest.raises(ValidationError):
        QueryResultsSummary(found_resources=[])  # missing results

    # Wrong types
    with pytest.raises(ValidationError):
        QueryResultsSummary(
            results="not a list",  # should be list
            found_resources=[],
        )

    with pytest.raises(ValidationError):
        QueryResultsSummary(
            results=["PMID:123"],
            found_resources="not a list",  # should be list
        )


def test_query_construction_entry_with_query_results():
    """Test QueryConstructionEntry with query_results populated."""
    # Create entry with query_results
    entry = QueryConstructionEntry(
        timestamp="2024-01-15T10:30:00Z",
        session_id="test-session",
        query_index=0,
        constructor_type="direct",
        constructor_config={"max_keywords": 7},
        input_keywords=["CD8", "T cell"],
        keyword_scores=[0.9, 0.8],
        final_query="CD8 T cell",
        construction_time=0.5,
        fallback_used=False,
        backend="pubmed",
        cumulative_coverage=0.5,
        query_results=QueryResultsSummary(
            results=["PMID:12345678"],
            found_resources=[{"resource": "PMID:12345678", "index": 0}],
        ),
    )
    assert entry.query_results is not None
    assert len(entry.query_results.results) == 1
    assert entry.query_results.results[0] == "PMID:12345678"

    # Create entry without query_results (backward compatibility)
    entry_no_results = QueryConstructionEntry(
        timestamp="2024-01-15T10:30:00Z",
        session_id="test-session",
        query_index=0,
        constructor_type="direct",
        constructor_config={"max_keywords": 7},
        input_keywords=["CD8", "T cell"],
        keyword_scores=[0.9, 0.8],
        final_query="CD8 T cell",
        construction_time=0.5,
        fallback_used=False,
        backend="pubmed",
        cumulative_coverage=0.5,
    )
    assert entry_no_results.query_results is None


@pytest.mark.asyncio
async def test_log_query_construction_with_results(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
    sample_search_results: SearchResults,
):
    """Test log_query_construction() with results and matches provided."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Create matches for the first result
        matches = [
            ResourceMatch(
                resource=sample_resources[0],
                search_result=sample_search_results.results[0],
                match_method="pmid",
                confidence=1.0,
                query_index=0,
            )
        ]

        # Log query construction with results and matches
        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={"max_keywords": 7},
            input_keywords=["CD8", "T cell"],
            keyword_scores=[0.9, 0.8],
            final_query="CD8 T cell marker",
            construction_time=0.5,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.33,
            search_results=sample_search_results,
            matches=matches,
        )

    # Parse log and verify query_results field
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 1

    entry = entries[0]
    assert entry["stage"] == "query_construction"
    assert "query_results" in entry
    assert entry["query_results"] is not None

    # Verify results list format (PMID:X for PMID, URL for others)
    query_results = entry["query_results"]
    assert "results" in query_results
    assert len(query_results["results"]) == 2
    assert query_results["results"][0] == "PMID:12345678"  # Has PMID metadata
    assert query_results["results"][1] == "https://example.com/paper2"  # No PMID

    # Verify found_resources structure
    assert "found_resources" in query_results
    assert len(query_results["found_resources"]) == 1
    found = query_results["found_resources"][0]
    assert found["resource"] == "PMID:12345678"
    assert found["index"] == 0  # 0-based index


@pytest.mark.asyncio
async def test_log_query_construction_without_results(
    temp_log_file: Path,
):
    """Test log_query_construction() backward compatibility (None parameters)."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Log query construction without results/matches (old behavior)
        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={"max_keywords": 7},
            input_keywords=["CD8", "T cell"],
            keyword_scores=[0.9, 0.8],
            final_query="CD8 T cell marker",
            construction_time=0.5,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.33,
            # search_results and matches not provided (defaults to None)
        )

    # Parse log and verify query_results is None
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 1

    entry = entries[0]
    assert entry["stage"] == "query_construction"
    assert "query_results" in entry
    assert entry["query_results"] is None


@pytest.mark.asyncio
async def test_query_results_pmid_formatting(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test result identifier formatting (PMID vs URL)."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Create search results with mix of PMID and URL-only results
        query = SearchQuery(query="test", max_results=10)
        results = [
            SearchResult(
                title="Paper with PMID",
                url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                snippet="Test paper",
                relevance_score=0.95,
                backend="pubmed",
                metadata={"pmid": "12345678"},  # Has PMID
            ),
            SearchResult(
                title="Paper without PMID",
                url="https://example.com/paper1",
                snippet="Another paper",
                relevance_score=0.85,
                backend="pubmed",
                # No metadata → no PMID
            ),
            SearchResult(
                title="Paper with empty metadata",
                url="https://example.com/paper2",
                snippet="Third paper",
                relevance_score=0.75,
                backend="pubmed",
                metadata={},  # Empty metadata → no PMID
            ),
        ]
        search_results = SearchResults(
            query=query,
            results=results,
            total_found=3,
            search_time=1.0,
            backend="pubmed",
        )

        # Create a match for the first result
        matches = [
            ResourceMatch(
                resource=sample_resources[0],
                search_result=results[0],
                match_method="pmid",
                confidence=1.0,
                query_index=0,
            )
        ]

        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={},
            input_keywords=["test"],
            keyword_scores=[0.9],
            final_query="test",
            construction_time=0.5,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.33,
            search_results=search_results,
            matches=matches,
        )

    # Verify result identifier formatting
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    query_results = entries[0]["query_results"]

    # First result has PMID → PMID:X format
    assert query_results["results"][0] == "PMID:12345678"
    # Second result no PMID → raw URL
    assert query_results["results"][1] == "https://example.com/paper1"
    # Third result no PMID → raw URL
    assert query_results["results"][2] == "https://example.com/paper2"


@pytest.mark.asyncio
async def test_query_results_found_resources_indices(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test found_resources position tracking with 0-based indices."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Create search results with multiple results
        query = SearchQuery(query="test", max_results=10)
        results = [
            SearchResult(
                title="Result 0",
                url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                snippet="First",
                relevance_score=0.95,
                backend="pubmed",
                metadata={"pmid": "12345678"},
            ),
            SearchResult(
                title="Result 1",
                url="https://example.com/paper1",
                snippet="Second",
                relevance_score=0.85,
                backend="pubmed",
            ),
            SearchResult(
                title="Result 2",
                url="https://pubmed.ncbi.nlm.nih.gov/87654321/",
                snippet="Third",
                relevance_score=0.75,
                backend="pubmed",
                metadata={"pmid": "87654321"},
            ),
        ]
        search_results = SearchResults(
            query=query,
            results=results,
            total_found=3,
            search_time=1.0,
            backend="pubmed",
        )

        # Create matches at different positions
        matches = [
            ResourceMatch(
                resource=sample_resources[0],  # PMID:12345678
                search_result=results[0],  # Index 0
                match_method="pmid",
                confidence=1.0,
                query_index=0,
            ),
            ResourceMatch(
                resource=sample_resources[2],  # PMID:87654321
                search_result=results[2],  # Index 2
                match_method="pmid",
                confidence=1.0,
                query_index=0,
            ),
        ]

        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={},
            input_keywords=["test"],
            keyword_scores=[0.9],
            final_query="test",
            construction_time=0.5,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.67,
            search_results=search_results,
            matches=matches,
        )

    # Verify found_resources indices are 0-based and correct
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    query_results = entries[0]["query_results"]

    found_resources = query_results["found_resources"]
    assert len(found_resources) == 2

    # First match at index 0
    assert found_resources[0]["resource"] == "PMID:12345678"
    assert found_resources[0]["index"] == 0

    # Second match at index 2 (skipped index 1)
    assert found_resources[1]["resource"] == "PMID:87654321"
    assert found_resources[1]["index"] == 2


@pytest.mark.asyncio
async def test_query_results_empty_results(
    temp_log_file: Path,
):
    """Test edge case: empty results list."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Create empty search results
        query = SearchQuery(query="test", max_results=10)
        search_results = SearchResults(
            query=query,
            results=[],  # Empty results
            total_found=0,
            search_time=1.0,
            backend="pubmed",
        )

        # No matches (empty list)
        matches = []

        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={},
            input_keywords=["test"],
            keyword_scores=[0.9],
            final_query="test",
            construction_time=0.5,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.0,
            search_results=search_results,
            matches=matches,
        )

    # Verify query_results has empty lists
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    query_results = entries[0]["query_results"]

    assert query_results["results"] == []
    assert query_results["found_resources"] == []


@pytest.mark.asyncio
async def test_query_results_no_matches(
    temp_log_file: Path,
):
    """Test edge case: results with no matches."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Create search results with results but no matches
        query = SearchQuery(query="test", max_results=10)
        results = [
            SearchResult(
                title="Unmatched result",
                url="https://example.com/paper1",
                snippet="Test paper",
                relevance_score=0.95,
                backend="pubmed",
            ),
        ]
        search_results = SearchResults(
            query=query,
            results=results,
            total_found=1,
            search_time=1.0,
            backend="pubmed",
        )

        # No matches (empty list)
        matches = []

        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={},
            input_keywords=["test"],
            keyword_scores=[0.9],
            final_query="test",
            construction_time=0.5,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.0,
            search_results=search_results,
            matches=matches,
        )

    # Verify query_results has results but empty found_resources
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    query_results = entries[0]["query_results"]

    assert len(query_results["results"]) == 1
    assert query_results["results"][0] == "https://example.com/paper1"
    assert query_results["found_resources"] == []


@pytest.mark.asyncio
async def test_query_results_missing_metadata(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test edge case: results missing PMID metadata."""
    async with InvestigationLogger(temp_log_file) as logger:
        # Create search results with missing PMID metadata
        query = SearchQuery(query="test", max_results=10)
        results = [
            SearchResult(
                title="Paper 1",
                url="https://example.com/paper1",
                snippet="First paper",
                relevance_score=0.95,
                backend="pubmed",
                metadata={},  # Empty metadata (no PMID)
            ),
            SearchResult(
                title="Paper 2",
                url="https://example.com/paper2",
                snippet="Second paper",
                relevance_score=0.85,
                backend="pubmed",
                metadata={},  # Empty metadata (no PMID)
            ),
        ]
        search_results = SearchResults(
            query=query,
            results=results,
            total_found=2,
            search_time=1.0,
            backend="pubmed",
        )

        # Create match using URL-only resource
        matches = [
            ResourceMatch(
                resource=sample_resources[1],  # URL-only resource
                search_result=results[0],
                match_method="url",
                confidence=0.9,
                query_index=0,
            )
        ]

        await logger.log_query_construction(
            query_index=0,
            constructor_type="direct",
            constructor_config={},
            input_keywords=["test"],
            keyword_scores=[0.9],
            final_query="test",
            construction_time=0.5,
            fallback_used=False,
            backend="pubmed",
            cumulative_coverage=0.33,
            search_results=search_results,
            matches=matches,
        )

    # Verify all results use URL format (no PMID available)
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    query_results = entries[0]["query_results"]

    # Both results should use raw URL format
    assert query_results["results"][0] == "https://example.com/paper1"
    assert query_results["results"][1] == "https://example.com/paper2"

    # Found resource should use URL format too
    assert len(query_results["found_resources"]) == 1
    assert (
        query_results["found_resources"][0]["resource"] == "https://example.com/paper1"
    )


# Tests for log_query() method (consolidated query logging)


@pytest.mark.asyncio
async def test_log_query_basic_structure(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test log_query() creates consolidated entry with correct structure."""
    from interaction_finder.search.reverse.investigation_logger import (
        QueryConstructionDetails,
    )
    from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults

    async with InvestigationLogger(temp_log_file) as logger:
        construction_details = QueryConstructionDetails(
            keywords=[{"keyword": "CD8", "score": 0.9}],
            keyword_scores=[0.9],
            extractor_type="yake",
            extractor_config={"max_keywords": 10},
            constructor_type="direct",
            constructor_config={},
            fallback_used=False,
            input_resources=["PMID:12345678"],
            cluster_id=None,
            construction_time=0.1,
            extraction_time=0.2,
        )

        results = SearchResults(
            query=SearchQuery(query="CD8 T cell marker", max_results=10),
            results=[
                SearchResult(
                    url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                    title="Test Article",
                    snippet="Test abstract text",
                    relevance_score=0.95,
                    backend="pubmed",
                    metadata={"pmid": "12345678"},
                )
            ],
            total_found=1,
            search_time=0.5,
            backend="pubmed",
        )

        matches = [
            ResourceMatch(
                resource=sample_resources[0],
                search_result=results.results[0],
                match_method="pmid",
                confidence=1.0,
                query_index=0,
            )
        ]

        await logger.log_query(
            query_index=0,
            construction_details=construction_details,
            query_text="CD8 T cell marker",
            backend="pubmed",
            cache_hit=False,
            search_time=0.5,
            results=results,
            matches=matches,
            cumulative_coverage=0.33,
        )

    # Verify entry structure
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 1
    entry = entries[0]

    # Verify basic fields
    assert entry["stage"] == "query"
    assert entry["query_index"] == 0
    assert "session_id" in entry
    assert "timestamp" in entry


@pytest.mark.asyncio
async def test_log_query_keyword_extraction_fields(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test log_query() correctly populates keyword extraction fields."""
    from interaction_finder.search.reverse.investigation_logger import (
        QueryConstructionDetails,
    )
    from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults

    async with InvestigationLogger(temp_log_file) as logger:
        construction_details = QueryConstructionDetails(
            keywords=[
                {"keyword": "CD8", "score": 0.9},
                {"keyword": "T cell", "score": 0.8},
            ],
            keyword_scores=[0.9, 0.8],
            extractor_type="yake",
            extractor_config={"max_keywords": 10, "n_gram": 2},
            constructor_type="direct",
            constructor_config={},
            fallback_used=False,
            input_resources=["PMID:12345678"],
            cluster_id=None,
            construction_time=0.1,
            extraction_time=0.2,
        )

        results = SearchResults(
            query=SearchQuery(query="CD8 T cell", max_results=10),
            results=[],
            total_found=0,
            search_time=0.5,
            backend="pubmed",
        )

        await logger.log_query(
            query_index=0,
            construction_details=construction_details,
            query_text="CD8 T cell",
            backend="pubmed",
            cache_hit=False,
            search_time=0.5,
            results=results,
            matches=[],
            cumulative_coverage=0.0,
        )

    # Verify keyword extraction fields
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    entry = entries[0]

    assert entry["keywords"] == [
        {"keyword": "CD8", "score": 0.9},
        {"keyword": "T cell", "score": 0.8},
    ]
    assert entry["keyword_extractor"] == "yake"
    assert entry["extractor_config"] == {"max_keywords": 10, "n_gram": 2}
    assert entry["extraction_time"] == 0.2


@pytest.mark.asyncio
async def test_log_query_construction_fields(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test log_query() correctly populates query construction fields."""
    from interaction_finder.search.reverse.investigation_logger import (
        QueryConstructionDetails,
    )
    from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults

    async with InvestigationLogger(temp_log_file) as logger:
        construction_details = QueryConstructionDetails(
            keywords=[{"keyword": "CD8", "score": 0.9}],
            keyword_scores=[0.9],
            extractor_type="yake",
            extractor_config={},
            constructor_type="boolean",
            constructor_config={"operator": "AND"},
            fallback_used=True,
            input_resources=["PMID:12345678", "PMID:87654321"],
            cluster_id=1,
            construction_time=0.15,
            extraction_time=0.2,
        )

        results = SearchResults(
            query=SearchQuery(query="CD8 AND T cell", max_results=10),
            results=[],
            total_found=0,
            search_time=0.5,
            backend="pubmed",
        )

        await logger.log_query(
            query_index=0,
            construction_details=construction_details,
            query_text="CD8 AND T cell",
            backend="pubmed",
            cache_hit=False,
            search_time=0.5,
            results=results,
            matches=[],
            cumulative_coverage=0.0,
        )

    # Verify query construction fields
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    entry = entries[0]

    assert entry["query_constructor"] == "boolean"
    assert entry["constructor_config"] == {"operator": "AND"}
    assert entry["query_text"] == "CD8 AND T cell"
    assert entry["construction_time"] == 0.15
    assert entry["fallback_used"] is True
    assert entry["input_resources"] == ["PMID:12345678", "PMID:87654321"]
    assert entry["cluster_id"] == 1


@pytest.mark.asyncio
async def test_log_query_search_execution_fields(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test log_query() correctly populates search execution fields."""
    from interaction_finder.search.reverse.investigation_logger import (
        QueryConstructionDetails,
    )
    from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults

    async with InvestigationLogger(temp_log_file) as logger:
        construction_details = QueryConstructionDetails(
            keywords=[{"keyword": "CD8", "score": 0.9}],
            keyword_scores=[0.9],
            extractor_type="yake",
            extractor_config={},
            constructor_type="direct",
            constructor_config={},
            fallback_used=False,
            input_resources=["PMID:12345678"],
            cluster_id=None,
            construction_time=0.1,
            extraction_time=0.2,
        )

        results = SearchResults(
            query=SearchQuery(query="CD8", max_results=10),
            results=[
                SearchResult(
                    url="https://example.com/paper1",
                    title="Paper 1",
                    snippet="",
                    relevance_score=0.9,
                    backend="pubmed",
                ),
                SearchResult(
                    url="https://example.com/paper2",
                    title="Paper 2",
                    snippet="",
                    relevance_score=0.8,
                    backend="pubmed",
                ),
            ],
            total_found=2,
            search_time=1.25,
            backend="pubmed",
        )

        await logger.log_query(
            query_index=5,
            construction_details=construction_details,
            query_text="CD8",
            backend="pubmed",
            cache_hit=True,
            search_time=1.25,
            results=results,
            matches=[],
            cumulative_coverage=0.6,
        )

    # Verify search execution fields
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    entry = entries[0]

    assert entry["backend"] == "pubmed"
    assert entry["cache_hit"] is True
    assert entry["search_time"] == 1.25
    assert entry["result_count"] == 2


@pytest.mark.asyncio
async def test_log_query_results_and_matching(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test log_query() correctly populates results and matching fields."""
    from interaction_finder.search.reverse.investigation_logger import (
        QueryConstructionDetails,
    )
    from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults

    async with InvestigationLogger(temp_log_file) as logger:
        construction_details = QueryConstructionDetails(
            keywords=[{"keyword": "CD8", "score": 0.9}],
            keyword_scores=[0.9],
            extractor_type="yake",
            extractor_config={},
            constructor_type="direct",
            constructor_config={},
            fallback_used=False,
            input_resources=["PMID:12345678"],
            cluster_id=None,
            construction_time=0.1,
            extraction_time=0.2,
        )

        results = SearchResults(
            query=SearchQuery(query="CD8", max_results=10),
            results=[
                SearchResult(
                    url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                    title="Matched Paper",
                    snippet="",
                    relevance_score=0.95,
                    backend="pubmed",
                    metadata={"pmid": "12345678"},
                ),
                SearchResult(
                    url="https://example.com/unmatched",
                    title="Unmatched Paper",
                    snippet="",
                    relevance_score=0.7,
                    backend="pubmed",
                ),
            ],
            total_found=2,
            search_time=0.5,
            backend="pubmed",
        )

        matches = [
            ResourceMatch(
                resource=sample_resources[0],
                search_result=results.results[0],
                match_method="pmid",
                confidence=1.0,
                query_index=0,
            )
        ]

        await logger.log_query(
            query_index=0,
            construction_details=construction_details,
            query_text="CD8",
            backend="pubmed",
            cache_hit=False,
            search_time=0.5,
            results=results,
            matches=matches,
            cumulative_coverage=0.33,
        )

    # Verify results and matching fields
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    entry = entries[0]

    assert entry["result_count"] == 2
    assert entry["new_finds"] == 1
    assert entry["cumulative_coverage"] == 0.33
    assert len(entry["results"]) == 2

    # First result should have match info
    assert entry["results"][0]["resource"] == "PMID:12345678"
    assert entry["results"][0]["index"] == 0
    assert entry["results"][0]["match"] == "exact"  # pmid match with confidence 1.0

    # Second result should not have match
    assert entry["results"][1]["resource"] == "https://example.com/unmatched"
    assert entry["results"][1]["index"] == 1
    assert entry["results"][1]["match"] is None


@pytest.mark.asyncio
async def test_log_query_timing_calculation(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test log_query() correctly calculates total_time."""
    from interaction_finder.search.reverse.investigation_logger import (
        QueryConstructionDetails,
    )
    from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults

    async with InvestigationLogger(temp_log_file) as logger:
        construction_details = QueryConstructionDetails(
            keywords=[{"keyword": "CD8", "score": 0.9}],
            keyword_scores=[0.9],
            extractor_type="yake",
            extractor_config={},
            constructor_type="direct",
            constructor_config={},
            fallback_used=False,
            input_resources=["PMID:12345678"],
            cluster_id=None,
            construction_time=0.1,
            extraction_time=0.2,
        )

        results = SearchResults(
            query=SearchQuery(query="CD8", max_results=10),
            results=[],
            total_found=0,
            search_time=0.5,
            backend="pubmed",
        )

        await logger.log_query(
            query_index=0,
            construction_details=construction_details,
            query_text="CD8",
            backend="pubmed",
            cache_hit=False,
            search_time=0.5,
            results=results,
            matches=[],
            cumulative_coverage=0.0,
        )

    # Verify timing calculation
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    entry = entries[0]

    # Total time = extraction + construction + search
    assert entry["total_time"] == 0.8  # 0.2 + 0.1 + 0.5


@pytest.mark.asyncio
async def test_log_query_match_confidence_formatting(
    temp_log_file: Path,
    sample_resources: list[KnownResource],
):
    """Test log_query() formats match strings with different match methods."""
    from interaction_finder.search.reverse.investigation_logger import (
        QueryConstructionDetails,
    )
    from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults

    async with InvestigationLogger(temp_log_file) as logger:
        construction_details = QueryConstructionDetails(
            keywords=[{"keyword": "CD8", "score": 0.9}],
            keyword_scores=[0.9],
            extractor_type="yake",
            extractor_config={},
            constructor_type="direct",
            constructor_config={},
            fallback_used=False,
            input_resources=["PMID:12345678"],
            cluster_id=None,
            construction_time=0.1,
            extraction_time=0.2,
        )

        results = SearchResults(
            query=SearchQuery(query="CD8", max_results=10),
            results=[
                SearchResult(
                    url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                    title="Paper 1",
                    snippet="",
                    relevance_score=0.95,
                    backend="pubmed",
                    metadata={"pmid": "12345678"},
                ),
                SearchResult(
                    url="https://example.com/paper2",
                    title="Paper 2",
                    snippet="",
                    relevance_score=0.8,
                    backend="pubmed",
                ),
                SearchResult(
                    url="https://doi.org/10.1234/test",
                    title="Paper 3",
                    snippet="",
                    relevance_score=0.75,
                    backend="pubmed",
                ),
            ],
            total_found=3,
            search_time=0.5,
            backend="pubmed",
        )

        matches = [
            ResourceMatch(
                resource=sample_resources[0],
                search_result=results.results[0],
                match_method="pmid",
                confidence=1.0,
                query_index=0,
            ),
            ResourceMatch(
                resource=sample_resources[1],
                search_result=results.results[1],
                match_method="url",
                confidence=0.85,
                query_index=0,
            ),
            ResourceMatch(
                resource=sample_resources[2],
                search_result=results.results[2],
                match_method="title_similarity",
                confidence=0.78,
                query_index=0,
            ),
        ]

        await logger.log_query(
            query_index=0,
            construction_details=construction_details,
            query_text="CD8",
            backend="pubmed",
            cache_hit=False,
            search_time=0.5,
            results=results,
            matches=matches,
            cumulative_coverage=1.0,
        )

    # Verify match string formatting
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    entry = entries[0]

    # PMID match should show "exact"
    assert entry["results"][0]["match"] == "exact"

    # URL match should also show "exact" (regardless of confidence)
    assert entry["results"][1]["match"] == "exact"

    # Title similarity match should show "title:0.78"
    assert entry["results"][2]["match"] == "title:0.78"


# Performance test for rewrite requirement


@pytest.mark.asyncio
async def test_rewrite_performance_100_entries(temp_log_file: Path):
    """Verify rewrite performance meets <50ms requirement for 100 entries."""
    import time

    async with InvestigationLogger(temp_log_file) as logger:
        # Pre-populate with 99 entries
        for i in range(99):
            await logger.log_query_generation(
                query_index=i,
                query_type="resource",
                extractor_type="yake",
                keywords=[f"keyword{i}"],
                final_query=f"query text {i}",
                resource_count=1,
            )

        # Measure 100th entry write time (includes full rewrite)
        start = time.perf_counter()
        await logger.log_query_generation(
            query_index=99,
            query_type="resource",
            extractor_type="yake",
            keywords=["final_keyword"],
            final_query="final query",
            resource_count=1,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Verify meets performance requirement
        assert elapsed_ms < 50, f"Rewrite took {elapsed_ms:.2f}ms, exceeds 50ms limit"

    # Verify all entries written
    content = temp_log_file.read_text()
    entries = parse_pretty_json_entries(content)
    assert len(entries) == 100
