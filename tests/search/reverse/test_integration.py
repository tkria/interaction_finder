"""
Comprehensive integration tests for reverse search end-to-end workflows.

Tests cover:
- Full workflow with mock backend (JSONL-like → parse → search → match → output)
- All three stopping criteria (coverage achieved, consecutive zeros, max queries)
- Edge cases (empty resources, malformed data, all found first query, none found)
- JSONL-like parsing validation
- Output structure validation
- Performance validation (<5 minutes for 100 resources)
- Multi-domain resources (domain-agnostic behavior)
"""

import asyncio
import json
import pytest
import time
from pathlib import Path
from typing import List
from unittest.mock import Mock, AsyncMock, patch

from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
    ResourceParseError,
    ReverseSearchSession,
)
from interaction_finder.search.reverse.searcher import ReverseSearcher
from interaction_finder.search.base import (
    SearchQuery,
    SearchResult,
    SearchResults,
)
from interaction_finder.search.cache import SearchCache


# Fixtures


@pytest.fixture
def sample_resources():
    """Sample known resources for testing."""
    return [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/"),
        KnownResource(pmid="87654321", url="https://pubmed.ncbi.nlm.nih.gov/87654321/"),
        KnownResource(
            url="https://example.com/paper",
            hint_fields={"title": "Test Paper", "celltype": "langerhans cell"},
        ),
    ]


@pytest.fixture
def sample_jsonl_file(tmp_path, sample_resources):
    """Create temporary JSONL file with sample resources."""
    jsonl_path = tmp_path / "test_resources.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for resource in sample_resources:
            entry = {"url": resource.url}
            if resource.pmid:
                entry["pmid"] = resource.pmid
            entry.update(resource.hint_fields)
            f.write(json.dumps(entry) + "\n")

    return jsonl_path


@pytest.fixture
def mock_backend():
    """Mock search backend for testing."""
    backend = Mock()
    backend.backend_name = "test_backend"
    return backend


@pytest.fixture
def mock_cache():
    """Mock search cache for testing."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)  # Always cache miss
    cache.set = AsyncMock()
    return cache


@pytest.fixture
def mock_fetcher():
    """Mock page fetcher for testing."""
    return Mock()


# Helper functions for JSONL parsing and output writing


def parse_known_resources_jsonl(jsonl_path: Path) -> List[KnownResource]:
    """
    Parse JSONL file into list of KnownResource objects.

    Parameters:
        jsonl_path: Path - Path to JSONL file

    Returns:
        List[KnownResource] - Parsed resources

    Raises:
        ResourceParseError: If file is empty, has invalid JSON, or missing required fields
    """
    resources = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("No valid resources found in JSONL file")

    for i, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise ResourceParseError(
                f"Invalid JSON on line {i}",
                context={"line": i, "error": str(e)},
            )

        # Validate required field
        if "url" not in data:
            raise ResourceParseError(
                f"Missing required field 'url' on line {i}",
                context={"line": i, "data": data},
            )

        # Extract fields
        url = data.pop("url")
        pmid = data.pop("pmid", None)
        hint_fields = data  # Remaining fields become hint_fields

        resources.append(KnownResource(url=url, pmid=pmid, hint_fields=hint_fields))

    return resources


def write_reverse_search_output(session: ReverseSearchSession, output_path: Path):
    """
    Write reverse search session results to JSONL output file.

    Format: One JSON object per line for each query result, plus a summary line.

    Parameters:
        session: ReverseSearchSession - Session to write
        output_path: Path - Output file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
        # Write per-query results
        for result in session.query_results:
            query_output = {
                "query": result.query,
                "query_index": result.query_index,
                "new_finds": result.new_finds,
                "cumulative_coverage": result.cumulative_coverage,
                "search_time": result.search_time,
                "backend": result.backend,
                "resources_found": [
                    {"pmid": r.pmid, "url": r.canonical_url}
                    for r in result.resources_found
                ],
            }
            f.write(json.dumps(query_output) + "\n")

        # Write summary
        summary = {
            "summary": True,
            "total_queries": session.total_queries,
            "final_coverage": session.final_coverage,
            "found_count": session.found_count,
            "unfound_count": len(session.unfound_resources),
            "total_time": session.total_time,
            "stopping_reason": session.stopping_reason,
        }
        f.write(json.dumps(summary) + "\n")


# Full workflow tests


@pytest.mark.asyncio
async def test_full_reverse_search_workflow(
    tmp_path,
    sample_jsonl_file,
    sample_resources,
    mock_backend,
    mock_cache,
    mock_fetcher,
):
    """Test complete reverse search workflow with mock backend."""

    # Mock backend to return results matching first two resources
    async def mock_search(query):
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title="Test Paper 1",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                    backend="test_backend",
                    metadata={"pmid": "12345678"},
                ),
                SearchResult(
                    title="Test Paper 2",
                    url="https://pubmed.ncbi.nlm.nih.gov/87654321/",
                    backend="test_backend",
                    metadata={"pmid": "87654321"},
                ),
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    # Create searcher
    config = ReverseSearchConfig(coverage_target=0.5)  # Minimum allowed
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)
    # Mock query generator
    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["test query"]
        # Execute search
        session = await searcher.search(sample_resources, verbose=False)
        # Validate session
        assert session.found_count >= 2  # Should find at least 2 resources
        assert session.final_coverage >= 0.5
        assert len(session.query_results) > 0
        assert session.total_queries > 0
        assert session.stopping_reason in [
            "coverage_achieved",
            "consecutive_zero_finds",
            "max_queries",
        ]
        # Validate output structure
        assert len(session.target_resources) == len(sample_resources)
        assert len(session.matches) >= 1
        assert session.total_time > 0


@pytest.mark.asyncio
async def test_jsonl_parsing_and_output(tmp_path, sample_jsonl_file):
    """Test JSONL input parsing and output formatting."""
    # Test parsing
    resources = parse_known_resources_jsonl(sample_jsonl_file)
    assert len(resources) == 3
    assert resources[0].pmid == "12345678"
    assert resources[2].hint_fields.get("celltype") == "langerhans cell"
    # Test output formatting (create minimal session)
    session = ReverseSearchSession(
        target_resources=resources,
        query_results=[],
        matches=[],
        total_queries=0,
        final_coverage=0.0,
        found_count=0,
        unfound_resources=resources,
        total_time=0.0,
        stopping_reason="max_queries",
    )

    output_path = tmp_path / "output.jsonl"
    write_reverse_search_output(session, output_path)
    # Validate output file exists and has summary
    assert output_path.exists()
    with open(output_path, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0
        # Last line should be summary
        summary = json.loads(lines[-1])
        assert summary.get("summary") is True
        assert summary.get("total_queries") == 0


# Stopping criteria tests


@pytest.mark.asyncio
async def test_stopping_coverage_achieved(
    sample_resources,
    mock_backend,
    mock_cache,
    mock_fetcher,
):
    """Test stopping when coverage target reached."""

    # Mock to match all resources
    async def mock_search(query):
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title=f"Paper {r.pmid or 'X'}",
                    url=r.canonical_url,
                    backend="test_backend",
                    metadata={"pmid": r.pmid} if r.pmid else {},
                )
                for r in sample_resources
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)

    config = ReverseSearchConfig(coverage_target=0.95)
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["comprehensive query"]

        session = await searcher.search(sample_resources, verbose=False)

        assert session.stopping_reason == "coverage_achieved"
        assert session.final_coverage >= 0.95
        assert session.found_count >= len(sample_resources) * 0.95


@pytest.mark.asyncio
async def test_stopping_consecutive_zeros(
    sample_resources,
    mock_backend,
    mock_cache,
    mock_fetcher,
):
    """Test stopping after consecutive queries with no finds."""

    # Mock to always return empty results
    async def mock_search(query):
        return SearchResults(query=query, results=[], backend="test_backend")

    mock_backend.search = AsyncMock(side_effect=mock_search)

    config = ReverseSearchConfig(consecutive_zero_limit=3)
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["q1", "q2", "q3", "q4", "q5"]

        session = await searcher.search(sample_resources, verbose=False)

        assert session.stopping_reason == "consecutive_zero_finds"
        assert len(session.query_results) == 3
        assert session.found_count == 0


@pytest.mark.asyncio
async def test_stopping_max_queries(
    sample_resources,
    mock_backend,
    mock_cache,
    mock_fetcher,
):
    """Test stopping when max queries reached."""
    # Mock to return one unrelated result per query (never find targets)
    # This ensures we don't reach coverage target before max queries

    async def mock_search(query):
        # Always return unrelated papers (not in our target set)
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title="Unrelated Paper",
                    url="https://example.com/other-paper",
                    backend="test_backend",
                    metadata={},
                )
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)

    config = ReverseSearchConfig(
        max_queries=5, coverage_target=1.0, consecutive_zero_limit=10
    )
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        # Generate many queries
        mock_gen.return_value = [f"query{i}" for i in range(10)]

        session = await searcher.search(sample_resources, verbose=False)

        assert session.stopping_reason == "max_queries"
        assert len(session.query_results) == 5


# Edge case tests


@pytest.mark.asyncio
async def test_empty_jsonl_file(tmp_path):
    """Test handling of empty JSONL file."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")

    with pytest.raises(ValueError, match="No valid resources"):
        parse_known_resources_jsonl(empty_file)


@pytest.mark.asyncio
async def test_malformed_jsonl(tmp_path):
    """Test handling of malformed JSONL."""
    malformed_file = tmp_path / "malformed.jsonl"
    malformed_file.write_text("not valid json\n")

    with pytest.raises(ResourceParseError, match="Invalid JSON"):
        parse_known_resources_jsonl(malformed_file)


@pytest.mark.asyncio
async def test_missing_required_field(tmp_path):
    """Test handling of JSONL entry missing 'url' field."""
    invalid_file = tmp_path / "invalid.jsonl"
    invalid_file.write_text('{"pmid": "123"}\n')  # Missing url

    with pytest.raises(ResourceParseError, match="Missing required field 'url'"):
        parse_known_resources_jsonl(invalid_file)


@pytest.mark.asyncio
async def test_all_resources_found_first_query(
    sample_resources,
    mock_backend,
    mock_cache,
    mock_fetcher,
):
    """Test when all resources found on first query."""

    # Mock to match all resources
    async def mock_search(query):
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title=f"Paper {i}",
                    url=r.canonical_url,
                    backend="test_backend",
                    metadata={"pmid": r.pmid} if r.pmid else {},
                )
                for i, r in enumerate(sample_resources)
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)

    config = ReverseSearchConfig()
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["perfect query"]

        session = await searcher.search(sample_resources, verbose=False)

        assert session.final_coverage == 1.0
        assert session.found_count == len(sample_resources)
        assert len(session.query_results) == 1
        assert session.stopping_reason == "coverage_achieved"


@pytest.mark.asyncio
async def test_no_resources_found(
    sample_resources,
    mock_backend,
    mock_cache,
    mock_fetcher,
):
    """Test when no resources ever found."""

    # Mock to never return matching results
    async def mock_search(query):
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title="Unrelated Paper",
                    url="https://example.com/unrelated",
                    backend="test_backend",
                    metadata={},
                )
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)

    config = ReverseSearchConfig(consecutive_zero_limit=3)
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["q1", "q2", "q3"]

        session = await searcher.search(sample_resources, verbose=False)

        assert session.found_count == 0
        assert session.final_coverage == 0.0
        assert len(session.unfound_resources) == len(sample_resources)
        assert session.stopping_reason == "consecutive_zero_finds"


# Multi-domain tests


@pytest.mark.asyncio
async def test_multi_domain_resources(tmp_path, mock_backend, mock_cache, mock_fetcher):
    """Test reverse search with resources from multiple domains."""
    # Create resources from different domains
    resources = [
        # Biomedical
        KnownResource(
            pmid="11111111",
            url="https://pubmed.ncbi.nlm.nih.gov/11111111/",
            hint_fields={"domain": "biomedical", "topic": "diabetes"},
        ),
        # Computer science
        KnownResource(
            url="https://arxiv.org/abs/1234.5678",
            hint_fields={"domain": "computer_science", "topic": "machine learning"},
        ),
        # Chemistry
        KnownResource(
            url="https://pubs.acs.org/doi/10.1021/example",
            hint_fields={"domain": "chemistry", "topic": "catalysis"},
        ),
    ]

    # Mock to match first two resources from different domains
    async def mock_search(query):
        # Return biomedical and computer science resources (first 2)
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title="Paper 0",
                    url=resources[0].canonical_url,
                    backend="test_backend",
                    metadata={"pmid": resources[0].pmid} if resources[0].pmid else {},
                ),
                SearchResult(
                    title="Paper 1",
                    url=resources[1].canonical_url,
                    backend="test_backend",
                    metadata={},
                ),
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)

    config = ReverseSearchConfig(coverage_target=0.6)
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["multi-domain query"]

        session = await searcher.search(resources, verbose=False)
        # Should work regardless of domain
        assert session.found_count >= 2
        assert session.final_coverage >= 0.6


# Performance tests


@pytest.mark.asyncio
@pytest.mark.slow
async def test_performance_with_caching(tmp_path, mock_backend, mock_fetcher):
    """Test performance with 100 resources (should complete in <5 minutes)."""
    # Create 100 test resources
    resources = [
        KnownResource(
            pmid=str(10000000 + i),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{10000000 + i}/",
        )
        for i in range(100)
    ]

    # Mock backend with realistic delay (100ms per search)
    # Track which query we're on to return different resources
    query_count = [0]

    async def mock_search(query):
        await asyncio.sleep(0.1)  # Simulate network delay
        # Match 10 different resources per query
        start_idx = query_count[0] * 10
        end_idx = min(start_idx + 10, len(resources))
        query_count[0] += 1

        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title=f"Paper {i}",
                    url=resources[i].canonical_url,
                    backend="test_backend",
                    metadata={"pmid": resources[i].pmid},
                )
                for i in range(start_idx, end_idx)
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    mock_backend.backend_name = "test_backend"
    # Use real cache
    cache = SearchCache(cache_dir=tmp_path / "cache", ttl_hours=24)

    config = ReverseSearchConfig(coverage_target=0.95)
    searcher = ReverseSearcher(config, mock_backend, cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        # Generate 20 queries (should hit coverage with caching)
        mock_gen.return_value = [f"query{i}" for i in range(20)]

        start_time = time.time()
        session = await searcher.search(resources, verbose=False)
        elapsed = time.time() - start_time
        # Should complete in reasonable time
        assert elapsed < 300  # 5 minutes
        assert session.found_count >= 95  # At least 95% found


# Additional edge case tests


@pytest.mark.asyncio
async def test_empty_target_resources(mock_backend, mock_cache, mock_fetcher):
    """Test handling of empty target resource list."""
    config = ReverseSearchConfig()
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with pytest.raises(ValueError, match="target_resources cannot be empty"):
        await searcher.search([], verbose=False)


@pytest.mark.asyncio
async def test_query_generation_failure(
    sample_resources,
    mock_backend,
    mock_cache,
    mock_fetcher,
):
    """Test handling of query generation failure."""
    config = ReverseSearchConfig()
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)
    # Mock query generator to raise error
    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.side_effect = Exception("Query generation failed")

        with pytest.raises(Exception, match="Failed to generate initial queries"):
            await searcher.search(sample_resources, verbose=False)


@pytest.mark.asyncio
async def test_backend_failure_recovery(
    sample_resources,
    mock_backend,
    mock_cache,
    mock_fetcher,
):
    """Test recovery from backend search failures."""
    call_count = [0]

    # Mock backend to fail first 2 queries, then succeed
    async def mock_search(query):
        call_count[0] += 1
        if call_count[0] <= 2:
            raise Exception("Backend error")
        # Third query succeeds
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title="Paper 1",
                    url=sample_resources[0].canonical_url,
                    backend="test_backend",
                    metadata={"pmid": sample_resources[0].pmid},
                )
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)

    config = ReverseSearchConfig(coverage_target=0.5)  # Minimum allowed
    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["q1", "q2", "q3"]

        session = await searcher.search(sample_resources, verbose=False)
        # Should have found something despite failures
        assert session.found_count >= 1
        # Should have attempted all queries
        assert call_count[0] == 3


@pytest.mark.asyncio
async def test_output_structure_validation(tmp_path, sample_resources):
    """Test that output JSONL has correct structure."""
    # Create a complete session with results
    from interaction_finder.search.reverse.models import (
        ReverseSearchResult,
        ResourceMatch,
    )

    query_result = ReverseSearchResult(
        query="test query",
        query_index=0,
        search_results=SearchResults(
            query=SearchQuery(query="test query"),
            results=[],
            backend="test_backend",
        ),
        resources_found=[sample_resources[0]],
        new_finds=1,
        cumulative_coverage=0.33,
        search_time=1.5,
        backend="test_backend",
    )

    match = ResourceMatch(
        resource=sample_resources[0],
        search_result=SearchResult(
            title="Test",
            url=sample_resources[0].canonical_url,
            backend="test_backend",
        ),
        match_method="pmid",
        confidence=1.0,
        query_index=0,
    )

    session = ReverseSearchSession(
        target_resources=sample_resources,
        query_results=[query_result],
        matches=[match],
        total_queries=1,
        final_coverage=0.33,
        found_count=1,
        unfound_resources=sample_resources[1:],
        total_time=2.0,
        stopping_reason="coverage_achieved",
    )

    output_path = tmp_path / "output.jsonl"
    write_reverse_search_output(session, output_path)
    # Validate structure
    with open(output_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2  # 1 query result + 1 summary
        # Check query result structure
        query_output = json.loads(lines[0])
        assert "query" in query_output
        assert "query_index" in query_output
        assert "new_finds" in query_output
        assert "cumulative_coverage" in query_output
        assert "search_time" in query_output
        assert "backend" in query_output
        assert "resources_found" in query_output
        # Check summary structure
        summary = json.loads(lines[1])
        assert summary["summary"] is True
        assert "total_queries" in summary
        assert "final_coverage" in summary
        assert "found_count" in summary
        assert "unfound_count" in summary
        assert "total_time" in summary
        assert "stopping_reason" in summary
