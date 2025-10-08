"""
Comprehensive unit tests for ReverseSearcher orchestration.

Tests cover:
- Full workflow with mock backend
- Stopping criteria (coverage achieved, consecutive zeros, max queries)
- Error handling (empty targets, query generation failures, backend failures)
- Cache hit/miss scenarios
- Coverage calculation correctness
- Refinement query generation
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from interaction_finder.search.reverse.searcher import ReverseSearcher
from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
    ReverseSearchError,
)
from interaction_finder.search.base import (
    SearchQuery,
    SearchResult,
    SearchResults,
)


@pytest.mark.asyncio
async def test_reverse_search_full_workflow():
    """Test complete reverse search workflow with mock backend."""
    config = ReverseSearchConfig(coverage_target=0.5)  # Lower for test
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"

    # Mock backend.search to return results matching first resource
    async def mock_search(query):
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title="Test Paper",
                    url="https://example.com/paper1",
                    backend="test_backend",
                    metadata={"pmid": "123"},
                ),
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    # Mock cache (always miss)
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    # Mock fetcher
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)
    # Mock query generator to return one query
    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["test query"]

        target_resources = [
            KnownResource(pmid="123", url="https://example.com/paper1"),
            KnownResource(pmid="456", url="https://example.com/paper2"),
        ]

        session = await searcher.search(target_resources, verbose=False)
        # Assertions
        assert session.found_count >= 1
        assert session.final_coverage >= 0.5
        assert len(session.query_results) > 0
        assert session.stopping_reason in [
            "coverage_achieved",
            "consecutive_zero_finds",
            "max_queries",
        ]


@pytest.mark.asyncio
async def test_stopping_criterion_coverage_achieved():
    """Test that search stops when coverage target reached."""
    config = ReverseSearchConfig(coverage_target=0.5)
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"

    # Mock to return results that match all resources
    async def mock_search(query):
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title="Paper 1",
                    url="https://example.com/paper1",
                    backend="test_backend",
                    metadata={"pmid": "123"},
                ),
                SearchResult(
                    title="Paper 2",
                    url="https://example.com/paper2",
                    backend="test_backend",
                    metadata={"pmid": "456"},
                ),
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["test query"]

        target_resources = [
            KnownResource(pmid="123", url="https://example.com/paper1"),
            KnownResource(pmid="456", url="https://example.com/paper2"),
        ]

        session = await searcher.search(target_resources, verbose=False)
        # Should stop after first query with 100% coverage
        assert session.final_coverage == 1.0
        assert session.stopping_reason == "coverage_achieved"
        assert len(session.query_results) == 1


@pytest.mark.asyncio
async def test_stopping_criterion_consecutive_zero_finds():
    """Test that search stops after N consecutive queries with no finds."""
    config = ReverseSearchConfig(consecutive_zero_limit=3, max_queries=100)
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"

    # Mock to always return empty results
    async def mock_search(query):
        return SearchResults(
            query=query,
            results=[],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)
    # Generate 5 queries initially (more than consecutive zero limit)
    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["query1", "query2", "query3", "query4", "query5"]

        target_resources = [
            KnownResource(pmid=str(i), url=f"https://example.com/paper{i}")
            for i in range(10)
        ]

        session = await searcher.search(target_resources, verbose=False)
        # Should stop after 3 consecutive zero finds
        assert session.stopping_reason == "consecutive_zero_finds"
        assert len(session.query_results) == 3
        assert session.final_coverage == 0.0


@pytest.mark.asyncio
async def test_stopping_criterion_max_queries():
    """Test that search stops when max queries reached."""
    config = ReverseSearchConfig(max_queries=5, consecutive_zero_limit=10)
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"
    # Mock to return one match per query (never reach coverage target)
    call_count = 0

    async def mock_search(query):
        nonlocal call_count
        call_count += 1
        # Return a different PMID each time
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title=f"Paper {call_count}",
                    url=f"https://example.com/paper{call_count}",
                    backend="test_backend",
                    metadata={"pmid": str(call_count)},
                ),
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)
    # Generate many queries initially
    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = [f"query{i}" for i in range(10)]

        target_resources = [
            KnownResource(pmid=str(i), url=f"https://example.com/paper{i}")
            for i in range(1, 21)  # 20 resources
        ]

        session = await searcher.search(target_resources, verbose=False)
        # Should stop after 5 queries (max_queries)
        assert session.stopping_reason == "max_queries"
        assert len(session.query_results) == 5
        assert session.found_count == 5  # One per query
        assert session.final_coverage == 0.25  # 5/20


@pytest.mark.asyncio
async def test_empty_target_resources():
    """Test that empty target_resources raises ValueError."""
    config = ReverseSearchConfig()
    mock_backend = Mock()
    mock_cache = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with pytest.raises(ValueError, match="target_resources cannot be empty"):
        await searcher.search([], verbose=False)


@pytest.mark.asyncio
async def test_query_generation_failure():
    """Test that query generation failure raises ReverseSearchError."""
    config = ReverseSearchConfig()
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"
    mock_cache = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)
    # Mock query generator to raise exception
    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.side_effect = Exception("Query generation failed")

        target_resources = [
            KnownResource(pmid="123", url="https://example.com/paper1"),
        ]

        with pytest.raises(
            ReverseSearchError, match="Failed to generate initial queries"
        ):
            await searcher.search(target_resources, verbose=False)


@pytest.mark.asyncio
async def test_backend_execution_failure_continues():
    """Test that backend execution failure continues with next query."""
    config = ReverseSearchConfig(consecutive_zero_limit=3, max_queries=10)
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"
    # First query fails, second succeeds
    call_count = 0

    async def mock_search(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Backend error")
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title="Paper 1",
                    url="https://example.com/paper1",
                    backend="test_backend",
                    metadata={"pmid": "123"},
                ),
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["query1", "query2"]

        target_resources = [
            KnownResource(pmid="123", url="https://example.com/paper1"),
        ]

        session = await searcher.search(target_resources, verbose=False)
        # Should continue after first failure and find resource with second query
        assert session.found_count == 1
        assert len(session.query_results) == 1  # Only successful query recorded


@pytest.mark.asyncio
async def test_cache_hit():
    """Test that cache hit returns cached results."""
    config = ReverseSearchConfig()
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"
    # Cached results
    cached_results = SearchResults(
        query=SearchQuery(query="test query"),
        results=[
            SearchResult(
                title="Cached Paper",
                url="https://example.com/paper1",
                backend="test_backend",
                metadata={"pmid": "123"},
            ),
        ],
        backend="test_backend",
    )
    # Mock cache to return cached results
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=cached_results)
    mock_cache.set = AsyncMock()
    # Mock backend should NOT be called
    mock_backend.search = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["test query"]

        target_resources = [
            KnownResource(pmid="123", url="https://example.com/paper1"),
        ]

        session = await searcher.search(target_resources, verbose=False)
        # Cache should be checked
        assert mock_cache.get.called
        # Backend should NOT be called
        assert not mock_backend.search.called
        # Should find resource from cache
        assert session.found_count == 1


@pytest.mark.asyncio
async def test_cache_miss_stores_results():
    """Test that cache miss executes query and stores results."""
    config = ReverseSearchConfig()
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"

    # Mock backend to return results
    async def mock_search(query):
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title="Paper 1",
                    url="https://example.com/paper1",
                    backend="test_backend",
                    metadata={"pmid": "123"},
                ),
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    # Mock cache to return None (miss)
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["test query"]

        target_resources = [
            KnownResource(pmid="123", url="https://example.com/paper1"),
        ]

        session = await searcher.search(target_resources, verbose=False)
        # Cache should be checked
        assert mock_cache.get.called
        # Backend should be called
        assert mock_backend.search.called
        # Results should be cached
        assert mock_cache.set.called
        # Should find resource
        assert session.found_count == 1


@pytest.mark.asyncio
async def test_coverage_calculation():
    """Test that coverage is correctly calculated."""
    config = ReverseSearchConfig(max_queries=10, consecutive_zero_limit=5)
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"
    # Return different resources per query
    call_count = 0

    async def mock_search(query):
        nonlocal call_count
        call_count += 1
        # First query finds resource 1, second finds resource 2
        pmid = str(call_count)
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title=f"Paper {pmid}",
                    url=f"https://example.com/paper{pmid}",
                    backend="test_backend",
                    metadata={"pmid": pmid},
                ),
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["query1", "query2"]

        target_resources = [
            KnownResource(pmid="1", url="https://example.com/paper1"),
            KnownResource(pmid="2", url="https://example.com/paper2"),
            KnownResource(pmid="3", url="https://example.com/paper3"),
            KnownResource(pmid="4", url="https://example.com/paper4"),
        ]

        session = await searcher.search(target_resources, verbose=False)
        # Should find 2 out of 4 resources
        assert session.found_count == 2
        assert session.final_coverage == 0.5
        # Check cumulative coverage in query results
        assert session.query_results[0].cumulative_coverage == 0.25  # 1/4
        assert session.query_results[1].cumulative_coverage == 0.5  # 2/4


@pytest.mark.asyncio
async def test_refinement_query_generation():
    """Test that refinement queries are generated when initial queries exhausted."""
    config = ReverseSearchConfig(
        coverage_target=1.0,  # Need 100% to stop
        consecutive_zero_limit=5,
        max_queries=10,
    )
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"
    # First query finds one resource, refinement query finds another
    call_count = 0

    async def mock_search(query):
        nonlocal call_count
        call_count += 1
        pmid = str(call_count)
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title=f"Paper {pmid}",
                    url=f"https://example.com/paper{pmid}",
                    backend="test_backend",
                    metadata={"pmid": pmid},
                ),
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)
    # Mock both initial and refinement query generation
    with (
        patch.object(
            searcher.query_generator, "generate_initial_queries"
        ) as mock_initial,
        patch.object(
            searcher.query_generator, "generate_refinement_queries"
        ) as mock_refine,
    ):
        mock_initial.return_value = ["initial_query"]
        mock_refine.return_value = ["refinement_query"]

        target_resources = [
            KnownResource(pmid="1", url="https://example.com/paper1"),
            KnownResource(pmid="2", url="https://example.com/paper2"),
        ]

        session = await searcher.search(target_resources, verbose=False)
        # Should call refinement query generation
        assert mock_refine.called
        # Should find both resources (one from initial, one from refinement)
        assert session.found_count == 2
        assert session.final_coverage == 1.0
        assert len(session.query_results) == 2


@pytest.mark.asyncio
async def test_no_refinement_queries_stops():
    """Test that search stops when refinement query generation returns empty."""
    config = ReverseSearchConfig(
        coverage_target=1.0,
        consecutive_zero_limit=5,
        max_queries=10,
    )
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"

    # Return empty results
    async def mock_search(query):
        return SearchResults(
            query=query,
            results=[],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with (
        patch.object(
            searcher.query_generator, "generate_initial_queries"
        ) as mock_initial,
        patch.object(
            searcher.query_generator, "generate_refinement_queries"
        ) as mock_refine,
    ):
        mock_initial.return_value = ["initial_query"]
        mock_refine.return_value = []  # No refinement queries

        target_resources = [
            KnownResource(pmid="1", url="https://example.com/paper1"),
            KnownResource(pmid="2", url="https://example.com/paper2"),
        ]

        session = await searcher.search(target_resources, verbose=False)
        # Should stop after initial query (refinement returned empty)
        assert len(session.query_results) == 1
        assert session.found_count == 0


@pytest.mark.asyncio
async def test_consecutive_zero_resets_on_find():
    """Test that consecutive zero finds counter resets when resources found."""
    config = ReverseSearchConfig(consecutive_zero_limit=2, max_queries=10)
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"
    # Alternate between empty and found results
    call_count = 0

    async def mock_search(query):
        nonlocal call_count
        call_count += 1
        # Queries 1,3,5 find nothing; queries 2,4 find resources
        if call_count % 2 == 0:
            pmid = str(call_count)
            return SearchResults(
                query=query,
                results=[
                    SearchResult(
                        title=f"Paper {pmid}",
                        url=f"https://example.com/paper{pmid}",
                        backend="test_backend",
                        metadata={"pmid": pmid},
                    ),
                ],
                backend="test_backend",
            )
        else:
            return SearchResults(query=query, results=[], backend="test_backend")

    mock_backend.search = AsyncMock(side_effect=mock_search)
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = [f"query{i}" for i in range(1, 6)]

        target_resources = [
            KnownResource(pmid=str(i), url=f"https://example.com/paper{i}")
            for i in range(2, 11)  # PMIDs 2,4,6,8,10
        ]

        session = await searcher.search(target_resources, verbose=False)
        # Should execute all 5 queries (consecutive zero resets on finds)
        # Query 1: 0 finds (consecutive=1)
        # Query 2: 1 find (consecutive=0)
        # Query 3: 0 finds (consecutive=1)
        # Query 4: 1 find (consecutive=0)
        # Query 5: 0 finds (consecutive=1)
        assert len(session.query_results) == 5
        assert session.found_count == 2


@pytest.mark.asyncio
async def test_unfound_resources_tracked():
    """Test that unfound resources are correctly tracked."""
    config = ReverseSearchConfig(consecutive_zero_limit=2, max_queries=5)
    mock_backend = Mock()
    mock_backend.backend_name = "test_backend"

    # Only find resource 1
    async def mock_search(query):
        return SearchResults(
            query=query,
            results=[
                SearchResult(
                    title="Paper 1",
                    url="https://example.com/paper1",
                    backend="test_backend",
                    metadata={"pmid": "1"},
                ),
            ],
            backend="test_backend",
        )

    mock_backend.search = AsyncMock(side_effect=mock_search)
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_fetcher = Mock()

    searcher = ReverseSearcher(config, mock_backend, mock_cache, mock_fetcher)

    with patch.object(searcher.query_generator, "generate_initial_queries") as mock_gen:
        mock_gen.return_value = ["query"]

        resource1 = KnownResource(pmid="1", url="https://example.com/paper1")
        resource2 = KnownResource(pmid="2", url="https://example.com/paper2")
        resource3 = KnownResource(pmid="3", url="https://example.com/paper3")

        target_resources = [resource1, resource2, resource3]

        session = await searcher.search(target_resources, verbose=False)
        # Should track unfound resources
        assert session.found_count == 1
        assert len(session.unfound_resources) == 2
        # Check that unfound are resource2 and resource3
        unfound_pmids = {r.pmid for r in session.unfound_resources}
        assert unfound_pmids == {"2", "3"}
