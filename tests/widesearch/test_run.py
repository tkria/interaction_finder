"""Tests for widesearch convenience entry point."""

import pytest
from pydantic_ai.models.test import TestModel

from interaction_finder.checkpoint import PipelineCheckpoint
from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchBackend, SearchQuery, SearchResult
from interaction_finder.settings import IfetcherConfig
from interaction_finder.widesearch import (
    fetch_and_populate_results,
    run_widesearch,
    run_widesearch_with_checkpoint,
)
from interaction_finder.widesearch.agents import (
    get_goal_planner_agent,
    get_query_generator_agent,
    get_reflector_agent,
    get_result_selector_agent,
)


@pytest.fixture
def test_config():
    """Config with CPU device to avoid GPU memory issues in tests."""
    config = IfetcherConfig()
    # Force CPU device for reranker to avoid CUDA OOM in tests
    config.tools.widesearch.reranker_device = "cpu"
    return config


class MockSearchBackend(SearchBackend):
    """Mock search backend for testing."""

    def __init__(self, results=None):
        super().__init__()
        self.results = results or []

    @property
    def name(self) -> str:
        return "mock"

    async def search(self, query: SearchQuery):
        """Return mock search results."""
        return self.results

    def healthy(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_run_widesearch_basic(test_config):
    """Test basic usage of run_widesearch convenience function."""
    mock_results = [
        SearchResult(
            title="Paper 1",
            url="https://example.com/1",
            snippet="content 1",
        ),
        SearchResult(
            title="Paper 2",
            url="https://example.com/2",
            snippet="content 2",
        ),
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    # Call factory functions to get agent instances, then override
    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config).override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        results = await run_widesearch(
            topic="diabetes treatment",
            keyphrases=["insulin", "glucose"],
            search_backend=backend,
            config=test_config,
            max_rounds=1,
        )

        # Should return list of SearchResult
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_run_widesearch_with_existing_pool(test_config):
    """Test run_widesearch with existing ResourcePool."""
    mock_results = [
        SearchResult(
            title="Paper",
            url="https://example.com/test",
            snippet="content",
        ),
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    # Create pool with pre-existing URL
    existing_pool = ResourcePool()
    existing_pool.register("https://example.com/existing")

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config).override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword"],
            search_backend=backend,
            resource_pool=existing_pool,
            config=test_config,
            max_rounds=1,
        )

        # Should have used existing pool
        assert "https://example.com/existing" in existing_pool
        assert isinstance(results, list)


@pytest.mark.asyncio
async def test_run_widesearch_disable_reranking():
    """Test run_widesearch with reranking disabled."""
    mock_results = [
        SearchResult(
            title="Paper",
            url="https://example.com/test",
            snippet="content",
        ),
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config).override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword"],
            search_backend=backend,
            max_rounds=1,
        )

        assert isinstance(results, list)


@pytest.mark.asyncio
async def test_run_widesearch_custom_max_rounds(test_config):
    """Test run_widesearch with custom max_rounds."""
    mock_results = [
        SearchResult(
            title="Paper",
            url="https://example.com/test",
            snippet="content",
        ),
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config).override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword"],
            search_backend=backend,
            config=test_config,
            max_rounds=2,  # Custom max_rounds
        )

        assert isinstance(results, list)


@pytest.mark.asyncio
async def test_run_widesearch_with_checkpoint_basic(test_config):
    """Test run_widesearch_with_checkpoint returns complete checkpoint."""
    mock_results = [
        SearchResult(
            title="Paper 1",
            url="https://example.com/1",
            snippet="content 1",
        ),
        SearchResult(
            title="Paper 2",
            url="https://example.com/2",
            snippet="content 2",
        ),
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    # Create input checkpoint with keywords data
    from interaction_finder.checkpoint import KeywordsStageData

    input_checkpoint = PipelineCheckpoint(
        topic="diabetes treatment",
        resources=ResourcePool(),
        keywords=KeywordsStageData(
            terms=["insulin", "glucose"],
            scores=[0.9, 0.8],
            total_documents_processed=5,
            rounds_completed=1,
            coverage_assessment="Good coverage of topic with multiple relevant sources and bridging terms",
            resource_urls=[],
        ),
    )

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config).override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        checkpoint = await run_widesearch_with_checkpoint(
            input_checkpoint=input_checkpoint,
            search_backend=backend,
            config=test_config,
            max_rounds=1,
        )

        # Should return PipelineCheckpoint
        assert isinstance(checkpoint, PipelineCheckpoint)
        # Check core fields present
        assert isinstance(checkpoint.resources, ResourcePool)
        assert checkpoint.topic == "diabetes treatment"
        # Check keywords data preserved
        assert checkpoint.keywords is not None
        assert checkpoint.keywords.terms == ["insulin", "glucose"]
        # Check search stage data present
        assert checkpoint.search is not None
        assert isinstance(checkpoint.search.results, list)
        assert isinstance(checkpoint.search.queries, list)
        assert isinstance(checkpoint.search.query_results, dict)
        assert checkpoint.search.keyphrases == ["insulin", "glucose"]
        assert checkpoint.search.rounds_completed >= 0


@pytest.mark.asyncio
async def test_checkpoint_serialization(test_config):
    """Test checkpoint can be serialized to JSON and back."""
    import json

    mock_results = [
        SearchResult(
            title="Paper",
            url="https://example.com/test",
            snippet="content",
        ),
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    # Create input checkpoint with keywords data
    from interaction_finder.checkpoint import KeywordsStageData

    input_checkpoint = PipelineCheckpoint(
        topic="test topic",
        resources=ResourcePool(),
        keywords=KeywordsStageData(
            terms=["keyword"],
            scores=[0.9],
            total_documents_processed=5,
            rounds_completed=1,
            coverage_assessment="Good coverage of topic with multiple relevant sources and bridging terms",
            resource_urls=[],
        ),
    )

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config).override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        checkpoint = await run_widesearch_with_checkpoint(
            input_checkpoint=input_checkpoint,
            search_backend=backend,
            config=test_config,
            max_rounds=1,
        )

        # Serialize to JSON
        checkpoint_dict = checkpoint.model_dump(mode="json")
        json_str = json.dumps(checkpoint_dict)

        # Should not raise errors
        assert json_str is not None
        assert len(json_str) > 0

        # Verify JSON structure is valid (new nested structure)
        loaded_dict = json.loads(json_str)
        assert "search" in loaded_dict
        assert "results" in loaded_dict["search"]
        assert "queries" in loaded_dict["search"]
        assert "query_results" in loaded_dict["search"]
        assert "resources" in loaded_dict
        assert "topic" in loaded_dict
        assert "keyphrases" in loaded_dict["search"]
        assert "rounds_completed" in loaded_dict["search"]

        # Full deserialization should now work with new ResourcePool serialization
        loaded_checkpoint = PipelineCheckpoint.model_validate(loaded_dict)

        # Verify round-trip preserves data
        assert loaded_checkpoint.topic == checkpoint.topic
        assert loaded_checkpoint.search.keyphrases == checkpoint.search.keyphrases
        assert (
            loaded_checkpoint.search.rounds_completed
            == checkpoint.search.rounds_completed
        )


@pytest.mark.asyncio
async def test_checkpoint_contains_all_data(test_config):
    """Test checkpoint contains queries, results, and resource pool."""
    mock_results = [
        SearchResult(
            title="Paper 1",
            url="https://example.com/1",
            snippet="content 1",
        ),
        SearchResult(
            title="Paper 2",
            url="https://example.com/2",
            snippet="content 2",
        ),
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    # Create input checkpoint with keywords data
    from interaction_finder.checkpoint import KeywordsStageData

    input_checkpoint = PipelineCheckpoint(
        topic="test topic",
        resources=ResourcePool(),
        keywords=KeywordsStageData(
            terms=["keyword1", "keyword2"],
            scores=[0.9, 0.8],
            total_documents_processed=5,
            rounds_completed=1,
            coverage_assessment="Good coverage of topic with multiple relevant sources and bridging terms",
            resource_urls=[],
        ),
    )

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config).override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        checkpoint = await run_widesearch_with_checkpoint(
            input_checkpoint=input_checkpoint,
            search_backend=backend,
            config=test_config,
            max_rounds=2,
        )

        # Should have executed queries
        assert len(checkpoint.search.queries) > 0
        # Should have results
        assert len(checkpoint.search.results) >= 0  # May be empty if no selection
        # Should have query_results mapping
        assert isinstance(checkpoint.search.query_results, dict)
        # Should have resource pool
        assert isinstance(checkpoint.resources, ResourcePool)
        # Topic and keyphrases should be preserved
        assert checkpoint.topic == "test topic"
        assert checkpoint.search.keyphrases == ["keyword1", "keyword2"]


@pytest.mark.asyncio
async def test_checkpoint_backward_compatibility(test_config):
    """Test original run_widesearch still works unchanged."""
    mock_results = [
        SearchResult(
            title="Paper",
            url="https://example.com/test",
            snippet="content",
        ),
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config).override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        # Original function should still return list[SearchResult]
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword"],
            search_backend=backend,
            config=test_config,
            max_rounds=1,
        )

        # Should return list, not checkpoint
        assert isinstance(results, list)
        assert not isinstance(results, PipelineCheckpoint)
        assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_fetch_and_populate_results_basic(test_config, tmp_path):
    """Test fetch_and_populate_results fetches and populates ResourcePool."""
    from unittest.mock import AsyncMock, patch

    # Create checkpoint with results
    from interaction_finder.checkpoint import SearchStageData

    results = [
        SearchResult(title="Paper 1", url="https://example.com/1", snippet="snippet 1"),
        SearchResult(title="Paper 2", url="https://example.com/2", snippet="snippet 2"),
    ]
    checkpoint = PipelineCheckpoint(
        topic="test topic",
        resources=ResourcePool(),
        search=SearchStageData(
            results=results,
            queries=["query1"],
            query_results={
                "query1": ["https://example.com/1", "https://example.com/2"]
            },
            keyphrases=["keyword"],
            rounds_completed=1,
        ),
    )

    # Configure test config with temp cache dir
    test_config.output.cache = str(tmp_path / "cache")

    # Mock PageFetcher methods
    from interaction_finder.fetcher import FetchedDocument

    mock_documents = {
        "https://example.com/1": FetchedDocument(
            url="https://example.com/1",
            content_markdown="# Paper 1 content",
            source_type="html",
            doi="10.1234/paper1",
            publication_date="2023-06-15",
        ),
        "https://example.com/2": FetchedDocument(
            url="https://example.com/2",
            content_markdown="# Paper 2 content",
            source_type="html",
            doi=None,
            publication_date=None,
        ),
    }
    # Mock PageFetcher.get_chunks to return mock chunks
    mock_chunks = [
        ["# Paper 1 content"],  # Single chunk for paper 1
        ["# Paper 2", " content"],  # Two chunks for paper 2
    ]

    with (
        patch(
            "interaction_finder.widesearch.run.PageFetcher.fetch_documents",
            new_callable=AsyncMock,
        ) as mock_fetch_documents,
        patch(
            "interaction_finder.widesearch.run.PageFetcher.get_chunks",
            new_callable=AsyncMock,
        ) as mock_get_chunks,
    ):
        mock_fetch_documents.return_value = mock_documents
        mock_get_chunks.return_value = mock_chunks

        # Run fetch_and_populate_results
        stats = await fetch_and_populate_results(checkpoint, test_config)

        # Verify stats
        assert stats["total"] == 2
        assert stats["fetched"] == 2
        assert stats["cached"] == 0
        assert stats["failed"] == 0

        # Verify resources were added to pool
        assert len(checkpoint.resources.resource_map) == 2

        # Verify Paper 1: single chunk
        resource1 = checkpoint.resources.get("https://example.com/1")
        assert resource1 is not None
        assert resource1.text == "# Paper 1 content"
        assert len(resource1.chunks) == 1
        assert resource1.chunks[0] == (0, len("# Paper 1 content"))
        assert resource1.doi == "10.1234/paper1"
        assert resource1.publication_date == "2023-06-15"

        # Verify Paper 2: two chunks
        resource2 = checkpoint.resources.get("https://example.com/2")
        assert resource2 is not None
        assert resource2.text == "# Paper 2 content"
        # Should have 2 chunks computed from ["# Paper 2", " content"]
        assert len(resource2.chunks) == 2
        # First chunk should be "# Paper 2" found at start
        assert resource2.chunks[0][0] == 0
        # Second chunk should be " content" found after first chunk
        assert resource2.chunks[1][0] > 0
        assert resource2.doi is None
        assert resource2.publication_date is None


@pytest.mark.asyncio
async def test_fetch_and_populate_results_with_existing_content(test_config, tmp_path):
    """Test fetch_and_populate_results skips already-fetched content."""
    from unittest.mock import AsyncMock, patch
    from interaction_finder.checkpoint import SearchStageData

    # Create resource pool with one URL already having content
    pool = ResourcePool()
    rid1 = pool.register("https://example.com/1")
    pool.add_content(rid1, "Paper 1", "# Existing content")
    pool.register("https://example.com/2")  # No content yet

    results = [
        SearchResult(title="Paper 1", url="https://example.com/1", snippet="snippet 1"),
        SearchResult(title="Paper 2", url="https://example.com/2", snippet="snippet 2"),
    ]
    checkpoint = PipelineCheckpoint(
        topic="test topic",
        resources=pool,
        search=SearchStageData(
            results=results,
            queries=["query1"],
            query_results={
                "query1": ["https://example.com/1", "https://example.com/2"]
            },
            keyphrases=["keyword"],
            rounds_completed=1,
        ),
    )

    test_config.output.cache = str(tmp_path / "cache")

    # Mock PageFetcher to return content only for URL 2
    from interaction_finder.fetcher import FetchedDocument

    mock_documents = {
        "https://example.com/2": FetchedDocument(
            url="https://example.com/2",
            content_markdown="# Paper 2 content",
            source_type="html",
            doi=None,
            publication_date=None,
        ),
    }

    with (
        patch(
            "interaction_finder.widesearch.run.PageFetcher.fetch_documents",
            new_callable=AsyncMock,
        ) as mock_fetch_documents,
        patch(
            "interaction_finder.widesearch.run.PageFetcher.get_chunks",
            new_callable=AsyncMock,
        ) as mock_get_chunks,
    ):
        mock_fetch_documents.return_value = mock_documents
        mock_get_chunks.return_value = [["# Paper 2 content"]]

        stats = await fetch_and_populate_results(checkpoint, test_config)

        # Should only fetch URL 2
        assert stats["total"] == 2
        assert stats["fetched"] == 1
        assert stats["cached"] == 1
        assert stats["failed"] == 0

        # Verify URL 1 still has original content
        resource1 = checkpoint.resources.get("https://example.com/1")
        assert resource1.text == "# Existing content"


@pytest.mark.asyncio
async def test_fetch_and_populate_results_handles_failures(test_config, tmp_path):
    """Test fetch_and_populate_results handles fetch failures gracefully."""
    from unittest.mock import AsyncMock, patch
    from interaction_finder.checkpoint import SearchStageData

    results = [
        SearchResult(title="Paper 1", url="https://example.com/1", snippet="snippet 1"),
        SearchResult(title="Paper 2", url="https://example.com/2", snippet="snippet 2"),
    ]
    checkpoint = PipelineCheckpoint(
        topic="test topic",
        resources=ResourcePool(),
        search=SearchStageData(
            results=results,
            queries=["query1"],
            query_results={
                "query1": ["https://example.com/1", "https://example.com/2"]
            },
            keyphrases=["keyword"],
            rounds_completed=1,
        ),
    )

    test_config.output.cache = str(tmp_path / "cache")

    # Mock PageFetcher to return content for first URL, None for second (failed)
    from interaction_finder.fetcher import FetchedDocument

    mock_documents = {
        "https://example.com/1": FetchedDocument(
            url="https://example.com/1",
            content_markdown="# Paper 1 content",
            source_type="html",
            doi=None,
            publication_date=None,
        ),
        # URL 2 has no entry, simulating a fetch failure
    }

    with (
        patch(
            "interaction_finder.widesearch.run.PageFetcher.fetch_documents",
            new_callable=AsyncMock,
        ) as mock_fetch_documents,
        patch(
            "interaction_finder.widesearch.run.PageFetcher.get_chunks",
            new_callable=AsyncMock,
        ) as mock_get_chunks,
    ):
        mock_fetch_documents.return_value = mock_documents
        mock_get_chunks.return_value = [["# Paper 1 content"], []]

        stats = await fetch_and_populate_results(checkpoint, test_config)

        # Should have 1 success, 1 failure
        assert stats["total"] == 2
        assert stats["fetched"] == 1
        assert stats["cached"] == 0
        assert stats["failed"] == 1

        # Verify successful URL has content
        resource1 = checkpoint.resources.get("https://example.com/1")
        assert resource1 is not None
        assert resource1.text == "# Paper 1 content"

        # Failed URL should still be registered but without content
        resource2 = checkpoint.resources.get("https://example.com/2")
        assert resource2 is None  # No content added


@pytest.mark.asyncio
async def test_fetch_and_populate_results_empty_checkpoint(test_config, tmp_path):
    """Test fetch_and_populate_results handles empty results gracefully."""
    from interaction_finder.checkpoint import SearchStageData

    checkpoint = PipelineCheckpoint(
        topic="test topic",
        resources=ResourcePool(),
        search=SearchStageData(
            results=[],
            queries=["query1"],
            query_results={},
            keyphrases=["keyword"],
            rounds_completed=1,
        ),
    )

    test_config.output.cache = str(tmp_path / "cache")

    stats = await fetch_and_populate_results(checkpoint, test_config)

    # Should return all zeros
    assert stats["total"] == 0
    assert stats["fetched"] == 0
    assert stats["cached"] == 0
    assert stats["failed"] == 0
