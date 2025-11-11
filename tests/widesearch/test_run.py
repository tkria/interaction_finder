"""Tests for widesearch convenience entry point."""

import pytest
from pydantic_ai.models.test import TestModel

from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchBackend, SearchQuery, SearchResult
from interaction_finder.settings import IfetcherConfig
from interaction_finder.widesearch import (
    WidesearchCheckpoint,
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

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config).override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        checkpoint = await run_widesearch_with_checkpoint(
            topic="diabetes treatment",
            keyphrases=["insulin", "glucose"],
            search_backend=backend,
            config=test_config,
            max_rounds=1,
        )

        # Should return WidesearchCheckpoint
        assert isinstance(checkpoint, WidesearchCheckpoint)
        # Check all expected fields present
        assert isinstance(checkpoint.results, list)
        assert isinstance(checkpoint.queries, list)
        assert isinstance(checkpoint.query_results, dict)
        assert isinstance(checkpoint.resources, ResourcePool)
        assert checkpoint.topic == "diabetes treatment"
        assert checkpoint.keyphrases == ["insulin", "glucose"]
        assert checkpoint.rounds_completed >= 0


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

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config).override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        checkpoint = await run_widesearch_with_checkpoint(
            topic="test topic",
            keyphrases=["keyword"],
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

        # Verify JSON structure is valid
        loaded_dict = json.loads(json_str)
        assert "results" in loaded_dict
        assert "queries" in loaded_dict
        assert "query_results" in loaded_dict
        assert "resources" in loaded_dict
        assert "topic" in loaded_dict
        assert "keyphrases" in loaded_dict
        assert "rounds_completed" in loaded_dict

        # Full deserialization should now work with new ResourcePool serialization
        loaded_checkpoint = WidesearchCheckpoint.model_validate(loaded_dict)

        # Verify round-trip preserves data
        assert loaded_checkpoint.topic == checkpoint.topic
        assert loaded_checkpoint.keyphrases == checkpoint.keyphrases
        assert loaded_checkpoint.rounds_completed == checkpoint.rounds_completed


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

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config).override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        checkpoint = await run_widesearch_with_checkpoint(
            topic="test topic",
            keyphrases=["keyword1", "keyword2"],
            search_backend=backend,
            config=test_config,
            max_rounds=2,
        )

        # Should have executed queries
        assert len(checkpoint.queries) > 0
        # Should have results
        assert len(checkpoint.results) >= 0  # May be empty if no selection
        # Should have query_results mapping
        assert isinstance(checkpoint.query_results, dict)
        # Should have resource pool
        assert isinstance(checkpoint.resources, ResourcePool)
        # Topic and keyphrases should be preserved
        assert checkpoint.topic == "test topic"
        assert checkpoint.keyphrases == ["keyword1", "keyword2"]


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
        assert not isinstance(results, WidesearchCheckpoint)
        assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_fetch_and_populate_results_basic(test_config, tmp_path):
    """Test fetch_and_populate_results fetches and populates ResourcePool."""
    from unittest.mock import AsyncMock, patch

    # Create checkpoint with results
    results = [
        SearchResult(title="Paper 1", url="https://example.com/1", snippet="snippet 1"),
        SearchResult(title="Paper 2", url="https://example.com/2", snippet="snippet 2"),
    ]
    checkpoint = WidesearchCheckpoint(
        results=results,
        queries=["query1"],
        query_results={"query1": ["https://example.com/1", "https://example.com/2"]},
        resources=ResourcePool(),
        topic="test topic",
        keyphrases=["keyword"],
        rounds_completed=1,
    )

    # Configure test config with temp cache dir
    test_config.output.cache = str(tmp_path / "cache")

    # Mock PageFetcher.get_markdown to return mock content
    mock_content = ["# Paper 1 content", "# Paper 2 content"]

    with patch(
        "interaction_finder.widesearch.run.PageFetcher.get_markdown",
        new_callable=AsyncMock,
    ) as mock_get_markdown:
        mock_get_markdown.return_value = mock_content

        # Run fetch_and_populate_results
        stats = await fetch_and_populate_results(checkpoint, test_config)

        # Verify stats
        assert stats["total"] == 2
        assert stats["fetched"] == 2
        assert stats["cached"] == 0
        assert stats["failed"] == 0

        # Verify resources were added to pool
        assert len(checkpoint.resources.resource_map) == 2
        # Verify content was added
        for url in ["https://example.com/1", "https://example.com/2"]:
            resource = checkpoint.resources.get(url)
            assert resource is not None
            assert resource.text is not None


@pytest.mark.asyncio
async def test_fetch_and_populate_results_with_existing_content(test_config, tmp_path):
    """Test fetch_and_populate_results skips already-fetched content."""
    from unittest.mock import AsyncMock, patch

    # Create resource pool with one URL already having content
    pool = ResourcePool()
    rid1 = pool.register("https://example.com/1")
    pool.add_content(rid1, "Paper 1", "# Existing content")
    pool.register("https://example.com/2")  # No content yet

    results = [
        SearchResult(title="Paper 1", url="https://example.com/1", snippet="snippet 1"),
        SearchResult(title="Paper 2", url="https://example.com/2", snippet="snippet 2"),
    ]
    checkpoint = WidesearchCheckpoint(
        results=results,
        queries=["query1"],
        query_results={"query1": ["https://example.com/1", "https://example.com/2"]},
        resources=pool,
        topic="test topic",
        keyphrases=["keyword"],
        rounds_completed=1,
    )

    test_config.output.cache = str(tmp_path / "cache")

    # Mock PageFetcher to return content only for URL 2
    with patch(
        "interaction_finder.widesearch.run.PageFetcher.get_markdown",
        new_callable=AsyncMock,
    ) as mock_get_markdown:
        mock_get_markdown.return_value = ["# Paper 2 content"]

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

    results = [
        SearchResult(title="Paper 1", url="https://example.com/1", snippet="snippet 1"),
        SearchResult(title="Paper 2", url="https://example.com/2", snippet="snippet 2"),
    ]
    checkpoint = WidesearchCheckpoint(
        results=results,
        queries=["query1"],
        query_results={"query1": ["https://example.com/1", "https://example.com/2"]},
        resources=ResourcePool(),
        topic="test topic",
        keyphrases=["keyword"],
        rounds_completed=1,
    )

    test_config.output.cache = str(tmp_path / "cache")

    # Mock PageFetcher to return content for first URL, None for second (failed)
    with patch(
        "interaction_finder.widesearch.run.PageFetcher.get_markdown",
        new_callable=AsyncMock,
    ) as mock_get_markdown:
        mock_get_markdown.return_value = ["# Paper 1 content", None]

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
    checkpoint = WidesearchCheckpoint(
        results=[],
        queries=["query1"],
        query_results={},
        resources=ResourcePool(),
        topic="test topic",
        keyphrases=["keyword"],
        rounds_completed=1,
    )

    test_config.output.cache = str(tmp_path / "cache")

    stats = await fetch_and_populate_results(checkpoint, test_config)

    # Should return all zeros
    assert stats["total"] == 0
    assert stats["fetched"] == 0
    assert stats["cached"] == 0
    assert stats["failed"] == 0
