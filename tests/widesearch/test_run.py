"""Tests for widesearch convenience entry point."""

import pytest
from pydantic_ai.models.test import TestModel

from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchBackend, SearchQuery, SearchResult
from interaction_finder.settings import IfetcherConfig
from interaction_finder.widesearch import (
    WidesearchCheckpoint,
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

    with (
        get_goal_planner_agent.override(model=test_model),
        get_query_generator_agent.override(model=test_model),
        get_result_selector_agent.override(model=test_model),
        get_reflector_agent.override(model=test_model),
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

    with (
        get_goal_planner_agent.override(model=test_model),
        get_query_generator_agent.override(model=test_model),
        get_result_selector_agent.override(model=test_model),
        get_reflector_agent.override(model=test_model),
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

    with (
        get_goal_planner_agent.override(model=test_model),
        get_query_generator_agent.override(model=test_model),
        get_result_selector_agent.override(model=test_model),
        get_reflector_agent.override(model=test_model),
    ):
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword"],
            search_backend=backend,
            max_rounds=1,
            enable_reranking=False,  # Disable reranking
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

    with (
        get_goal_planner_agent.override(model=test_model),
        get_query_generator_agent.override(model=test_model),
        get_result_selector_agent.override(model=test_model),
        get_reflector_agent.override(model=test_model),
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

    with (
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
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

    with (
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
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

    with (
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
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

    with (
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
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
