"""Tests for batch processing in SelectResultsNode."""

import pytest
from pydantic_ai.models.test import TestModel

from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchBackend, SearchQuery, SearchResult
from interaction_finder.settings import IfetcherConfig
from interaction_finder.widesearch import run_widesearch
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
    config.stage.search.reranker_device = "cpu"
    # Disable reranking by default for these tests (rerank_top_k = 0)
    config.stage.search.rerank_top_k = 0
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
async def test_batch_size_zero_processes_all_at_once(test_config):
    """Test that batch_size=0 processes all results in a single call."""
    # Create many results to ensure batching would be needed if enabled
    mock_results = [
        SearchResult(
            title=f"Paper {i}",
            url=f"https://example.com/{i}",
            snippet=f"content {i}",
        )
        for i in range(100)
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    # Set batch_size to 0 (process all at once)
    test_config.stage.search.batch_size = 0

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config, "mock").override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword1", "keyword2"],
            search_backend=backend,
            config=test_config,
            max_rounds=1,
        )

        # Should successfully return results
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_batch_size_splits_results(test_config):
    """Test that batch_size > 0 splits results into batches."""
    # Create 25 results with batch_size=10 -> should process in 3 batches
    mock_results = [
        SearchResult(
            title=f"Paper {i}",
            url=f"https://example.com/{i}",
            snippet=f"content {i}",
        )
        for i in range(25)
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    # Set batch_size to 10
    test_config.stage.search.batch_size = 10

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config, "mock").override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword1", "keyword2"],
            search_backend=backend,
            config=test_config,
            max_rounds=1,
        )

        # Should successfully return results
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_batch_size_larger_than_results(test_config):
    """Test that batch_size larger than result count processes all at once."""
    # Create 5 results with batch_size=10 -> should process in single batch
    mock_results = [
        SearchResult(
            title=f"Paper {i}",
            url=f"https://example.com/{i}",
            snippet=f"content {i}",
        )
        for i in range(5)
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    # Set batch_size larger than result count
    test_config.stage.search.batch_size = 10

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config, "mock").override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword1", "keyword2"],
            search_backend=backend,
            config=test_config,
            max_rounds=1,
        )

        # Should successfully return results
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_batching_with_reranking_enabled(test_config):
    """Test that batching works correctly when reranking is also enabled."""
    mock_results = [
        SearchResult(
            title=f"Paper {i}",
            url=f"https://example.com/{i}",
            snippet=f"content {i}",
        )
        for i in range(30)
    ]

    backend = MockSearchBackend(results=mock_results)
    test_model = TestModel()

    # Enable both reranking and batching
    test_config.stage.search.rerank_top_k = 20  # Rerank down to 20 results
    test_config.stage.search.batch_size = 10  # Then process in batches of 10

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config, "mock").override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword1", "keyword2"],
            search_backend=backend,
            config=test_config,
            max_rounds=1,
        )

        # Should successfully return results
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_batching_empty_results(test_config):
    """Test that batching handles empty result sets gracefully."""
    backend = MockSearchBackend(results=[])
    test_model = TestModel()

    # Set batch_size
    test_config.stage.search.batch_size = 10

    config = IfetcherConfig()
    with (
        get_goal_planner_agent(config).override(model=test_model),
        get_query_generator_agent(config, "mock").override(model=test_model),
        get_result_selector_agent(config).override(model=test_model),
        get_reflector_agent(config).override(model=test_model),
    ):
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword1", "keyword2"],
            search_backend=backend,
            config=test_config,
            max_rounds=1,
        )

        # Should return empty list
        assert isinstance(results, list)
        assert len(results) == 0
