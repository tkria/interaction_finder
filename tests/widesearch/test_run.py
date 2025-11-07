"""Tests for widesearch convenience entry point."""

import pytest
from pydantic_ai.models.test import TestModel

from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchBackend, SearchQuery, SearchResult
from interaction_finder.widesearch import run_widesearch
from interaction_finder.widesearch.agents import (
    goal_planner_agent,
    query_generator_agent,
    reflector_agent,
    result_selector_agent,
)


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
async def test_run_widesearch_basic():
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
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
    ):
        results = await run_widesearch(
            topic="diabetes treatment",
            keyphrases=["insulin", "glucose"],
            search_backend=backend,
            max_rounds=1,
        )

        # Should return list of SearchResult
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_run_widesearch_with_existing_pool():
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
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
    ):
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword"],
            search_backend=backend,
            resource_pool=existing_pool,
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
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
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
async def test_run_widesearch_custom_max_rounds():
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
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
    ):
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword"],
            search_backend=backend,
            max_rounds=2,  # Custom max_rounds
        )

        assert isinstance(results, list)
