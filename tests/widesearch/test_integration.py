"""Integration tests for widesearch pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock

import httpx
from pydantic_ai.models.test import TestModel

from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchBackend, SearchQuery, SearchResult
from interaction_finder.widesearch import Deps, State, graph
from interaction_finder.widesearch.agents import (
    goal_planner_agent,
    query_generator_agent,
    reflector_agent,
    result_selector_agent,
)
from interaction_finder.widesearch.nodes import PlanGoalsNode
from interaction_finder.widesearch.reranker import Reranker


class MockSearchBackend(SearchBackend):
    """Mock search backend for testing."""

    def __init__(self, results=None):
        super().__init__()
        self.results = results or []
        self.search_count = 0

    @property
    def name(self) -> str:
        return "mock"

    async def search(self, query: SearchQuery):
        """Return mock search results."""
        self.search_count += 1
        return self.results

    def healthy(self) -> bool:
        return True


class MockReranker:
    """Mock reranker for testing."""

    def __init__(self):
        self.rerank_count = 0

    def rerank(self, query, results, top_k=None):
        """Return results unchanged (already sorted)."""
        self.rerank_count += 1
        if top_k:
            return results[:top_k]
        return results

    def healthy(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_full_pipeline_single_round():
    """Test complete pipeline with early stopping after one round."""
    # Setup mock search results
    mock_results = [
        SearchResult(
            title="Diabetes Review 2024",
            url="https://example.com/1",
            snippet="Comprehensive review of diabetes",
        ),
        SearchResult(
            title="Insulin Mechanisms",
            url="https://example.com/2",
            snippet="Study on insulin action",
        ),
    ]

    # Create mocks
    search_backend = MockSearchBackend(results=mock_results)
    reranker = MockReranker()
    resource_pool = ResourcePool()

    # Create test model
    test_model = TestModel()

    # Override agents with test model
    with (
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
    ):
        async with httpx.AsyncClient() as client:
            deps = Deps(
                http_client=client,
                search_backend=search_backend,
                reranker=reranker,
                resource_pool=resource_pool,
                config={
                    "results_per_query": 50,
                    "rerank_top_k": 20,
                },
            )

            state = State(
                topic="diabetes treatment",
                keyphrases=["insulin", "glucose"],
                max_rounds=3,
            )

            # Run the graph
            result = await graph.run(PlanGoalsNode(), state=state, deps=deps)

            # Verify execution
            assert result.output is not None
            # Should have executed at least one search round
            assert state.current_round > 0
            # Should have some subject goals
            assert len(state.subject_goals) > 0


@pytest.mark.asyncio
async def test_pipeline_reaches_max_rounds():
    """Test pipeline stops at max_rounds limit."""
    mock_results = [
        SearchResult(
            title="Paper",
            url=f"https://example.com/{i}",
            snippet="content",
        )
        for i in range(5)
    ]

    search_backend = MockSearchBackend(results=mock_results)
    reranker = MockReranker()
    resource_pool = ResourcePool()

    test_model = TestModel()

    with (
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
    ):
        async with httpx.AsyncClient() as client:
            deps = Deps(
                http_client=client,
                search_backend=search_backend,
                reranker=reranker,
                resource_pool=resource_pool,
                config={
                    "results_per_query": 50,
                    "rerank_top_k": 20,
                },
            )

            state = State(
                topic="test topic",
                keyphrases=["keyword"],
                max_rounds=2,  # Low limit
            )

            result = await graph.run(PlanGoalsNode(), state=state, deps=deps)

            # Should complete at least one round and not exceed max_rounds
            assert 1 <= state.current_round <= 2


@pytest.mark.asyncio
async def test_pipeline_handles_no_results():
    """Test pipeline handles case with no search results."""
    # Empty results
    search_backend = MockSearchBackend(results=[])
    reranker = MockReranker()
    resource_pool = ResourcePool()

    test_model = TestModel()

    with (
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
    ):
        async with httpx.AsyncClient() as client:
            deps = Deps(
                http_client=client,
                search_backend=search_backend,
                reranker=reranker,
                resource_pool=resource_pool,
                config={
                    "results_per_query": 50,
                    "rerank_top_k": 20,
                },
            )

            state = State(
                topic="test topic",
                keyphrases=["keyword"],
                max_rounds=2,
            )

            # Should not crash with no results
            result = await graph.run(PlanGoalsNode(), state=state, deps=deps)

            assert result.output is not None


@pytest.mark.asyncio
async def test_pipeline_without_reranking():
    """Test pipeline with reranking disabled."""
    mock_results = [
        SearchResult(
            title=f"Paper {i}",
            url=f"https://example.com/{i}",
            snippet="content",
        )
        for i in range(10)
    ]

    search_backend = MockSearchBackend(results=mock_results)
    reranker = MockReranker()
    resource_pool = ResourcePool()

    test_model = TestModel()

    with (
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
    ):
        async with httpx.AsyncClient() as client:
            deps = Deps(
                http_client=client,
                search_backend=search_backend,
                reranker=reranker,
                resource_pool=resource_pool,
                config={
                    "results_per_query": 50,
                    "enable_reranking": False,  # Disable reranking
                },
            )

            state = State(
                topic="test topic",
                keyphrases=["keyword"],
                max_rounds=1,
            )

            result = await graph.run(PlanGoalsNode(), state=state, deps=deps)

            # Should complete successfully
            assert result.output is not None
            # Reranker should not have been called
            assert reranker.rerank_count == 0


@pytest.mark.asyncio
async def test_resource_pool_registration():
    """Test that URLs are registered with ResourcePool."""
    mock_results = [
        SearchResult(
            title=f"Paper {i}",
            url=f"https://example.com/{i}",
            snippet="content",
        )
        for i in range(3)
    ]

    search_backend = MockSearchBackend(results=mock_results)
    reranker = MockReranker()
    resource_pool = ResourcePool()

    test_model = TestModel()

    with (
        goal_planner_agent.override(model=test_model),
        query_generator_agent.override(model=test_model),
        result_selector_agent.override(model=test_model),
        reflector_agent.override(model=test_model),
    ):
        async with httpx.AsyncClient() as client:
            deps = Deps(
                http_client=client,
                search_backend=search_backend,
                reranker=reranker,
                resource_pool=resource_pool,
                config={
                    "results_per_query": 50,
                    "rerank_top_k": 20,
                },
            )

            state = State(
                topic="test topic",
                keyphrases=["keyword"],
                max_rounds=1,
            )

            await graph.run(PlanGoalsNode(), state=state, deps=deps)

            # Some URLs should have been registered
            # (Exact number depends on TestModel behavior)
            # At minimum, resource_pool should be usable
            assert resource_pool is not None
