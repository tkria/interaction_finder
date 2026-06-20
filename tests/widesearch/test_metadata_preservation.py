"""Test that SearchResult metadata is preserved through the pipeline.

Regression test for bug where ReflectNode was creating new SearchResult
objects with empty titles instead of retrieving the stored objects.
"""

import contextlib

import pytest
from pydantic_ai.models.test import TestModel

from interaction_finder.search.models import SearchQuery, SearchResult, SearchBackend
from interaction_finder.settings import IfetcherConfig
from interaction_finder.widesearch import run_widesearch
from interaction_finder.widesearch.agents import (
    get_goal_planner_agent,
    get_query_generator_agent,
    get_reflector_agent,
    get_result_selector_agent,
)


@contextlib.contextmanager
def mock_widesearch_agents():
    """Override every widesearch LLM agent with TestModel.

    run_widesearch builds the graph with real agents; without this the run
    makes live OpenAI calls. The agents are cached by config signature, so
    overriding those fetched from the default config applies to the run.
    """
    config = IfetcherConfig()
    model = TestModel()
    with (
        get_goal_planner_agent(config).override(model=model),
        get_query_generator_agent(config, "mock").override(model=model),
        get_result_selector_agent(config).override(model=model),
        get_reflector_agent(config).override(model=model),
    ):
        yield


class MockSearchBackend(SearchBackend):
    """Mock search backend that returns results with full metadata."""

    def __init__(self, results_per_query=5):
        super().__init__()
        self.results_per_query = results_per_query
        self.search_count = 0

    @property
    def name(self) -> str:
        return "mock"

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Return mock results with titles, snippets, and relevance."""
        self.search_count += 1
        results = []
        for i in range(min(query.max_results, self.results_per_query)):
            idx = self.search_count * 100 + i
            results.append(
                SearchResult(
                    title=f"Test Article {idx}: {query.query[:30]}",
                    url=f"https://example.com/article/{idx}",
                    snippet=f"This is a snippet for article {idx} about {query.query}",
                    relevance=0.9 - (i * 0.1),
                )
            )
        return results

    def healthy(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_search_result_metadata_preserved():
    """Test that titles, snippets, and relevance are preserved in final results."""
    backend = MockSearchBackend(results_per_query=3)

    with mock_widesearch_agents():
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword1", "keyword2"],
            search_backend=backend,
            max_rounds=1,  # Single round to simplify
        )

    # Should have at least some results
    assert len(results) > 0, "Expected some results from widesearch"

    # Check that metadata is preserved
    for result in results:
        assert result.title != "", f"Title should not be empty for {result.url}"
        assert result.title.startswith("Test Article"), (
            f"Expected proper title, got: {result.title}"
        )
        assert result.snippet is not None, (
            f"Snippet should not be None for {result.url}"
        )
        assert "snippet for article" in result.snippet
        assert result.relevance is not None, f"Relevance should not be None"
        assert 0.0 <= result.relevance <= 1.0


@pytest.mark.asyncio
async def test_max_rounds_metadata_preserved():
    """Test metadata preservation when max_rounds is reached."""
    backend = MockSearchBackend(results_per_query=2)

    with mock_widesearch_agents():
        results = await run_widesearch(
            topic="test topic",
            keyphrases=["keyword1"],
            search_backend=backend,
            max_rounds=2,  # Force max_rounds termination
        )

    assert len(results) > 0

    # Verify all results have metadata
    for result in results:
        assert result.title != "", f"Title empty after max_rounds: {result.url}"
        assert result.snippet is not None


@pytest.mark.asyncio
async def test_checkpoint_metadata_preserved():
    """Test that checkpoint results preserve metadata."""
    from interaction_finder.widesearch import run_widesearch_with_checkpoint
    from interaction_finder.checkpoint import KeywordsStageData, PipelineCheckpoint
    from interaction_finder.resources import ResourcePool

    backend = MockSearchBackend(results_per_query=3)

    # Create input checkpoint with keywords data
    input_checkpoint = PipelineCheckpoint(
        topic="test topic",
        resources=ResourcePool(),
        keywords=KeywordsStageData(
            terms=["keyword1"],
            scores=[0.9],
            total_documents_processed=5,
            rounds_completed=1,
            coverage_assessment="Good coverage of topic with multiple relevant sources and bridging terms",
            resource_urls=[],
        ),
    )

    with mock_widesearch_agents():
        checkpoint = await run_widesearch_with_checkpoint(
            input_checkpoint=input_checkpoint,
            search_backend=backend,
            max_rounds=1,
        )

    # Check results in checkpoint (nested in search stage data)
    assert len(checkpoint.search.results) > 0
    for result in checkpoint.search.results:
        assert result.title != "", f"Empty title in checkpoint: {result.url}"
        assert result.snippet is not None
        assert result.relevance is not None

    # Verify SearchResult objects can be serialized individually
    for result in checkpoint.search.results:
        result_dict = result.model_dump()
        assert result_dict["title"] != "", "Empty title after model_dump"
        assert result_dict["snippet"] is not None
