"""Tests for Pydantic-AI agents."""

import pytest
from pydantic_ai.models.test import TestModel

from interaction_finder.keywords.agents import (
    get_document_summarizer_agent,
    get_keyword_evaluator_agent,
    get_query_expander_agent,
    get_reflector_agent,
    get_result_selector_agent,
)
from interaction_finder.keywords.deps import Deps
from interaction_finder.keywords.extractors import RAKEExtractor
from interaction_finder.keywords.models import (
    DocumentSummaryOut,
    KeywordEvaluationOut,
    QueryExpansionOut,
    ReflectionOut,
    ResultSelectionOut,
)
from interaction_finder.keywords.reranker import Reranker


@pytest.fixture
def mock_deps():
    """Create mock Deps for testing."""
    # We don't need real dependencies for these tests
    # TestModel doesn't call external services
    return None


class TestQueryExpanderAgent:
    """Test get_query_expander_agent."""

    @pytest.mark.asyncio
    async def test_agent_returns_query_expansion(self, mock_deps):
        """Test agent returns QueryExpansionOut structure."""
        # Override with TestModel
        test_model = TestModel()
        with get_query_expander_agent.override(model=test_model):
            result = await get_query_expander_agent.run(
                "pulmonary arterial hypertension", deps=mock_deps
            )
            # TestModel generates valid data matching the schema
            assert isinstance(result.output, QueryExpansionOut)
            assert isinstance(result.output.queries, list)
            assert isinstance(result.output.reasoning, str)

    def test_agent_has_correct_output_type(self):
        """Test agent is configured with correct output type."""
        # Agent should have output_type configured
        assert hasattr(get_query_expander_agent, "_output_type")


class TestResultSelectorAgent:
    """Test get_result_selector_agent."""

    @pytest.mark.asyncio
    async def test_agent_returns_result_selection(self, mock_deps):
        """Test agent returns ResultSelectionOut structure."""
        test_model = TestModel()
        with get_result_selector_agent.override(model=test_model):
            # Simulate providing search results context
            result = await get_result_selector_agent.run(
                "Select results about machine learning",
                deps=mock_deps,
            )
            assert isinstance(result.output, ResultSelectionOut)
            assert isinstance(result.output.selected_indices, list)
            assert isinstance(result.output.reasoning, str)


class TestKeywordEvaluatorAgent:
    """Test get_keyword_evaluator_agent."""

    @pytest.mark.asyncio
    async def test_agent_returns_keyword_evaluation(self, mock_deps):
        """Test agent returns KeywordEvaluationOut structure."""
        test_model = TestModel()
        with get_keyword_evaluator_agent.override(model=test_model):
            result = await get_keyword_evaluator_agent.run(
                "Evaluate these keywords: term1, term2, term3",
                deps=mock_deps,
            )
            assert isinstance(result.output, KeywordEvaluationOut)
            assert isinstance(result.output.bridging_terms, list)
            assert isinstance(result.output.reasoning, str)


class TestDocumentSummarizerAgent:
    """Test get_document_summarizer_agent."""

    @pytest.mark.asyncio
    async def test_agent_returns_document_summary(self, mock_deps):
        """Test agent returns DocumentSummaryOut structure."""
        test_model = TestModel()
        with get_document_summarizer_agent.override(model=test_model):
            result = await get_document_summarizer_agent.run(
                "Summarize this document about machine learning",
                deps=mock_deps,
            )
            assert isinstance(result.output, DocumentSummaryOut)
            assert isinstance(result.output.summary, str)
            assert isinstance(result.output.related_areas, list)
            assert isinstance(result.output.bridging_terms, list)
            assert isinstance(result.output.coverage_contribution, str)


class TestReflectorAgent:
    """Test get_reflector_agent."""

    @pytest.mark.asyncio
    async def test_agent_returns_reflection(self, mock_deps):
        """Test agent returns ReflectionOut structure."""
        test_model = TestModel()
        with get_reflector_agent.override(model=test_model):
            result = await get_reflector_agent.run(
                "Review coverage and decide whether to continue",
                deps=mock_deps,
            )
            assert isinstance(result.output, ReflectionOut)
            assert result.output.decision in ["continue", "stop"]
            assert isinstance(result.output.reasoning, str)
            assert isinstance(result.output.new_search_angles, list)


class TestAgentConfiguration:
    """Test agent configuration."""

    def test_all_agents_have_system_prompts(self):
        """Test all agents are configured with system prompts."""
        agents = [
            get_query_expander_agent,
            get_result_selector_agent,
            get_keyword_evaluator_agent,
            get_document_summarizer_agent,
            get_reflector_agent,
        ]
        for agent in agents:
            # Agents should have system prompts configured
            assert agent._system_prompts is not None
            assert len(agent._system_prompts) > 0  # Has prompts

    def test_all_agents_have_deps_type(self):
        """Test all agents are configured with Deps type."""
        agents = [
            get_query_expander_agent,
            get_result_selector_agent,
            get_keyword_evaluator_agent,
            get_document_summarizer_agent,
            get_reflector_agent,
        ]
        for agent in agents:
            assert agent._deps_type == Deps
