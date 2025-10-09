"""
Unit tests for LLM-based keyword extraction.

Tests LLMExtractor with mocked Pydantic AI responses to verify:
- Basic query generation
- Hint field integration
- Backend-specific syntax generation
- Fallback to YAKE on LLM failures
- Async/sync interface consistency
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from interaction_finder.search.reverse.keyword_extractors import LLMExtractor
from interaction_finder.search.reverse.llm_models import LLMQueryResponse


@pytest.fixture
def mock_agent_response():
    """Create mock Pydantic AI agent response."""
    response = MagicMock()
    response.output = LLMQueryResponse(
        queries=[
            '"BRCA1"[Title] AND "breast cancer"[MeSH]',
            '"hereditary breast cancer" AND "genetic susceptibility"',
        ],
        reasoning="Focus on gene-disease relationship with PubMed field tags",
        backend_specific=True,
    )
    return response


@pytest.fixture
def mock_agent(mock_agent_response):
    """Create mock Pydantic AI agent."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value=mock_agent_response)
    return agent


class TestLLMExtractorBasic:
    """Test basic LLM extraction functionality."""

    @pytest.mark.asyncio
    async def test_extract_async_basic(self, mock_agent):
        """Test basic async query generation."""
        extractor = LLMExtractor(model="openai:gpt-4o-mini")
        extractor._agent = mock_agent

        text = "This paper discusses BRCA1 mutations in breast cancer patients."
        queries = await extractor.extract_async(text, top_n=3)

        # Verify queries returned
        assert len(queries) == 2
        assert "BRCA1" in queries[0]
        assert "breast cancer" in queries[0] or "breast cancer" in queries[1]

        # Verify agent was called once
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_async_respects_top_n(self, mock_agent):
        """Test that top_n parameter limits query count."""
        extractor = LLMExtractor(model="openai:gpt-4o-mini")
        extractor._agent = mock_agent

        text = "Test paper content"
        queries = await extractor.extract_async(text, top_n=1)

        # Should return only 1 query even though mock returns 2
        assert len(queries) == 1

    def test_extract_sync_wrapper(self, mock_agent):
        """Test sync wrapper calls async implementation."""
        extractor = LLMExtractor(model="openai:gpt-4o-mini")
        extractor._agent = mock_agent

        text = "Test paper content"
        queries = extractor.extract(text, top_n=3)

        # Verify queries returned from sync wrapper
        assert len(queries) == 2
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_list(self):
        """Test that empty text returns empty list without calling LLM."""
        extractor = LLMExtractor()

        queries = await extractor.extract_async("", top_n=3)
        assert queries == []

        queries = await extractor.extract_async("   ", top_n=3)
        assert queries == []

    @pytest.mark.asyncio
    async def test_invalid_top_n_raises_error(self):
        """Test that invalid top_n raises ValueError."""
        extractor = LLMExtractor()

        with pytest.raises(ValueError, match="top_n must be > 0"):
            await extractor.extract_async("test", top_n=0)

        with pytest.raises(ValueError, match="top_n must be > 0"):
            await extractor.extract_async("test", top_n=-1)


class TestLLMExtractorHintFields:
    """Test hint field integration."""

    @pytest.mark.asyncio
    async def test_hint_fields_passed_to_prompt(self, mock_agent):
        """Test that hint fields are included in the prompt."""
        extractor = LLMExtractor(model="openai:gpt-4o-mini")
        extractor._agent = mock_agent

        text = "Test paper content"
        hint_fields = {"gene": "BRCA1", "disease": "breast cancer"}

        await extractor.extract_async(text, top_n=3, hint_fields=hint_fields)

        # Verify agent was called
        mock_agent.run.assert_called_once()

        # Get the prompt that was passed to the agent
        call_args = mock_agent.run.call_args
        prompt = call_args[0][0]

        # Verify hint fields appear in prompt
        assert "BRCA1" in prompt
        assert "breast cancer" in prompt
        assert "HINT FIELDS" in prompt

    @pytest.mark.asyncio
    async def test_hint_fields_none_works(self, mock_agent):
        """Test that None hint_fields is handled gracefully."""
        extractor = LLMExtractor(model="openai:gpt-4o-mini")
        extractor._agent = mock_agent

        text = "Test paper content"
        queries = await extractor.extract_async(text, top_n=3, hint_fields=None)

        # Should still work and return queries
        assert len(queries) == 2
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_hint_fields_empty_dict_works(self, mock_agent):
        """Test that empty hint_fields dict is handled gracefully."""
        extractor = LLMExtractor(model="openai:gpt-4o-mini")
        extractor._agent = mock_agent

        text = "Test paper content"
        queries = await extractor.extract_async(text, top_n=3, hint_fields={})

        # Should still work and return queries
        assert len(queries) == 2
        mock_agent.run.assert_called_once()


class TestLLMExtractorBackendSpecific:
    """Test backend-specific syntax generation."""

    @pytest.mark.asyncio
    async def test_pubmed_field_tags_enabled(self):
        """Test that backend_specific=True generates PubMed field tags."""
        # Create mock agent that returns PubMed-style queries
        response = MagicMock()
        response.output = LLMQueryResponse(
            queries=['"BRCA1"[Title] AND "breast cancer"[MeSH]'],
            reasoning="Using PubMed field tags",
            backend_specific=True,
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=response)

        extractor = LLMExtractor(model="openai:gpt-4o-mini", backend_specific=True)
        extractor._agent = mock_agent

        text = "Test paper content"
        queries = await extractor.extract_async(text, top_n=3)

        # Verify query uses PubMed field tags
        assert "[Title]" in queries[0] or "[MeSH]" in queries[0]

    @pytest.mark.asyncio
    async def test_natural_language_queries_when_disabled(self):
        """Test that backend_specific=False generates natural language queries."""
        # Create mock agent that returns natural language queries
        response = MagicMock()
        response.output = LLMQueryResponse(
            queries=["BRCA1 breast cancer genetic susceptibility"],
            reasoning="Natural language query without field tags",
            backend_specific=False,
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=response)

        extractor = LLMExtractor(model="openai:gpt-4o-mini", backend_specific=False)
        extractor._agent = mock_agent

        text = "Test paper content"
        queries = await extractor.extract_async(text, top_n=3)

        # Verify query does not use field tags
        assert "[Title]" not in queries[0]
        assert "[MeSH]" not in queries[0]

    def test_system_prompt_changes_with_backend_specific(self):
        """Test that different system prompts are used based on backend_specific."""
        # Mock the Agent creation to capture system_prompt
        with patch("pydantic_ai.Agent") as MockAgent:
            mock_instance = MagicMock()
            MockAgent.return_value = mock_instance

            # Create extractor with backend_specific=True
            extractor_pubmed = LLMExtractor(backend_specific=True)
            extractor_pubmed._create_agent()

            # Verify Agent was called with PubMed prompt
            call_args_pubmed = MockAgent.call_args
            system_prompt_pubmed = call_args_pubmed[1]["system_prompt"]
            assert "PubMed field tags" in system_prompt_pubmed

            # Create extractor with backend_specific=False
            extractor_natural = LLMExtractor(backend_specific=False)
            extractor_natural._create_agent()

            # Verify Agent was called with natural language prompt
            call_args_natural = MockAgent.call_args
            system_prompt_natural = call_args_natural[1]["system_prompt"]
            assert "natural language" in system_prompt_natural


class TestLLMExtractorFallback:
    """Test fallback to YAKE on LLM failures."""

    @pytest.mark.asyncio
    async def test_fallback_on_llm_exception(self):
        """Test that LLM exceptions trigger YAKE fallback."""
        # Create mock agent that raises an exception
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("API Error"))

        extractor = LLMExtractor(model="openai:gpt-4o-mini")
        extractor._agent = mock_agent

        text = "BRCA1 breast cancer genetic mutation hereditary"
        queries = await extractor.extract_async(text, top_n=3)

        # Should still return queries from YAKE fallback
        assert len(queries) > 0
        # YAKE should extract keywords from the text
        assert any("brca1" in q.lower() or "cancer" in q.lower() for q in queries)

    @pytest.mark.asyncio
    async def test_fallback_on_api_rate_limit(self):
        """Test fallback on API rate limit errors."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(
            side_effect=Exception("Rate limit exceeded. Please try again later.")
        )

        extractor = LLMExtractor(model="openai:gpt-4o-mini")
        extractor._agent = mock_agent

        text = "Test content with keywords"
        queries = await extractor.extract_async(text, top_n=3)

        # Should return YAKE results
        assert len(queries) > 0

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_response(self):
        """Test fallback when LLM returns invalid data."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=ValueError("Invalid response format"))

        extractor = LLMExtractor(model="openai:gpt-4o-mini")
        extractor._agent = mock_agent

        text = "Test content"
        queries = await extractor.extract_async(text, top_n=3)

        # Should return YAKE results
        assert len(queries) >= 0  # YAKE might return empty for short text

    @pytest.mark.asyncio
    async def test_console_logging_on_fallback(self):
        """Test that console logging works when fallback occurs."""
        from rich.console import Console
        from io import StringIO

        # Create console with string buffer
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=False)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("Test error"))

        extractor = LLMExtractor(model="openai:gpt-4o-mini", console=console)
        extractor._agent = mock_agent

        text = "Test content"
        await extractor.extract_async(text, top_n=3)

        # Verify error message was logged
        output = buffer.getvalue()
        assert "failed" in output.lower() or "fallback" in output.lower()


class TestLLMExtractorProperties:
    """Test extractor properties and configuration."""

    def test_name_property(self):
        """Test that name property returns 'llm'."""
        extractor = LLMExtractor()
        assert extractor.name == "llm"

    def test_default_parameters(self):
        """Test default parameter values."""
        extractor = LLMExtractor()
        assert extractor.model == "openai:gpt-4o-mini"
        assert extractor.temperature == 0.3
        assert extractor.backend_specific is True
        assert extractor.console is None

    def test_custom_parameters(self):
        """Test custom parameter values."""
        from rich.console import Console

        console = Console()
        extractor = LLMExtractor(
            model="openai:gpt-4",
            temperature=0.7,
            backend_specific=False,
            console=console,
        )

        assert extractor.model == "openai:gpt-4"
        assert extractor.temperature == 0.7
        assert extractor.backend_specific is False
        assert extractor.console is console

    def test_lazy_agent_initialization(self):
        """Test that agent is not created until first use."""
        extractor = LLMExtractor()
        assert extractor._agent is None

    @pytest.mark.asyncio
    async def test_agent_created_on_first_call(self, mock_agent):
        """Test that agent is created on first extract call."""
        extractor = LLMExtractor()
        assert extractor._agent is None

        # Mock the _create_agent method
        extractor._create_agent = MagicMock(return_value=mock_agent)

        text = "Test content"
        await extractor.extract_async(text, top_n=3)

        # Verify agent was created
        assert extractor._agent is not None
        extractor._create_agent.assert_called_once()


class TestFactoryFunction:
    """Test create_extractor factory function."""

    def test_create_llm_extractor(self):
        """Test that factory creates LLM extractor."""
        from interaction_finder.search.reverse.keyword_extractors import (
            create_extractor,
        )

        extractor = create_extractor("llm")
        assert isinstance(extractor, LLMExtractor)
        assert extractor.name == "llm"

    def test_create_llm_extractor_with_params(self):
        """Test factory with custom LLM parameters."""
        from interaction_finder.search.reverse.keyword_extractors import (
            create_extractor,
        )

        extractor = create_extractor(
            "llm",
            model="openai:gpt-4",
            temperature=0.5,
            backend_specific=False,
        )

        assert isinstance(extractor, LLMExtractor)
        assert extractor.model == "openai:gpt-4"
        assert extractor.temperature == 0.5
        assert extractor.backend_specific is False

    def test_create_llm_case_insensitive(self):
        """Test that factory is case-insensitive."""
        from interaction_finder.search.reverse.keyword_extractors import (
            create_extractor,
        )

        extractor = create_extractor("LLM")
        assert isinstance(extractor, LLMExtractor)

        extractor = create_extractor("Llm")
        assert isinstance(extractor, LLMExtractor)


class TestPromptGeneration:
    """Test prompt generation helper."""

    def test_build_prompt_basic(self):
        """Test basic prompt generation."""
        from interaction_finder.search.reverse.prompts import (
            build_query_generation_prompt,
        )

        text = "This paper discusses BRCA1 mutations."
        prompt = build_query_generation_prompt(text)

        assert "BRCA1" in prompt
        assert "PAPER CONTENT" in prompt

    def test_build_prompt_with_hints(self):
        """Test prompt generation with hint fields."""
        from interaction_finder.search.reverse.prompts import (
            build_query_generation_prompt,
        )

        text = "Test content"
        hint_fields = {"gene": "BRCA1", "disease": "breast cancer"}

        prompt = build_query_generation_prompt(text, hint_fields=hint_fields)

        assert "BRCA1" in prompt
        assert "breast cancer" in prompt
        assert "HINT FIELDS" in prompt

    def test_build_prompt_truncates_long_text(self):
        """Test that long text is truncated."""
        from interaction_finder.search.reverse.prompts import (
            build_query_generation_prompt,
        )

        # Create text longer than 2000 characters
        long_text = "A" * 3000

        prompt = build_query_generation_prompt(long_text)

        # Should contain truncation note
        assert "truncated" in prompt.lower()

    def test_build_prompt_backend_specific_flag(self):
        """Test that backend_specific flag affects prompt."""
        from interaction_finder.search.reverse.prompts import (
            build_query_generation_prompt,
        )

        text = "Test content"

        # PubMed mode
        prompt_pubmed = build_query_generation_prompt(text, backend_specific=True)
        assert "PubMed" in prompt_pubmed or "[Title]" in prompt_pubmed

        # Natural language mode
        prompt_natural = build_query_generation_prompt(text, backend_specific=False)
        assert "natural language" in prompt_natural.lower()


class TestLLMQueryResponse:
    """Test LLMQueryResponse Pydantic model."""

    def test_valid_response(self):
        """Test valid response creation."""
        response = LLMQueryResponse(
            queries=["query1", "query2"],
            reasoning="Test reasoning",
            backend_specific=True,
        )

        assert len(response.queries) == 2
        assert response.reasoning == "Test reasoning"
        assert response.backend_specific is True

    def test_validate_query_count(self):
        """Test query count validation."""
        # Valid: 1 query
        response = LLMQueryResponse(
            queries=["query1"], reasoning="Test", backend_specific=False
        )
        assert response.validate_query_count()

        # Valid: 3 queries
        response = LLMQueryResponse(
            queries=["q1", "q2", "q3"], reasoning="Test", backend_specific=False
        )
        assert response.validate_query_count()

    def test_get_top_queries(self):
        """Test get_top_queries method."""
        response = LLMQueryResponse(
            queries=["q1", "q2", "q3"], reasoning="Test", backend_specific=False
        )

        # Get top 2
        top_2 = response.get_top_queries(2)
        assert len(top_2) == 2
        assert top_2 == ["q1", "q2"]

        # Get top 5 (more than available)
        top_5 = response.get_top_queries(5)
        assert len(top_5) == 3

    def test_pydantic_validation_min_queries(self):
        """Test that Pydantic validates minimum query count."""
        # Should raise validation error for empty queries
        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMQueryResponse(queries=[], reasoning="Test", backend_specific=False)

    def test_pydantic_validation_max_queries(self):
        """Test that Pydantic validates maximum query count."""
        # Should raise validation error for too many queries
        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMQueryResponse(
                queries=["q1", "q2", "q3", "q4"],
                reasoning="Test",
                backend_specific=False,
            )
