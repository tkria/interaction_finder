"""
Tests for query constructor implementations.

Tests both DirectQueryConstructor and LLMQueryConstructor with mocked agents,
verifying all code paths including fallback behavior.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from pydantic_ai.result import AgentRunResult
from rich.console import Console

from interaction_finder.search.reverse.llm_models import LLMQueryConstructionResponse
from interaction_finder.search.reverse.models import (
    QueryConstructionContext,
    QueryGenerationError,
)
from interaction_finder.search.reverse.query_constructors import (
    DirectQueryConstructor,
    LLMQueryConstructor,
    QueryConstructor,
    create_constructor,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def basic_context() -> QueryConstructionContext:
    """Basic query construction context with keywords and hint terms."""
    return QueryConstructionContext(
        keywords=["BRCA1", "breast cancer", "mutation"],
        keyword_scores=[0.05, 0.12, 0.18],
        hint_terms=["mammary epithelial cell"],
        backend="pubmed",
        resource_content=None,
        extractor_used="yake",
    )


@pytest.fixture
def context_with_content() -> QueryConstructionContext:
    """Context with full resource content."""
    return QueryConstructionContext(
        keywords=["FBLN4", "calcification"],
        keyword_scores=[0.03, 0.15],
        hint_terms=["arterial stiffness"],
        backend="pubmed",
        resource_content="FBLN4 mutations cause vascular calcification and arterial stiffness in patients with cutis laxa. The protein plays a critical role in elastic fiber assembly.",
        extractor_used="yake",
    )


@pytest.fixture
def minimal_context() -> QueryConstructionContext:
    """Minimal context with only keywords, no scores or content."""
    return QueryConstructionContext(
        keywords=["gene", "disease"],
        keyword_scores=None,
        hint_terms=[],
        backend="pubmed",
        resource_content=None,
        extractor_used="rake",
    )


@pytest.fixture
def empty_context() -> QueryConstructionContext:
    """Context with no keywords or hint terms."""
    return QueryConstructionContext(
        keywords=[],
        keyword_scores=None,
        hint_terms=[],
        backend="pubmed",
        resource_content=None,
        extractor_used="yake",
    )


@pytest.fixture
def mock_console():
    """Mock Rich console for logging tests."""
    return Mock(spec=Console)


# ==============================================================================
# DirectQueryConstructor Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_direct_constructor_basic(basic_context):
    """Test DirectQueryConstructor with basic context."""
    constructor = DirectQueryConstructor()
    query = await constructor.construct(basic_context)

    # Should concatenate keywords and hint terms
    assert "BRCA1" in query
    assert "breast cancer" in query
    assert "mutation" in query
    assert "mammary epithelial cell" in query


@pytest.mark.asyncio
async def test_direct_constructor_empty_context(empty_context):
    """Test DirectQueryConstructor with empty context."""
    constructor = DirectQueryConstructor()
    query = await constructor.construct(empty_context)

    # Should handle empty keywords gracefully
    assert query == ""


@pytest.mark.asyncio
async def test_direct_constructor_max_keywords():
    """Test DirectQueryConstructor respects max_keywords limit."""
    context = QueryConstructionContext(
        keywords=["k1", "k2", "k3", "k4", "k5"],
        keyword_scores=None,
        hint_terms=[],
        backend="pubmed",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = DirectQueryConstructor(max_keywords=3, include_hints=True)
    query = await constructor.construct(context)

    # Should only include first 3 keywords
    assert "k1" in query
    assert "k2" in query
    assert "k3" in query
    assert "k4" not in query
    assert "k5" not in query


@pytest.mark.asyncio
async def test_direct_constructor_no_hints():
    """Test DirectQueryConstructor excluding hint terms."""
    context = QueryConstructionContext(
        keywords=["BRCA1"],
        keyword_scores=None,
        hint_terms=["TP53", "PTEN"],
        backend="pubmed",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = DirectQueryConstructor(include_hints=False)
    query = await constructor.construct(context)

    # Should not include hint terms
    assert "BRCA1" in query
    assert "TP53" not in query
    assert "PTEN" not in query


def test_direct_constructor_name():
    """Test DirectQueryConstructor name property."""
    constructor = DirectQueryConstructor()
    assert constructor.name == "direct"


# ==============================================================================
# LLMQueryConstructor Tests - Successful Path
# ==============================================================================


@pytest.mark.asyncio
async def test_llm_constructor_successful(basic_context):
    """Test LLMQueryConstructor successful query construction."""
    # Mock agent response
    mock_response = LLMQueryConstructionResponse(
        query='BRCA1[Title/Abstract] AND "breast cancer"',
        reasoning="Combined gene with disease term for precision",
    )
    mock_result = Mock(spec=AgentRunResult)
    mock_result.output = mock_response

    # Create constructor
    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, enable_fallback=False
    )

    # Mock agent.run
    with patch.object(constructor, "_create_agent") as mock_create_agent:
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        query = await constructor.construct(basic_context)

        # Verify query returned
        assert query == 'BRCA1[Title/Abstract] AND "breast cancer"'

        # Verify agent was called
        mock_agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_llm_constructor_with_content(context_with_content):
    """Test LLMQueryConstructor with full resource content."""
    # Mock agent response
    mock_response = LLMQueryConstructionResponse(
        query='FBLN4[Title] AND "vascular calcification"',
        reasoning="Used keyword as anchor, content revealed vascular context",
    )
    mock_result = Mock(spec=AgentRunResult)
    mock_result.output = mock_response

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, enable_fallback=False
    )

    with patch.object(constructor, "_create_agent") as mock_create_agent:
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        query = await constructor.construct(context_with_content)

        assert query == 'FBLN4[Title] AND "vascular calcification"'
        mock_agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_llm_constructor_minimal_context(minimal_context):
    """Test LLMQueryConstructor with minimal context."""
    mock_response = LLMQueryConstructionResponse(
        query="gene AND disease", reasoning="Simple keyword combination"
    )
    mock_result = Mock(spec=AgentRunResult)
    mock_result.output = mock_response

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, enable_fallback=False
    )

    with patch.object(constructor, "_create_agent") as mock_create_agent:
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        query = await constructor.construct(minimal_context)

        assert query == "gene AND disease"


# ==============================================================================
# LLMQueryConstructor Tests - Fallback Behavior
# ==============================================================================


@pytest.mark.asyncio
async def test_llm_constructor_fallback_enabled(basic_context, mock_console):
    """Test LLMQueryConstructor falls back to DirectQueryConstructor on error."""
    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini",
        temperature=0.3,
        enable_fallback=True,
        console=mock_console,
    )

    # Mock agent to raise an error
    with patch.object(constructor, "_create_agent") as mock_create_agent:
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("API rate limit exceeded")
        mock_create_agent.return_value = mock_agent

        query = await constructor.construct(basic_context)

        # Should fall back to DirectQueryConstructor
        assert query  # Should return non-empty query
        assert "BRCA1" in query or "breast cancer" in query

        # Verify console logging
        assert mock_console.print.call_count >= 1


@pytest.mark.asyncio
async def test_llm_constructor_fallback_disabled(basic_context):
    """Test LLMQueryConstructor raises error when fallback disabled."""
    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, enable_fallback=False
    )

    # Mock agent to raise an error
    with patch.object(constructor, "_create_agent") as mock_create_agent:
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("API error")
        mock_create_agent.return_value = mock_agent

        # Should raise QueryGenerationError
        with pytest.raises(QueryGenerationError) as exc_info:
            await constructor.construct(basic_context)

        # Verify error context
        error = exc_info.value
        assert "fallback disabled" in str(error).lower()
        assert error.context["model"] == "openai:gpt-4o-mini"
        assert error.context["extractor"] == "yake"


# ==============================================================================
# LLMQueryConstructor Tests - Lazy Initialization
# ==============================================================================


def test_llm_constructor_lazy_agent_initialization():
    """Test LLMQueryConstructor doesn't create agent in __init__."""
    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, enable_fallback=False
    )

    # Agent should be None initially
    assert constructor._agent is None


@pytest.mark.asyncio
async def test_llm_constructor_agent_created_on_first_use(basic_context):
    """Test LLMQueryConstructor creates agent on first construct() call."""
    mock_response = LLMQueryConstructionResponse(query="test query", reasoning="test")
    mock_result = Mock(spec=AgentRunResult)
    mock_result.output = mock_response

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, enable_fallback=False
    )

    # Agent should be None before first call
    assert constructor._agent is None

    with patch.object(constructor, "_create_agent") as mock_create_agent:
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        await constructor.construct(basic_context)

        # Agent should be created
        mock_create_agent.assert_called_once()
        assert constructor._agent is not None


# ==============================================================================
# LLMQueryConstructor Tests - Prompt Building
# ==============================================================================


def test_llm_build_prompt_with_scores(basic_context):
    """Test LLMQueryConstructor builds prompt with keyword scores."""
    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, include_scores=True
    )

    prompt = constructor._build_prompt(basic_context)

    # Should include keywords with scores
    assert "BRCA1" in prompt
    assert "0.05" in prompt  # First score (formatted)
    assert "yake" in prompt.lower()  # Extractor name


def test_llm_build_prompt_without_scores(basic_context):
    """Test LLMQueryConstructor builds prompt without scores."""
    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, include_scores=False
    )

    prompt = constructor._build_prompt(basic_context)

    # Should include keywords without scores
    assert "BRCA1" in prompt
    assert "score:" not in prompt.lower()


def test_llm_build_prompt_with_content(context_with_content):
    """Test LLMQueryConstructor includes full content in prompt."""
    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, include_scores=True
    )

    prompt = constructor._build_prompt(context_with_content)

    # Should include resource content
    assert "FBLN4 mutations" in prompt
    assert "vascular calcification" in prompt
    assert "FULL RESOURCE CONTENT" in prompt


def test_llm_build_prompt_max_keywords():
    """Test LLMQueryConstructor truncates keywords to max_keywords."""
    context = QueryConstructionContext(
        keywords=[f"keyword{i}" for i in range(20)],
        keyword_scores=[0.1 * i for i in range(20)],
        hint_terms=[],
        backend="pubmed",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, max_keywords=5, include_scores=True
    )

    prompt = constructor._build_prompt(context)

    # Should include only first 5 keywords
    assert "keyword0" in prompt
    assert "keyword4" in prompt
    assert "keyword5" not in prompt
    assert "keyword15" not in prompt


def test_llm_build_prompt_backend_specific(basic_context):
    """Test LLMQueryConstructor includes backend-specific instructions."""
    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=True
    )

    prompt = constructor._build_prompt(basic_context)

    # Should include PubMed field tag instructions
    assert "pubmed" in prompt.lower()
    assert "[Title]" in prompt or "field tags" in prompt.lower()


def test_llm_build_prompt_natural_language():
    """Test LLMQueryConstructor with natural language mode."""
    context = QueryConstructionContext(
        keywords=["BRCA1", "cancer"],
        keyword_scores=None,
        hint_terms=[],
        backend="generic",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=False
    )

    prompt = constructor._build_prompt(context)

    # Should indicate natural language
    assert "natural language" in prompt.lower() or "generic" in prompt.lower()


def test_llm_constructor_name():
    """Test LLMQueryConstructor name property."""
    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, enable_fallback=False
    )
    assert constructor.name == "llm"


# ==============================================================================
# Factory Function Tests
# ==============================================================================


def test_create_constructor_direct():
    """Test factory creates DirectQueryConstructor."""
    constructor = create_constructor("direct", max_keywords=10, include_hints=True)

    assert isinstance(constructor, DirectQueryConstructor)
    assert constructor.name == "direct"


def test_create_constructor_llm():
    """Test factory creates LLMQueryConstructor."""
    constructor = create_constructor(
        "llm", model="openai:gpt-4o-mini", temperature=0.3, enable_fallback=True
    )

    assert isinstance(constructor, LLMQueryConstructor)
    assert constructor.name == "llm"


def test_create_constructor_case_insensitive():
    """Test factory is case-insensitive."""
    constructor_lower = create_constructor("direct")
    constructor_upper = create_constructor("DIRECT")
    constructor_mixed = create_constructor("Direct")

    assert all(
        isinstance(c, DirectQueryConstructor)
        for c in [constructor_lower, constructor_upper, constructor_mixed]
    )


def test_create_constructor_unknown():
    """Test factory raises error for unknown constructor."""
    with pytest.raises(ValueError) as exc_info:
        create_constructor("unknown")

    assert "Unknown constructor name" in str(exc_info.value)
    assert "direct" in str(exc_info.value).lower()
    assert "llm" in str(exc_info.value).lower()


# ==============================================================================
# Abstract Interface Tests
# ==============================================================================


def test_query_constructor_abstract():
    """Test QueryConstructor cannot be instantiated directly."""
    with pytest.raises(TypeError):
        QueryConstructor()


def test_query_constructor_incomplete_subclass():
    """Test incomplete QueryConstructor subclass raises TypeError."""

    class IncompleteConstructor(QueryConstructor):
        # Missing construct() and name implementations
        pass

    with pytest.raises(TypeError):
        IncompleteConstructor()


# ==============================================================================
# Backend Syntax Guidance Tests
# ==============================================================================


def test_llm_prompt_pubmed_syntax():
    """Test LLMQueryConstructor prompt contains PubMed field tag instructions."""
    context = QueryConstructionContext(
        keywords=["BRCA1", "breast cancer"],
        keyword_scores=[0.05, 0.12],
        hint_terms=["hereditary"],
        backend="pubmed",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=True
    )

    prompt = constructor._build_prompt(context)

    # Should contain PubMed-specific instructions
    assert "pubmed" in prompt.lower()
    assert "[Title]" in prompt or "[Abstract]" in prompt or "[Gene]" in prompt
    assert "field tags" in prompt.lower()
    # Should NOT contain negative instructions about PubMed tags
    assert "Do NOT use PubMed" not in prompt


def test_llm_prompt_perplexica_syntax():
    """Test LLMQueryConstructor prompt contains negative instructions for Perplexica."""
    context = QueryConstructionContext(
        keywords=["FBLN4", "calcification"],
        keyword_scores=[0.03, 0.15],
        hint_terms=["arterial stiffness"],
        backend="perplexica",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=True
    )

    prompt = constructor._build_prompt(context)

    # Should contain Perplexica backend name
    assert "perplexica" in prompt.lower()
    # Should contain negative instructions about PubMed tags
    assert "Do NOT use PubMed field tags" in prompt or "Do NOT use field tags" in prompt
    # Should recommend natural language
    assert "natural language" in prompt.lower()


def test_llm_prompt_openai_syntax():
    """Test LLMQueryConstructor prompt contains negative instructions for OpenAI."""
    context = QueryConstructionContext(
        keywords=["TP53", "tumor suppressor"],
        keyword_scores=[0.08, 0.14],
        hint_terms=["p53 pathway"],
        backend="openai",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=True
    )

    prompt = constructor._build_prompt(context)

    # Should contain OpenAI backend name
    assert "openai" in prompt.lower()
    # Should contain negative instructions about field tags
    assert "Do NOT use field tags" in prompt
    # Should allow Boolean operators
    assert "Boolean operators" in prompt or "AND" in prompt or "OR" in prompt


def test_llm_prompt_pmc_syntax():
    """Test LLMQueryConstructor prompt contains negative instructions for PMC."""
    context = QueryConstructionContext(
        keywords=["collagen", "fibrosis"],
        keyword_scores=[0.07, 0.11],
        hint_terms=["extracellular matrix"],
        backend="pmc",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=True
    )

    prompt = constructor._build_prompt(context)

    # Should contain PMC backend name
    assert "pmc" in prompt.lower()
    # Should contain negative instructions about PubMed tags
    assert "Do NOT use PubMed field tags" in prompt
    # Should allow natural language and Boolean operators
    assert "natural language" in prompt.lower() or "Boolean" in prompt


def test_llm_prompt_google_syntax():
    """Test LLMQueryConstructor prompt contains negative instructions for Google."""
    context = QueryConstructionContext(
        keywords=["hypertension", "BMPR2"],
        keyword_scores=[0.09, 0.16],
        hint_terms=["pulmonary arterial"],
        backend="google",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=True
    )

    prompt = constructor._build_prompt(context)

    # Should contain Google backend name
    assert "google" in prompt.lower()
    # Should contain negative instructions about PubMed tags
    assert "Do NOT use PubMed field tags" in prompt or "Do NOT use field tags" in prompt
    # Should allow Boolean operators
    assert "Boolean" in prompt or "natural language" in prompt.lower()


def test_llm_prompt_semantic_scholar_syntax():
    """Test LLMQueryConstructor prompt contains negative instructions for Semantic Scholar."""
    context = QueryConstructionContext(
        keywords=["Marfan", "fibrillin"],
        keyword_scores=[0.06, 0.13],
        hint_terms=["connective tissue"],
        backend="semantic_scholar",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=True
    )

    prompt = constructor._build_prompt(context)

    # Should contain Semantic Scholar backend name
    assert "semantic_scholar" in prompt.lower()
    # Should contain negative instructions about field tags
    assert "Do NOT use field tags" in prompt
    # Should allow natural language and Boolean operators
    assert "natural language" in prompt.lower() or "Boolean" in prompt


def test_llm_prompt_unknown_backend_fallback():
    """Test LLMQueryConstructor uses DEFAULT_SYNTAX_GUIDANCE for unknown backends."""
    context = QueryConstructionContext(
        keywords=["gene", "disease"],
        keyword_scores=[0.10, 0.20],
        hint_terms=["phenotype"],
        backend="unknown_backend",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=True
    )

    prompt = constructor._build_prompt(context)

    # Should contain the unknown backend name
    assert "unknown_backend" in prompt.lower()
    # Should contain default negative instructions about PubMed tags
    assert "Do NOT use PubMed field tags" in prompt
    # Should recommend natural language with Boolean operators
    assert "natural language" in prompt.lower()
    assert "Boolean" in prompt or "AND" in prompt


def test_llm_prompt_backend_specific_false():
    """Test LLMQueryConstructor with backend_specific=False uses natural language mode."""
    context = QueryConstructionContext(
        keywords=["BRCA1", "cancer"],
        keyword_scores=[0.05, 0.15],
        hint_terms=["mutation"],
        backend="pubmed",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=False
    )

    prompt = constructor._build_prompt(context)

    # Should indicate natural language syntax
    assert "natural language syntax" in prompt.lower()
    # Should NOT contain backend-specific field tag instructions
    assert "[Title]" not in prompt
    assert "[Abstract]" not in prompt
    assert "[Gene]" not in prompt
    assert "field tags" not in prompt.lower()


def test_llm_prompt_case_insensitive_backend():
    """Test LLMQueryConstructor handles case-insensitive backend names."""
    # Test with uppercase backend name
    context_upper = QueryConstructionContext(
        keywords=["test"],
        keyword_scores=[0.10],
        hint_terms=[],
        backend="PUBMED",
        resource_content=None,
        extractor_used="yake",
    )

    # Test with mixed case backend name
    context_mixed = QueryConstructionContext(
        keywords=["test"],
        keyword_scores=[0.10],
        hint_terms=[],
        backend="PubMed",
        resource_content=None,
        extractor_used="yake",
    )

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=True
    )

    prompt_upper = constructor._build_prompt(context_upper)
    prompt_mixed = constructor._build_prompt(context_mixed)

    # Both should contain PubMed field tag instructions (case-insensitive lookup)
    assert "[Title]" in prompt_upper or "[Abstract]" in prompt_upper
    assert "[Title]" in prompt_mixed or "[Abstract]" in prompt_mixed


# ==============================================================================
# Integration Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_integration_direct_constructor_full_workflow(basic_context):
    """Test DirectQueryConstructor end-to-end workflow."""
    constructor = create_constructor("direct", max_keywords=10, include_hints=True)

    query = await constructor.construct(basic_context)

    # Verify query contains expected terms
    assert query
    assert "BRCA1" in query
    assert constructor.name == "direct"


@pytest.mark.asyncio
async def test_integration_llm_constructor_with_mock(basic_context):
    """Test LLMQueryConstructor end-to-end with mocked agent."""
    mock_response = LLMQueryConstructionResponse(
        query='BRCA1[Title] AND "breast cancer" AND mutation',
        reasoning="Combined all keywords with PubMed field tags",
    )
    mock_result = Mock(spec=AgentRunResult)
    mock_result.output = mock_response

    constructor = create_constructor(
        "llm", model="openai:gpt-4o-mini", temperature=0.3, enable_fallback=False
    )

    with patch.object(constructor, "_create_agent") as mock_create_agent:
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        query = await constructor.construct(basic_context)

        assert query == 'BRCA1[Title] AND "breast cancer" AND mutation'
        assert constructor.name == "llm"


@pytest.mark.asyncio
async def test_integration_pubmed_backend_maintains_compatibility(basic_context):
    """Test PubMed backend query generation maintains backward compatibility."""
    # Mock agent to return PubMed-style query with field tags
    mock_response = LLMQueryConstructionResponse(
        query='BRCA1[Gene] AND "breast cancer"[MeSH]',
        reasoning="Used PubMed field tags for precision",
    )
    mock_result = Mock(spec=AgentRunResult)
    mock_result.output = mock_response

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=True
    )

    with patch.object(constructor, "_create_agent") as mock_create_agent:
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        query = await constructor.construct(basic_context)

        # Should successfully construct PubMed query with field tags
        assert "[Gene]" in query
        assert "[MeSH]" in query
        assert "BRCA1" in query

        # Verify prompt passed to agent contains PubMed instructions
        call_args = mock_agent.run.call_args
        prompt_arg = call_args[0][0]  # First positional argument
        assert "[Title]" in prompt_arg or "[Abstract]" in prompt_arg


@pytest.mark.asyncio
async def test_integration_perplexica_backend_avoids_pubmed_syntax():
    """Test Perplexica backend receives negative instructions about PubMed syntax."""
    context = QueryConstructionContext(
        keywords=["FBLN4", "calcification"],
        keyword_scores=[0.03, 0.15],
        hint_terms=["arterial stiffness"],
        backend="perplexica",
        resource_content=None,
        extractor_used="yake",
    )

    # Mock agent to return natural language query (no field tags)
    mock_response = LLMQueryConstructionResponse(
        query="FBLN4 vascular calcification arterial stiffness",
        reasoning="Used natural language for Perplexica compatibility",
    )
    mock_result = Mock(spec=AgentRunResult)
    mock_result.output = mock_response

    constructor = LLMQueryConstructor(
        model="openai:gpt-4o-mini", temperature=0.3, backend_specific=True
    )

    with patch.object(constructor, "_create_agent") as mock_create_agent:
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result
        mock_create_agent.return_value = mock_agent

        query = await constructor.construct(context)

        # Should successfully construct natural language query
        assert "FBLN4" in query
        assert "calcification" in query
        # Should NOT contain PubMed field tags
        assert "[" not in query
        assert "]" not in query

        # Verify prompt passed to agent contains negative instructions
        call_args = mock_agent.run.call_args
        prompt_arg = call_args[0][0]  # First positional argument
        assert (
            "Do NOT use PubMed field tags" in prompt_arg
            or "Do NOT use field tags" in prompt_arg
        )
