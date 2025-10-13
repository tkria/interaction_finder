"""
Comprehensive edge case tests for two-stage query generation.

Tests cover:
- Empty input handling (empty keywords, empty content, empty hint terms)
- LLM failure scenarios (timeout, API error, malformed response)
- Score interpretation (YAKE lower=better, RAKE higher=better, TF-IDF no scores)
- Content edge cases (very long, Unicode, missing metadata, failed fetch)
- Configuration edge cases (invalid combinations, missing required fields)
- Boundary conditions (single resource, hundreds of resources, zero top_n)
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from pydantic_ai.result import AgentRunResult

from interaction_finder.search.reverse.query_generator import QueryGenerator
from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
    QueryConstructionContext,
    QueryGenerationError,
    ConfigurationError,
)
from interaction_finder.search.reverse.llm_models import LLMQueryConstructionResponse


# ==============================================================================
# Empty Input Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_empty_keywords_list():
    """Test handling when extractor returns empty keyword list."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        use_hint_fields=False,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    # Mock metadata with no extractable keywords
    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {"12345678": {"title": "!!!", "abstract": "..."}}

        queries = await generator.generate_initial_queries_individual(resources)

        # Should handle gracefully
        assert isinstance(queries, list)


@pytest.mark.asyncio
async def test_empty_hint_terms():
    """Test handling when resource has empty hint_fields."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        use_hint_fields=True,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(
            pmid="12345678",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            hint_fields={},  # Empty
        )
    ]

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {"title": "BRCA1 mutations", "abstract": "Gene analysis"}
        }

        queries = await generator.generate_initial_queries_individual(resources)

        assert len(queries) == 1
        assert isinstance(queries[0], str)


@pytest.mark.asyncio
async def test_empty_resource_content():
    """Test handling when resource content is empty/missing."""
    config = ReverseSearchConfig(
        keyword_extractor="none",  # Relies on content
        query_constructor="llm",
        use_hint_fields=False,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    # Mock metadata with empty content
    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {"12345678": {"title": "", "abstract": ""}}

        # Mock LLM to handle empty content
        mock_response = LLMQueryConstructionResponse(
            query='"default query"', reasoning="No content available"
        )
        mock_result = Mock(spec=AgentRunResult)
        mock_result.output = mock_response
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch.object(
            generator.constructor, "_create_agent", return_value=mock_agent
        ):
            queries = await generator.generate_initial_queries_individual(resources)

            # Should still generate query (LLM handles empty content)
            assert len(queries) == 1


@pytest.mark.asyncio
async def test_empty_resources_list():
    """Test handling when resources list is empty."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    queries = await generator.generate_initial_queries_individual([])

    assert queries == []


# ==============================================================================
# LLM Failure Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_llm_api_timeout():
    """Test handling of LLM API timeout with fallback enabled."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="llm",
        keywords_per_query=5,
        enable_clustering=False,
        query_construction_config={"enable_fallback": True},
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {"title": "BRCA1 mutations", "abstract": "Gene analysis"}
        }

        # Mock LLM to raise timeout
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = TimeoutError("API timeout")

        with patch.object(
            generator.constructor, "_create_agent", return_value=mock_agent
        ):
            queries = await generator.generate_initial_queries_individual(resources)

            # Should fall back to direct constructor
            assert len(queries) == 1
            assert isinstance(queries[0], str)


@pytest.mark.asyncio
async def test_llm_api_error_no_fallback():
    """Test LLM API error with fallback disabled raises QueryGenerationError."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="llm",
        keywords_per_query=5,
        enable_clustering=False,
        query_construction_config={"enable_fallback": False},
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {"title": "BRCA1 mutations", "abstract": "Gene analysis"}
        }

        # Mock LLM to raise error
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("API error")

        with patch.object(
            generator.constructor, "_create_agent", return_value=mock_agent
        ):
            with pytest.raises(QueryGenerationError):
                await generator.generate_initial_queries_individual(resources)


@pytest.mark.asyncio
async def test_llm_malformed_response():
    """Test handling of malformed LLM response."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="llm",
        keywords_per_query=5,
        enable_clustering=False,
        query_construction_config={"enable_fallback": True},
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {"title": "BRCA1 mutations", "abstract": "Gene analysis"}
        }

        # Mock LLM to return malformed data
        mock_result = Mock(spec=AgentRunResult)
        mock_result.output = None  # Malformed
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch.object(
            generator.constructor, "_create_agent", return_value=mock_agent
        ):
            # Should fall back
            queries = await generator.generate_initial_queries_individual(resources)
            assert len(queries) == 1


@pytest.mark.asyncio
async def test_llm_empty_query_response():
    """Test handling when LLM returns minimal query string."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="llm",
        keywords_per_query=5,
        enable_clustering=False,
        query_construction_config={"enable_fallback": True},
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {"title": "BRCA1 mutations", "abstract": "Gene analysis"}
        }

        # Mock LLM to return minimal query (empty string is invalid per Pydantic validation)
        mock_response = LLMQueryConstructionResponse(
            query="*", reasoning="Unable to construct meaningful query"
        )
        mock_result = Mock(spec=AgentRunResult)
        mock_result.output = mock_response
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch.object(
            generator.constructor, "_create_agent", return_value=mock_agent
        ):
            queries = await generator.generate_initial_queries_individual(resources)

            # Should return the minimal query
            assert len(queries) == 1
            assert queries[0] == "*"


# ==============================================================================
# Score Interpretation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_yake_scores_lower_is_better():
    """Test that YAKE scores (lower is better) are passed correctly."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="llm",
        keywords_per_query=5,
        enable_clustering=False,
        query_construction_config={"include_scores_in_prompt": True},
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    captured_contexts = []

    async def capture_construct(context: QueryConstructionContext):
        captured_contexts.append(context)
        return '"test"'

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {
                "title": "BRCA1 breast cancer mutation genetics",
                "abstract": "Study of BRCA1 mutations in breast cancer patients.",
            }
        }

        with patch.object(
            generator.constructor, "construct", side_effect=capture_construct
        ):
            await generator.generate_initial_queries_individual(resources)

            assert len(captured_contexts) == 1
            context = captured_contexts[0]

            # YAKE should provide scores (lower is better)
            if context.keyword_scores:
                # Verify scores are present and numeric
                assert all(isinstance(s, (int, float)) for s in context.keyword_scores)


@pytest.mark.asyncio
async def test_rake_scores_higher_is_better():
    """Test that RAKE scores (higher is better) are passed correctly."""
    config = ReverseSearchConfig(
        keyword_extractor="rake",
        query_constructor="llm",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    captured_contexts = []

    async def capture_construct(context: QueryConstructionContext):
        captured_contexts.append(context)
        return '"test"'

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {
                "title": "BRCA1 breast cancer mutation genetics",
                "abstract": "Study of BRCA1 mutations in breast cancer patients.",
            }
        }

        with patch.object(
            generator.constructor, "construct", side_effect=capture_construct
        ):
            await generator.generate_initial_queries_individual(resources)

            assert len(captured_contexts) == 1
            context = captured_contexts[0]

            # RAKE may provide scores (higher is better)
            # Just verify they're numeric if present
            if context.keyword_scores:
                assert all(isinstance(s, (int, float)) for s in context.keyword_scores)


@pytest.mark.asyncio
async def test_tfidf_no_scores():
    """Test that TF-IDF provides None scores (not score-based)."""
    config = ReverseSearchConfig(
        keyword_extractor="tfidf",
        query_constructor="llm",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    captured_contexts = []

    async def capture_construct(context: QueryConstructionContext):
        captured_contexts.append(context)
        return '"test"'

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {
                "title": "BRCA1 breast cancer mutation genetics",
                "abstract": "Study of BRCA1 mutations in breast cancer patients.",
            }
        }

        with patch.object(
            generator.constructor, "construct", side_effect=capture_construct
        ):
            await generator.generate_initial_queries_individual(resources)

            assert len(captured_contexts) == 1
            context = captured_contexts[0]

            # TF-IDF should have None scores
            assert context.keyword_scores is None


# ==============================================================================
# Content Edge Cases
# ==============================================================================


@pytest.mark.asyncio
async def test_very_long_content():
    """Test handling of very long resource content."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="llm",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    # Create very long content (10KB)
    very_long_abstract = "BRCA1 mutation analysis. " * 500

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {"title": "BRCA1 study", "abstract": very_long_abstract}
        }

        mock_response = LLMQueryConstructionResponse(
            query='"BRCA1"', reasoning="Extracted from long content"
        )
        mock_result = Mock(spec=AgentRunResult)
        mock_result.output = mock_response
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch.object(
            generator.constructor, "_create_agent", return_value=mock_agent
        ):
            queries = await generator.generate_initial_queries_individual(resources)

            # Should handle long content (may truncate in prompt)
            assert len(queries) == 1


@pytest.mark.asyncio
async def test_unicode_content():
    """Test handling of Unicode characters in content."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {
                "title": "α-synuclein in Parkinson's disease",
                "abstract": "神经退行性疾病研究。β-amyloid and γ-secretase pathways.",
            }
        }

        queries = await generator.generate_initial_queries_individual(resources)

        # Should handle Unicode
        assert len(queries) == 1


@pytest.mark.asyncio
async def test_missing_pmid_metadata():
    """Test fallback when PMID metadata fetch returns nothing."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    # Mock PageFetcher for content fallback
    mock_fetcher = AsyncMock()
    mock_doc = Mock()
    mock_doc.content_markdown = "Fallback content from full paper fetch."
    mock_fetcher.fetch_documents = AsyncMock(
        return_value={"https://pubmed.ncbi.nlm.nih.gov/12345678/": mock_doc}
    )
    generator.fetcher = mock_fetcher

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {}  # No metadata

        queries = await generator.generate_initial_queries_individual(resources)

        # Should fall back to content
        assert len(queries) == 1


@pytest.mark.asyncio
async def test_failed_url_fetch():
    """Test handling when both metadata and URL fetch fail."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        use_hint_fields=True,  # Will use hint fields as last resort
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(
            pmid="12345678",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            hint_fields={"disease": "breast cancer", "gene": "BRCA1"},  # Fallback
        )
    ]

    # Mock PageFetcher to return empty
    mock_fetcher = AsyncMock()
    mock_fetcher.fetch_documents = AsyncMock(return_value={})
    generator.fetcher = mock_fetcher

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {}  # No metadata

        queries = await generator.generate_initial_queries_individual(resources)

        # Should fall back to hint fields
        assert len(queries) == 1


# ==============================================================================
# Configuration Edge Cases
# ==============================================================================


def test_invalid_extractor_constructor_combo():
    """Test that none + direct raises ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        ReverseSearchConfig(
            keyword_extractor="none",
            query_constructor="direct",
        )

    error = exc_info.value
    assert "Invalid query generation configuration" in error.message
    assert error.context["keyword_extractor"] == "none"
    assert error.context["query_constructor"] == "direct"


def test_llm_constructor_requires_llm_config():
    """Test that llm constructor requires non-empty llm_query_config."""
    with pytest.raises(ConfigurationError):
        ReverseSearchConfig(
            keyword_extractor="none",
            query_constructor="llm",
            llm_query_config={},  # Empty not allowed
        )


# ==============================================================================
# Boundary Condition Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_single_resource():
    """Test query generation with single resource."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {"title": "BRCA1 mutations", "abstract": "Gene analysis"}
        }

        queries = await generator.generate_initial_queries_individual(resources)

        assert len(queries) == 1


@pytest.mark.asyncio
async def test_many_resources():
    """Test query generation with many resources."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=3,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    # Create 50 resources
    resources = [
        KnownResource(pmid=f"{i:08d}", url=f"https://pubmed.ncbi.nlm.nih.gov/{i:08d}/")
        for i in range(50)
    ]

    metadata = {
        f"{i:08d}": {"title": f"Study {i}", "abstract": f"Research content {i}"}
        for i in range(50)
    }

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = metadata

        queries = await generator.generate_initial_queries_individual(resources)

        assert len(queries) == 50


def test_zero_top_n():
    """Test that keywords_per_query=0 is rejected by config validation."""
    # Pydantic validation should reject keywords_per_query < 3
    with pytest.raises(ValueError) as exc_info:
        ReverseSearchConfig(
            keyword_extractor="yake",
            query_constructor="direct",
            keywords_per_query=0,  # Invalid: below minimum of 3
            enable_clustering=False,
        )

    # Verify it's a validation error for keywords_per_query
    assert "keywords_per_query" in str(exc_info.value).lower()


# ==============================================================================
# Special Character Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_special_characters_in_keywords():
    """Test handling of special characters in keywords."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid="12345678", url="https://pubmed.ncbi.nlm.nih.gov/12345678/")
    ]

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = {
            "12345678": {
                "title": 'Gene "BRCA1" & pathway',
                "abstract": "Study of <gene> expression (p < 0.05).",
            }
        }

        queries = await generator.generate_initial_queries_individual(resources)

        # Should handle special characters (may quote or escape)
        assert len(queries) == 1
        assert isinstance(queries[0], str)


# ==============================================================================
# Concurrent Generation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_concurrent_query_generation():
    """Test that query generation handles concurrent operations correctly."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=5,
        enable_clustering=False,
    )
    generator = QueryGenerator(config, backend_name="pubmed")

    resources = [
        KnownResource(pmid=f"{i:08d}", url=f"https://pubmed.ncbi.nlm.nih.gov/{i:08d}/")
        for i in range(10)
    ]

    metadata = {
        f"{i:08d}": {"title": f"Study {i}", "abstract": f"Research {i}"}
        for i in range(10)
    }

    with patch.object(generator, "_fetch_pmid_metadata_batch") as mock_fetch:
        mock_fetch.return_value = metadata

        # Generate queries (internally may use concurrency)
        queries = await generator.generate_initial_queries_individual(resources)

        assert len(queries) == 10
        # All queries should be valid
        assert all(isinstance(q, str) for q in queries)
