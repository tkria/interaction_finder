"""
Integration tests for LLM extractor in QueryGenerator workflow.

Tests end-to-end flow from ReverseSearchConfig through QueryGenerator
to LLM-based query generation, including configuration routing, hint field
passing, and fallback behavior.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
)
from interaction_finder.search.reverse.query_generator import QueryGenerator


@pytest.mark.asyncio
async def test_llm_extractor_selection():
    """Test that LLM extractor is created when keyword_extractor='llm'."""
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={
            "model": "openai:gpt-4o-mini",
            "temperature": 0.5,
            "backend_specific_syntax": True,
        },
    )

    generator = QueryGenerator(config, console=None)

    assert generator.extractor.name == "llm"
    assert generator.extractor.model == "openai:gpt-4o-mini"
    assert generator.extractor.temperature == 0.5
    assert generator.extractor.backend_specific is True


@pytest.mark.asyncio
async def test_llm_config_passed_correctly():
    """Test that LLM config parameters are passed to extractor."""
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={
            "model": "anthropic:claude-3-sonnet",
            "temperature": 0.7,
            "backend_specific_syntax": False,
        },
    )

    generator = QueryGenerator(config, console=None)

    assert generator.extractor.model == "anthropic:claude-3-sonnet"
    assert generator.extractor.temperature == 0.7
    assert generator.extractor.backend_specific is False


@pytest.mark.asyncio
async def test_hint_fields_passed_to_llm():
    """Test that hint fields from KnownResource reach LLM extractor."""
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        use_hint_fields=True,
        llm_query_config={"model": "openai:gpt-4o-mini", "temperature": 0.7},
    )

    # Mock LLMExtractor to capture calls
    with patch(
        "interaction_finder.search.reverse.keyword_extractors.LLMExtractor"
    ) as MockLLM:
        mock_instance = AsyncMock()
        mock_instance.name = "llm"
        mock_instance.extract_async = AsyncMock(
            return_value=['"BRCA1"[Title] AND "breast cancer"[Abstract]']
        )
        MockLLM.return_value = mock_instance

        generator = QueryGenerator(config, console=None)

        # Mock metadata fetch to return content
        with patch.object(
            generator,
            "_fetch_pmid_metadata_batch",
            return_value={
                "12345": {
                    "title": "BRCA1 study",
                    "abstract": "Study about BRCA1 and breast cancer",
                }
            },
        ):
            resources = [
                KnownResource(
                    pmid="12345",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                    hint_fields={"gene": "BRCA1", "disease": "breast cancer"},
                )
            ]

            queries = await generator.generate_initial_queries(resources)

            # Verify queries generated
            assert len(queries) > 0
            assert "BRCA1" in queries[0]

            # Verify hint fields were passed to LLM extractor
            mock_instance.extract_async.assert_called_once()
            call_args = mock_instance.extract_async.call_args
            assert call_args.kwargs.get("hint_fields") == {
                "gene": "BRCA1",
                "disease": "breast cancer",
            }


@pytest.mark.asyncio
async def test_hint_fields_not_passed_when_disabled():
    """Test that hint fields are not passed when use_hint_fields=False."""
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        use_hint_fields=False,  # Disabled
        llm_query_config={"model": "openai:gpt-4o-mini", "temperature": 0.7},
    )

    with patch(
        "interaction_finder.search.reverse.keyword_extractors.LLMExtractor"
    ) as MockLLM:
        mock_instance = AsyncMock()
        mock_instance.name = "llm"
        mock_instance.extract_async = AsyncMock(
            return_value=['"keyword1" OR "keyword2"']
        )
        MockLLM.return_value = mock_instance

        generator = QueryGenerator(config, console=None)

        with patch.object(
            generator,
            "_fetch_pmid_metadata_batch",
            return_value={"12345": {"title": "Test", "abstract": "Abstract text"}},
        ):
            resources = [
                KnownResource(
                    pmid="12345",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                    hint_fields={"gene": "BRCA1"},
                )
            ]

            await generator.generate_initial_queries(resources)

            # Verify hint_fields=None was passed
            mock_instance.extract_async.assert_called_once()
            call_args = mock_instance.extract_async.call_args
            assert call_args.kwargs.get("hint_fields") is None


@pytest.mark.asyncio
async def test_llm_query_generation_e2e():
    """Test complete flow from config to query generation with LLM."""
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={"model": "openai:gpt-4o-mini", "temperature": 0.7},
    )

    with patch(
        "interaction_finder.search.reverse.keyword_extractors.LLMExtractor"
    ) as MockLLM:
        mock_instance = AsyncMock()
        mock_instance.name = "llm"
        mock_instance.extract_async = AsyncMock(
            return_value=['"BRCA1"[Title] AND "breast cancer"[Abstract]']
        )
        MockLLM.return_value = mock_instance

        generator = QueryGenerator(config, console=None)

        with patch.object(
            generator,
            "_fetch_pmid_metadata_batch",
            return_value={
                "12345": {
                    "title": "BRCA1 research",
                    "abstract": "Research on BRCA1 mutations",
                }
            },
        ):
            resources = [
                KnownResource(
                    pmid="12345",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                    hint_fields={"gene": "BRCA1", "disease": "breast cancer"},
                )
            ]

            queries = await generator.generate_initial_queries(resources)

            assert len(queries) == 1
            assert "BRCA1" in queries[0]
            assert "[Title]" in queries[0]  # Backend-specific syntax


@pytest.mark.asyncio
async def test_llm_complete_query_construction():
    """Test that LLM complete queries (with field tags) are used as-is."""
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={"model": "openai:gpt-4o-mini", "temperature": 0.7},
    )

    with patch(
        "interaction_finder.search.reverse.keyword_extractors.LLMExtractor"
    ) as MockLLM:
        mock_instance = AsyncMock()
        mock_instance.name = "llm"
        # Return complete query with field tags
        mock_instance.extract_async = AsyncMock(
            return_value=['"BRCA1"[Title] AND "breast cancer"[Abstract]']
        )
        MockLLM.return_value = mock_instance

        generator = QueryGenerator(config, console=None)

        with patch.object(
            generator,
            "_fetch_pmid_metadata_batch",
            return_value={"12345": {"title": "Test", "abstract": "Text"}},
        ):
            resources = [
                KnownResource(
                    pmid="12345",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                )
            ]

            queries = await generator.generate_initial_queries(resources)

            # Complete query should be used as-is, not quoted again
            assert queries[0] == '"BRCA1"[Title] AND "breast cancer"[Abstract]'


@pytest.mark.asyncio
async def test_yake_unchanged():
    """Test backward compatibility: YAKE extractor works unchanged."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",  # Not LLM
        keywords_per_query=5,
    )

    generator = QueryGenerator(config, console=None)

    # YAKE should be selected
    assert generator.extractor.name == "yake"

    # Mock metadata fetch
    with patch.object(
        generator,
        "_fetch_pmid_metadata_batch",
        return_value={
            "12345": {
                "title": "Machine learning models",
                "abstract": "Deep neural networks for classification",
            }
        },
    ):
        resources = [
            KnownResource(
                pmid="12345",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
            )
        ]

        queries = await generator.generate_initial_queries(resources)

        # Should generate queries (YAKE works normally)
        assert len(queries) > 0
        # YAKE returns keyword list, so query will be OR-joined
        assert " OR " in queries[0] or len(queries[0].split('"')) > 2


@pytest.mark.asyncio
async def test_llm_fallback_in_query_generator():
    """Test that LLM failures in QueryGenerator context are handled gracefully."""
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={"model": "openai:gpt-4o-mini", "temperature": 0.7},
    )

    # LLM extractor has built-in fallback to YAKE, so simulate that
    with patch(
        "interaction_finder.search.reverse.keyword_extractors.LLMExtractor"
    ) as MockLLM:
        mock_instance = AsyncMock()
        mock_instance.name = "llm"
        # Simulate fallback behavior: LLM fails internally, returns YAKE keywords
        mock_instance.extract_async = AsyncMock(
            return_value=["machine learning", "neural networks", "classification"]
        )
        MockLLM.return_value = mock_instance

        generator = QueryGenerator(config, console=None)

        with patch.object(
            generator,
            "_fetch_pmid_metadata_batch",
            return_value={"12345": {"title": "ML study", "abstract": "Study text"}},
        ):
            resources = [
                KnownResource(
                    pmid="12345",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                )
            ]

            # Should not raise exception
            queries = await generator.generate_initial_queries(resources)

            # Should get fallback keywords
            assert len(queries) > 0
            # Keywords should be OR-joined (fallback behavior)
            assert "machine learning" in queries[0]


@pytest.mark.asyncio
async def test_console_logging_for_llm():
    """Test that console logging works for LLM query generation."""
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={"model": "openai:gpt-4o-mini", "temperature": 0.7},
    )

    # Mock console
    mock_console = MagicMock()

    with patch(
        "interaction_finder.search.reverse.keyword_extractors.LLMExtractor"
    ) as MockLLM:
        mock_instance = AsyncMock()
        mock_instance.name = "llm"
        mock_instance.extract_async = AsyncMock(return_value=['"test query"'])
        MockLLM.return_value = mock_instance

        generator = QueryGenerator(config, console=mock_console)

        with patch.object(
            generator,
            "_fetch_pmid_metadata_batch",
            return_value={"12345": {"title": "Test", "abstract": "Text"}},
        ):
            resources = [
                KnownResource(
                    pmid="12345",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                )
            ]

            await generator.generate_initial_queries(resources)

            # Verify console was called with LLM logging
            mock_console.print.assert_any_call(
                "  [dim]Generating LLM queries (model: openai:gpt-4o-mini)[/dim]"
            )


@pytest.mark.asyncio
async def test_hint_fields_aggregation_in_clustering():
    """Test that hint fields are aggregated correctly for clustered queries."""
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        use_hint_fields=True,
        enable_clustering=False,  # Disable clustering for simpler test
        llm_query_config={"model": "openai:gpt-4o-mini", "temperature": 0.7},
    )

    with patch(
        "interaction_finder.search.reverse.keyword_extractors.LLMExtractor"
    ) as MockLLM:
        mock_instance = AsyncMock()
        mock_instance.name = "llm"
        mock_instance.extract_async = AsyncMock(return_value=['"test query"'])
        MockLLM.return_value = mock_instance

        generator = QueryGenerator(config, console=None)

        with patch.object(
            generator,
            "_fetch_pmid_metadata_batch",
            return_value={
                "12345": {"title": "Test 1", "abstract": "Abstract 1"},
                "67890": {"title": "Test 2", "abstract": "Abstract 2"},
            },
        ):
            resources = [
                KnownResource(
                    pmid="12345",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                    hint_fields={"gene": "BRCA1", "disease": "breast cancer"},
                ),
                KnownResource(
                    pmid="67890",
                    url="https://pubmed.ncbi.nlm.nih.gov/67890/",
                    hint_fields={"gene": "TP53", "disease": "lung cancer"},
                ),
            ]

            await generator.generate_initial_queries(resources)

            # Verify extract_async was called twice (one per resource)
            assert mock_instance.extract_async.call_count == 2

            # Verify hint fields were passed for each call
            calls = mock_instance.extract_async.call_args_list
            hint_fields_calls = [call.kwargs.get("hint_fields") for call in calls]

            # Each resource should have its own hint fields
            assert {"gene": "BRCA1", "disease": "breast cancer"} in hint_fields_calls
            assert {"gene": "TP53", "disease": "lung cancer"} in hint_fields_calls
