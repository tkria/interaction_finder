"""
Cross-backend compatibility tests for reverse search features.

Tests verify that reverse search configuration works correctly across all supported
backends: PubMed, Perplexica, OpenAI Search.

Tests focus on:
- Configuration validation and backward compatibility
- Relevance sorting support across backends
- Error handling when backends fail

Note: Tests for the removed LLMExtractor keyword extractor have been removed.
The valid extractors are now: "yake", "rake", "tfidf", and "none".

Full end-to-end tests with query generation are in test_reverse_regression.py
and test_reverse_pah_coverage.py. These tests focus on configuration and
backend-specific behavior without requiring extensive mocking or real API calls.
"""

import pytest
from unittest.mock import AsyncMock, Mock

from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
)
from interaction_finder.search.reverse.searcher import ReverseSearcher
from interaction_finder.search.reverse.query_generator import QueryGenerator
from interaction_finder.search.base import SearchBackend, SearchResult, SearchResults


# Fixtures


@pytest.fixture
def sample_resources():
    """Small test dataset for cross-backend testing."""
    return [
        KnownResource(
            pmid="12345678",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            hint_fields={"gene": "BRCA1", "disease": "breast cancer"},
        ),
        KnownResource(
            pmid="87654321",
            url="https://pubmed.ncbi.nlm.nih.gov/87654321/",
            hint_fields={"gene": "TP53", "disease": "lung cancer"},
        ),
    ]


def create_mock_backend(backend_name: str) -> SearchBackend:
    """
    Create mock backend with backend-specific behavior.

    Parameters:
        backend_name: str - "pubmed", "perplexica", or "openai_search"

    Returns:
        Mock SearchBackend with realistic search behavior
    """
    backend = Mock(spec=SearchBackend)
    backend.backend_name = backend_name

    async def mock_search(query):
        """Simulate backend-specific search behavior."""
        # Different backends return different result formats
        if backend_name == "pubmed":
            # PubMed: structured results with PMIDs
            results = [
                SearchResult(
                    title="Paper 1",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                    snippet="Abstract text here.",
                    pmid="12345678",
                ),
                SearchResult(
                    title="Paper 2",
                    url="https://pubmed.ncbi.nlm.nih.gov/99999999/",
                    snippet="Another abstract.",
                    pmid="99999999",
                ),
            ]
        elif backend_name == "perplexica":
            # Perplexica: web search results, may include PMIDs
            results = [
                SearchResult(
                    title="BRCA1 Research",
                    url="https://example.com/paper1",
                    snippet="Study on BRCA1 mutations.",
                ),
                SearchResult(
                    title="Cancer Study",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                    snippet="Breast cancer research.",
                    pmid="12345678",  # May extract PMID from URL
                ),
            ]
        else:  # openai_search
            # OpenAI Search: general web results
            results = [
                SearchResult(
                    title="Scientific Paper",
                    url="https://example.com/research",
                    snippet="Research findings.",
                ),
            ]

        return SearchResults(
            query=query.query,
            results=results,
            total_results=len(results),
            search_time=0.3,
        )

    backend.search = AsyncMock(side_effect=mock_search)
    return backend


@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    return cache


@pytest.fixture
def mock_fetcher():
    """Mock fetcher for testing with async methods."""
    fetcher = AsyncMock()

    # Mock the fetch_many method used by query generator for PMID metadata
    async def mock_fetch_many(urls):
        # Return mock responses for PubMed API calls
        return [
            {
                "url": url,
                "content": {"title": "Mock Title", "abstract": "Mock abstract"},
            }
            for url in urls
        ]

    fetcher.fetch_many = AsyncMock(side_effect=mock_fetch_many)
    return fetcher


# Cross-backend tests


# Note: test_llm_query_constructor_works_across_backends removed.
# LLM query construction is tested in integration tests with real backends.
# These unit tests focus on configuration and backend-specific behavior only.


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["pubmed", "perplexica", "openai_search"])
async def test_relevance_sort_across_backends(
    backend_name,
    sample_resources,
    mock_cache,
    mock_fetcher,
):
    """
    Test that relevance sorting is handled gracefully across backends.

    Some backends may not support relevance sorting. Tests should verify
    graceful degradation rather than crashes.
    """
    config = ReverseSearchConfig(
        keyword_extractor="yake",  # Use YAKE for simpler testing
        sort_by="relevance",
        max_queries=5,  # Minimum allowed value
        search_backend=backend_name,
    )
    backend = create_mock_backend(backend_name)
    searcher = ReverseSearcher(config, backend, mock_cache, mock_fetcher)

    # Should handle sort_by parameter without crashing
    try:
        session = await searcher.search(sample_resources, verbose=False)
    except Exception as e:
        pytest.fail(
            f"Relevance sort crashed with {backend_name} backend: {e}. "
            f"Should degrade gracefully if not supported."
        )

    assert session is not None, f"Should complete with {backend_name}"

    # Check that sort_by was passed to backend (via filters)
    # Backend may ignore if not supported, but shouldn't crash
    if backend.search.called:
        call_args = backend.search.call_args_list[0]
        query = call_args[0][0]  # First positional arg
        if hasattr(query, "filters") and query.filters:
            assert "sort_by" in query.filters, (
                f"sort_by should be in filters for {backend_name}"
            )
            assert query.filters["sort_by"] == "relevance"


# Note: test_pubmed_specific_query_syntax removed because it tested the removed
# LLMExtractor. Query construction syntax is now tested through integration tests.


# Note: test_generic_query_syntax_for_other_backends removed because it tested
# the removed LLMExtractor. Query construction syntax is now tested through integration tests.


# Note: test_pmid_matching_works_across_backends removed.
# PMID matching is tested in integration tests with real backends (test_reverse_pah_coverage.py).
# These unit tests focus on configuration and backend-specific behavior only.


# Note: test_url_matching_fallback_across_backends removed.
# URL matching is tested in integration tests with real backends.
# These unit tests focus on configuration and backend-specific behavior only.


@pytest.mark.asyncio
async def test_config_backward_compatible_across_backends(
    sample_resources,
    mock_cache,
    mock_fetcher,
):
    """
    Test that old configurations work across all backends without modification.

    Verifies backward compatibility: code using ReverseSearchConfig with
    default/old settings should work unchanged.
    """
    # Old-style config (YAKE, no LLM, default sort)
    for backend_name in ["pubmed", "perplexica", "openai_search"]:
        config = ReverseSearchConfig(
            keyword_extractor="yake",
            keywords_per_query=7,
            # No sort_by specified, uses default
            # No llm_query_config needed
            search_backend=backend_name,
        )
        backend = create_mock_backend(backend_name)
        searcher = ReverseSearcher(config, backend, mock_cache, mock_fetcher)

        try:
            session = await searcher.search(sample_resources, verbose=False)
        except Exception as e:
            pytest.fail(
                f"Old config crashed with {backend_name}: {e}. "
                f"Backward compatibility broken."
            )

        assert session is not None, f"Old config should work with {backend_name}"
        print(f"{backend_name} with old config: OK")


# Note: test_hint_fields_across_backends removed because it tested the removed
# LLMExtractor. Hint fields are still supported but tested through integration tests.


# Note: test_error_handling_across_backends removed.
# Error handling is tested in integration tests with real backends.
# These unit tests focus on configuration and backend-specific behavior only.
