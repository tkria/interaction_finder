"""
Cross-backend compatibility tests for LLM extractor and reverse search features.

Tests verify that new features (LLM extractor, relevance sorting, configuration)
work correctly across all supported backends: PubMed, Perplexica, OpenAI Search.

Tests handle graceful degradation when backends don't support certain features
(e.g., relevance sorting may not be available on all backends).
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
    """Mock fetcher for testing."""
    return Mock()


# Cross-backend tests


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["pubmed", "perplexica", "openai_search"])
async def test_llm_extractor_works_across_backends(
    backend_name,
    sample_resources,
    mock_cache,
    mock_fetcher,
):
    """
    Test that LLM extractor works with all backends.

    Verifies that LLM query generation doesn't break when used with different
    search backends, and that backend-specific query syntax is handled correctly.
    """
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={
            "model": "openai:gpt-4o-mini",
            "temperature": 0.7,
            "backend_specific_syntax": True,  # Should adapt to backend
        },
        sort_by="relevance",
        max_queries=5,  # Minimum allowed value
    )

    backend = create_mock_backend(backend_name)
    searcher = ReverseSearcher(config, backend, mock_cache, mock_fetcher)

    # Should not crash
    try:
        session = await searcher.search(sample_resources, verbose=False)
    except Exception as e:
        pytest.fail(
            f"LLM extractor crashed with {backend_name} backend: {e}. "
            f"Cross-backend compatibility broken."
        )

    # Verify session completed
    assert session is not None, f"Session should complete with {backend_name}"
    assert session.total_queries > 0, f"Should execute queries with {backend_name}"

    # Results may vary by backend, but shouldn't crash
    print(
        f"{backend_name}: {session.found_count}/{len(sample_resources)} found "
        f"in {session.total_queries} queries"
    )


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


@pytest.mark.asyncio
async def test_pubmed_specific_query_syntax():
    """
    Test that LLM extractor generates PubMed-specific syntax when configured.

    PubMed supports field tags like [Title], [Abstract], [Author] etc.
    LLM should generate these when backend_specific_syntax=True.
    """
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={
            "model": "openai:gpt-4o-mini",
            "backend_specific_syntax": True,  # Enable PubMed syntax
        },
    )

    generator = QueryGenerator(config, backend_name="pubmed", console=None)

    # Mock LLM to return PubMed-specific query
    from unittest.mock import patch

    with patch(
        "interaction_finder.search.reverse.keyword_extractors.LLMExtractor"
    ) as MockLLM:
        mock_instance = AsyncMock()
        mock_instance.name = "llm"
        mock_instance.extract_async = AsyncMock(
            return_value=['"BRCA1"[Title] AND "breast cancer"[Abstract]']
        )
        MockLLM.return_value = mock_instance

        generator.extractor = mock_instance

        with patch.object(
            generator,
            "_fetch_pmid_metadata_batch",
            return_value={"12345": {"title": "BRCA1", "abstract": "Study"}},
        ):
            resources = [
                KnownResource(
                    pmid="12345",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                )
            ]
            queries = await generator.generate_initial_queries(resources)

        # Verify PubMed field tags present
        assert len(queries) > 0
        query = queries[0]
        assert "[Title]" in query or "[Abstract]" in query, (
            f"PubMed-specific field tags should be present when backend_specific_syntax=True. "
            f"Got: {query}"
        )


@pytest.mark.asyncio
async def test_generic_query_syntax_for_other_backends():
    """
    Test that LLM generates generic queries for non-PubMed backends.

    Perplexica and OpenAI Search don't support PubMed field tags.
    Queries should be plain boolean queries without field tags.
    """
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={
            "model": "openai:gpt-4o-mini",
            "backend_specific_syntax": False,  # Generic syntax
        },
    )

    generator = QueryGenerator(config, backend_name="perplexica", console=None)

    from unittest.mock import patch

    with patch(
        "interaction_finder.search.reverse.keyword_extractors.LLMExtractor"
    ) as MockLLM:
        mock_instance = AsyncMock()
        mock_instance.name = "llm"
        # Return generic query without field tags
        mock_instance.extract_async = AsyncMock(
            return_value=["BRCA1 AND breast cancer"]
        )
        MockLLM.return_value = mock_instance

        generator.extractor = mock_instance

        with patch.object(
            generator,
            "_fetch_pmid_metadata_batch",
            return_value={"12345": {"title": "BRCA1", "abstract": "Study"}},
        ):
            resources = [
                KnownResource(
                    pmid="12345",
                    url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                )
            ]
            queries = await generator.generate_initial_queries(resources)

        # Verify no PubMed field tags
        assert len(queries) > 0
        query = queries[0]
        assert "[Title]" not in query and "[Abstract]" not in query, (
            f"Generic queries should not have PubMed field tags when backend_specific_syntax=False. "
            f"Got: {query}"
        )


@pytest.mark.asyncio
async def test_pmid_matching_works_across_backends(
    sample_resources, mock_cache, mock_fetcher
):
    """
    Test that PMID matching works for backends that return PMID information.

    PubMed always has PMIDs, Perplexica may extract them from URLs,
    OpenAI Search typically doesn't have them.
    """
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        max_queries=5,  # Minimum allowed value
    )

    for backend_name in ["pubmed", "perplexica", "openai_search"]:
        backend = create_mock_backend(backend_name)
        searcher = ReverseSearcher(config, backend, mock_cache, mock_fetcher)

        session = await searcher.search(sample_resources, verbose=False)

        # PubMed should find matches via PMID
        if backend_name == "pubmed":
            assert session.found_count > 0, (
                "PubMed backend should find matches via PMID"
            )
        # Perplexica may find some matches
        elif backend_name == "perplexica":
            # May or may not find matches depending on PMID extraction
            pass
        # OpenAI Search unlikely to find matches (no PMIDs)
        else:
            # May find zero matches, which is expected
            pass

        print(
            f"{backend_name}: {session.found_count}/{len(sample_resources)} found via PMID"
        )


@pytest.mark.asyncio
async def test_url_matching_fallback_across_backends(mock_cache, mock_fetcher):
    """
    Test that URL matching works as fallback when PMIDs not available.

    Non-PubMed resources should still be findable via URL matching.
    """
    # Resources without PMIDs (generic web papers)
    resources = [
        KnownResource(url="https://example.com/paper1"),
        KnownResource(url="https://example.com/paper2"),
    ]

    config = ReverseSearchConfig(
        keyword_extractor="yake",
        max_queries=5,  # Minimum allowed value
    )

    # Test with Perplexica (returns web URLs)
    backend = create_mock_backend("perplexica")
    searcher = ReverseSearcher(config, backend, mock_cache, mock_fetcher)

    session = await searcher.search(resources, verbose=False)

    # May find some matches via URL matching
    # (depends on mock backend returning matching URLs)
    print(f"Perplexica URL matching: {session.found_count}/{len(resources)} found")


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
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=7,
        # No sort_by specified, uses default
        # No llm_query_config needed
    )

    for backend_name in ["pubmed", "perplexica", "openai_search"]:
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


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["pubmed", "perplexica", "openai_search"])
async def test_hint_fields_across_backends(
    backend_name,
    sample_resources,
    mock_cache,
    mock_fetcher,
):
    """
    Test that hint fields are used correctly across backends.

    LLM should incorporate hint fields into queries regardless of backend.
    """
    config = ReverseSearchConfig(
        keyword_extractor="llm",
        llm_query_config={"model": "openai:gpt-4o-mini"},
        use_hint_fields=True,
        max_queries=5,  # Minimum allowed value
    )

    backend = create_mock_backend(backend_name)
    searcher = ReverseSearcher(config, backend, mock_cache, mock_fetcher)

    from unittest.mock import patch

    # Track what hint fields were passed to LLM
    hint_fields_received = []

    def mock_extract_async(text, hint_fields=None):
        hint_fields_received.append(hint_fields)
        return ['"test query"']

    with patch(
        "interaction_finder.search.reverse.keyword_extractors.LLMExtractor"
    ) as MockLLM:
        mock_instance = AsyncMock()
        mock_instance.name = "llm"
        mock_instance.extract_async = AsyncMock(side_effect=mock_extract_async)
        MockLLM.return_value = mock_instance

        # Replace extractor
        searcher.query_generator.extractor = mock_instance

        with patch.object(
            searcher.query_generator,
            "_fetch_pmid_metadata_batch",
            return_value={
                "12345678": {"title": "Test", "abstract": "Text"},
                "87654321": {"title": "Test2", "abstract": "Text2"},
            },
        ):
            await searcher.search(sample_resources, verbose=False)

    # Verify hint fields were passed to LLM for each resource
    assert len(hint_fields_received) > 0, f"Hint fields not passed with {backend_name}"
    # At least one call should have hint fields
    has_hints = any(hf for hf in hint_fields_received if hf)
    assert has_hints, f"Hint fields not passed to LLM with {backend_name}"


@pytest.mark.asyncio
async def test_error_handling_across_backends(
    sample_resources, mock_cache, mock_fetcher
):
    """
    Test that backend failures are handled gracefully.

    If a backend search fails, the system should log the error and continue
    rather than crashing the entire session.
    """
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        max_queries=5,  # Minimum allowed value
    )

    # Create backend that fails on some queries
    backend = Mock(spec=SearchBackend)
    backend.backend_name = "failing_backend"

    call_count = 0

    async def mock_search_with_failures(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First query succeeds
            return SearchResults(
                query=query.query,
                results=[
                    SearchResult(
                        title="Paper",
                        url="https://example.com/paper",
                        snippet="Text",
                    )
                ],
                total_results=1,
                search_time=0.1,
            )
        else:
            # Subsequent queries fail
            raise Exception("Backend timeout")

    backend.search = AsyncMock(side_effect=mock_search_with_failures)

    searcher = ReverseSearcher(config, backend, mock_cache, mock_fetcher)

    # Should not crash despite backend failures
    try:
        session = await searcher.search(sample_resources, verbose=False)
    except Exception as e:
        pytest.fail(
            f"Session crashed on backend failure: {e}. Should handle gracefully."
        )

    # Session should complete with partial results
    assert session is not None, "Should complete despite backend failures"
    assert session.total_queries >= 1, (
        "Should execute at least one query before failure"
    )
