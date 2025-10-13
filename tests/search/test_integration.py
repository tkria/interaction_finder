"""Integration tests for search functionality."""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from interaction_finder.search import SearchQuery, SearchCache
from interaction_finder.search.backends.pubmed import PubMedBackend
from interaction_finder.search.expansion.llm import LLMQueryExpander
from interaction_finder.search.evaluation import EvaluationRunner, EvaluationConfig
from interaction_finder.search.config import (
    SearchConfig,
    PubMedConfig,
    LLMExpansionConfig,
)


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def search_config():
    """Create search configuration for testing."""
    return SearchConfig(
        enabled_backends=["pubmed"],
        default_backend="pubmed",
        max_results=20,
        pubmed=PubMedConfig(
            email="test@example.com",
            rate_limit=1.0,  # Low rate for testing
        ),
    )


class TestSearchIntegration:
    """Integration tests for complete search workflow."""

    @pytest.mark.asyncio
    async def test_basic_search_workflow(self, search_config):
        """Test basic search workflow without external dependencies."""
        # Mock PubMed backend to avoid external API calls
        with patch(
            "interaction_finder.search.backends.pubmed.PubMedBackend.search"
        ) as mock_search:
            from interaction_finder.search.base import SearchResults, SearchResult

            # Mock response
            mock_search.return_value = SearchResults(
                query="BRCA1 mutations",
                backend="pubmed",
                results=[
                    SearchResult(
                        title="BRCA1 mutations in breast cancer",
                        url="https://pubmed.ncbi.nlm.nih.gov/123456",
                        pmid="123456",
                    ),
                    SearchResult(
                        title="Another BRCA1 study",
                        url="https://pubmed.ncbi.nlm.nih.gov/789012",
                        pmid="789012",
                    ),
                ],
                total_found=100,
                search_time_seconds=1.0,
            )

            # Test search
            backend_config = search_config.get_backend_config("pubmed")
            backend = PubMedBackend(backend_config)

            query = SearchQuery(query="BRCA1 mutations", max_results=20)

            async with backend:
                results = await backend.search(query)

                assert results.query == "BRCA1 mutations"
                assert results.backend == "pubmed"
                assert len(results.results) == 2
                assert results.total_found == 100

    @pytest.mark.asyncio
    async def test_search_with_caching(self, search_config, temp_cache_dir):
        """Test search workflow with caching."""
        from interaction_finder.search.base import SearchResults, SearchResult

        # Create cache
        cache = SearchCache(temp_cache_dir, ttl_hours=1)

        # Mock backend
        with patch(
            "interaction_finder.search.backends.pubmed.PubMedBackend.search"
        ) as mock_search:
            mock_results = SearchResults(
                query="test query",
                backend="pubmed",
                results=[
                    SearchResult(title="Test Paper", url="https://example.com/1"),
                ],
                total_found=10,
            )
            mock_search.return_value = mock_results

            query = SearchQuery(query="test query")
            backend_config = search_config.get_backend_config("pubmed")
            backend = PubMedBackend(backend_config)

            # First search - cache miss
            cached_results = await cache.get(query, "pubmed")
            assert cached_results is None

            async with backend:
                results = await backend.search(query)

            # Cache the results
            await cache.set(query, "pubmed", results)

            # Second search - cache hit
            cached_results = await cache.get(query, "pubmed")
            assert cached_results is not None
            assert cached_results.query == results.query
            assert len(cached_results.results) == len(results.results)

            # Verify backend was called only once
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_expansion(self):
        """Test search workflow with query expansion."""
        # Mock LLM expansion to avoid external API calls
        with patch(
            "interaction_finder.search.expansion.llm.LLMQueryExpander.expand_query"
        ) as mock_expand:
            from interaction_finder.search.expansion.base import (
                ExpandedQuery,
                ExpansionTerm,
            )

            # Mock expansion response
            mock_expand.return_value = ExpandedQuery(
                original_query="BRCA1",
                expanded_terms=[
                    ExpansionTerm(
                        term="breast cancer",
                        confidence=0.9,
                        source="llm",
                        category="related",
                    ),
                    ExpansionTerm(
                        term="BRCA1 gene",
                        confidence=0.8,
                        source="llm",
                        category="synonym",
                    ),
                ],
                expansion_method="llm",
                total_confidence=0.85,
            )

            # Test expansion
            config = LLMExpansionConfig(model_name="openai:gpt-4o-mini")
            expander = LLMQueryExpander(model_name=config.model_name)

            expanded = await expander.expand_query("BRCA1")

            assert expanded.original_query == "BRCA1"
            assert len(expanded.expanded_terms) == 2
            assert expanded.expansion_method == "llm"

            # Verify terms are properly structured
            terms = expanded.expanded_terms
            assert terms[0].term == "breast cancer"
            assert terms[0].confidence == 0.9
            assert terms[1].term == "BRCA1 gene"
            assert terms[1].confidence == 0.8

    @pytest.mark.asyncio
    async def test_evaluation_workflow(self, search_config):
        """Test evaluation workflow."""
        from interaction_finder.search.base import SearchResults, SearchResult

        # Mock backends for evaluation
        with patch(
            "interaction_finder.search.backends.pubmed.PubMedBackend.search"
        ) as mock_search:
            mock_search.return_value = SearchResults(
                query="test query",
                backend="pubmed",
                results=[
                    SearchResult(title="Paper 1", url="https://example.com/1"),
                    SearchResult(title="Paper 2", url="https://example.com/2"),
                ],
                total_found=50,
                search_time_seconds=1.2,
            )

            # Set up evaluation
            eval_config = EvaluationConfig(
                max_results_to_evaluate=10,
                timeout_seconds=30,
            )
            runner = EvaluationRunner(eval_config)

            # Run evaluation
            queries = ["query 1", "query 2"]
            backend_config = search_config.get_backend_config("pubmed")

            results = await runner.run_backend_evaluation(
                queries=queries,
                backend_name="pubmed",
                backend_config=backend_config,
            )

            assert results["backend"] == "pubmed"
            assert len(results["individual_results"]) == 2
            assert "aggregate_metrics" in results

            # Check aggregate metrics
            aggregates = results["aggregate_metrics"]
            assert aggregates["total_queries"] == 2
            assert aggregates["successful_queries"] == 2
            assert aggregates["error_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, search_config, temp_cache_dir):
        """Test concurrent search operations."""
        from interaction_finder.search.base import SearchResults, SearchResult

        # Create cache
        cache = SearchCache(temp_cache_dir, ttl_hours=1)

        # Mock backend
        with patch(
            "interaction_finder.search.backends.pubmed.PubMedBackend.search"
        ) as mock_search:

            def mock_search_func(query):
                return SearchResults(
                    query=query.query,
                    backend="pubmed",
                    results=[
                        SearchResult(
                            title=f"Paper for {query.query}",
                            url=f"https://example.com/{query.query.replace(' ', '_')}",
                        )
                    ],
                    total_found=1,
                    search_time_seconds=0.5,
                )

            mock_search.side_effect = mock_search_func

            backend_config = search_config.get_backend_config("pubmed")
            backend = PubMedBackend(backend_config)

            # Create multiple queries
            queries = [SearchQuery(query=f"query {i}") for i in range(5)]

            # Run concurrent searches
            async with backend:
                tasks = [backend.search(query) for query in queries]
                results = await asyncio.gather(*tasks)

            # Verify results
            assert len(results) == 5
            for i, result in enumerate(results):
                assert result.query == f"query {i}"
                assert len(result.results) == 1

            # Cache all results
            cache_tasks = [
                cache.set(queries[i], "pubmed", results[i]) for i in range(5)
            ]
            await asyncio.gather(*cache_tasks)

            # Verify all cached
            for i, query in enumerate(queries):
                cached = await cache.get(query, "pubmed")
                assert cached is not None
                assert cached.query == f"query {i}"

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, search_config):
        """Test error handling throughout the search pipeline."""
        from interaction_finder.search.base import SearchError

        # Mock backend that raises errors
        with patch(
            "interaction_finder.search.backends.pubmed.PubMedBackend.search"
        ) as mock_search:
            mock_search.side_effect = SearchError("Backend unavailable")

            backend_config = search_config.get_backend_config("pubmed")
            backend = PubMedBackend(backend_config)

            query = SearchQuery(query="test query")

            # Test search error handling
            with pytest.raises(SearchError):
                async with backend:
                    await backend.search(query)

        # Test evaluation with errors
        eval_config = EvaluationConfig(max_results_to_evaluate=5)
        runner = EvaluationRunner(eval_config)

        with patch(
            "interaction_finder.search.backends.pubmed.PubMedBackend.search"
        ) as mock_search:
            mock_search.side_effect = Exception("Network error")

            results = await runner.run_backend_evaluation(
                queries=["failing query"],
                backend_name="pubmed",
                backend_config=backend_config,
            )

            # Should handle errors gracefully
            individual_results = results["individual_results"]
            assert len(individual_results) == 1
            assert len(individual_results[0]["errors"]) > 0
            assert "Network error" in individual_results[0]["errors"][0]

    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation in integration."""
        # Test invalid backend configuration
        invalid_config = SearchConfig(
            enabled_backends=["nonexistent"],
            default_backend="nonexistent",
        )

        with pytest.raises(ValueError):
            invalid_config.get_backend_config("nonexistent")

        # Test backend availability validation
        valid_config = SearchConfig(enabled_backends=["pubmed", "perplexica"])
        availability = valid_config.validate_backend_availability()

        assert "pubmed" in availability
        assert availability["pubmed"] is True  # PubMed is always available
        assert "perplexica" in availability
        # Perplexica availability depends on local installation
