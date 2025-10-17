"""Tests for search base classes and models."""

import pytest
from datetime import datetime, date
from pydantic import ValidationError

from interaction_finder.search.base import (
    SearchQuery,
    SearchResult,
    SearchResults,
    SearchError,
    SearchTimeoutError,
    SearchRateLimitError,
    SearchUnavailableError,
)


class TestSearchQuery:
    """Test SearchQuery model."""

    def test_basic_query(self):
        """Test basic query creation."""
        query = SearchQuery(query="test query")
        assert query.query == "test query"
        assert query.max_results == 100  # default
        assert query.expanded_terms == []

    def test_query_with_options(self):
        """Test query with all options."""
        query = SearchQuery(
            query="BRCA1 mutations",
            max_results=50,
            expanded_terms=["BRCA1", "breast cancer"],
        )
        assert query.query == "BRCA1 mutations"
        assert query.max_results == 50
        assert query.expanded_terms == ["BRCA1", "breast cancer"]

    def test_invalid_max_results(self):
        """Test validation of max_results."""
        with pytest.raises(ValidationError):
            SearchQuery(query="test", max_results=0)

        with pytest.raises(ValidationError):
            SearchQuery(query="test", max_results=2000)  # Too high

    def test_empty_query(self):
        """Test empty query is allowed but creates valid query object."""
        query = SearchQuery(query="")
        assert query.query == ""  # Empty queries are actually allowed


class TestSearchResult:
    """Test SearchResult model."""

    def test_minimal_result(self):
        """Test minimal result creation."""
        result = SearchResult(
            title="Test Paper", url="https://example.com", backend="pubmed"
        )
        assert result.title == "Test Paper"
        assert result.url == "https://example.com"
        assert result.backend == "pubmed"
        assert result.metadata == {}

    def test_full_result(self):
        """Test result with all fields."""
        result = SearchResult(
            title="BRCA1 mutations in breast cancer",
            url="https://pubmed.ncbi.nlm.nih.gov/123456",
            relevance_score=0.95,
            backend="pubmed",
            metadata={
                "authors": ["Smith, J.", "Doe, J."],
                "abstract": "This study investigates BRCA1 mutations...",
                "journal": "Nature Genetics",
                "publication_date": datetime(2023, 1, 15),
                "doi": "10.1038/s41588-023-01234-5",
                "pmid": "123456",
                "source": "pubmed",
            },
        )

        assert result.title == "BRCA1 mutations in breast cancer"
        assert result.metadata["authors"] == ["Smith, J.", "Doe, J."]
        assert "BRCA1 mutations" in result.metadata["abstract"]
        assert result.metadata["journal"] == "Nature Genetics"
        assert result.metadata["publication_date"] == datetime(2023, 1, 15)
        assert result.metadata["pmid"] == "123456"
        assert result.relevance_score == 0.95

    def test_invalid_relevance_score(self):
        """Test relevance score validation."""
        with pytest.raises(ValidationError):
            SearchResult(
                title="Test",
                url="https://example.com",
                backend="pubmed",
                relevance_score=1.5,
            )

        with pytest.raises(ValidationError):
            SearchResult(
                title="Test",
                url="https://example.com",
                backend="pubmed",
                relevance_score=-0.1,
            )


class TestSearchResults:
    """Test SearchResults collection."""

    def test_empty_results(self):
        """Test empty results collection."""
        query = SearchQuery(query="test query")
        results = SearchResults(
            query=query, backend="pubmed", results=[], total_found=0
        )
        assert results.query.query == "test query"
        assert results.backend == "pubmed"
        assert len(results.results) == 0
        assert results.total_found == 0

    def test_results_with_data(self):
        """Test results with actual data."""
        search_results = [
            SearchResult(
                title="Paper 1", url="https://example.com/1", backend="pubmed"
            ),
            SearchResult(
                title="Paper 2", url="https://example.com/2", backend="pubmed"
            ),
        ]
        query = SearchQuery(query="BRCA1")

        results = SearchResults(
            query=query,
            backend="pubmed",
            results=search_results,
            total_found=250,
            search_time=1.23,
        )

        assert results.query.query == "BRCA1"
        assert results.backend == "pubmed"
        assert len(results.results) == 2
        assert results.total_found == 250
        assert results.search_time == 1.23

    def test_results_timestamp(self):
        """Test that timestamp is automatically set."""
        query = SearchQuery(query="test")
        results = SearchResults(
            query=query, backend="pubmed", results=[], total_found=0
        )

        assert results.timestamp is not None
        # Should be very recent
        time_diff = datetime.now() - results.timestamp
        assert time_diff.total_seconds() < 1.0


class TestSearchErrors:
    """Test search error types."""

    def test_search_error(self):
        """Test basic SearchError."""
        error = SearchError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert isinstance(error, Exception)

    def test_timeout_error(self):
        """Test SearchTimeoutError."""
        error = SearchTimeoutError("Request timed out", backend="pubmed", query="test")
        assert "timed out" in str(error)
        assert error.backend == "pubmed"
        assert error.query == "test"

    def test_unavailable_error(self):
        """Test SearchUnavailableError."""
        error = SearchUnavailableError("Service unavailable", backend="perplexica")
        assert "unavailable" in str(error)
        assert error.backend == "perplexica"
