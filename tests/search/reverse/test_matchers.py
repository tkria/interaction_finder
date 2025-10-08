"""Tests for resource matching logic."""

import pytest

from interaction_finder.search.reverse.matchers import ResourceMatcher
from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
)
from interaction_finder.search.base import SearchResult, SearchResults, SearchQuery


# Fixtures
@pytest.fixture
def config():
    """Basic configuration for matcher tests."""
    return ReverseSearchConfig()


@pytest.fixture
def matcher(config):
    """ResourceMatcher instance for testing."""
    return ResourceMatcher(config)


# PMID matching tests
def test_pmid_exact_match_successful():
    """PMID exact match should succeed when PMIDs match."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(pmid="12345678", url="https://example.com/paper")
    result = SearchResult(
        title="Test Paper",
        url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        backend="pubmed",
        metadata={"pmid": "12345678"},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="pubmed",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 1
    assert matches[0].match_method == "pmid"
    assert matches[0].confidence == 1.0
    assert matches[0].resource == target


def test_pmid_no_match():
    """PMID matching should fail when PMIDs don't match."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(pmid="12345678", url="https://example.com/paper1")
    result = SearchResult(
        title="Test Paper",
        url="https://example.com/paper2",
        backend="pubmed",
        metadata={"pmid": "87654321"},  # Different PMID
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="pubmed",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 0


def test_pmid_match_with_whitespace():
    """PMID matching should handle whitespace differences."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(pmid=" 12345678 ", url="https://example.com/paper")
    result = SearchResult(
        title="Test Paper",
        url="https://example.com/paper2",
        backend="pubmed",
        metadata={"pmid": "12345678  "},  # Extra whitespace
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="pubmed",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 1
    assert matches[0].match_method == "pmid"
    assert matches[0].confidence == 1.0


def test_pmid_missing_in_result():
    """Should skip PMID strategy when PMID missing from result metadata."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    # Note: target has only URL (no PMID), so canonical_url will be normalized URL
    target = KnownResource(url="https://example.com/paper")
    result = SearchResult(
        title="Test Paper",
        url="https://example.com/paper",  # URL matches
        backend="test",
        metadata={},  # No PMID in metadata
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    # Should match by URL since PMID is missing
    assert len(matches) == 1
    assert matches[0].match_method == "url"


# URL matching tests
def test_url_exact_match_successful():
    """URL exact match should succeed after normalization."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(url="https://example.com/paper")
    result = SearchResult(
        title="Test Paper",
        url="https://example.com/paper",
        backend="test",
        metadata={},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 1
    assert matches[0].match_method == "url"
    assert matches[0].confidence == 1.0


def test_url_match_with_tracking_params():
    """URL matching should work after removing tracking parameters."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(url="https://example.com/paper")
    result = SearchResult(
        title="Test Paper",
        url="https://example.com/paper?utm_source=twitter&utm_campaign=test",
        backend="test",
        metadata={},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 1
    assert matches[0].match_method == "url"


def test_url_match_with_case_differences():
    """URL matching should be case-insensitive."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(url="https://example.com/paper")
    result = SearchResult(
        title="Test Paper",
        url="HTTPS://Example.com/Paper/",  # Different case, trailing slash
        backend="test",
        metadata={},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 1
    assert matches[0].match_method == "url"
    assert matches[0].confidence == 1.0


def test_url_no_match():
    """URL matching should fail when URLs don't match."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(url="https://example.com/paper1")
    result = SearchResult(
        title="Test Paper",
        url="https://example.com/paper2",
        backend="test",
        metadata={},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 0


# Title similarity tests
def test_title_similarity_above_threshold():
    """Title similarity should match when above threshold."""
    config = ReverseSearchConfig(title_similarity_threshold=0.7)
    matcher = ResourceMatcher(config)

    # These titles have ~0.747 similarity
    target = KnownResource(
        url="https://example.com/paper1",
        hint_fields={"title": "Machine Learning for Medical Diagnosis"},
    )
    result = SearchResult(
        title="Machine Learning for Medical Diagnosis and Treatment",
        url="https://example.com/paper2",
        backend="test",
        metadata={},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 1
    assert matches[0].match_method == "title_similarity"
    assert 0.7 <= matches[0].confidence <= 1.0


def test_title_similarity_below_threshold():
    """Title similarity should not match when below threshold."""
    config = ReverseSearchConfig(title_similarity_threshold=0.8)
    matcher = ResourceMatcher(config)

    target = KnownResource(
        url="https://example.com/paper1",
        hint_fields={"title": "Machine Learning for Medical Diagnosis"},
    )
    result = SearchResult(
        title="Quantum Computing in Physics",  # Very different
        url="https://example.com/paper2",
        backend="test",
        metadata={},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 0


def test_title_similarity_at_threshold():
    """Title similarity should match when exactly at threshold."""
    config = ReverseSearchConfig(title_similarity_threshold=0.5)
    matcher = ResourceMatcher(config)

    # Create titles with known similarity around 0.5
    target = KnownResource(
        url="https://example.com/paper1",
        hint_fields={"title": "machine learning methods"},
    )
    result = SearchResult(
        title="machine learning",  # Subset, should be around 0.5-0.7 similarity
        url="https://example.com/paper2",
        backend="test",
        metadata={},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    # Should match since similarity >= threshold
    assert len(matches) == 1
    assert matches[0].match_method == "title_similarity"
    assert matches[0].confidence >= 0.5


def test_title_missing_in_result():
    """Should skip title strategy when title missing from result."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(
        url="https://example.com/paper1",
        hint_fields={"title": "Machine Learning"},
    )
    result = SearchResult(
        title="",  # Empty title
        url="https://example.com/paper2",
        backend="test",
        metadata={},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 0


def test_title_missing_in_resource():
    """Title matching should fail when resource has no title in hint_fields."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(
        url="https://example.com/paper1",
        hint_fields={},  # No title
    )
    result = SearchResult(
        title="Machine Learning",
        url="https://example.com/paper2",
        backend="test",
        metadata={},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 0


# Priority ordering tests
def test_priority_pmid_over_url():
    """PMID matching should take priority over URL matching."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(pmid="12345678", url="https://example.com/paper")
    result = SearchResult(
        title="Test Paper",
        url="https://example.com/paper",  # URL also matches
        backend="pubmed",
        metadata={"pmid": "12345678"},  # But PMID should win
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="pubmed",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 1
    assert matches[0].match_method == "pmid"  # PMID, not URL


def test_priority_url_over_title():
    """URL matching should take priority over title similarity."""
    config = ReverseSearchConfig(
        title_similarity_threshold=0.5
    )  # Minimum threshold allowed
    matcher = ResourceMatcher(config)

    target = KnownResource(
        url="https://example.com/paper",
        hint_fields={"title": "Machine Learning"},
    )
    result = SearchResult(
        title="Machine Learning",  # Title also matches
        url="https://example.com/paper",  # URL matches
        backend="test",
        metadata={},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 1
    assert matches[0].match_method == "url"  # URL, not title_similarity


# Multiple results and edge cases
def test_multiple_results_partial_matches():
    """Should correctly handle multiple results with partial matches."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    # Note: target2 has no PMID, so its canonical_url will be the normalized URL
    target1 = KnownResource(pmid="111", url="https://example.com/paper1")
    target2 = KnownResource(url="https://example.com/paper2")

    result1 = SearchResult(
        title="Paper 1",
        url="https://example.com/result1",
        backend="test",
        metadata={"pmid": "111"},  # Matches target1
    )
    result2 = SearchResult(
        title="Paper 2",
        url="https://example.com/result2",
        backend="test",
        metadata={"pmid": "999"},  # No match
    )
    result3 = SearchResult(
        title="Paper 3",
        url="https://example.com/paper2",  # Matches target2 by URL
        backend="test",
        metadata={},
    )

    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result1, result2, result3],
        backend="test",
    )

    matches = matcher.match_results(results, {target1, target2}, query_index=0)

    assert len(matches) == 2
    # result1 matches target1 by PMID
    assert any(
        m.search_result == result1
        and m.resource == target1
        and m.match_method == "pmid"
        for m in matches
    )
    # result3 matches target2 by URL
    assert any(
        m.search_result == result3 and m.resource == target2 and m.match_method == "url"
        for m in matches
    )
    # result2 doesn't match anything
    assert not any(m.search_result == result2 for m in matches)


def test_empty_target_resources():
    """Should return no matches when target_resources is empty."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    result = SearchResult(
        title="Test Paper",
        url="https://example.com/paper",
        backend="test",
        metadata={"pmid": "12345678"},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, set(), query_index=0)

    assert len(matches) == 0


def test_empty_search_results():
    """Should return no matches when search results are empty."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(pmid="12345678", url="https://example.com/paper")
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[],  # Empty results
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=0)

    assert len(matches) == 0


def test_query_index_preserved():
    """Should preserve query_index in matches."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    target = KnownResource(pmid="12345678", url="https://example.com/paper")
    result = SearchResult(
        title="Test Paper",
        url="https://example.com/paper",
        backend="test",
        metadata={"pmid": "12345678"},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target}, query_index=42)

    assert len(matches) == 1
    assert matches[0].query_index == 42


def test_first_match_wins():
    """When multiple targets could match, first match should win."""
    config = ReverseSearchConfig()
    matcher = ResourceMatcher(config)

    # Two targets with same PMID (edge case)
    target1 = KnownResource(pmid="12345678", url="https://example.com/paper1")
    target2 = KnownResource(pmid="12345678", url="https://example.com/paper2")

    result = SearchResult(
        title="Test Paper",
        url="https://example.com/paper3",
        backend="test",
        metadata={"pmid": "12345678"},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target1, target2}, query_index=0)

    # Should match exactly one target (first found)
    assert len(matches) == 1
    assert matches[0].match_method == "pmid"
    assert matches[0].resource in {target1, target2}


def test_title_similarity_selects_best_match():
    """Title similarity should select the best matching resource."""
    config = ReverseSearchConfig(title_similarity_threshold=0.5)
    matcher = ResourceMatcher(config)

    # Two targets with different title similarities
    target1 = KnownResource(
        url="https://example.com/paper1",
        hint_fields={"title": "machine learning methods"},
    )
    target2 = KnownResource(
        url="https://example.com/paper2",
        hint_fields={"title": "machine learning neural networks deep learning"},
    )

    result = SearchResult(
        title="machine learning neural networks",
        url="https://example.com/paper3",
        backend="test",
        metadata={},
    )
    results = SearchResults(
        query=SearchQuery(query="test"),
        results=[result],
        backend="test",
    )

    matches = matcher.match_results(results, {target1, target2}, query_index=0)

    # Should match the better title (target2)
    assert len(matches) == 1
    assert matches[0].resource == target2
    assert matches[0].match_method == "title_similarity"


# PMID extraction unit tests (Task 01)
class TestPMIDExtraction:
    """Unit tests for _extract_pmid_from_metadata helper."""

    def test_extract_pmid_direct(self, matcher):
        """Extract PMID from direct metadata location."""
        metadata = {"pmid": "12345678"}
        assert matcher._extract_pmid_from_metadata(metadata) == "12345678"

    def test_extract_pmid_pubmed_nested(self, matcher):
        """Extract PMID from PubMed nested metadata."""
        metadata = {"pubmed_summary": {"pmid": "12345678"}}
        assert matcher._extract_pmid_from_metadata(metadata) == "12345678"

    def test_extract_pmid_missing(self, matcher):
        """Return None when PMID not present in metadata."""
        metadata = {"other_field": "value"}
        assert matcher._extract_pmid_from_metadata(metadata) is None

    def test_extract_pmid_priority(self, matcher):
        """Direct PMID takes priority over nested when both present."""
        metadata = {"pmid": "11111111", "pubmed_summary": {"pmid": "22222222"}}
        assert matcher._extract_pmid_from_metadata(metadata) == "11111111"

    def test_extract_pmid_normalization(self, matcher):
        """Integer PMID converted to string, whitespace stripped."""
        # Test integer conversion
        metadata = {"pmid": 12345678}
        assert matcher._extract_pmid_from_metadata(metadata) == "12345678"
        # Test whitespace stripping (direct)
        metadata = {"pmid": "  12345678  "}
        assert matcher._extract_pmid_from_metadata(metadata) == "12345678"
        # Test whitespace stripping (nested)
        metadata = {"pubmed_summary": {"pmid": "  12345678  "}}
        assert matcher._extract_pmid_from_metadata(metadata) == "12345678"

    def test_extract_pmid_malformed_pubmed_summary(self, matcher):
        """Handle malformed pubmed_summary gracefully."""
        # pubmed_summary is not a dict
        metadata = {"pubmed_summary": "not a dict"}
        assert matcher._extract_pmid_from_metadata(metadata) is None
        # pubmed_summary is None
        metadata = {"pubmed_summary": None}
        assert matcher._extract_pmid_from_metadata(metadata) is None

    def test_extract_pmid_empty_string(self, matcher):
        """Empty string PMID is treated as missing."""
        metadata = {"pmid": ""}
        assert matcher._extract_pmid_from_metadata(metadata) is None

    def test_extract_pmid_empty_metadata(self, matcher):
        """Empty metadata returns None."""
        metadata = {}
        assert matcher._extract_pmid_from_metadata(metadata) is None


# DOI normalization unit tests (Task 01)
class TestDOINormalization:
    """Unit tests for _normalize_doi helper."""

    def test_normalize_doi_with_prefix(self, matcher):
        """Normalize DOI with 'doi:' prefix."""
        assert matcher._normalize_doi("doi:10.1234/ABC") == "10.1234/abc"

    def test_normalize_doi_with_https(self, matcher):
        """Normalize DOI with HTTPS URL."""
        assert matcher._normalize_doi("https://doi.org/10.1234/ABC") == "10.1234/abc"

    def test_normalize_doi_with_dx(self, matcher):
        """Normalize DOI with http://dx.doi.org/ prefix."""
        assert matcher._normalize_doi("http://dx.doi.org/10.1234/ABC") == "10.1234/abc"

    def test_normalize_doi_plain(self, matcher):
        """Plain DOI already normalized remains unchanged."""
        assert matcher._normalize_doi("10.1234/abc") == "10.1234/abc"

    def test_normalize_doi_case_insensitive_prefix(self, matcher):
        """DOI: prefix is case-insensitive."""
        assert matcher._normalize_doi("DOI:10.1234/ABC") == "10.1234/abc"
        assert matcher._normalize_doi("Doi:10.1234/ABC") == "10.1234/abc"
        assert matcher._normalize_doi("dOi:10.1234/ABC") == "10.1234/abc"

    def test_normalize_doi_whitespace(self, matcher):
        """Whitespace is trimmed before and after normalization."""
        assert matcher._normalize_doi("  doi:10.1234/ABC  ") == "10.1234/abc"
        assert (
            matcher._normalize_doi("  https://doi.org/10.1234/ABC  ") == "10.1234/abc"
        )
        assert matcher._normalize_doi("  10.1234/ABC  ") == "10.1234/abc"

    def test_normalize_doi_lowercase_conversion(self, matcher):
        """DOI is converted to lowercase."""
        assert matcher._normalize_doi("10.1234/ABC") == "10.1234/abc"
        assert matcher._normalize_doi("10.ABCD/XYZ.test") == "10.abcd/xyz.test"

    def test_normalize_doi_complex_suffix(self, matcher):
        """DOI with complex suffix normalized correctly."""
        # Real-world DOI examples
        assert (
            matcher._normalize_doi("doi:10.1038/nature12345") == "10.1038/nature12345"
        )
        assert (
            matcher._normalize_doi("https://doi.org/10.1016/j.cell.2023.01.001")
            == "10.1016/j.cell.2023.01.001"
        )

    def test_normalize_doi_only_first_prefix_stripped(self, matcher):
        """Only the first matching prefix is stripped."""
        # Edge case: DOI containing "doi:" in suffix
        result = matcher._normalize_doi("doi:10.1234/contains-doi:-text")
        assert result == "10.1234/contains-doi:-text"
