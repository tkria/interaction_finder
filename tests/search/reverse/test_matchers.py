"""Tests for resource matching logic."""

from interaction_finder.search.reverse.matchers import ResourceMatcher
from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
)
from interaction_finder.search.base import SearchResult, SearchResults, SearchQuery


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
