"""
Tests for ResourceMatcher investigation logging integration.

Tests that ResourceMatcher properly tracks detailed matching strategy information
for investigation logging without changing existing matching behavior.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from interaction_finder.search.base import SearchResult, SearchResults, SearchQuery
from interaction_finder.search.reverse.matchers import ResourceMatcher
from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
)
from interaction_finder.search.reverse.investigation_logger import InvestigationLogger


@pytest.fixture
def config():
    """Basic reverse search configuration."""
    return ReverseSearchConfig(title_similarity_threshold=0.8)


@pytest.fixture
def mock_fetcher():
    """Mock fetcher for DOI lookups."""
    fetcher = Mock()
    fetcher.get_doi = AsyncMock(return_value=None)
    return fetcher


@pytest.fixture
def test_query():
    """Sample search query for tests."""
    return SearchQuery(query="test query")


@pytest.fixture
def sample_resources():
    """Sample target resources for matching."""
    return {
        KnownResource(
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            pmid="12345678",
            hint_fields={"title": "Sample PMID Paper"},
        ),
        KnownResource(
            url="https://example.com/paper1",
            hint_fields={"title": "URL Match Paper"},
        ),
        KnownResource(
            url="https://doi.org/10.1234/test",
            hint_fields={"title": "DOI Match Paper"},
        ),
        KnownResource(
            url="https://example.com/paper2",
            hint_fields={"title": "Title Match Paper Test"},
        ),
    }


@pytest.mark.asyncio
async def test_matcher_without_logger(
    config, mock_fetcher, sample_resources, test_query
):
    """ResourceMatcher works normally without investigation_logger."""
    # Create matcher without logger
    matcher = ResourceMatcher(config, mock_fetcher)
    # Create search results with PMID match
    results = SearchResults(
        query=test_query,
        results=[
            SearchResult(
                url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                title="Sample PMID Paper",
                backend="test",
                metadata={"pmid": "12345678"},
            )
        ],
        backend="test",
    )
    # Match results
    matches = await matcher.match_results(results, sample_resources, query_index=0)
    # Verify match found
    assert len(matches) == 1
    assert matches[0].match_method == "pmid"
    # Verify last_matching_details is populated (details tracked regardless of logger)
    assert len(matcher.last_matching_details) == 1
    assert matcher.last_matching_details[0]["final_match"]["matched"] is True


@pytest.mark.asyncio
async def test_matching_details_tracked(
    config, mock_fetcher, sample_resources, test_query
):
    """Matching details are tracked when logger provided."""
    # Create matcher with logger (logger itself not used in matcher)
    matcher = ResourceMatcher(config, mock_fetcher, investigation_logger=AsyncMock())
    # Create search results with mix of matches
    results = SearchResults(
        query=test_query,
        results=[
            SearchResult(
                url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                title="Sample PMID Paper",
                backend="test",
                metadata={"pmid": "12345678"},
            ),
            SearchResult(
                url="https://example.com/paper1",
                title="URL Match Paper",
                backend="test",
                metadata={},
            ),
            SearchResult(
                url="https://other.com/unmatched",
                title="No Match Paper",
                backend="test",
                metadata={},
            ),
        ],
        backend="test",
    )
    # Match results
    matches = await matcher.match_results(results, sample_resources, query_index=0)
    # Verify matches
    assert len(matches) == 2
    # Verify last_matching_details populated
    assert len(matcher.last_matching_details) == 3  # One entry per result
    # Check structure of each detail
    for detail in matcher.last_matching_details:
        assert "result_url" in detail
        assert "result_title" in detail
        assert "strategies_attempted" in detail
        assert "final_match" in detail
        # Verify strategies_attempted has all 4 strategies
        assert "pmid" in detail["strategies_attempted"]
        assert "url" in detail["strategies_attempted"]
        assert "doi" in detail["strategies_attempted"]
        assert "title" in detail["strategies_attempted"]


@pytest.mark.asyncio
async def test_pmid_strategy_logged(config, mock_fetcher, sample_resources, test_query):
    """PMID strategy details are properly logged."""
    matcher = ResourceMatcher(config, mock_fetcher, investigation_logger=AsyncMock())
    results = SearchResults(
        query=test_query,
        results=[
            SearchResult(
                url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                title="Sample PMID Paper",
                backend="test",
                metadata={"pmid": "12345678"},
            )
        ],
        backend="test",
    )
    matches = await matcher.match_results(results, sample_resources, query_index=0)
    # Verify match
    assert len(matches) == 1
    assert matches[0].match_method == "pmid"
    # Check detail structure
    detail = matcher.last_matching_details[0]
    pmid_detail = detail["strategies_attempted"]["pmid"]
    assert pmid_detail["attempted"] is True
    assert pmid_detail["result_pmid"] == "12345678"
    assert (
        pmid_detail["matched_resource"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    )
    assert pmid_detail["confidence"] == 1.0
    # Other strategies not attempted (PMID matched)
    assert detail["strategies_attempted"]["url"]["attempted"] is False
    assert detail["strategies_attempted"]["doi"]["attempted"] is False
    assert detail["strategies_attempted"]["title"]["attempted"] is False


@pytest.mark.asyncio
async def test_url_strategy_logged(config, mock_fetcher, sample_resources, test_query):
    """URL strategy details are properly logged."""
    matcher = ResourceMatcher(config, mock_fetcher, investigation_logger=AsyncMock())
    # Result without PMID but matching URL
    results = SearchResults(
        query=test_query,
        results=[
            SearchResult(
                url="https://example.com/paper1",
                title="URL Match Paper",
                backend="test",
                metadata={},
            )
        ],
        backend="test",
    )
    matches = await matcher.match_results(results, sample_resources, query_index=0)
    # Verify match
    assert len(matches) == 1
    assert matches[0].match_method == "url"
    # Check detail structure
    detail = matcher.last_matching_details[0]
    pmid_detail = detail["strategies_attempted"]["pmid"]
    assert pmid_detail["attempted"] is False  # No PMID in metadata
    url_detail = detail["strategies_attempted"]["url"]
    assert url_detail["attempted"] is True
    assert "normalized_result_url" in url_detail
    assert "normalized_target_urls" in url_detail
    assert isinstance(url_detail["normalized_target_urls"], list)
    assert url_detail["matched_resource"] == "https://example.com/paper1"
    assert url_detail["confidence"] == 1.0
    # DOI and title not attempted (URL matched)
    assert detail["strategies_attempted"]["doi"]["attempted"] is False
    assert detail["strategies_attempted"]["title"]["attempted"] is False


@pytest.mark.asyncio
async def test_doi_strategy_logged(config, mock_fetcher, sample_resources, test_query):
    """DOI strategy details are properly logged."""
    # Mock DOI fetch to return matching DOI
    mock_fetcher.get_doi = AsyncMock(return_value="10.1234/test")
    matcher = ResourceMatcher(config, mock_fetcher, investigation_logger=AsyncMock())
    # Result without PMID or matching URL but with DOI
    results = SearchResults(
        query=test_query,
        results=[
            SearchResult(
                url="https://other.com/doi-paper",
                title="DOI Match Paper",  # High similarity to trigger DOI fetch
                backend="test",
                metadata={},
            )
        ],
        backend="test",
    )
    matches = await matcher.match_results(results, sample_resources, query_index=0)
    # Verify match
    assert len(matches) == 1
    assert matches[0].match_method == "doi"
    # Check detail structure
    detail = matcher.last_matching_details[0]
    doi_detail = detail["strategies_attempted"]["doi"]
    assert doi_detail["attempted"] is True
    assert doi_detail["fetched_doi"] == "10.1234/test"
    assert doi_detail["title_similarity_threshold_met"] is True
    assert doi_detail["matched_resource"] == "https://doi.org/10.1234/test"
    assert doi_detail["confidence"] == 1.0


@pytest.mark.asyncio
async def test_title_strategy_logged(
    config, mock_fetcher, sample_resources, test_query
):
    """Title strategy details are properly logged."""
    matcher = ResourceMatcher(config, mock_fetcher, investigation_logger=AsyncMock())
    # Result with similar title but no PMID/URL/DOI match
    results = SearchResults(
        query=test_query,
        results=[
            SearchResult(
                url="https://other.com/title-match",
                title="Title Match Paper Test",  # Matches hint_fields title
                backend="test",
                metadata={},
            )
        ],
        backend="test",
    )
    matches = await matcher.match_results(results, sample_resources, query_index=0)
    # Verify match
    assert len(matches) == 1
    assert matches[0].match_method == "title_similarity"
    # Check detail structure
    detail = matcher.last_matching_details[0]
    title_detail = detail["strategies_attempted"]["title"]
    assert title_detail["attempted"] is True
    assert "similarities" in title_detail
    assert isinstance(title_detail["similarities"], list)
    # Each similarity entry has resource_url and score
    for sim in title_detail["similarities"]:
        assert "resource_url" in sim
        assert "score" in sim
    assert "best_score" in title_detail
    assert title_detail["best_score"] >= config.title_similarity_threshold
    assert title_detail["threshold"] == config.title_similarity_threshold
    assert title_detail["matched_resource"] == "https://example.com/paper2"
    assert title_detail["confidence"] == title_detail["best_score"]


@pytest.mark.asyncio
async def test_no_match_all_strategies_tried(
    config, mock_fetcher, sample_resources, test_query
):
    """All strategies attempted when no match found."""
    matcher = ResourceMatcher(config, mock_fetcher, investigation_logger=AsyncMock())
    # Result that doesn't match anything
    results = SearchResults(
        query=test_query,
        results=[
            SearchResult(
                url="https://completely-different.com/paper",
                title="Completely Unrelated Paper",
                backend="test",
                metadata={},
            )
        ],
        backend="test",
    )
    matches = await matcher.match_results(results, sample_resources, query_index=0)
    # Verify no match
    assert len(matches) == 0
    # Check all strategies attempted
    detail = matcher.last_matching_details[0]
    assert detail["strategies_attempted"]["pmid"]["attempted"] is False
    assert detail["strategies_attempted"]["url"]["attempted"] is True
    assert detail["strategies_attempted"]["url"]["matched_resource"] is None
    assert (
        detail["strategies_attempted"]["doi"]["attempted"] is False
    )  # Title similarity too low
    assert detail["strategies_attempted"]["title"]["attempted"] is True
    assert detail["strategies_attempted"]["title"]["matched_resource"] is None
    # Final match shows no match
    assert detail["final_match"]["matched"] is False
    assert detail["final_match"]["resource_url"] is None
    assert detail["final_match"]["method"] is None


@pytest.mark.asyncio
async def test_search_execution_logging_integration(
    config, mock_fetcher, sample_resources, test_query, tmp_path
):
    """Full integration with logger for search execution (placeholder test)."""
    # This test verifies the matcher provides data for logging
    # Actual logging is done by ReverseSearcher (Task 02)
    log_file = tmp_path / "test_investigation.jsonl"
    async with InvestigationLogger(log_file) as logger:
        matcher = ResourceMatcher(config, mock_fetcher, investigation_logger=logger)
        results = SearchResults(
            query=test_query,
            results=[
                SearchResult(
                    url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                    title="Sample Paper",
                    backend="test",
                    metadata={"pmid": "12345678"},
                )
            ],
            backend="test",
            search_time=0.5,
        )
        matches = await matcher.match_results(results, sample_resources, query_index=0)
        # Verify matcher provides detailed information
        assert len(matcher.last_matching_details) == 1
        assert matcher.last_matching_details[0]["final_match"]["matched"] is True
        # Note: Actual logging to file is done by ReverseSearcher.search()
        # which calls await logger.log_matching(...)


@pytest.mark.asyncio
async def test_matching_logging_integration_placeholder(
    config, mock_fetcher, sample_resources, test_query, tmp_path
):
    """Placeholder for full matching entry logging (handled by ReverseSearcher)."""
    # This test shows what data is available for logging
    matcher = ResourceMatcher(config, mock_fetcher, investigation_logger=AsyncMock())
    results = SearchResults(
        query=test_query,
        results=[
            SearchResult(
                url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
                title="Sample Paper",
                backend="test",
                metadata={"pmid": "12345678"},
            ),
            SearchResult(
                url="https://example.com/paper1",
                title="URL Match Paper",
                backend="test",
                metadata={},
            ),
        ],
        backend="test",
    )
    matches = await matcher.match_results(results, sample_resources, query_index=0)
    # Verify data available for logging
    assert len(matches) == 2
    assert len(matcher.last_matching_details) == 2
    # Verify structure suitable for MatchingEntry
    details = matcher.last_matching_details
    assert all("result_url" in d for d in details)
    assert all("result_title" in d for d in details)
    assert all("strategies_attempted" in d for d in details)
    assert all("final_match" in d for d in details)
    # Extract strategies attempted (for MatchingEntry.strategies_attempted field)
    all_strategies = set()
    for detail in details:
        for strategy_name, strategy_info in detail["strategies_attempted"].items():
            if strategy_info.get("attempted", False):
                all_strategies.add(strategy_name)
    # Verify at least PMID and URL were attempted
    assert "pmid" in all_strategies or "url" in all_strategies
