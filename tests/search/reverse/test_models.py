"""
Comprehensive unit tests for reverse search models.

Tests cover all acceptance criteria:
1. KnownResource model validation and canonical URL computation
2. ResourceMatch model with method and confidence validation
3. ReverseSearchResult model with metrics validation
4. ReverseSearchSession model with coverage_pct property
5. ReverseSearchConfig model with all constraint validation
6. ReverseSearchError hierarchy with context handling
7. Model docstrings and field descriptions (manual verification)
8. Hash/equality operations for KnownResource
"""


import pytest
from pydantic import ValidationError

from interaction_finder.search.base import SearchQuery, SearchResult, SearchResults
from interaction_finder.search.reverse import (
    KnownResource,
    MatchingError,
    QueryGenerationError,
    ResourceMatch,
    ResourceParseError,
    ReverseSearchConfig,
    ReverseSearchError,
    ReverseSearchResult,
    ReverseSearchSession,
)

# KnownResource tests


def test_known_resource_canonical_url_from_pmid():
    """Test that canonical URL is derived from PMID when available."""
    resource = KnownResource(pmid="12345678", url="https://example.com/paper")
    assert resource.canonical_url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_known_resource_canonical_url_from_pmid_with_whitespace():
    """Test that PMID whitespace is stripped in canonical URL."""
    resource = KnownResource(pmid="  12345678  ", url="https://example.com/paper")
    assert resource.canonical_url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_known_resource_canonical_url_normalization():
    """Test basic URL normalization when PMID is not available."""
    resource = KnownResource(url="HTTPS://Example.com/Paper/")
    # Basic normalization: lowercase scheme/domain, preserve path case, strip trailing slash
    assert resource.canonical_url == "https://example.com/Paper"


def test_known_resource_canonical_url_no_trailing_slash():
    """Test URL normalization handles URLs without trailing slash."""
    resource = KnownResource(url="https://example.com/paper")
    assert resource.canonical_url == "https://example.com/paper"


def test_known_resource_with_hint_fields():
    """Test KnownResource stores hint_fields correctly."""
    resource = KnownResource(
        url="https://example.com/paper",
        hint_fields={"celltype": "CD8+ T cell", "marker": "CD8A"},
    )
    assert resource.hint_fields == {"celltype": "CD8+ T cell", "marker": "CD8A"}


def test_known_resource_empty_hint_fields():
    """Test KnownResource defaults to empty dict for hint_fields."""
    resource = KnownResource(url="https://example.com/paper")
    assert resource.hint_fields == {}


def test_known_resource_hash_uses_canonical_url():
    """Test that hash is based on canonical_url."""
    res1 = KnownResource(pmid="12345678", url="https://example.com/paper")
    res2 = KnownResource(pmid="12345678", url="https://different.com/paper")
    # Same PMID → same canonical_url → same hash
    assert hash(res1) == hash(res2)


def test_known_resource_equality_uses_canonical_url():
    """Test that equality is based on canonical_url."""
    res1 = KnownResource(pmid="12345678", url="https://example.com/paper")
    res2 = KnownResource(pmid="12345678", url="https://different.com/paper")
    # Same PMID → same canonical_url → equal
    assert res1 == res2


def test_known_resource_inequality_different_canonical_url():
    """Test that different canonical URLs result in inequality."""
    res1 = KnownResource(url="https://example.com/paper1")
    res2 = KnownResource(url="https://example.com/paper2")
    assert res1 != res2


def test_known_resource_set_deduplication():
    """Test that KnownResource works correctly in sets for deduplication."""
    res1 = KnownResource(pmid="12345678", url="https://example.com/paper")
    res2 = KnownResource(pmid="12345678", url="https://different.com/paper")
    res3 = KnownResource(url="https://unique.com/paper")
    # Set should deduplicate res1 and res2 (same canonical_url)
    resources = {res1, res2, res3}
    assert len(resources) == 2


# ResourceMatch tests


def test_resource_match_validation():
    """Test ResourceMatch validates all fields correctly."""
    resource = KnownResource(url="https://example.com/paper", pmid="12345678")
    result = SearchResult(
        title="Test Paper",
        url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        backend="pubmed",
    )
    match = ResourceMatch(
        resource=resource,
        search_result=result,
        match_method="pmid",
        confidence=1.0,
        query_index=0,
    )
    assert match.resource == resource
    assert match.search_result == result
    assert match.match_method == "pmid"
    assert match.confidence == 1.0
    assert match.query_index == 0


def test_resource_match_confidence_bounds():
    """Test ResourceMatch validates confidence is between 0.0 and 1.0."""
    resource = KnownResource(url="https://example.com/paper")
    result = SearchResult(title="Test", url="https://example.com", backend="pubmed")

    # Valid confidence values
    match = ResourceMatch(
        resource=resource,
        search_result=result,
        match_method="url",
        confidence=0.5,
        query_index=0,
    )
    assert match.confidence == 0.5

    # Invalid: confidence > 1.0
    with pytest.raises(ValidationError) as exc_info:
        ResourceMatch(
            resource=resource,
            search_result=result,
            match_method="url",
            confidence=1.5,
            query_index=0,
        )
    assert "confidence" in str(exc_info.value).lower()

    # Invalid: confidence < 0.0
    with pytest.raises(ValidationError) as exc_info:
        ResourceMatch(
            resource=resource,
            search_result=result,
            match_method="url",
            confidence=-0.1,
            query_index=0,
        )
    assert "confidence" in str(exc_info.value).lower()


def test_resource_match_method_literal():
    """Test ResourceMatch validates match_method is one of allowed values."""
    resource = KnownResource(url="https://example.com/paper")
    result = SearchResult(title="Test", url="https://example.com", backend="pubmed")

    # Valid methods
    for method in ["pmid", "url", "title_similarity"]:
        match = ResourceMatch(
            resource=resource,
            search_result=result,
            match_method=method,
            confidence=0.9,
            query_index=0,
        )
        assert match.match_method == method

    # Invalid method
    with pytest.raises(ValidationError) as exc_info:
        ResourceMatch(
            resource=resource,
            search_result=result,
            match_method="invalid_method",
            confidence=0.9,
            query_index=0,
        )
    assert "match_method" in str(exc_info.value).lower()


# ReverseSearchResult tests


def test_reverse_search_result_validation():
    """Test ReverseSearchResult validates all fields correctly."""
    query = SearchQuery(query="BRCA1 breast cancer")
    results = SearchResults(
        query=query,
        results=[
            SearchResult(title="Paper 1", url="https://example.com/1", backend="pubmed")
        ],
        backend="pubmed",
    )
    resource = KnownResource(url="https://example.com/1")

    result = ReverseSearchResult(
        query="BRCA1 breast cancer",
        query_index=0,
        search_results=results,
        resources_found=[resource],
        new_finds=1,
        cumulative_coverage=0.2,
        search_time=1.5,
        backend="pubmed",
    )
    assert result.query == "BRCA1 breast cancer"
    assert result.query_index == 0
    assert result.search_results == results
    assert result.resources_found == [resource]
    assert result.new_finds == 1
    assert result.cumulative_coverage == 0.2
    assert result.search_time == 1.5
    assert result.backend == "pubmed"


def test_reverse_search_result_cumulative_coverage_bounds():
    """Test ReverseSearchResult validates cumulative_coverage is between 0.0 and 1.0."""
    query = SearchQuery(query="test")
    results = SearchResults(query=query, results=[], backend="pubmed")

    # Valid coverage
    result = ReverseSearchResult(
        query="test",
        query_index=0,
        search_results=results,
        resources_found=[],
        new_finds=0,
        cumulative_coverage=0.75,
        search_time=1.0,
        backend="pubmed",
    )
    assert result.cumulative_coverage == 0.75

    # Invalid: coverage > 1.0
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchResult(
            query="test",
            query_index=0,
            search_results=results,
            resources_found=[],
            new_finds=0,
            cumulative_coverage=1.5,
            search_time=1.0,
            backend="pubmed",
        )
    assert "cumulative_coverage" in str(exc_info.value).lower()


# ReverseSearchSession tests


def test_reverse_search_session_validation():
    """Test ReverseSearchSession validates all fields correctly."""
    resource = KnownResource(url="https://example.com/paper")
    query = SearchQuery(query="test")
    results = SearchResults(query=query, results=[], backend="pubmed")
    query_result = ReverseSearchResult(
        query="test",
        query_index=0,
        search_results=results,
        resources_found=[resource],
        new_finds=1,
        cumulative_coverage=1.0,
        search_time=1.0,
        backend="pubmed",
    )

    session = ReverseSearchSession(
        target_resources=[resource],
        query_results=[query_result],
        matches=[],
        total_queries=1,
        final_coverage=1.0,
        found_count=1,
        unfound_resources=[],
        total_time=1.5,
        stopping_reason="coverage_achieved",
    )
    assert session.target_resources == [resource]
    assert session.query_results == [query_result]
    assert session.total_queries == 1
    assert session.final_coverage == 1.0
    assert session.found_count == 1
    assert session.unfound_resources == []
    assert session.total_time == 1.5
    assert session.stopping_reason == "coverage_achieved"


def test_reverse_search_session_coverage_pct_property():
    """Test ReverseSearchSession.coverage_pct converts final_coverage to percentage."""
    query = SearchQuery(query="test")
    results = SearchResults(query=query, results=[], backend="pubmed")
    query_result = ReverseSearchResult(
        query="test",
        query_index=0,
        search_results=results,
        resources_found=[],
        new_finds=0,
        cumulative_coverage=0.67,
        search_time=1.0,
        backend="pubmed",
    )

    session = ReverseSearchSession(
        target_resources=[],
        query_results=[query_result],
        matches=[],
        total_queries=1,
        final_coverage=0.67,
        found_count=0,
        unfound_resources=[],
        total_time=1.0,
        stopping_reason="max_queries",
    )
    assert session.coverage_pct == 67.0


def test_reverse_search_session_stopping_reason_literal():
    """Test ReverseSearchSession validates stopping_reason is one of allowed values."""
    query = SearchQuery(query="test")
    results = SearchResults(query=query, results=[], backend="pubmed")
    query_result = ReverseSearchResult(
        query="test",
        query_index=0,
        search_results=results,
        resources_found=[],
        new_finds=0,
        cumulative_coverage=0.5,
        search_time=1.0,
        backend="pubmed",
    )

    # Valid stopping reasons
    for reason in ["coverage_achieved", "consecutive_zero_finds", "max_queries"]:
        session = ReverseSearchSession(
            target_resources=[],
            query_results=[query_result],
            matches=[],
            total_queries=1,
            final_coverage=0.5,
            found_count=0,
            unfound_resources=[],
            total_time=1.0,
            stopping_reason=reason,
        )
        assert session.stopping_reason == reason

    # Invalid stopping reason
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchSession(
            target_resources=[],
            query_results=[query_result],
            matches=[],
            total_queries=1,
            final_coverage=0.5,
            found_count=0,
            unfound_resources=[],
            total_time=1.0,
            stopping_reason="invalid_reason",
        )
    assert "stopping_reason" in str(exc_info.value).lower()


# ReverseSearchConfig tests


def test_reverse_search_config_defaults():
    """Test ReverseSearchConfig provides sensible defaults."""
    config = ReverseSearchConfig()
    assert config.coverage_target == 0.95
    assert config.consecutive_zero_limit == 3
    assert config.max_queries == 100
    assert config.keyword_extractor == "yake"
    assert config.keywords_per_query == 7
    assert config.use_hint_fields is True
    assert config.enable_clustering is True
    assert config.min_cluster_size == 3
    assert config.target_clusters == 4
    assert config.title_similarity_threshold == 0.8
    assert config.batch_pmid_fetch_size == 200
    assert config.cache_ttl_days == 7
    assert config.pmid_metadata_cache_days == 30
    assert config.search_backend == "pubmed"
    assert config.max_results_per_query == 100


def test_reverse_search_config_custom_values():
    """Test ReverseSearchConfig accepts valid custom values."""
    config = ReverseSearchConfig(
        coverage_target=0.90,
        consecutive_zero_limit=5,
        max_queries=50,
        keyword_extractor="rake",
        keywords_per_query=10,
        use_hint_fields=False,
        enable_clustering=False,
        min_cluster_size=2,
        target_clusters=6,
        title_similarity_threshold=0.7,
        batch_pmid_fetch_size=100,
        cache_ttl_days=14,
        pmid_metadata_cache_days=60,
        search_backend="perplexica",
        max_results_per_query=200,
    )
    assert config.coverage_target == 0.90
    assert config.consecutive_zero_limit == 5
    assert config.max_queries == 50
    assert config.keyword_extractor == "rake"
    assert config.keywords_per_query == 10
    assert config.use_hint_fields is False
    assert config.enable_clustering is False
    assert config.min_cluster_size == 2
    assert config.target_clusters == 6
    assert config.title_similarity_threshold == 0.7
    assert config.batch_pmid_fetch_size == 100
    assert config.cache_ttl_days == 14
    assert config.pmid_metadata_cache_days == 60
    assert config.search_backend == "perplexica"
    assert config.max_results_per_query == 200


def test_reverse_search_config_coverage_target_bounds():
    """Test ReverseSearchConfig validates coverage_target is between 0.5 and 1.0."""
    # Valid bounds
    config = ReverseSearchConfig(coverage_target=0.5)
    assert config.coverage_target == 0.5
    config = ReverseSearchConfig(coverage_target=1.0)
    assert config.coverage_target == 1.0

    # Invalid: below lower bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(coverage_target=0.4)
    assert "coverage_target" in str(exc_info.value).lower()

    # Invalid: above upper bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(coverage_target=1.1)
    assert "coverage_target" in str(exc_info.value).lower()


def test_reverse_search_config_consecutive_zero_limit_bounds():
    """Test ReverseSearchConfig validates consecutive_zero_limit is between 1 and 10."""
    # Valid bounds
    config = ReverseSearchConfig(consecutive_zero_limit=1)
    assert config.consecutive_zero_limit == 1
    config = ReverseSearchConfig(consecutive_zero_limit=10)
    assert config.consecutive_zero_limit == 10

    # Invalid: below lower bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(consecutive_zero_limit=0)
    assert "consecutive_zero_limit" in str(exc_info.value).lower()

    # Invalid: above upper bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(consecutive_zero_limit=11)
    assert "consecutive_zero_limit" in str(exc_info.value).lower()


def test_reverse_search_config_max_queries_bounds():
    """Test ReverseSearchConfig validates max_queries is between 5 and 200."""
    # Valid bounds
    config = ReverseSearchConfig(max_queries=5)
    assert config.max_queries == 5
    config = ReverseSearchConfig(max_queries=200)
    assert config.max_queries == 200

    # Invalid: below lower bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(max_queries=4)
    assert "max_queries" in str(exc_info.value).lower()

    # Invalid: above upper bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(max_queries=201)
    assert "max_queries" in str(exc_info.value).lower()


def test_reverse_search_config_keyword_extractor_literal():
    """Test ReverseSearchConfig validates keyword_extractor is one of allowed values."""
    # Valid extractors
    for extractor in ["yake", "rake", "tfidf"]:
        config = ReverseSearchConfig(keyword_extractor=extractor)
        assert config.keyword_extractor == extractor

    # Invalid extractor
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(keyword_extractor="invalid")
    assert "keyword_extractor" in str(exc_info.value).lower()


def test_reverse_search_config_keywords_per_query_bounds():
    """Test ReverseSearchConfig validates keywords_per_query is between 3 and 15."""
    # Valid bounds
    config = ReverseSearchConfig(keywords_per_query=3)
    assert config.keywords_per_query == 3
    config = ReverseSearchConfig(keywords_per_query=15)
    assert config.keywords_per_query == 15

    # Invalid: below lower bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(keywords_per_query=2)
    assert "keywords_per_query" in str(exc_info.value).lower()

    # Invalid: above upper bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(keywords_per_query=16)
    assert "keywords_per_query" in str(exc_info.value).lower()


def test_reverse_search_config_min_cluster_size_bounds():
    """Test ReverseSearchConfig validates min_cluster_size is >= 2."""
    # Valid bounds
    config = ReverseSearchConfig(min_cluster_size=2)
    assert config.min_cluster_size == 2
    config = ReverseSearchConfig(min_cluster_size=100)
    assert config.min_cluster_size == 100

    # Invalid: below lower bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(min_cluster_size=1)
    assert "min_cluster_size" in str(exc_info.value).lower()


def test_reverse_search_config_target_clusters_bounds():
    """Test ReverseSearchConfig validates target_clusters is between 2 and 10."""
    # Valid bounds
    config = ReverseSearchConfig(target_clusters=2)
    assert config.target_clusters == 2
    config = ReverseSearchConfig(target_clusters=10)
    assert config.target_clusters == 10

    # Invalid: below lower bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(target_clusters=1)
    assert "target_clusters" in str(exc_info.value).lower()

    # Invalid: above upper bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(target_clusters=11)
    assert "target_clusters" in str(exc_info.value).lower()


def test_reverse_search_config_title_similarity_threshold_bounds():
    """Test ReverseSearchConfig validates title_similarity_threshold is between 0.5 and 1.0."""
    # Valid bounds
    config = ReverseSearchConfig(title_similarity_threshold=0.5)
    assert config.title_similarity_threshold == 0.5
    config = ReverseSearchConfig(title_similarity_threshold=1.0)
    assert config.title_similarity_threshold == 1.0

    # Invalid: below lower bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(title_similarity_threshold=0.4)
    assert "title_similarity_threshold" in str(exc_info.value).lower()

    # Invalid: above upper bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(title_similarity_threshold=1.1)
    assert "title_similarity_threshold" in str(exc_info.value).lower()


def test_reverse_search_config_batch_pmid_fetch_size_bounds():
    """Test ReverseSearchConfig validates batch_pmid_fetch_size is between 1 and 200."""
    # Valid bounds
    config = ReverseSearchConfig(batch_pmid_fetch_size=1)
    assert config.batch_pmid_fetch_size == 1
    config = ReverseSearchConfig(batch_pmid_fetch_size=200)
    assert config.batch_pmid_fetch_size == 200

    # Invalid: below lower bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(batch_pmid_fetch_size=0)
    assert "batch_pmid_fetch_size" in str(exc_info.value).lower()

    # Invalid: above upper bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(batch_pmid_fetch_size=201)
    assert "batch_pmid_fetch_size" in str(exc_info.value).lower()


def test_reverse_search_config_cache_ttl_days_bounds():
    """Test ReverseSearchConfig validates cache_ttl_days is between 1 and 365."""
    # Valid bounds
    config = ReverseSearchConfig(cache_ttl_days=1)
    assert config.cache_ttl_days == 1
    config = ReverseSearchConfig(cache_ttl_days=365)
    assert config.cache_ttl_days == 365

    # Invalid: below lower bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(cache_ttl_days=0)
    assert "cache_ttl_days" in str(exc_info.value).lower()

    # Invalid: above upper bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(cache_ttl_days=366)
    assert "cache_ttl_days" in str(exc_info.value).lower()


def test_reverse_search_config_pmid_metadata_cache_days_bounds():
    """Test ReverseSearchConfig validates pmid_metadata_cache_days is between 1 and 365."""
    # Valid bounds
    config = ReverseSearchConfig(pmid_metadata_cache_days=1)
    assert config.pmid_metadata_cache_days == 1
    config = ReverseSearchConfig(pmid_metadata_cache_days=365)
    assert config.pmid_metadata_cache_days == 365

    # Invalid: below lower bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(pmid_metadata_cache_days=0)
    assert "pmid_metadata_cache_days" in str(exc_info.value).lower()

    # Invalid: above upper bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(pmid_metadata_cache_days=366)
    assert "pmid_metadata_cache_days" in str(exc_info.value).lower()


def test_reverse_search_config_max_results_per_query_bounds():
    """Test ReverseSearchConfig validates max_results_per_query is between 10 and 500."""
    # Valid bounds
    config = ReverseSearchConfig(max_results_per_query=10)
    assert config.max_results_per_query == 10
    config = ReverseSearchConfig(max_results_per_query=500)
    assert config.max_results_per_query == 500

    # Invalid: below lower bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(max_results_per_query=9)
    assert "max_results_per_query" in str(exc_info.value).lower()

    # Invalid: above upper bound
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(max_results_per_query=501)
    assert "max_results_per_query" in str(exc_info.value).lower()


# ReverseSearchError tests


def test_reverse_search_error_basic():
    """Test ReverseSearchError with just a message."""
    error = ReverseSearchError("Something went wrong")
    assert error.message == "Something went wrong"
    assert error.context == {}
    assert str(error) == "Something went wrong"


def test_reverse_search_error_with_context():
    """Test ReverseSearchError with message and context."""
    error = ReverseSearchError(
        "Query generation failed", context={"resources": 5, "extractor": "yake"}
    )
    assert error.message == "Query generation failed"
    assert error.context == {"resources": 5, "extractor": "yake"}
    assert "Query generation failed" in str(error)
    assert "Context:" in str(error)
    assert "resources" in str(error)


def test_reverse_search_error_as_exception():
    """Test ReverseSearchError can be raised and caught as an exception."""
    with pytest.raises(ReverseSearchError) as exc_info:
        raise ReverseSearchError("Test error", context={"foo": "bar"})
    assert exc_info.value.message == "Test error"
    assert exc_info.value.context == {"foo": "bar"}


def test_resource_parse_error():
    """Test ResourceParseError is a ReverseSearchError subclass."""
    error = ResourceParseError("Invalid JSON on line 5", context={"line": 5})
    assert isinstance(error, ReverseSearchError)
    assert error.message == "Invalid JSON on line 5"
    assert error.context == {"line": 5}


def test_query_generation_error():
    """Test QueryGenerationError is a ReverseSearchError subclass."""
    error = QueryGenerationError(
        "No keywords extracted", context={"resources": 3, "extractor": "yake"}
    )
    assert isinstance(error, ReverseSearchError)
    assert error.message == "No keywords extracted"
    assert error.context == {"resources": 3, "extractor": "yake"}


def test_matching_error():
    """Test MatchingError is a ReverseSearchError subclass."""
    error = MatchingError(
        "PMID extraction failed",
        context={"result_url": "https://example.com", "backend": "pubmed"},
    )
    assert isinstance(error, ReverseSearchError)
    assert error.message == "PMID extraction failed"
    assert error.context == {"result_url": "https://example.com", "backend": "pubmed"}


def test_error_subclasses_can_be_caught_as_base():
    """Test that error subclasses can be caught as ReverseSearchError."""
    with pytest.raises(ReverseSearchError):
        raise ResourceParseError("Parse failed")

    with pytest.raises(ReverseSearchError):
        raise QueryGenerationError("Generation failed")

    with pytest.raises(ReverseSearchError):
        raise MatchingError("Matching failed")
