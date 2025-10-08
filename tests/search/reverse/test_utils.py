"""
Tests for URL normalization utilities.

Validates comprehensive URL normalization including lowercase conversion,
trailing slash removal, tracking parameter cleanup, and fragment stripping.
"""

from interaction_finder.search.reverse.utils import TRACKING_PARAMS, normalize_url

# Lowercase and trailing slash tests


def test_normalize_url_lowercase_scheme_and_domain():
    """Lowercase scheme and domain components."""
    url = "HTTPS://Example.Com/page"
    normalized = normalize_url(url)
    assert normalized == "https://example.com/page"


def test_normalize_url_strip_trailing_slash():
    """Remove trailing slash from path."""
    url = "https://example.com/page/"
    normalized = normalize_url(url)
    assert normalized == "https://example.com/page"


def test_normalize_url_lowercase_and_strip_slash():
    """Lowercase and strip trailing slash together."""
    url = "HTTPS://Example.com/Page/"
    normalized = normalize_url(url)
    assert normalized == "https://example.com/Page"


def test_normalize_url_preserve_path_case():
    """Path case is preserved (only scheme and domain lowercased)."""
    url = "https://example.com/MyPage"
    normalized = normalize_url(url)
    # Path case should be preserved
    assert normalized == "https://example.com/MyPage"


# Tracking parameter tests


def test_normalize_url_remove_single_tracking_param():
    """Remove single tracking parameter."""
    url = "https://example.com/paper?id=123&utm_source=twitter"
    normalized = normalize_url(url)
    assert normalized == "https://example.com/paper?id=123"
    assert "utm_source" not in normalized


def test_normalize_url_remove_multiple_tracking_params():
    """Remove multiple tracking parameters."""
    url = "https://example.com/paper?id=123&utm_source=twitter&utm_campaign=spring&fbclid=abc"
    normalized = normalize_url(url)
    assert normalized == "https://example.com/paper?id=123"
    assert "utm_source" not in normalized
    assert "utm_campaign" not in normalized
    assert "fbclid" not in normalized


def test_normalize_url_preserve_non_tracking_params():
    """Preserve non-tracking parameters."""
    url = "https://example.com/paper?id=123&page=5&format=pdf"
    normalized = normalize_url(url)
    # All params should be preserved (order may vary due to sorting)
    assert "id=123" in normalized
    assert "page=5" in normalized
    assert "format=pdf" in normalized


def test_normalize_url_all_tracking_params():
    """Test that all defined tracking params are removed."""
    base_url = "https://example.com/page"
    # Build URL with all tracking params
    params = "&".join(f"{param}=value" for param in TRACKING_PARAMS)
    url_with_tracking = f"{base_url}?{params}&id=123"
    normalized = normalize_url(url_with_tracking)
    assert normalized == f"{base_url}?id=123"
    # Verify no tracking params remain
    for param in TRACKING_PARAMS:
        assert param not in normalized


def test_normalize_url_empty_tracking_param_values():
    """Remove tracking params even with empty values."""
    url = "https://example.com/page?utm_source=&id=123"
    normalized = normalize_url(url)
    assert normalized == "https://example.com/page?id=123"
    assert "utm_source" not in normalized


def test_normalize_url_only_tracking_params():
    """When all params are tracking params, result has no query string."""
    url = "https://example.com/page?utm_source=twitter&fbclid=abc&gclid=xyz"
    normalized = normalize_url(url)
    assert normalized == "https://example.com/page"
    assert "?" not in normalized


# Fragment tests


def test_normalize_url_remove_fragment():
    """Remove fragment identifier."""
    url = "https://example.com/page#section"
    normalized = normalize_url(url)
    assert normalized == "https://example.com/page"
    assert "#" not in normalized


def test_normalize_url_remove_fragment_with_params():
    """Remove fragment while preserving query params."""
    url = "https://example.com/page?id=123#section"
    normalized = normalize_url(url)
    assert normalized == "https://example.com/page?id=123"
    assert "#section" not in normalized


# Edge case tests


def test_normalize_url_empty_string():
    """Empty string input returns empty string."""
    assert normalize_url("") == ""


def test_normalize_url_whitespace_handling():
    """Strip leading/trailing whitespace."""
    url = "  https://example.com/page  "
    normalized = normalize_url(url)
    assert normalized == "https://example.com/page"


def test_normalize_url_whitespace_only():
    """Whitespace-only input returns empty string."""
    assert normalize_url("   ") == ""


def test_normalize_url_no_scheme():
    """Handle URL without scheme gracefully."""
    url = "example.com/page"
    normalized = normalize_url(url)
    # Should lowercase and strip trailing slash without crashing
    assert normalized == "example.com/page"


def test_normalize_url_malformed_url():
    """Handle malformed URL without exceptions."""
    url = "ht!tp://example.com/page"
    # Should not crash
    normalized = normalize_url(url)
    # Fallback behavior: lowercase and strip trailing slash
    assert isinstance(normalized, str)


def test_normalize_url_relative_path():
    """Handle relative path gracefully."""
    url = "/path/to/page"
    normalized = normalize_url(url)
    assert normalized == "/path/to/page"


# Idempotence tests


def test_normalize_url_idempotent_simple():
    """Normalization is idempotent for simple URL."""
    url = "https://example.com/page"
    normalized_once = normalize_url(url)
    normalized_twice = normalize_url(normalized_once)
    assert normalized_once == normalized_twice


def test_normalize_url_idempotent_complex():
    """Normalization is idempotent for complex URL."""
    url = "HTTPS://Example.com/Page/?utm_source=twitter&id=123&fbclid=abc#section"
    normalized_once = normalize_url(url)
    normalized_twice = normalize_url(normalized_once)
    assert normalized_once == normalized_twice


def test_normalize_url_idempotent_multiple_applications():
    """Normalization is idempotent for multiple applications."""
    url = "HTTPS://Example.com/Page/?utm_campaign=email"
    normalized_1 = normalize_url(url)
    normalized_2 = normalize_url(normalized_1)
    normalized_3 = normalize_url(normalized_2)
    assert normalized_1 == normalized_2 == normalized_3


# Query parameter ordering tests


def test_normalize_url_sorted_query_params():
    """Query parameters are sorted for determinism."""
    url1 = "https://example.com/page?b=2&a=1&c=3"
    url2 = "https://example.com/page?c=3&a=1&b=2"
    normalized1 = normalize_url(url1)
    normalized2 = normalize_url(url2)
    # Both should produce same result due to sorting
    assert normalized1 == normalized2
    # Parameters should be in alphabetical order
    assert normalized1 == "https://example.com/page?a=1&b=2&c=3"


# Comprehensive integration tests


def test_normalize_url_comprehensive_transformation():
    """Apply all transformations at once."""
    url = "HTTPS://Example.COM/Paper/?utm_source=twitter&utm_campaign=email&id=123&fbclid=xyz#abstract"
    normalized = normalize_url(url)
    expected = "https://example.com/Paper?id=123"
    assert normalized == expected


def test_normalize_url_real_world_pubmed():
    """Normalize realistic PubMed URL."""
    url = "https://pubmed.ncbi.nlm.nih.gov/12345678/?utm_source=newsletter"
    normalized = normalize_url(url)
    assert normalized == "https://pubmed.ncbi.nlm.nih.gov/12345678"
    assert "utm_source" not in normalized


def test_normalize_url_real_world_doi():
    """Normalize realistic DOI URL."""
    url = "https://doi.org/10.1234/example?utm_medium=social"
    normalized = normalize_url(url)
    assert normalized == "https://doi.org/10.1234/example"


def test_normalize_url_real_world_arxiv():
    """Normalize realistic arXiv URL."""
    url = "https://arxiv.org/abs/2301.00001?utm_campaign=research"
    normalized = normalize_url(url)
    assert normalized == "https://arxiv.org/abs/2301.00001"


# Parameter preservation tests


def test_normalize_url_preserve_special_chars_in_params():
    """Preserve special characters in parameter values."""
    url = "https://example.com/page?query=hello+world&id=123"
    normalized = normalize_url(url)
    # Should preserve encoded spaces and other special chars
    assert "query=" in normalized
    assert "id=123" in normalized


def test_normalize_url_preserve_empty_param_values():
    """Preserve non-tracking params with empty values."""
    url = "https://example.com/page?id=&format=pdf"
    normalized = normalize_url(url)
    assert "id=" in normalized
    assert "format=pdf" in normalized


# TRACKING_PARAMS constant tests


def test_tracking_params_contains_common_trackers():
    """TRACKING_PARAMS includes all common tracking parameters."""
    # Google Analytics
    assert "utm_source" in TRACKING_PARAMS
    assert "utm_medium" in TRACKING_PARAMS
    assert "utm_campaign" in TRACKING_PARAMS
    assert "utm_term" in TRACKING_PARAMS
    assert "utm_content" in TRACKING_PARAMS
    # Facebook
    assert "fbclid" in TRACKING_PARAMS
    assert "fb_action_ids" in TRACKING_PARAMS
    # Google Ads
    assert "gclid" in TRACKING_PARAMS
    assert "gclsrc" in TRACKING_PARAMS
    # Mailchimp
    assert "mc_cid" in TRACKING_PARAMS
    assert "mc_eid" in TRACKING_PARAMS
    # Generic
    assert "ref" in TRACKING_PARAMS
    assert "referrer" in TRACKING_PARAMS
    # Google Analytics client IDs
    assert "_ga" in TRACKING_PARAMS
    assert "_gl" in TRACKING_PARAMS


def test_tracking_params_is_set():
    """TRACKING_PARAMS is a set for efficient lookups."""
    assert isinstance(TRACKING_PARAMS, set)
