"""Tests for CLI fetch command cache clearing functionality."""

import pytest
from interaction_finder.cli_fetch import match_url_pattern


class TestMatchUrlPattern:
    """Tests for URL pattern matching logic."""

    def test_exact_substring_match(self):
        """Test exact substring matching is case-insensitive."""
        assert match_url_pattern("https://pubmed.ncbi.nlm.nih.gov/123/", "pubmed")
        assert match_url_pattern(
            "https://pubmed.ncbi.nlm.nih.gov/123/", "ncbi.nlm.nih.gov"
        )
        assert match_url_pattern("https://example.com/path", "example.com")

    def test_case_insensitive(self):
        """Test pattern matching is case-insensitive."""
        assert match_url_pattern("https://PubMed.NCBI.nlm.nih.gov/123/", "pubmed")
        assert match_url_pattern("https://pubmed.ncbi.nlm.nih.gov/123/", "PUBMED")
        assert match_url_pattern("https://Example.COM/path", "example.com")

    def test_prefix_match(self):
        """Test pattern matches URL prefixes."""
        assert match_url_pattern("https://pubmed.ncbi.nlm.nih.gov/123/", "https://")
        assert match_url_pattern(
            "https://pubmed.ncbi.nlm.nih.gov/123/", "https://pubmed"
        )

    def test_glob_pattern_with_star(self):
        """Test glob pattern matching with asterisks."""
        assert match_url_pattern("https://pubmed.ncbi.nlm.nih.gov/12345/", "*123*")
        assert match_url_pattern("https://pubmed.ncbi.nlm.nih.gov/abc123def/", "*123*")
        assert match_url_pattern("https://example.com/path", "https://example.*")
        assert not match_url_pattern("https://example.com/path", "*foo*")

    def test_question_mark_not_glob(self):
        """Test that ? in pattern is NOT treated as glob wildcard (for URL query params)."""
        # Question marks should be treated literally, not as glob wildcards
        assert match_url_pattern("https://example.com/path?query=value", "?query=value")
        assert match_url_pattern("https://example.com/path?foo=bar", "?foo")
        # This should NOT match like a glob (? = single char wildcard)
        assert not match_url_pattern("https://example.com/abc", "?bc")

    def test_no_match(self):
        """Test patterns that should not match."""
        assert not match_url_pattern(
            "https://pubmed.ncbi.nlm.nih.gov/123/", "example.com"
        )
        assert not match_url_pattern("https://example.com/path", "pubmed")

    def test_optional_https_prefix(self):
        """Test that https:// prefix is optional in patterns."""
        # Both with and without https:// should work
        assert match_url_pattern("https://pubmed.ncbi.nlm.nih.gov/123/", "pubmed")
        assert match_url_pattern(
            "https://pubmed.ncbi.nlm.nih.gov/123/", "https://pubmed"
        )

    def test_partial_domain_match(self):
        """Test matching partial domain names."""
        assert match_url_pattern("https://pmc.ncbi.nlm.nih.gov/articles/123", "pmc")
        assert match_url_pattern(
            "https://pmc.ncbi.nlm.nih.gov/articles/123", "pmc.ncbi"
        )
        assert match_url_pattern(
            "https://www.ncbi.nlm.nih.gov/pmc/articles/123", "ncbi"
        )

    def test_path_matching(self):
        """Test matching URL paths."""
        assert match_url_pattern("https://example.com/foo/bar", "/foo/")
        assert match_url_pattern("https://example.com/foo/bar", "/bar")
        assert match_url_pattern("https://example.com/api/v1/users", "/api/v1")

    def test_glob_with_full_url(self):
        """Test glob patterns with full URL structure."""
        assert match_url_pattern(
            "https://pubmed.ncbi.nlm.nih.gov/123456/", "https://pubmed*/123*"
        )
        assert match_url_pattern("https://example.com/path123", "*/path*")
        assert not match_url_pattern(
            "https://example.com/other", "https://pubmed*/123*"
        )
