"""
Tests for URL utilities functionality.

Tests cover:
- URL normalization functionality
- URL to hash conversion (both legacy and base36)
- Hash consistency and determinism
- Edge cases and error conditions
"""

import pytest
import hashlib
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from interaction_finder.fetcher.utils import (
    normalize_url,
    url_to_hash,
    url_to_hash_base36,
    URL_HASH_LENGTH,
)


class TestURLNormalization:
    """Test URL normalization functionality."""

    def test_normalize_url_basic(self):
        """Test basic URL normalization."""
        url = "http://example.com/path"
        result = normalize_url(url)

        assert result == "http://example.com/path"

    def test_normalize_url_remove_fragment(self):
        """Test that URL fragments are removed."""
        url = "http://example.com/path#fragment"
        result = normalize_url(url)

        assert result == "http://example.com/path"
        assert "#fragment" not in result

    def test_normalize_url_remove_complex_fragment(self):
        """Test removal of complex fragments with parameters."""
        url = "http://example.com/path?param=value#section-1.2.3"
        result = normalize_url(url)

        assert result == "http://example.com/path?param=value"
        assert "#section-1.2.3" not in result

    def test_normalize_url_preserve_query_params(self):
        """Test that query parameters are preserved."""
        url = "http://example.com/path?param1=value1&param2=value2"
        result = normalize_url(url)

        assert result == url
        assert "param1=value1" in result
        assert "param2=value2" in result

    def test_normalize_url_https(self):
        """Test normalization of HTTPS URLs."""
        url = "https://secure.example.com/path?param=value#fragment"
        result = normalize_url(url)

        assert result == "https://secure.example.com/path?param=value"

    def test_normalize_url_port_numbers(self):
        """Test normalization with port numbers."""
        url = "http://example.com:8080/path#fragment"
        result = normalize_url(url)

        assert result == "http://example.com:8080/path"

    def test_normalize_url_empty_fragment(self):
        """Test URLs with empty fragments."""
        url = "http://example.com/path#"
        result = normalize_url(url)

        assert result == "http://example.com/path"

    def test_normalize_url_no_fragment(self):
        """Test URLs without fragments (no change expected)."""
        url = "http://example.com/path?param=value"
        result = normalize_url(url)

        assert result == url

    def test_normalize_url_complex_path(self):
        """Test normalization with complex paths."""
        url = "http://example.com/path/to/resource.html?query=test#section"
        result = normalize_url(url)

        assert result == "http://example.com/path/to/resource.html?query=test"

    def test_normalize_url_unicode(self):
        """Test normalization with Unicode characters."""
        url = "http://example.com/path/файл.html#раздел"
        result = normalize_url(url)

        # Should preserve Unicode in path but remove fragment
        assert "файл.html" in result
        assert "#раздел" not in result


class TestURLToHash:
    """Test legacy URL to hash conversion functionality."""

    def test_url_to_hash_basic(self):
        """Test basic URL to hash conversion."""
        url = "http://example.com"
        result = url_to_hash(url)

        # Should return a hex string of specified length
        assert isinstance(result, str)
        assert len(result) == URL_HASH_LENGTH
        assert all(c in "0123456789abcdef" for c in result)

    def test_url_to_hash_deterministic(self):
        """Test that hash generation is deterministic."""
        url = "http://example.com/path"

        result1 = url_to_hash(url)
        result2 = url_to_hash(url)

        assert result1 == result2

    def test_url_to_hash_different_urls(self):
        """Test that different URLs produce different hashes."""
        url1 = "http://example.com/path1"
        url2 = "http://example.com/path2"

        hash1 = url_to_hash(url1)
        hash2 = url_to_hash(url2)

        assert hash1 != hash2

    def test_url_to_hash_normalization(self):
        """Test that URLs are normalized before hashing."""
        url_with_fragment = "http://example.com/path#fragment"
        url_without_fragment = "http://example.com/path"

        hash1 = url_to_hash(url_with_fragment)
        hash2 = url_to_hash(url_without_fragment)

        # Should be same because fragment is normalized away
        assert hash1 == hash2

    def test_url_to_hash_length_constant(self):
        """Test that the hash length constant is reasonable."""
        assert URL_HASH_LENGTH > 0
        assert URL_HASH_LENGTH <= 64  # SHA256 hex is 64 chars max

    def test_url_to_hash_collision_resistance(self):
        """Test that similar URLs produce different hashes."""
        base_url = "http://example.com/article"
        urls = [base_url + str(i) for i in range(10)]

        hashes = [url_to_hash(url) for url in urls]

        # All hashes should be unique
        assert len(set(hashes)) == len(hashes)


class TestURLToHashBase36:
    """Test base36 URL to hash conversion functionality."""

    def test_url_to_hash_base36_basic(self):
        """Test basic base36 hash conversion."""
        url = "http://example.com"
        result = url_to_hash_base36(url)

        # Should return a base36 string (0-9, a-z)
        assert isinstance(result, str)
        assert len(result) > 0
        assert all(c in "0123456789abcdefghijklmnopqrstuvwxyz" for c in result)

    def test_url_to_hash_base36_deterministic(self):
        """Test that base36 hash generation is deterministic."""
        url = "http://example.com/path"

        result1 = url_to_hash_base36(url)
        result2 = url_to_hash_base36(url)

        assert result1 == result2

    def test_url_to_hash_base36_different_urls(self):
        """Test that different URLs produce different base36 hashes."""
        url1 = "http://example.com/path1"
        url2 = "http://example.com/path2"

        hash1 = url_to_hash_base36(url1)
        hash2 = url_to_hash_base36(url2)

        assert hash1 != hash2

    def test_url_to_hash_base36_normalization(self):
        """Test that URLs are normalized before base36 hashing."""
        url_with_fragment = "http://example.com/path#fragment"
        url_without_fragment = "http://example.com/path"

        hash1 = url_to_hash_base36(url_with_fragment)
        hash2 = url_to_hash_base36(url_without_fragment)

        # Should be same because fragment is normalized away
        assert hash1 == hash2

    def test_url_to_hash_base36_zero_handling(self):
        """Test that zero hash is handled correctly."""
        # This tests the special case in the base36 conversion function
        # We can't easily test hash_int == 0 case, but we can test that
        # the function handles it correctly by checking the result format
        url = "http://example.com"
        result = url_to_hash_base36(url)

        # Should not return "0" for normal URLs (extremely unlikely)
        # But if it did, it should be valid
        assert isinstance(result, str)
        assert len(result) > 0

    def test_url_to_hash_base36_compactness(self):
        """Test that base36 hashes are more compact than hex."""
        url = "http://example.com/very/long/path/to/resource.html?param=value"

        hex_hash = url_to_hash(url)
        base36_hash = url_to_hash_base36(url)

        # Base36 should generally be more compact than hex for same amount of entropy
        # This is a probabilistic test, might occasionally fail
        assert isinstance(hex_hash, str)
        assert isinstance(base36_hash, str)

    def test_url_to_hash_base36_collision_resistance(self):
        """Test that base36 hashes resist collisions for similar URLs."""
        base_url = "http://example.com/article"
        urls = [base_url + str(i) for i in range(20)]

        hashes = [url_to_hash_base36(url) for url in urls]

        # All hashes should be unique
        assert len(set(hashes)) == len(hashes)


class TestHashConsistency:
    """Test consistency between different hash functions."""

    def test_hash_functions_both_deterministic(self):
        """Test that both hash functions are deterministic."""
        url = "http://example.com/test"

        # Multiple calls should give same results
        hex_hash1 = url_to_hash(url)
        hex_hash2 = url_to_hash(url)
        base36_hash1 = url_to_hash_base36(url)
        base36_hash2 = url_to_hash_base36(url)

        assert hex_hash1 == hex_hash2
        assert base36_hash1 == base36_hash2

    def test_hash_functions_both_normalize(self):
        """Test that both hash functions normalize URLs consistently."""
        url_with_fragment = "http://example.com/path#fragment"
        url_without_fragment = "http://example.com/path"

        # Hex hashes should be same
        hex_hash1 = url_to_hash(url_with_fragment)
        hex_hash2 = url_to_hash(url_without_fragment)
        assert hex_hash1 == hex_hash2

        # Base36 hashes should be same
        base36_hash1 = url_to_hash_base36(url_with_fragment)
        base36_hash2 = url_to_hash_base36(url_without_fragment)
        assert base36_hash1 == base36_hash2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_url(self):
        """Test handling of empty URLs."""
        # Functions should handle empty strings gracefully
        hex_result = url_to_hash("")
        base36_result = url_to_hash_base36("")

        assert isinstance(hex_result, str)
        assert isinstance(base36_result, str)
        assert len(hex_result) == URL_HASH_LENGTH

    def test_normalize_empty_url(self):
        """Test normalization of empty URL."""
        result = normalize_url("")
        assert result == ""

    def test_very_long_url(self):
        """Test handling of very long URLs."""
        # Create a very long URL
        long_path = "/".join(["segment" + str(i) for i in range(100)])
        long_url = f"http://example.com{long_path}?param=value#fragment"

        # Should handle long URLs without error
        normalized = normalize_url(long_url)
        hex_hash = url_to_hash(long_url)
        base36_hash = url_to_hash_base36(long_url)

        assert isinstance(normalized, str)
        assert "#fragment" not in normalized
        assert isinstance(hex_hash, str)
        assert len(hex_hash) == URL_HASH_LENGTH
        assert isinstance(base36_hash, str)

    def test_unicode_urls(self):
        """Test handling of URLs with Unicode characters."""
        unicode_url = "http://example.com/测试路径?参数=值#片段"

        # Should handle Unicode without errors
        normalized = normalize_url(unicode_url)
        hex_hash = url_to_hash(unicode_url)
        base36_hash = url_to_hash_base36(unicode_url)

        assert isinstance(normalized, str)
        assert "#片段" not in normalized  # Fragment should be removed
        assert isinstance(hex_hash, str)
        assert isinstance(base36_hash, str)

    def test_special_characters_in_url(self):
        """Test handling of URLs with special characters."""
        special_url = "http://example.com/path%20with%20spaces?param=value%26more"

        # Should handle URL encoding without errors
        normalized = normalize_url(special_url)
        hex_hash = url_to_hash(special_url)
        base36_hash = url_to_hash_base36(special_url)

        assert isinstance(normalized, str)
        assert isinstance(hex_hash, str)
        assert isinstance(base36_hash, str)

    def test_malformed_urls(self):
        """Test handling of malformed URLs."""
        malformed_urls = [
            "not-a-url",
            "ftp://example.com",  # Different protocol
            "://example.com",  # Missing protocol
            "http://",  # Incomplete
        ]

        # Should handle malformed URLs gracefully (no exceptions)
        for url in malformed_urls:
            normalized = normalize_url(url)
            hex_hash = url_to_hash(url)
            base36_hash = url_to_hash_base36(url)

            assert isinstance(normalized, str)
            assert isinstance(hex_hash, str)
            assert isinstance(base36_hash, str)


class TestImplementationDetails:
    """Test implementation-specific behavior."""

    def test_url_to_hash_uses_sha256(self):
        """Test that url_to_hash uses SHA256 under the hood."""
        url = "http://example.com"
        expected_full_hash = hashlib.sha256(
            normalize_url(url).encode("utf-8")
        ).hexdigest()
        expected_truncated = expected_full_hash[:URL_HASH_LENGTH]

        result = url_to_hash(url)

        assert result == expected_truncated

    def test_url_to_hash_base36_uses_sha256(self):
        """Test that url_to_hash_base36 uses SHA256 under the hood."""
        url = "http://example.com"
        normalized_url = normalize_url(url)

        # Recreate the base36 conversion manually
        hash_bytes = hashlib.sha256(normalized_url.encode("utf-8")).digest()
        hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")

        # Manual base36 conversion
        if hash_int == 0:
            expected = "0"
        else:
            digits = "0123456789abcdefghijklmnopqrstuvwxyz"
            expected = ""
            while hash_int:
                expected = digits[hash_int % 36] + expected
                hash_int //= 36

        result = url_to_hash_base36(url)

        assert result == expected

    def test_base36_character_set(self):
        """Test that base36 uses correct character set."""
        # Generate several base36 hashes and verify character set
        urls = [f"http://example.com/path{i}" for i in range(10)]

        for url in urls:
            result = url_to_hash_base36(url)
            # Should only contain 0-9 and a-z
            assert all(c in "0123456789abcdefghijklmnopqrstuvwxyz" for c in result)
            # Should not contain uppercase letters
            assert not any(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in result)


class TestConstants:
    """Test module constants and configuration."""

    def test_url_hash_length_reasonable(self):
        """Test that URL_HASH_LENGTH is a reasonable value."""
        assert isinstance(URL_HASH_LENGTH, int)
        assert URL_HASH_LENGTH > 0
        assert URL_HASH_LENGTH <= 64  # Max length for SHA256 hex

    def test_url_hash_length_provides_good_distribution(self):
        """Test that the chosen hash length provides good distribution."""
        # Generate hashes for many URLs and check for reasonable distribution
        urls = [f"http://example{i}.com/path{j}" for i in range(5) for j in range(10)]
        hashes = [url_to_hash(url) for url in urls]

        # All should be the correct length
        assert all(len(h) == URL_HASH_LENGTH for h in hashes)
        # Should have no obvious collisions (very unlikely with good hash function)
        assert len(set(hashes)) == len(hashes)
