"""URL utilities for the fetcher package."""

import hashlib
from urllib.parse import urlparse, urlunparse

# URL hashing constants
URL_HASH_LENGTH = 16


def normalize_url(url: str) -> str:
    """Normalize URL by removing fragment and other client-side only components."""
    parsed = urlparse(url)
    # Remove fragment (everything after #) as it's client-side only
    return urlunparse(parsed._replace(fragment=""))


def url_to_hash(url: str) -> str:
    """Convert URL to a filesystem-safe hash (backward compatibility)."""
    normalized_url = normalize_url(url)
    return hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:URL_HASH_LENGTH]


def url_to_hash_base36(url: str) -> str:
    """Convert URL to a base36 hash for compact representation."""
    normalized_url = normalize_url(url)
    # Use deterministic SHA256 hash instead of Python's non-deterministic hash()
    hash_bytes = hashlib.sha256(normalized_url.encode("utf-8")).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")  # Use first 8 bytes

    # Manual base36 conversion
    if hash_int == 0:
        return "0"

    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while hash_int:
        result = digits[hash_int % 36] + result
        hash_int //= 36
    return result
