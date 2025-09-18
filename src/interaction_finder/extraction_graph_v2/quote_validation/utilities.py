"""
Text matching and analysis utilities for quote validation.
"""

import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...resources import Resource


def find_longest_matching_prefix(
    quote_text: str, resource: "Resource"
) -> Optional[str]:
    """
    Find the longest matching prefix by searching all occurrences of the first 3 words.

    Args:
        quote_text: Quote text to find prefix for
        resource: Resource to search within

    Returns:
        Longest matching prefix string or None if no match found
    """
    words = quote_text.strip().split()
    if len(words) < 3:
        return None

    # Search for all occurrences of the first 3 words
    first_three = " ".join(words[:3])
    longest_match = None

    try:
        # Find all locations where first 3 words appear
        first_three_quote = resource.quote(first_three)

        # For each location, try to expand as far as possible
        for start_pos, _ in first_three_quote.spans:
            # Try expanding word by word
            for end_word in range(3, len(words) + 1):
                candidate = " ".join(words[:end_word])
                try:
                    resource.quote(candidate)
                    longest_match = candidate
                except:
                    break  # Can't extend further from this position

        return longest_match

    except:
        return None


def find_longest_matching_suffix(
    quote_text: str, resource: "Resource"
) -> Optional[str]:
    """
    Find the longest matching suffix by searching all occurrences of the last 3 words.

    Args:
        quote_text: Quote text to find suffix for
        resource: Resource to search within

    Returns:
        Longest matching suffix string or None if no match found
    """
    words = quote_text.strip().split()
    if len(words) < 3:
        return None

    # Search for all occurrences of the last 3 words
    last_three = " ".join(words[-3:])
    longest_match = None

    try:
        # Find all locations where last 3 words appear
        last_three_quote = resource.quote(last_three)

        # For each location, try to expand backwards as far as possible
        for _, end_pos in last_three_quote.spans:
            # Try expanding word by word backwards
            for start_word in range(len(words) - 3, -1, -1):
                candidate = " ".join(words[start_word:])
                try:
                    resource.quote(candidate)
                    longest_match = candidate
                except:
                    break  # Can't extend further from this position

        return longest_match

    except:
        return None


def compute_bag_of_words_similarity(text1: str, text2: str) -> float:
    """
    Compute bag-of-words similarity between two texts.

    Uses Jaccard similarity: |intersection| / |union|

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Compute Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def detect_split_quote(quote_text: str, resource: "Resource") -> Optional[dict]:
    """
    Detect if a quote combines text from different locations (split quote).

    Uses binary search to find the longest matching prefix, then checks if
    the remaining suffix exists separately in the document.

    Args:
        quote_text: Quote text to analyze
        resource: Resource to search within

    Returns:
        Dict with prefix/suffix info if split detected, None otherwise
    """
    # Try to find the longest matching prefix
    longest_prefix = find_longest_matching_prefix(quote_text, resource)

    if not longest_prefix:
        return None

    # Get the remaining suffix after removing the prefix
    prefix_words = longest_prefix.split()
    all_words = quote_text.split()

    if len(prefix_words) >= len(all_words):
        return None  # Prefix covers the entire quote

    suffix_words = all_words[len(prefix_words) :]
    suffix_text = " ".join(suffix_words)

    # Check if the suffix exists separately in the document
    try:
        resource.quote(suffix_text)

        # Check that prefix and suffix are not contiguous
        # (if they were contiguous, the original quote should have worked)
        try:
            full_quote = resource.quote(quote_text)
            # If we can find the full quote, it's not actually split
            return None
        except:
            # Suffix exists separately AND full quote doesn't exist = split quote
            return {"prefix": longest_prefix, "suffix": suffix_text}

    except:
        # Suffix doesn't exist separately
        return None


def find_longest_matching_subquote(
    quote_text: str, resource: "Resource"
) -> Optional[str]:
    """
    Find the longest contiguous substring that exists in the resource.

    This is useful for finding partial matches when the full quote fails.

    Args:
        quote_text: Quote text to find substring for
        resource: Resource to search within

    Returns:
        Longest matching substring or None if no match found
    """
    words = quote_text.strip().split()
    if len(words) < 2:
        return None

    longest_match = None
    max_length = 0

    # Try all possible substrings, starting with longer ones
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):  # At least 2 words
            candidate = " ".join(words[start:end])
            try:
                resource.quote(candidate)
                if len(candidate) > max_length:
                    longest_match = candidate
                    max_length = len(candidate)
            except:
                continue

    return longest_match
