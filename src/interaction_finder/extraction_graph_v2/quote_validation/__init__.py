"""
Quote validation and correction utilities for entity extraction.

This package handles:
- Quote text validation against source documents
- Automatic correction of high-confidence quote errors
- Error message generation for LLM feedback
- Text matching and similarity utilities
"""

from .validators import QuoteValidator
from .corrections import QuoteCorrector
from .error_messages import build_retry_message
from .utilities import (
    find_longest_matching_prefix,
    find_longest_matching_suffix,
    compute_bag_of_words_similarity,
    detect_split_quote,
    find_longest_matching_subquote,
)

__all__ = [
    "QuoteValidator",
    "QuoteCorrector",
    "build_retry_message",
    "find_longest_matching_prefix",
    "find_longest_matching_suffix",
    "compute_bag_of_words_similarity",
    "detect_split_quote",
    "find_longest_matching_subquote",
]
