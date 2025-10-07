"""
Sequence alignment-based quote validation and correction system.

This package provides:
- Word-level sequence alignment using difflib.SequenceMatcher
- Precision error classification (split quotes, substitutions, etc.)
- Intelligent correction suggestions based on alignment quality
- Rich error analysis with detailed alignment information
- Backward compatibility with existing validation interfaces
"""

# New alignment-based core system
from .alignment import (
    SequenceAligner,
    AlignmentResult,
    MatchingBlock,
    CorrectionSuggestion,
    ErrorType,
    Gap,
)

# Enhanced validators using alignment system
from .validators import QuoteValidator
from .corrections import QuoteCorrector
from .error_messages import build_retry_message

# Alignment-based utilities (replacing old bag-of-words approach)
from .alignment_utilities import (
    compute_alignment_similarity,
    analyze_quote_alignment,
    find_best_alignment_segments,
    get_alignment_details,
    classify_quote_error,
    get_correction_suggestions,
    is_quote_high_quality,
    # Backward compatibility aliases
    compute_bag_of_words_similarity,
    detect_split_quote,
    find_longest_matching_subquote,
)

__all__ = [
    # Core alignment system
    "SequenceAligner",
    "AlignmentResult",
    "MatchingBlock",
    "CorrectionSuggestion",
    "ErrorType",
    "Gap",
    # Enhanced validators
    "QuoteValidator",
    "QuoteCorrector",
    "build_retry_message",
    # Alignment utilities
    "compute_alignment_similarity",
    "analyze_quote_alignment",
    "find_best_alignment_segments",
    "get_alignment_details",
    "classify_quote_error",
    "get_correction_suggestions",
    "is_quote_high_quality",
    # Backward compatibility
    "compute_bag_of_words_similarity",
    "detect_split_quote",
    "find_longest_matching_subquote",
]
