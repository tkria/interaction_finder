"""
Alignment-based utilities replacing the old bag-of-words approach.

These functions provide the main interface for sequence alignment-based
quote validation, with backward compatibility for existing code.
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from .alignment import SequenceAligner, AlignmentResult, ErrorType

if TYPE_CHECKING:
    from ...resources import Resource


# Global aligner instance for consistency
_default_aligner = SequenceAligner()


def compute_alignment_similarity(quote_text: str, resource: "Resource") -> float:
    """
    Compute sequence alignment similarity between quote and resource text.

    This is the primary similarity function using SequenceMatcher-based alignment.

    Args:
        quote_text: Quote text to analyze
        resource: Resource to align against

    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    alignment = _default_aligner.align_quote_to_resource(quote_text, resource)
    return alignment.similarity_ratio


def analyze_quote_alignment(quote_text: str, resource: "Resource") -> AlignmentResult:
    """
    Perform comprehensive quote alignment analysis.

    This replaces detect_split_quote() with much more detailed analysis.

    Args:
        quote_text: Quote text to analyze
        resource: Resource to align against

    Returns:
        Complete alignment analysis with error classification and suggestions
    """
    return _default_aligner.align_quote_to_resource(quote_text, resource)


def find_best_alignment_segments(quote_text: str, resource: "Resource") -> List[str]:
    """
    Find the best matching text segments using alignment analysis.

    This replaces find_longest_matching_subquote() with alignment-based approach.

    Args:
        quote_text: Quote text to find segments for
        resource: Resource to search within

    Returns:
        List of best matching text segments, ordered by quality
    """
    alignment = _default_aligner.align_quote_to_resource(quote_text, resource)

    # Extract suggestion texts from alignment corrections
    segments = []
    for correction in alignment.corrections:
        if correction.text and correction.text not in segments:
            segments.append(correction.text)

    # If no corrections available, try to extract from matching blocks
    if not segments and alignment.matching_blocks:
        for block in alignment.matching_blocks:
            if block.length >= 2:  # Minimum meaningful length
                segment_text = " ".join(
                    alignment.doc_words[block.doc_start : block.doc_end]
                )
                if segment_text not in segments:
                    segments.append(segment_text)

    return segments


# Backward compatibility functions
def compute_bag_of_words_similarity(text1: str, text2: str) -> float:
    """
    DEPRECATED: Compute bag-of-words similarity for backward compatibility.

    This function is maintained for backward compatibility but internally
    uses sequence alignment for better accuracy.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    # Create a temporary resource-like object for alignment
    # This is a simplified approach for compatibility
    from difflib import SequenceMatcher

    words1 = text1.lower().split()
    words2 = text2.lower().split()

    matcher = SequenceMatcher(None, words1, words2)
    return matcher.ratio()


def detect_split_quote(
    quote_text: str, resource: "Resource"
) -> Optional[Dict[str, Any]]:
    """
    DEPRECATED: Detect split quotes using alignment analysis.

    This function is maintained for backward compatibility but uses the new
    alignment system for much more accurate split detection.

    Args:
        quote_text: Quote text to analyze
        resource: Resource to search within

    Returns:
        Dict with split info if detected, None otherwise
    """
    alignment = analyze_quote_alignment(quote_text, resource)

    if alignment.error_type == ErrorType.SPLIT_QUOTE or alignment.is_likely_split:
        # Convert alignment result to old format for compatibility
        if len(alignment.matching_blocks) >= 2:
            # Extract prefix from first block
            first_block = alignment.matching_blocks[0]
            prefix_words = alignment.doc_words[
                first_block.doc_start : first_block.doc_end
            ]
            prefix = " ".join(prefix_words)

            # Extract suffix from last block
            last_block = alignment.matching_blocks[-1]
            suffix_words = alignment.doc_words[
                last_block.doc_start : last_block.doc_end
            ]
            suffix = " ".join(suffix_words)

            # Try to create ResourceQuote objects for backward compatibility
            prefix_quote = None
            suffix_quote = None
            try:
                prefix_quote = resource.quote(prefix)
            except:
                pass
            try:
                suffix_quote = resource.quote(suffix)
            except:
                pass

            result = {
                "prefix": prefix,
                "suffix": suffix,
                "alignment_result": alignment,  # Extra info for enhanced functionality
            }

            # Add ResourceQuote objects if they were successfully created
            if prefix_quote:
                result["prefix_quote"] = prefix_quote
            if suffix_quote:
                result["suffix_quote"] = suffix_quote

            return result

    return None


def find_longest_matching_subquote(
    quote_text: str, resource: "Resource"
) -> Optional[str]:
    """
    DEPRECATED: Find longest matching substring using alignment analysis.

    This function is maintained for backward compatibility but uses alignment
    analysis for better results.

    Args:
        quote_text: Quote text to find substring for
        resource: Resource to search within

    Returns:
        Longest matching substring or None if no match found
    """
    segments = find_best_alignment_segments(quote_text, resource)

    # Return the longest segment for compatibility
    if segments:
        return max(segments, key=len)

    return None


# Enhanced utilities that leverage full alignment power
def get_alignment_details(quote_text: str, resource: "Resource") -> Dict[str, Any]:
    """
    Get detailed alignment information for debugging and analysis.

    Args:
        quote_text: Quote text to analyze
        resource: Resource to align against

    Returns:
        Dictionary with comprehensive alignment details
    """
    alignment = analyze_quote_alignment(quote_text, resource)

    return {
        "similarity_ratio": alignment.similarity_ratio,
        "error_type": alignment.error_type.value,
        "is_contiguous": alignment.contiguous,
        "coverage_ratio": alignment.coverage_ratio,
        "total_matching_words": alignment.total_matching_words,
        "num_matching_blocks": len(alignment.matching_blocks),
        "num_gaps": len(alignment.gaps),
        "quality_metrics": alignment.quality_metrics,
        "is_high_quality": alignment.is_high_quality,
        "is_likely_split": alignment.is_likely_split,
        "corrections_available": len(alignment.corrections),
        "best_correction": alignment.corrections[0].text
        if alignment.corrections
        else None,
    }


def classify_quote_error(quote_text: str, resource: "Resource") -> str:
    """
    Classify the type of quote error using alignment analysis.

    Args:
        quote_text: Quote text to analyze
        resource: Resource to align against

    Returns:
        Error type classification string
    """
    alignment = analyze_quote_alignment(quote_text, resource)
    return alignment.error_type.value


def get_correction_suggestions(
    quote_text: str, resource: "Resource", max_suggestions: int = 3
) -> List[Dict[str, Any]]:
    """
    Get ranked correction suggestions with detailed metrics.

    Args:
        quote_text: Quote text to get corrections for
        resource: Resource to align against
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of suggestion dictionaries with text, confidence, and explanation
    """
    alignment = analyze_quote_alignment(quote_text, resource)

    suggestions = []
    for correction in alignment.corrections[:max_suggestions]:
        suggestions.append(
            {
                "text": correction.text,
                "confidence": correction.confidence,
                "alignment_score": correction.alignment_score,
                "contiguity_score": correction.contiguity_score,
                "boundary_quality": correction.boundary_quality,
                "explanation": correction.explanation,
            }
        )

    return suggestions


def is_quote_high_quality(quote_text: str, resource: "Resource") -> bool:
    """
    Check if a quote represents a high-quality alignment.

    Args:
        quote_text: Quote text to evaluate
        resource: Resource to align against

    Returns:
        True if the quote is high quality
    """
    alignment = analyze_quote_alignment(quote_text, resource)
    return alignment.is_high_quality
