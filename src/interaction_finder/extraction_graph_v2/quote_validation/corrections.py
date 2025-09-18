"""
Quote correction and auto-accept mechanisms.
"""

import logging
import re
from typing import List, Optional, Tuple, TYPE_CHECKING

from .utilities import (
    find_longest_matching_prefix,
    find_longest_matching_suffix,
    find_longest_matching_subquote,
)

if TYPE_CHECKING:
    from ...resources import Resource

logger = logging.getLogger(__name__)


class QuoteCorrector:
    """Handles quote correction and auto-accept logic."""

    def __init__(self, auto_accept_threshold: float = 85.0):
        """
        Initialize quote corrector.

        Args:
            auto_accept_threshold: Percentage threshold for auto-accepting corrections
        """
        self.auto_accept_threshold = auto_accept_threshold

    def generate_suggestions(
        self, quote_text: str, resource: "Resource"
    ) -> List[Tuple[str, float]]:
        """
        Generate correction suggestions for a failed quote.

        Args:
            quote_text: The failed quote text
            resource: Resource to search within

        Returns:
            List of (suggestion, confidence_percentage) tuples
        """
        suggestions = []

        # Try to find the longest matching subquote
        longest_match = find_longest_matching_subquote(quote_text, resource)
        if longest_match:
            confidence = self._calculate_match_percentage(quote_text, longest_match)

            # Try to extend the match in both directions for better context
            extended_suggestions = self._extend_match_for_context(
                longest_match, resource
            )

            for extended in extended_suggestions:
                ext_confidence = self._calculate_match_percentage(quote_text, extended)
                suggestions.append((extended, ext_confidence))

            # Always include the basic longest match
            suggestions.append((longest_match, confidence))

        # Try prefix-based corrections
        prefix_match = find_longest_matching_prefix(quote_text, resource)
        if prefix_match:
            confidence = self._calculate_match_percentage(quote_text, prefix_match)

            # Try to extend prefix match for better context
            extended_prefix = self._extend_match_for_context(prefix_match, resource)
            for extended in extended_prefix:
                ext_confidence = self._calculate_match_percentage(quote_text, extended)
                suggestions.append((extended, ext_confidence))

            suggestions.append((prefix_match, confidence))

        # Try suffix-based corrections
        suffix_match = find_longest_matching_suffix(quote_text, resource)
        if suffix_match:
            confidence = self._calculate_match_percentage(quote_text, suffix_match)

            # Try to extend suffix match for better context
            extended_suffix = self._extend_match_for_context(suffix_match, resource)
            for extended in extended_suffix:
                ext_confidence = self._calculate_match_percentage(quote_text, extended)
                suggestions.append((extended, ext_confidence))

            suggestions.append((suffix_match, confidence))

        # Remove duplicates and sort by confidence
        unique_suggestions = {}
        for suggestion, confidence in suggestions:
            cleaned = self._clean_suggestion(suggestion)
            if cleaned and len(cleaned.strip()) > 10:  # Minimum length filter
                if (
                    cleaned not in unique_suggestions
                    or confidence > unique_suggestions[cleaned]
                ):
                    unique_suggestions[cleaned] = confidence

        # Convert back to list and sort by confidence
        result = [(text, conf) for text, conf in unique_suggestions.items()]
        result.sort(key=lambda x: x[1], reverse=True)

        return result[:3]  # Return top 3 suggestions

    def should_auto_accept(self, suggestions: List[Tuple[str, float]]) -> bool:
        """
        Check if the best suggestion should be auto-accepted.

        Args:
            suggestions: List of (suggestion, confidence) tuples

        Returns:
            True if should auto-accept the best suggestion
        """
        if not suggestions:
            return False

        best_confidence = suggestions[0][1]
        return best_confidence >= self.auto_accept_threshold

    def get_best_suggestion(
        self, suggestions: List[Tuple[str, float]]
    ) -> Optional[str]:
        """
        Get the best suggestion text.

        Args:
            suggestions: List of (suggestion, confidence) tuples

        Returns:
            Best suggestion text or None
        """
        if not suggestions:
            return None
        return suggestions[0][0]

    def _calculate_match_percentage(self, original: str, suggestion: str) -> float:
        """Calculate percentage match between original and suggestion."""
        if not original.strip() or not suggestion.strip():
            return 0.0

        orig_words = original.lower().split()
        sugg_words = suggestion.lower().split()

        # Count matching words
        matching_words = 0
        orig_set = set(orig_words)
        sugg_set = set(sugg_words)

        # Use intersection for word-level matching
        matching_words = len(orig_set.intersection(sugg_set))
        total_unique_words = len(orig_set.union(sugg_set))

        if total_unique_words == 0:
            return 0.0

        # Calculate percentage based on word overlap
        return (matching_words / len(orig_set)) * 100.0

    def _extend_match_for_context(
        self, match_text: str, resource: "Resource"
    ) -> List[str]:
        """
        Try to extend a matching text in both directions for better context.

        Args:
            match_text: Text that already matches in the resource
            resource: Resource to search within

        Returns:
            List of extended matches
        """
        extensions = []

        try:
            # Find the match in the resource to get its position
            quote_obj = resource.quote(match_text)

            # For each occurrence, try to extend
            for start_pos, end_pos in quote_obj.spans:
                # Try extending backward (find sentence start)
                extended_back = self._extend_to_sentence_boundary(
                    resource.text, start_pos, end_pos, extend_backward=True
                )
                if extended_back and extended_back != match_text:
                    extensions.append(extended_back)

                # Try extending forward (find sentence end)
                extended_forward = self._extend_to_sentence_boundary(
                    resource.text, start_pos, end_pos, extend_backward=False
                )
                if extended_forward and extended_forward != match_text:
                    extensions.append(extended_forward)

        except Exception:
            pass  # Match not found or other error

        return extensions

    def _extend_to_sentence_boundary(
        self, full_text: str, start_pos: int, end_pos: int, extend_backward: bool = True
    ) -> Optional[str]:
        """
        Extend text to natural sentence boundaries.

        Args:
            full_text: Full text of the resource
            start_pos: Start position of current match
            end_pos: End position of current match
            extend_backward: Whether to extend backward or forward

        Returns:
            Extended text or None
        """
        if extend_backward:
            # Look backward for sentence start
            search_start = max(0, start_pos - 200)  # Look back up to 200 chars
            search_text = full_text[search_start:end_pos]

            # Find sentence boundaries (. ! ?)
            sentence_starts = [m.end() for m in re.finditer(r"[.!?]\s+", search_text)]

            if sentence_starts:
                # Use the last sentence boundary found
                actual_start = search_start + sentence_starts[-1]
                return full_text[actual_start:end_pos].strip()
            else:
                # No sentence boundary found, extend by a reasonable amount
                actual_start = max(search_start, start_pos - 50)
                return full_text[actual_start:end_pos].strip()

        else:
            # Look forward for sentence end
            search_end = min(
                len(full_text), end_pos + 200
            )  # Look ahead up to 200 chars
            search_text = full_text[start_pos:search_end]

            # Find sentence boundaries
            sentence_end_match = re.search(r"[.!?]", search_text)

            if sentence_end_match:
                # Include the punctuation
                actual_end = start_pos + sentence_end_match.end()
                return full_text[start_pos:actual_end].strip()
            else:
                # No sentence boundary found, extend by a reasonable amount
                actual_end = min(search_end, end_pos + 50)
                return full_text[start_pos:actual_end].strip()

    def _clean_suggestion(self, suggestion: str) -> str:
        """
        Clean up a suggestion by removing artifacts and formatting issues.

        Args:
            suggestion: Raw suggestion text

        Returns:
            Cleaned suggestion text
        """
        if not suggestion:
            return ""

        # Remove common artifacts
        cleaned = suggestion.strip()

        # Remove reference numbers (e.g., "protein22" -> "protein")
        cleaned = re.sub(r"([a-zA-Z]+)\d+\b", r"\1", cleaned)

        # Remove standalone reference numbers
        cleaned = re.sub(r"\b\d+\.\s*$", "", cleaned)

        # Remove metadata artifacts
        metadata_patterns = [
            r"\bPubMed\b",
            r"\bGoogle Scholar\b",
            r"\bCrossref\b",
            r"\bScopus\b",
            r"\b\d+\.\s*$",  # Trailing reference numbers
            r"\*\*[^*]+\*\*",  # Bold text
            r"_[^_]+_",  # Italic text (but preserve gene names like _GENE_)
        ]

        for pattern in metadata_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Clean up extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Remove trailing punctuation if it seems artificial
        if cleaned.endswith((".", ",", ";")) and not cleaned.endswith("..."):
            # Only remove if the sentence doesn't seem naturally complete
            words = cleaned.split()
            if len(words) < 5:  # Short fragments probably don't need ending punctuation
                cleaned = cleaned[:-1].strip()

        return cleaned
