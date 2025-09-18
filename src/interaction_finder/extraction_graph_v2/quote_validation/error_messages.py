"""
Error message construction for quote validation failures.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
from .utilities import detect_split_quote

if TYPE_CHECKING:
    from ...resources import Resource


def build_retry_message(
    entity_name: str,
    entity_kind: str,
    failed_quote: str,
    resource: "Resource",
    suggestions: List[Tuple[str, float]],
) -> str:
    """
    Build a comprehensive retry message for quote validation failures.

    Args:
        entity_name: Name of the entity that failed
        entity_kind: Kind of entity (gene, disease, etc.)
        failed_quote: The quote text that failed validation
        resource: Resource where the quote was supposed to exist
        suggestions: List of (suggestion, confidence) correction suggestions

    Returns:
        Formatted retry message for the LLM
    """
    message = f"Quote validation failed for entity '{entity_name}':\n"
    message += f'Failed Quote: "{failed_quote}"\n\n'

    # Check if this is a split quote
    split_info = detect_split_quote(failed_quote, resource)

    if split_info:
        message += "❗ QUOTE APPEARS TO BE INCORRECTLY COMBINED:\n"
        message += f'  • Prefix: "{split_info["prefix"]}"\n'
        message += f'  • Suffix: "{split_info["suffix"]}"\n'
        message += (
            '  This quote combines text from different parts without using "...".\n\n'
        )

        message += "INSTRUCTIONS FOR THIS QUOTE:\n"
        message += "• Use EXACT text - no paraphrasing or synonym substitution\n"
        message += "• Copy text character-for-character as it appears in the document\n"
        message += (
            "• For split quotes, use separate quotes for each part with correct text"
        )

    elif suggestions:
        # Show correction suggestions
        best_suggestion, best_confidence = suggestions[0]

        message += f"✅ POTENTIAL CORRECTION (partial match found):\n"

        # Find the matching part for display
        matching_part = _find_matching_part(failed_quote, best_suggestion)
        if matching_part:
            message += f'  Matched part: "{matching_part}"\n'

        message += "  Suggested corrections:\n"
        for i, (suggestion, confidence) in enumerate(suggestions[:2], 1):
            message += f'  • "{suggestion}"\n'

        message += "\nINSTRUCTIONS FOR THIS QUOTE:\n"
        message += "• Use EXACT text - no paraphrasing or synonym substitution\n"
        message += "• Copy text character-for-character as it appears in the document\n"
        message += "• If corrections are suggested above, use the exact suggested text"

    else:
        # No specific corrections available
        message += "INSTRUCTIONS FOR THIS QUOTE:\n"
        message += "• Use EXACT text - no paraphrasing or synonym substitution\n"
        message += "• Copy text character-for-character as it appears in the document"

    return message


def _find_matching_part(original: str, suggestion: str) -> Optional[str]:
    """
    Find the longest matching substring between original and suggestion.

    Args:
        original: Original failed quote
        suggestion: Suggested correction

    Returns:
        Longest matching substring or None
    """
    if not original or not suggestion:
        return None

    orig_words = original.lower().split()
    sugg_words = suggestion.lower().split()

    # Find longest common subsequence
    longest_match = ""

    for i in range(len(orig_words)):
        for j in range(i + 1, len(orig_words) + 1):
            candidate = " ".join(orig_words[i:j])
            if candidate.lower() in suggestion.lower() and len(candidate) > len(
                longest_match
            ):
                longest_match = candidate

    return longest_match if len(longest_match) > 10 else None
