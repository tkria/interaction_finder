"""
Error message construction for quote validation failures.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
from .alignment import SequenceAligner, ErrorType
from .alignment_utilities import analyze_quote_alignment

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
    Build a comprehensive retry message using alignment analysis for precise feedback.

    Args:
        entity_name: Name of the entity that failed
        entity_kind: Kind of entity (gene, disease, etc.)
        failed_quote: The quote text that failed validation
        resource: Resource where the quote was supposed to exist
        suggestions: List of (suggestion, confidence) correction suggestions

    Returns:
        Formatted retry message for the LLM with detailed alignment information
    """
    message = f"Quote validation failed for entity '{entity_name}':\n"
    message += f'Failed Quote: "{failed_quote}"\n\n'

    # Use alignment analysis for detailed error information
    alignment = analyze_quote_alignment(failed_quote, resource)

    # Generate specific error messages based on alignment analysis
    if alignment.error_type == ErrorType.SPLIT_QUOTE or alignment.is_likely_split:
        message += _build_split_quote_message(alignment)
    elif alignment.error_type == ErrorType.WORD_SUBSTITUTION:
        message += _build_substitution_message(alignment, suggestions)
    elif alignment.error_type == ErrorType.INSERTION:
        message += _build_insertion_message(alignment)
    elif alignment.error_type == ErrorType.DELETION:
        message += _build_deletion_message(alignment)
    elif alignment.error_type == ErrorType.PARAPHRASE:
        message += _build_paraphrase_message(alignment, suggestions)
    else:
        message += _build_general_error_message(suggestions)

    # Always add general instructions
    message += "\n\nGENERAL INSTRUCTIONS:\n"
    message += "• Use EXACT text - no paraphrasing or synonym substitution\n"
    message += "• Copy text character-for-character as it appears in the document\n"
    message += "• Ensure quotes are verbatim from the source document\n"

    return message


def _build_split_quote_message(alignment) -> str:
    """Build error message for split quote errors."""
    message = "❗ QUOTE APPEARS TO BE INCORRECTLY COMBINED:\n"

    if len(alignment.matching_blocks) >= 2:
        first_block = alignment.matching_blocks[0]
        last_block = alignment.matching_blocks[-1]

        prefix_words = alignment.doc_words[first_block.doc_start : first_block.doc_end]
        suffix_words = alignment.doc_words[last_block.doc_start : last_block.doc_end]

        message += f'  • First part: "{" ".join(prefix_words)}"\n'
        message += f'  • Last part: "{" ".join(suffix_words)}"\n'

        gap_words = last_block.doc_start - first_block.doc_end
        if gap_words > 0:
            message += f"  • Gap of {gap_words} words between parts\n"

    message += "  This quote combines text from different locations without proper ellipsis.\n\n"
    message += "SPECIFIC INSTRUCTIONS FOR SPLIT QUOTES:\n"
    message += "• Use separate quotes for each part, OR\n"
    message += "• Use exact text with proper '...' to indicate skipped content\n"
    message += "• Each quote must be from a single, contiguous location\n"

    return message


def _build_substitution_message(alignment, suggestions) -> str:
    """Build error message for word substitution errors."""
    message = "🔄 WORD SUBSTITUTION DETECTED:\n"
    message += f"  • Overall similarity: {alignment.similarity_ratio:.1%}\n"
    message += f"  • Word coverage: {alignment.coverage_ratio:.1%}\n"

    if alignment.gaps:
        message += "  • Differences found in specific word positions\n"

    if suggestions:
        message += "\n✅ SUGGESTED CORRECTIONS:\n"
        for i, (suggestion, confidence) in enumerate(suggestions[:2], 1):
            message += f'  {i}. "{suggestion}" (confidence: {confidence:.1f}%)\n'

    message += "\nSPECIFIC INSTRUCTIONS FOR SUBSTITUTIONS:\n"
    message += "• Check each word against the original document\n"
    message += "• Replace any paraphrased or synonymous words with exact text\n"
    message += "• Maintain the exact spelling and capitalization\n"

    return message


def _build_insertion_message(alignment) -> str:
    """Build error message for insertion errors (extra words in quote)."""
    message = "➕ EXTRA WORDS DETECTED:\n"
    message += f"  • Quote contains words not found in the document\n"
    message += f"  • Matched content: {alignment.coverage_ratio:.1%} of quote\n"

    if alignment.gaps:
        extra_words = []
        for gap in alignment.gaps:
            if gap.gap_type == "deletion":  # Words in quote but not in doc
                extra_words.extend(gap.words)

        if extra_words:
            message += f"  • Extra words: {', '.join(extra_words[:5])}\n"

    message += "\nSPECIFIC INSTRUCTIONS FOR EXTRA WORDS:\n"
    message += "• Remove any words not present in the original document\n"
    message += "• Use only text that appears verbatim in the source\n"
    message += "• Check for added explanatory or connecting words\n"

    return message


def _build_deletion_message(alignment) -> str:
    """Build error message for deletion errors (missing words from quote)."""
    message = "➖ MISSING WORDS DETECTED:\n"
    message += f"  • Quote appears to be incomplete or shortened\n"
    message += f"  • Matched content: {alignment.coverage_ratio:.1%} of quote\n"

    message += "\nSPECIFIC INSTRUCTIONS FOR MISSING WORDS:\n"
    message += "• Include all words from the relevant sentence or phrase\n"
    message += "• Do not abbreviate or shorten the original text\n"
    message += "• Extend quote to include complete meaningful context\n"

    return message


def _build_paraphrase_message(alignment, suggestions) -> str:
    """Build error message for paraphrasing errors."""
    message = "📝 PARAPHRASING DETECTED:\n"
    message += (
        f"  • Low similarity to original text: {alignment.similarity_ratio:.1%}\n"
    )
    message += "  • Content appears to be rewritten or summarized\n"

    if suggestions:
        message += "\n✅ SUGGESTED VERBATIM TEXT:\n"
        for i, (suggestion, confidence) in enumerate(suggestions[:2], 1):
            message += f'  {i}. "{suggestion}"\n'

    message += "\nSPECIFIC INSTRUCTIONS FOR PARAPHRASING:\n"
    message += "• Replace paraphrased content with exact original text\n"
    message += "• Do not summarize, interpret, or rewrite quotes\n"
    message += "• Copy the text word-for-word as it appears in the document\n"

    return message


def _build_general_error_message(suggestions) -> str:
    """Build general error message when specific classification is unclear."""
    message = "❓ QUOTE NOT FOUND IN DOCUMENT:\n"

    if suggestions:
        message += "\n✅ POTENTIAL CORRECTIONS FOUND:\n"
        for i, (suggestion, confidence) in enumerate(suggestions[:3], 1):
            message += f'  {i}. "{suggestion}" (confidence: {confidence:.1f}%)\n'
        message += "\nSPECIFIC INSTRUCTIONS:\n"
        message += "• Use one of the suggested corrections if appropriate\n"
        message += "• Ensure the selected text contains the relevant entity mention\n"
    else:
        message += "\nSPECIFIC INSTRUCTIONS:\n"
        message += "• Carefully re-read the document for the entity mention\n"
        message += "• Use exact text surrounding the entity name\n"
        message += "• If entity is not clearly mentioned, return empty quotes array\n"

    return message


# Legacy function kept for compatibility but now uses alignment analysis
def _find_matching_part(original: str, suggestion: str) -> Optional[str]:
    """
    Find the longest matching substring using sequence alignment.

    Args:
        original: Original failed quote
        suggestion: Suggested correction

    Returns:
        Longest matching substring or None
    """
    if not original or not suggestion:
        return None

    from difflib import SequenceMatcher

    orig_words = original.lower().split()
    sugg_words = suggestion.lower().split()

    # Use SequenceMatcher to find matching blocks
    matcher = SequenceMatcher(None, orig_words, sugg_words)

    longest_match = ""
    for block in matcher.get_matching_blocks():
        if block.size >= 2:  # At least 2 words
            match_words = orig_words[block.a : block.a + block.size]
            candidate = " ".join(match_words)
            if len(candidate) > len(longest_match):
                longest_match = candidate

    return longest_match if len(longest_match) > 10 else None
