"""
Resource management models for document sources and quote tracking.

This module provides structured management of document resources and validates
that extracted information can be traced back to specific documents with
supporting quotes.
"""

import hashlib
from urllib.parse import urlparse, urlunparse
import re
from dataclasses import dataclass
from enum import Enum
from difflib import SequenceMatcher
from typing import List, Optional, Tuple, TYPE_CHECKING
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    ValidationInfo,
    model_serializer,
)
from pydantic_core import core_schema
from rapidfuzz import fuzz

from interaction_finder.text_mapping import NormalizedTextMapper

if TYPE_CHECKING:
    pass  # For forward references


# Fuzzy matching thresholds
FUZZY_SUGGESTION_THRESHOLD: float = 0.75  # Minimum similarity for suggestions


class ErrorType(Enum):
    """Types of quote validation errors based on alignment patterns."""

    SPLIT_QUOTE = "split_quote"  # Non-contiguous matching blocks
    WORD_SUBSTITUTION = "word_substitution"  # High similarity, specific mismatches
    INSERTION = "insertion"  # Extra words in quote
    DELETION = "deletion"  # Missing words in quote
    PARAPHRASE = "paraphrase"  # Low similarity, some structure preserved
    REORDERING = "reordering"  # Same words, wrong order
    NOT_FOUND = "not_found"  # No significant alignment


@dataclass
class MatchingBlock:
    """Represents a contiguous matching block between quote and document."""

    quote_start: int  # Start position in quote words
    quote_end: int  # End position in quote words
    doc_start: int  # Start position in document words
    doc_end: int  # End position in document words
    length: int  # Number of matching words

    @property
    def quote_span(self) -> Tuple[int, int]:
        """Get quote span as (start, end) tuple."""
        return (self.quote_start, self.quote_end)

    @property
    def doc_span(self) -> Tuple[int, int]:
        """Get document span as (start, end) tuple."""
        return (self.doc_start, self.doc_end)


@dataclass
class CorrectionSuggestion:
    """A suggested correction for a failed quote match."""

    text: str  # The suggested quote text
    confidence: float  # Overall confidence score (0.0 to 1.0)
    similarity: float  # Fuzzy similarity score
    explanation: str  # Human-readable explanation

    # Optional detailed metrics (populated by V2 alignment system)
    alignment_score: Optional[float] = None
    contiguity_score: Optional[float] = None
    boundary_quality: Optional[float] = None


class QuoteValidationError(ValueError):
    """
    Base class for all quote validation failures.

    Inherits from ValueError for backwards compatibility with existing
    code that catches ValueError.
    """

    def __init__(
        self, quote_text: str, resource: "Resource", similarity_threshold: float
    ):
        self.quote_text = quote_text
        self.resource = resource
        self.similarity_threshold = similarity_threshold
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        """Subclasses override to provide specific messages."""
        return f"Quote validation failed: {self.quote_text!r}"


class QuoteNotFoundError(QuoteValidationError):
    """
    Raised when quote has no match at all in the resource.

    This represents a complete failure - no similar text was found even
    at the minimum FUZZY_SUGGESTION_THRESHOLD.
    """

    def _build_message(self) -> str:
        return f"Quote not found in resource: {self.quote_text!r}"


class QuoteNearMatchError(QuoteValidationError):
    """Base for near-match errors with suggestions."""

    def __init__(
        self,
        quote_text: str,
        resource: "Resource",
        similarity_threshold: float,
        suggestions: List[CorrectionSuggestion],
        similarity: float,
    ):
        self.suggestions = suggestions
        self.similarity = similarity
        super().__init__(quote_text, resource, similarity_threshold)

    def best_suggestion(self) -> CorrectionSuggestion:
        """Get the highest-confidence suggestion."""
        return self.suggestions[0]


class SplitQuoteError(QuoteNearMatchError):
    """Quote matches in multiple non-contiguous locations."""

    def __init__(
        self,
        quote_text: str,
        resource: "Resource",
        similarity_threshold: float,
        suggestions: List[CorrectionSuggestion],
        similarity: float,
        matching_blocks: List[MatchingBlock],
        coverage_ratio: float,
    ):
        self.matching_blocks = matching_blocks
        self.coverage_ratio = coverage_ratio
        super().__init__(
            quote_text, resource, similarity_threshold, suggestions, similarity
        )

    def get_block_gaps(self) -> List[int]:
        """Get distances between matching blocks in document."""
        if len(self.matching_blocks) <= 1:
            return []

        gaps = []
        for i in range(len(self.matching_blocks) - 1):
            gap = (
                self.matching_blocks[i + 1].doc_start - self.matching_blocks[i].doc_end
            )
            gaps.append(gap)
        return gaps

    def suggest_ellipsis_format(self) -> str:
        """Suggest an ellipsis-formatted quote based on matching blocks."""
        parts = []
        for block in self.matching_blocks:
            words = self.resource.normalized_text.split()[
                block.doc_start : block.doc_end
            ]
            parts.append(" ".join(words))
        return " ... ".join(parts)

    def _build_message(self) -> str:
        gaps = self.get_block_gaps()
        return (
            f"Quote split across {len(self.matching_blocks)} non-contiguous sections: {self.quote_text!r}\n"
            f"Coverage: {self.coverage_ratio:.1%}, Gaps: {gaps}\n"
            f"Suggested ellipsis format: {self.suggest_ellipsis_format()!r}"
        )


class WordSubstitutionError(QuoteNearMatchError):
    """Quote has specific word substitutions."""

    def __init__(
        self,
        quote_text: str,
        resource: "Resource",
        similarity_threshold: float,
        suggestions: List[CorrectionSuggestion],
        similarity: float,
        substitutions: List[Tuple[int, str, str]],  # (position, quote_word, doc_word)
    ):
        self.substitutions = substitutions
        super().__init__(
            quote_text, resource, similarity_threshold, suggestions, similarity
        )

    def _build_message(self) -> str:
        sub_details = ", ".join(
            f"'{quote_word}' → '{doc_word}' at pos {pos}"
            for pos, quote_word, doc_word in self.substitutions[:3]  # Show first 3
        )
        return (
            f"Quote has word substitutions: {self.quote_text!r}\n"
            f"Substitutions: {sub_details}\n"
            f"Suggested: {self.best_suggestion().text!r}"
        )


class InsertionError(QuoteNearMatchError):
    """Quote contains extra words not in document."""

    def __init__(
        self,
        quote_text: str,
        resource: "Resource",
        similarity_threshold: float,
        suggestions: List[CorrectionSuggestion],
        similarity: float,
        insertions: List[Tuple[int, str]],  # (position, inserted_word)
    ):
        self.insertions = insertions
        super().__init__(
            quote_text, resource, similarity_threshold, suggestions, similarity
        )

    def _build_message(self) -> str:
        inserted_words = [word for _, word in self.insertions]
        return (
            f"Quote contains extra words: {self.quote_text!r}\n"
            f"Inserted words: {', '.join(inserted_words[:5])}\n"
            f"Suggested (with insertions removed): {self.best_suggestion().text!r}"
        )


class DeletionError(QuoteNearMatchError):
    """Quote is missing words from document."""

    def __init__(
        self,
        quote_text: str,
        resource: "Resource",
        similarity_threshold: float,
        suggestions: List[CorrectionSuggestion],
        similarity: float,
        deletions: List[Tuple[int, str]],  # (position, missing_word)
    ):
        self.deletions = deletions
        super().__init__(
            quote_text, resource, similarity_threshold, suggestions, similarity
        )

    def _build_message(self) -> str:
        missing_words = [word for _, word in self.deletions]
        return (
            f"Quote is missing words: {self.quote_text!r}\n"
            f"Missing words: {', '.join(missing_words[:5])}\n"
            f"Suggested (with missing words added): {self.best_suggestion().text!r}"
        )


class ParaphraseError(QuoteNearMatchError):
    """Quote is paraphrased - low similarity but some structure preserved."""

    def __init__(
        self,
        quote_text: str,
        resource: "Resource",
        similarity_threshold: float,
        suggestions: List[CorrectionSuggestion],
        similarity: float,
        coverage_ratio: float,
        matching_words: List[str],
    ):
        self.coverage_ratio = coverage_ratio
        self.matching_words = matching_words
        super().__init__(
            quote_text, resource, similarity_threshold, suggestions, similarity
        )

    def _build_message(self) -> str:
        return (
            f"Quote is paraphrased: {self.quote_text!r}\n"
            f"Similarity: {self.similarity:.1%} (threshold: {self.similarity_threshold:.1%})\n"
            f"Coverage: {self.coverage_ratio:.1%}\n"
            f"Matching words: {', '.join(self.matching_words[:10])}\n"
            f"Closest match: {self.best_suggestion().text!r}"
        )


class ReorderingError(QuoteNearMatchError):
    """Quote has words in wrong order."""

    def __init__(
        self,
        quote_text: str,
        resource: "Resource",
        similarity_threshold: float,
        suggestions: List[CorrectionSuggestion],
        similarity: float,
        expected_order: List[Tuple[int, str]],  # (expected_position, word)
        actual_order: List[Tuple[int, str]],  # (actual_position, word)
    ):
        self.expected_order = expected_order
        self.actual_order = actual_order
        super().__init__(
            quote_text, resource, similarity_threshold, suggestions, similarity
        )

    def _build_message(self) -> str:
        return (
            f"Quote has words in wrong order: {self.quote_text!r}\n"
            f"Suggested (with correct order): {self.best_suggestion().text!r}"
        )


def expand_scientific_shorthand(text: str) -> List[str]:
    """
    Expand scientific shorthand notation into individual components.

    Handles common scientific notation patterns:
    - Comma-separated: "ISCA1,2" → ["ISCA1", "ISCA2"]
    - Slash-separated: "COL1A1/A2" → ["COL1A1", "COL1A2"]
    - Numeric ranges: "exons 2-4" → ["exon 2", "exon 3", "exon 4"]

    For text containing shorthand, returns variants where the shorthand
    is replaced with expanded forms.

    Args:
        text: Text potentially containing scientific shorthand

    Returns:
        List of expanded forms (includes original if no expansion possible)
    """
    import re

    expanded = []
    text = text.strip()

    # Try to find and expand patterns within the text

    # Pattern 1a: Simple numeric variant like "ISCA1,2"
    numeric_pattern = r"(\w+)(\d+)([,/])(\d+)"
    numeric_match = re.search(numeric_pattern, text)
    if numeric_match:
        base = numeric_match.group(1)
        first_num = numeric_match.group(2)
        _separator = numeric_match.group(3)
        second_num = numeric_match.group(4)

        # Create expanded versions by replacing the pattern
        original_pattern = numeric_match.group(0)
        first_replacement = f"{base}{first_num}"
        second_replacement = f"{base}{second_num}"

        expanded.append(text.replace(original_pattern, first_replacement))
        expanded.append(text.replace(original_pattern, second_replacement))
        return expanded

    # Pattern 1b: Letter variant like "COL1A1,A2"
    letter_pattern = r"(\w+)([A-Z]\d+)([,/])([A-Z]\d+)"
    letter_match = re.search(letter_pattern, text)
    if letter_match:
        base = letter_match.group(1)
        first_suffix = letter_match.group(2)
        _separator = letter_match.group(3)
        second_suffix = letter_match.group(4)

        # Create expanded versions by replacing the pattern
        original_pattern = letter_match.group(0)
        first_replacement = f"{base}{first_suffix}"
        second_replacement = f"{base}{second_suffix}"

        expanded.append(text.replace(original_pattern, first_replacement))
        expanded.append(text.replace(original_pattern, second_replacement))
        return expanded

    # Pattern 2: Numeric ranges like "exons 2-4" or "chapters 1-3"
    range_pattern = r"(\w+s?)\s+(\d+)-(\d+)"
    range_match = re.search(range_pattern, text, re.IGNORECASE)
    if range_match:
        base_word = range_match.group(1).rstrip("s")  # Remove plural 's'
        start_num = int(range_match.group(2))
        end_num = int(range_match.group(3))

        # Generate range (limit to reasonable size)
        if end_num - start_num <= 20:  # Prevent huge expansions
            original_pattern = range_match.group(0)
            for num in range(start_num, end_num + 1):
                replacement = f"{base_word} {num}"
                expanded.append(text.replace(original_pattern, replacement))
            return expanded

    # Pattern 3a: Slash with Greek letters like "p53α/β"
    greek_slash_pattern = (
        r"(\w+)([αβγδεζηθικλμνξοπρστυφχψω]+)/([αβγδεζηθικλμνξοπρστυφχψω]+)"
    )
    greek_match = re.search(greek_slash_pattern, text)
    if greek_match:
        base = greek_match.group(1)
        first_variant = greek_match.group(2)
        second_variant = greek_match.group(3)

        original_pattern = greek_match.group(0)
        first_replacement = f"{base}{first_variant}"
        second_replacement = f"{base}{second_variant}"

        expanded.append(text.replace(original_pattern, first_replacement))
        expanded.append(text.replace(original_pattern, second_replacement))
        return expanded

    # Pattern 3b: Slash with alphanumeric variants like "p53a/b" or "IgG1/2"
    alphanumeric_slash_pattern = r"(\w+)([a-zA-Z0-9]+)/([a-zA-Z0-9]+)"
    alphanumeric_match = re.search(alphanumeric_slash_pattern, text)
    if alphanumeric_match:
        base = alphanumeric_match.group(1)
        first_variant = alphanumeric_match.group(2)
        second_variant = alphanumeric_match.group(3)

        # Only expand if:
        # 1. Variants are short (avoid false positives like "protein/function")
        # 2. Base looks like scientific identifier (has digits OR is short)
        # 3. Variants look like suffixes (single chars/short alphanumeric)
        if (
            len(first_variant) <= 3
            and len(second_variant) <= 3
            and len(base) >= 2
            and len(base) <= 10  # Reasonable identifier length
            and (
                any(c.isdigit() for c in base)  # Contains numbers: p53, IgG1, CD4
                or (
                    len(base) <= 4 and base.isupper()
                )  # Short uppercase: DNA->no, IL->yes
                or base[-1].isdigit()  # Ends with digit: p53, CD4
            )
        ):
            original_pattern = alphanumeric_match.group(0)
            first_replacement = f"{base}{first_variant}"
            second_replacement = f"{base}{second_variant}"

            expanded.append(text.replace(original_pattern, first_replacement))
            expanded.append(text.replace(original_pattern, second_replacement))
            return expanded

    # No expansion possible, return original
    return [text]


class FuzzyMatchResult(BaseModel):
    """
    Result from fuzzy matching with auto-corrected quote text.

    Contains the corrected quote that exists in the document, similarity score,
    and alignment blocks for debugging/validation.
    """

    corrected_quote: str = Field(
        description="Auto-corrected quote text that exists in the document"
    )
    similarity: float = Field(
        description="Similarity score between LLM quote and document (0.0-1.0)"
    )
    match_blocks: List[Tuple[int, int, int]] = Field(
        description="Matching blocks from SequenceMatcher (llm_pos, doc_pos, length)"
    )


def fuzzy_match_quote(
    llm_quote: str,
    document_segment: str,
    threshold: float = 0.90,
) -> Optional[FuzzyMatchResult]:
    """
    Fuzzy match LLM-generated quote against document segment using hybrid algorithm.

    Uses three-stage approach:
    1. RapidFuzz for fast approximate location finding
    2. Window extraction with margin for boundary capture
    3. difflib for precise entity-preserving alignment

    Args:
        llm_quote: Quote text from LLM (may be slightly paraphrased)
        document_segment: Segment of document to match against
        threshold: Minimum similarity for auto-correction (default: 0.90)

    Returns:
        FuzzyMatchResult with corrected quote and metadata, or None if below threshold

    Notes:
        - Uses normalized text for comparison (via NormalizedTextMapper.normalize())
        - Threshold of 0.90 is conservative to avoid false corrections
        - Very short quotes (<5 chars) are unreliable and return None
    """
    # Edge case: skip very short quotes (too unreliable)
    if len(llm_quote.strip()) < 5:
        return None

    # Normalize both for comparison
    normalized_llm = NormalizedTextMapper.normalize(llm_quote)
    normalized_doc = NormalizedTextMapper.normalize(document_segment)

    # Edge case: empty after normalization
    if not normalized_llm or not normalized_doc:
        return None

    # Stage 1: RapidFuzz finds approximate match location
    # Convert threshold from 0.0-1.0 scale to 0-100 scale for RapidFuzz
    score_cutoff = threshold * 100
    alignment = fuzz.partial_ratio_alignment(
        normalized_llm, normalized_doc, score_cutoff=score_cutoff
    )

    # Return None if RapidFuzz found no match above threshold
    if alignment is None:
        return None

    # Stage 2: Extract window with margin for boundary capture
    # Use 0.10x query length as margin to keep window tight for difflib Stage 3
    # Larger margins (e.g. 1.5x) dilute similarity too much for 0.75 threshold
    margin = int(len(normalized_llm) * 0.10)
    window_start = max(0, alignment.dest_start - margin)
    window_end = min(len(normalized_doc), alignment.dest_end + margin)
    window = normalized_doc[window_start:window_end]

    # Stage 3: difflib precise alignment on window
    matcher = SequenceMatcher(None, normalized_llm, window)
    similarity = matcher.ratio()

    # Return None if difflib refinement rejects the match
    if similarity < threshold:
        return None

    # Extract aligned text from window using matching blocks
    corrected_quote, match_blocks = _auto_correct_quote_from_alignment(
        normalized_llm, window, matcher, window_start
    )

    return FuzzyMatchResult(
        corrected_quote=corrected_quote,
        similarity=similarity,
        match_blocks=match_blocks,
    )


def _auto_correct_quote_from_alignment(
    normalized_llm: str,
    window: str,
    matcher: SequenceMatcher,
    window_start: int,
) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Extract corrected quote from window and translate coordinates to document.

    Builds corrected quote by extracting aligned portions from the window,
    then translates match blocks from window coordinates to document coordinates.

    Args:
        normalized_llm: Normalized LLM quote text
        window: Normalized document window text
        matcher: Pre-configured SequenceMatcher for the texts
        window_start: Start position of window in document (for coordinate translation)

    Returns:
        Tuple of (corrected_quote, translated_match_blocks) where match blocks
        are in document coordinates: (src_start, doc_start, length)

    Notes:
        - Uses get_matching_blocks() to identify aligned segments
        - Extracts text from window at aligned positions
        - Translates match blocks from window-relative to document-absolute coordinates
        - Preserves small gaps (single spaces) between consecutive blocks in window,
          but omits larger gaps (these represent truly disjoint quote segments)
    """
    # Get matching blocks: (llm_pos, window_pos, length) tuples
    blocks = matcher.get_matching_blocks()

    # Build corrected quote by extracting matching segments
    # Preserve small gaps (word boundaries) but not large omissions
    corrected_parts = []
    prev_window_end = None

    for llm_pos, window_pos, length in blocks:
        if length == 0:  # Skip dummy block at end
            continue

        # Check if there's a gap in the window between previous block and this one
        if prev_window_end is not None and window_pos > prev_window_end:
            gap = window[prev_window_end:window_pos]
            # Only preserve very small gaps (1-2 spaces) that represent word boundaries
            # Larger gaps represent content that was omitted (disjoint quotes)
            if len(gap) <= 2 and gap.strip() == "":
                # Single space or double space → preserve as word boundary
                corrected_parts.append(" ")
            # Larger gaps are skipped (represent disjoint sections)

        # Extract the matching segment
        segment = window[window_pos : window_pos + length]
        corrected_parts.append(segment)
        prev_window_end = window_pos + length

    # Join all parts
    corrected = "".join(corrected_parts)

    # Clean up multiple spaces and strip
    corrected = " ".join(corrected.split())

    # Translate match blocks from window coordinates to document coordinates
    translated_blocks = [
        (src_start, window_start + dest_start, length)
        for src_start, dest_start, length in blocks
    ]

    return corrected, translated_blocks


def compute_chunk_spans(
    full_text: str, chunk_texts: List[str]
) -> List[Tuple[int, int]]:
    """
    Efficiently compute chunk spans by sequential forward search.

    Args:
        full_text: Complete document text (e.g., from get_markdown())
        chunk_texts: List of chunk texts (e.g., from get_chunks())

    Returns:
        List of (start, end) positions for each chunk in the full text
    """
    spans = []
    search_start = 0

    for i, chunk_text in enumerate(chunk_texts):
        if not chunk_text.strip():  # Skip empty chunks
            continue

        # Try exact match first
        chunk_start = full_text.find(chunk_text, search_start)

        if chunk_start == -1:
            # Try with whitespace normalization fallback
            chunk_normalized = " ".join(chunk_text.split())
            if not chunk_normalized:
                continue  # Skip empty normalized chunks

            search_text = full_text[search_start:]

            # Look for normalized version within a reasonable window
            window_size = min(len(search_text), len(chunk_normalized) * 2)
            for j in range(
                min(window_size - len(chunk_normalized) + 1, 1000)
            ):  # Limit search to avoid performance issues
                candidate_end = min(j + len(chunk_normalized), len(search_text))
                candidate = " ".join(search_text[j:candidate_end].split())
                if candidate == chunk_normalized:
                    chunk_start = search_start + j
                    break

        if chunk_start == -1:
            # Last resort: try normalized text matching
            normalized_chunk = NormalizedTextMapper.normalize(chunk_text)
            normalized_search = NormalizedTextMapper.normalize(full_text[search_start:])

            if normalized_chunk and normalized_search:
                norm_pos = normalized_search.find(normalized_chunk)
                if norm_pos != -1:
                    # Approximate mapping back (this is imprecise but better than nothing)
                    chunk_start = search_start + norm_pos

        if chunk_start == -1:
            # Skip this chunk if we still can't find it
            continue

        chunk_end = chunk_start + len(chunk_text)
        spans.append((chunk_start, chunk_end))

        # Next search starts after this chunk
        search_start = chunk_end

    return spans


def _expand_to_word_boundaries(text: str, start: int, end: int) -> Tuple[int, int]:
    """
    Expand span boundaries to align with word boundaries.

    Ensures quotes don't begin or end mid-word by expanding outward to the
    nearest word boundaries. A word boundary is defined as a transition between
    alphanumeric and non-alphanumeric characters.

    Args:
        text: Full text content
        start: Start position of span
        end: End position of span

    Returns:
        Tuple of (adjusted_start, adjusted_end) aligned to word boundaries
    """
    text_length = len(text)
    adjusted_start = start
    adjusted_end = end

    # Expand start backward if we're mid-word
    while (
        adjusted_start > 0
        and text[adjusted_start - 1].isalnum()
        and text[adjusted_start].isalnum()
    ):
        adjusted_start -= 1

    # Expand end forward if we're mid-word
    while (
        adjusted_end < text_length
        and text[adjusted_end - 1].isalnum()
        and text[adjusted_end].isalnum()
    ):
        adjusted_end += 1

    return adjusted_start, adjusted_end


class ResourceId(BaseModel):
    """
    Lightweight identifier for a document resource.

    Automatically generates stable hash-based ID from URL and counter.
    """

    id: str = Field(description="Stable hash-based resource identifier")
    url: str = Field(description="Original URL of the resource")

    def __init__(self, url: str = None, counter: int = None, **data):
        """
        Create ResourceId with automatic ID generation or from serialized data.

        Two modes:
        1. Construction: ResourceId(url="...", counter=1) - generates id
        2. Deserialization: ResourceId(id="1_abc", url="...") - uses provided id

        Args:
            url: Document URL
            counter: Sequential counter (required for construction mode)
        """
        # Mode 1: Deserialization (id already in data)
        if "id" in data:
            super().__init__(url=url, **data)
            return

        # Mode 2: Construction (generate id from url + counter)
        if url is None or counter is None:
            raise ValueError("ResourceId requires either (url, counter) or (id, url)")

        # Generate stable ID using same scheme as cache hashing
        parsed = urlparse(url)
        url_to_hash = urlunparse(parsed._replace(fragment=""))

        # Compute base36 of first 8 bytes of sha256, then take first 8 chars
        hash_bytes = hashlib.sha256(url_to_hash.encode()).digest()[:8]
        hash_int = int.from_bytes(hash_bytes, byteorder="big")

        # Manual base36 conversion
        if hash_int == 0:
            base36 = "0"
        else:
            digits = "0123456789abcdefghijklmnopqrstuvwxyz"
            chars = []
            while hash_int:
                chars.append(digits[hash_int % 36])
                hash_int //= 36
            base36 = "".join(reversed(chars))

        # Ensure at least 8 characters (pad with leading zeros), then truncate to 8
        url_hash = base36.rjust(8, "0")[:8]
        resource_id = f"{counter}_{url_hash}"

        super().__init__(id=resource_id, url=url, **data)

    def __hash__(self):
        """Make ResourceId hashable for use as dict keys."""
        return hash(self.id)

    def __eq__(self, other):
        """Equality comparison for ResourceId objects."""
        if not isinstance(other, ResourceId):
            return False
        return self.id == other.id

    def __repr__(self) -> str:
        """Clean representation for REPL display."""
        return f"ResourceId('{self.id}' → '{self.url}')"


class Resource(BaseModel):
    """
    Complete document resource with content and precomputed mappings.

    Self-contained resource that includes both identification and content,
    with normalized text and position offset mapping cached for efficient
    quote matching and position translation. Optionally includes chunk boundaries
    for mapping quotes to specific document chunks.
    """

    id: ResourceId = Field(description="Resource identifier")
    title: str = Field(description="Human-readable document title")
    text: str = Field(description="Full document text content")
    normalized_text: str = Field(
        description="Precomputed normalized text for quote matching"
    )
    chunks: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="List of (start, end) character positions for document chunks",
    )
    doi: Optional[str] = Field(
        default=None, description="Digital Object Identifier (DOI) if available"
    )
    publication_date: Optional[str] = Field(
        default=None, description="Publication date (YYYY-MM-DD) if available"
    )

    # Private attribute for position mapping with normalization
    _position_mapper: Optional[NormalizedTextMapper] = PrivateAttr(default=None)

    def __init__(
        self,
        id: ResourceId,
        title: str,
        text: str,
        chunks: Optional[List[Tuple[int, int]]] = None,
        doi: Optional[str] = None,
        publication_date: Optional[str] = None,
        **data,
    ):
        """
        Create Resource with automatic normalized text and position mapping.

        Args:
            id: ResourceId for the document
            title: Human-readable document title
            text: Full document text content
            chunks: Optional list of (start, end) positions for document chunks.
                   Defaults to single chunk spanning entire document.
            doi: Optional Digital Object Identifier
            publication_date: Optional publication date (YYYY-MM-DD)
        """
        # Create normalized text mapper (handles Greek letters automatically)
        mapper = NormalizedTextMapper.from_text(text)

        # Default chunks to entire document if not provided
        if chunks is None:
            chunks = [(0, len(text))]

        super().__init__(
            id=id,
            title=title,
            text=text,
            normalized_text=mapper.source,  # Get normalized text from mapper
            chunks=chunks,
            doi=doi,
            publication_date=publication_date,
            **data,
        )
        # Store the mapper
        self._position_mapper = mapper

    def quote(self, text: str, similarity_threshold: float = 0.8) -> "ResourceQuote":
        """
        Create a ResourceQuote by finding all occurrences of the given text in this resource.

        Args:
            text: Text to find and quote
            similarity_threshold: Minimum similarity for matches (default: 1.0 for exact only).
                - 1.0: Exact matching only
                - <1.0: Enable fuzzy matching with auto-accept above this threshold

        Returns:
            ResourceQuote with all occurrences

        Raises:
            QuoteNotFoundError: If quote not found (no match at all)
            QuoteNearMatchError subclasses: If similarity between FUZZY_SUGGESTION_THRESHOLD
                and similarity_threshold (includes error-specific diagnostics)
        """
        return ResourceQuote(self, text, similarity_threshold=similarity_threshold)

    def __repr__(self) -> str:
        """Informative representation for REPL display."""
        text_preview = (self.text[:50] + "...") if len(self.text) > 50 else self.text
        return (
            f"Resource(id='{self.id.id}', title='{self.title}', "
            f"text='{text_preview}', {len(self.text)} chars)"
        )

    def map_normalized_to_original_position(
        self, norm_start: int, norm_length: int
    ) -> Tuple[int, int]:
        """
        Map character positions from normalized text back to original text.

        Uses the precomputed position mapper for efficient O(log n) lookups.

        Args:
            norm_start: Start position in normalized text
            norm_length: Length of span in normalized text

        Returns:
            Tuple of (original_start, original_end)
        """
        if self._position_mapper is None:
            raise RuntimeError("Position mapper not initialized")
        return self._position_mapper.targetspan(norm_start, norm_start + norm_length)

    def get_chunk_for_position(self, pos: int) -> Optional[int]:
        """
        Get chunk index containing the given character position.

        Args:
            pos: Character position in the document text

        Returns:
            Index of chunk containing the position, or None if not found
        """
        for i, (start, end) in enumerate(self.chunks):
            if start <= pos < end:
                return i
        return None

    def get_chunks_for_span(self, start: int, end: int) -> List[int]:
        """
        Get all chunk indices that overlap with the given character span.

        Args:
            start: Start character position
            end: End character position

        Returns:
            List of chunk indices that overlap with the span
        """
        chunks = []
        for i, (chunk_start, chunk_end) in enumerate(self.chunks):
            if chunk_start < end and chunk_end > start:  # Overlap check
                chunks.append(i)
        return chunks

    def get_chunk_text(self, chunk_index: int) -> Optional[str]:
        """
        Get the text content of a specific chunk.

        Args:
            chunk_index: Index of the chunk to retrieve

        Returns:
            Text content of the chunk, or None if index is invalid
        """
        if 0 <= chunk_index < len(self.chunks):
            start, end = self.chunks[chunk_index]
            return self.text[start:end]
        return None


def _validate_resource_pool(obj):
    """Validator for ResourcePool that handles list/dict formats before Pydantic validation.

    Accepts:
    - Direct list: [{url, title?, text?, chunks?, id?}, ...]
    - Wrapped dict: {"resources": [{...}]}
    - ResourcePool instance: passed through
    """
    if isinstance(obj, ResourcePool):
        return obj

    resources_list = None

    if isinstance(obj, list):
        # Direct list format
        resources_list = obj
    elif isinstance(obj, dict):
        # Wrapped format
        if "resources" in obj and isinstance(obj["resources"], list):
            resources_list = obj["resources"]

    if resources_list is not None:
        pool = ResourcePool()
        for idx, entry in enumerate(resources_list, start=1):
            # Determine counter
            if "id" in entry:
                counter = int(entry["id"].split("_")[0])
            else:
                counter = idx

            # Reconstruct ResourceId
            resource_id = ResourceId(url=entry["url"], counter=counter)

            # Add to map
            if "text" in entry and "title" in entry:
                resource = Resource(
                    id=resource_id,
                    title=entry["title"],
                    text=entry["text"],
                    chunks=entry.get("chunks", []),
                    doi=entry.get("doi"),
                    publication_date=entry.get("publication_date"),
                )
                pool.resource_map[resource_id] = resource
            else:
                pool.resource_map[resource_id] = None
        return pool

    # Fallback to object as-is for default Pydantic handling
    return obj


class ResourcePool(BaseModel):
    """
    Collection of document resources with separate ID registration and content storage.

    Supports registering resource IDs first, then loading content separately,
    allowing for flexible workflows where resource lists are known before content.
    """

    resource_map: dict[ResourceId, Optional[Resource]] = Field(default_factory=dict)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Custom schema that applies validator before default validation."""
        # Get the default schema for ResourcePool
        python_schema = handler(source_type)

        # Wrap it with our custom validator
        return core_schema.no_info_before_validator_function(
            _validate_resource_pool,
            python_schema,
        )

    def register(self, url: str) -> ResourceId:
        """
        Register a new resource ID without content.

        Args:
            url: Document URL (must be unique)

        Returns:
            ResourceId for the registered resource

        Raises:
            ValueError: If URL already exists in pool
        """
        # Check if URL already registered
        url_plain = urlunparse(urlparse(url)._replace(fragment=""))
        for resource_id in self.resource_map.keys():
            if resource_id.url == url_plain:
                raise ValueError(f"Resource with url {url_plain} already exists")

        # Generate new ResourceId using map size as counter
        counter = len(self.resource_map) + 1
        resource_id = ResourceId(url=url, counter=counter)

        # Register with None content initially
        self.resource_map[resource_id] = None

        return resource_id

    def add_content(
        self,
        resource_id: ResourceId,
        title: str,
        document_text: str,
        chunks: Optional[List[Tuple[int, int]]] = None,
        doi: Optional[str] = None,
        publication_date: Optional[str] = None,
    ) -> Resource:
        """
        Add content to a previously registered resource.

        Args:
            resource_id: Previously registered ResourceId
            title: Human-readable document title
            document_text: Full text content of the document
            chunks: Optional list of (start, end) positions for document chunks
            doi: Optional Digital Object Identifier
            publication_date: Optional publication date (YYYY-MM-DD)

        Returns:
            Complete Resource with content

        Raises:
            KeyError: If resource_id not found in pool
            ValueError: If resource already has content
        """
        if resource_id not in self.resource_map:
            raise KeyError(f"Resource ID {resource_id.id} not found in pool")

        if self.resource_map[resource_id] is not None:
            raise ValueError(f"Resource {resource_id.id} already has content")

        # Create Resource with content
        resource = Resource(
            id=resource_id,
            title=title,
            text=document_text,
            chunks=chunks,
            doi=doi,
            publication_date=publication_date,
        )
        self.resource_map[resource_id] = resource

        return resource

    def add(
        self,
        url: str,
        title: str,
        document_text: str,
        chunks: Optional[List[Tuple[int, int]]] = None,
        doi: Optional[str] = None,
        publication_date: Optional[str] = None,
    ) -> Resource:
        """
        Add a complete resource (register ID + content) in one step.

        Args:
            url: Document URL (must be unique)
            title: Human-readable document title
            document_text: Full text content of the document
            chunks: Optional list of (start, end) positions for document chunks
            doi: Optional Digital Object Identifier
            publication_date: Optional publication date (YYYY-MM-DD)

        Returns:
            Complete Resource

        Raises:
            ValueError: If URL already exists in pool
        """
        resource_id = self.register(url)
        return self.add_content(
            resource_id, title, document_text, chunks, doi, publication_date
        )

    def _find_resource_id(self, key) -> Optional[ResourceId]:
        """
        Find ResourceId by key (ResourceId, ID string, or URL).

        Args:
            key: ResourceId object, ID string, or URL string

        Returns:
            ResourceId if found, None otherwise
        """
        if isinstance(key, ResourceId):
            return key if key in self.resource_map else None
        elif isinstance(key, str):
            for resource_id in self.resource_map.keys():
                if resource_id.id == key or resource_id.url == key:
                    return resource_id
        return None

    def get(self, key) -> Optional[Resource]:
        """
        Retrieve resource by ResourceId, ID string, or URL.

        Args:
            key: ResourceId object, ID string, or URL string

        Returns:
            Resource if found, None otherwise
        """
        resource_id = self._find_resource_id(key)
        return self.resource_map.get(resource_id) if resource_id else None

    def __getitem__(self, key) -> Resource:
        """
        Dictionary-style access with KeyError for missing resources.

        Args:
            key: ResourceId object, ID string, or URL

        Returns:
            Resource for the key

        Raises:
            KeyError: If resource not found or has no content
        """
        resource = self.get(key)
        if resource is None:
            if key in self:
                raise KeyError(f"Resource {key} registered but has no content")
            else:
                raise KeyError(f"Resource {key} not found")
        return resource

    def __contains__(self, item) -> bool:
        """
        Check if resource exists in pool using 'in' operator.

        Supports URL strings, ID strings, and ResourceId objects.

        Examples:
            "https://example.com" in pool
            "1_a1b2c3d4" in pool
            resource_id in pool
        """
        return self._find_resource_id(item) is not None

    @property
    def resources(self) -> List[Resource]:
        """Get list of all resources with content."""
        return [
            resource for resource in self.resource_map.values() if resource is not None
        ]

    def __repr__(self) -> str:
        """Informative representation for REPL display."""
        total_resources = len(self.resource_map)
        loaded_resources = len(self.resources)
        return f"ResourcePool({loaded_resources}/{total_resources} resources loaded)"

    @model_serializer(mode="wrap")
    def _serialize(self, serializer, info):
        """Custom serializer that converts resource_map to a list format.

        Serializes as list of dicts with structure:
        [{url: str, title?: str, text?: str, chunks?: [...], id?: str}]

        Resources are serialized in counter order (sorted by counter extracted from ID).
        The id field is only included if it doesn't match the expected pattern
        (counter = list index + 1). This makes the common case more compact.

        Resources with None content only include url (and id if non-standard).
        """
        if info.mode == "json":
            # Sort resources by counter (extracted from ID) for predictable ordering
            sorted_items = sorted(
                self.resource_map.items(),
                key=lambda item: int(item[0].id.split("_")[0]),
            )

            resources_list = []
            for idx, (resource_id, resource) in enumerate(sorted_items, start=1):
                entry = {"url": resource_id.url}

                # Only include content fields if resource is not None
                if resource is not None:
                    entry["title"] = resource.title
                    entry["text"] = resource.text
                    entry["chunks"] = resource.chunks
                    # Only include DOI if present
                    if resource.doi is not None:
                        entry["doi"] = resource.doi
                    # Only include publication_date if present
                    if resource.publication_date is not None:
                        entry["publication_date"] = resource.publication_date

                # Only include id if it doesn't match expected pattern
                # Expected pattern: "{counter}_{hash}" where counter = idx
                expected_counter = idx
                actual_counter = int(resource_id.id.split("_")[0])
                if actual_counter != expected_counter:
                    # Non-standard counter, must include full ID
                    entry["id"] = resource_id.id

                resources_list.append(entry)

            # Return just the list, not wrapped in {"resources": ...}
            # When serialized as a field, Pydantic will handle the field name
            return resources_list
        else:
            # For non-JSON modes, use default serialization
            return serializer(self)


class ResourceQuote(BaseModel):
    """
    All occurrences of a quote phrase within a document resource.

    Associates extracted content with all locations in the source document
    where the phrase appears, enabling comprehensive validation and citation.
    """

    resource: Resource = Field(description="Source document resource with full content")
    query_text: str = Field(description="Original query text that was searched for")
    spans: List[Tuple[int, int]] = Field(
        description="List of (start, end) character spans for all segments"
    )
    is_disjoint: bool = Field(
        description="True if this quote contains ellipsis markers"
    )

    # Fuzzy matching metadata (optional, populated when fuzzy matching used)
    fuzzy_corrected: bool = Field(
        default=False, description="Whether quote was auto-corrected via fuzzy matching"
    )
    original_query: Optional[str] = Field(
        default=None, description="Original query text if fuzzy-corrected"
    )
    fuzzy_similarity: Optional[float] = Field(
        default=None, description="Fuzzy similarity score if fuzzy matching was used"
    )

    @model_serializer(mode="wrap")
    def _serialize(self, serializer, info):
        """Replace Resource with resource_url reference during JSON serialization."""
        if info.mode == "json":
            data = serializer(self)
            # Replace full resource with just URL reference
            data["resource_url"] = self.resource.id.url
            del data["resource"]
            return data
        else:
            return serializer(self)

    def __init__(
        self,
        resource: Resource = None,
        text: str = None,
        similarity_threshold: float = 1.0,
        **data,
    ):
        """
        Create ResourceQuote by finding all occurrences of text in resource.

        Tries multiple strategies in order:
        0. Verbatim match in original text
        1. Normalized match (handles formatting)
        2. Shorthand expansion (e.g., "ISCA1,2" → "ISCA1")
        3. Fuzzy matching (if similarity_threshold < 1.0)

        Args:
            resource: Resource to search within
            text: Text quote to locate
            similarity_threshold: Minimum similarity for matches (default: 1.0).
                - 1.0: Exact matching only (strategies 0-2)
                - <1.0: Enable fuzzy matching, auto-accept above this threshold

        Raises:
            QuoteNotFoundError: If quote not found (no match at all)
            QuoteNearMatchError subclasses: If similarity between FUZZY_SUGGESTION_THRESHOLD
                and similarity_threshold (includes error-specific diagnostics)
        """
        # Allow direct construction if spans and query_text are provided (for deserialization)
        if "spans" in data and "query_text" in data:
            # Set is_disjoint if not provided
            if "is_disjoint" not in data:
                data["is_disjoint"] = (
                    "..." in data["query_text"] or "…" in data["query_text"]
                )
            super().__init__(resource=resource, **data)
            return

        # Normal construction requires resource and text
        if resource is None or text is None:
            raise ValueError("resource and text are required for quote matching")

        # Strategy 0: Try verbatim match in original text (preserves formatting)
        verbatim_spans = self._find_verbatim(resource.text, text)
        if verbatim_spans:
            super().__init__(
                resource=resource,
                query_text=text,
                spans=verbatim_spans,
                is_disjoint=False,
                fuzzy_corrected=False,
            )
            return

        # Strategy 1: Try normalized match (handles formatting differences)
        try:
            spans, is_disjoint = self._find_normalized(resource, text)
            super().__init__(
                resource=resource,
                query_text=text,
                spans=spans,
                is_disjoint=is_disjoint,
                fuzzy_corrected=False,
            )
            return
        except ValueError:
            pass

        # Strategy 2: Try shorthand expansion
        for variant in expand_scientific_shorthand(text):
            try:
                spans, is_disjoint = self._find_normalized(resource, variant)
                super().__init__(
                    resource=resource,
                    query_text=variant,
                    spans=spans,
                    is_disjoint=is_disjoint,
                    fuzzy_corrected=False,
                )
                return
            except ValueError:
                continue

        # Strategy 3: Fuzzy matching (if enabled via similarity_threshold < 1.0)
        if similarity_threshold < 1.0:
            fuzzy_result = fuzzy_match_quote(
                text, resource.normalized_text, threshold=FUZZY_SUGGESTION_THRESHOLD
            )

            if fuzzy_result and fuzzy_result.similarity >= similarity_threshold:
                # Auto-accept: create ResourceQuote with corrected text
                try:
                    spans, is_disjoint = self._find_normalized(
                        resource, fuzzy_result.corrected_quote
                    )
                    query_text = fuzzy_result.corrected_quote
                except ValueError:
                    (
                        spans,
                        is_disjoint,
                        query_text,
                    ) = self._spans_from_fuzzy_alignment(resource, fuzzy_result)

                super().__init__(
                    resource=resource,
                    query_text=query_text,
                    spans=spans,
                    is_disjoint=is_disjoint,
                    fuzzy_corrected=True,
                    original_query=text,
                    fuzzy_similarity=fuzzy_result.similarity,
                )
                return

            # Below threshold: raise error with suggestions
            if fuzzy_result:
                # Simple fuzzy suggestion (for now, just use ParaphraseError)
                # Later can integrate V2 alignment system for richer error classification
                suggestions = [
                    CorrectionSuggestion(
                        text=fuzzy_result.corrected_quote,
                        confidence=fuzzy_result.similarity,
                        similarity=fuzzy_result.similarity,
                        explanation="Fuzzy match from RapidFuzz + difflib alignment",
                    )
                ]
                raise ParaphraseError(
                    quote_text=text,
                    resource=resource,
                    similarity_threshold=similarity_threshold,
                    suggestions=suggestions,
                    similarity=fuzzy_result.similarity,
                    coverage_ratio=fuzzy_result.similarity,  # Approximate
                    matching_words=[],  # TODO: extract from fuzzy_result if needed
                )

        # No match found with any strategy
        raise QuoteNotFoundError(
            quote_text=text,
            resource=resource,
            similarity_threshold=similarity_threshold,
        )

    def _find_verbatim(
        self, original_text: str, quote_text: str
    ) -> List[Tuple[int, int]]:
        """Find all verbatim occurrences in original text."""
        spans = []
        pos = 0
        while True:
            idx = original_text.find(quote_text, pos)
            if idx == -1:
                break
            spans.append((idx, idx + len(quote_text)))
            pos = idx + 1
        return spans

    def _find_normalized(
        self, resource: Resource, text: str
    ) -> Tuple[List[Tuple[int, int]], bool]:
        """
        Find normalized match and return (spans, is_disjoint).

        Raises ValueError if not found.
        """
        # Check if this is a disjoint quote (both ... and …)
        segments = re.split(r"\s*(?:\.{3,}|…)\s*", text)
        segments = [s.strip() for s in segments if s.strip()]
        is_disjoint = len(segments) > 1

        if is_disjoint:
            # Find all occurrences where segments appear in order
            normalized_text = resource.normalized_text
            normalized_segments = [
                NormalizedTextMapper.normalize(seg) for seg in segments
            ]

            all_spans = []
            pos = 0
            while True:
                # Find first segment starting from pos
                first_match = self._find_next_match(
                    normalized_text, normalized_segments[0], pos
                )
                if not first_match:
                    break

                # Try to find remaining segments in order
                spans = [first_match]
                current_pos = first_match[1]

                for seg in normalized_segments[1:]:
                    next_match = self._find_next_match(
                        normalized_text, seg, current_pos
                    )
                    if not next_match:
                        break
                    spans.append(next_match)
                    current_pos = next_match[1]

                if len(spans) == len(normalized_segments):
                    # All segments found - add to results
                    all_spans.extend(spans)

                pos = first_match[0] + 1  # Continue searching

            if not all_spans:
                raise ValueError(f"Quote text not found in resource: {text!r}")

            spans = self._original_positions(resource, all_spans)
            return (spans, True)
        else:
            # Continuous quote - find all matches by looping
            normalized_text = resource.normalized_text
            normalized_pattern = NormalizedTextMapper.normalize(text)

            norm_spans = []
            pos = 0
            while True:
                match = self._find_next_match(normalized_text, normalized_pattern, pos)
                if not match:
                    break
                norm_spans.append(match)
                pos = match[0] + 1  # Continue from next character

        if not norm_spans:
            raise ValueError(f"Quote text not found in resource: {text!r}")

        spans = self._original_positions(resource, norm_spans)
        return (spans, False)

    def _spans_from_fuzzy_alignment(
        self, resource: Resource, fuzzy_result: FuzzyMatchResult
    ) -> Tuple[List[Tuple[int, int]], bool, str]:
        """
        Convert RapidFuzz alignment data to original document spans.

        This acts as a robust fallback when the normalized lookup fails—rather than
        re-searching for the corrected quote, we directly map the alignment blocks
        back to the original text, ensuring we can always surface a quote span when
        similarity is above the acceptance threshold.
        """
        candidate_spans: List[Tuple[int, int]] = []

        for _, norm_start, length in fuzzy_result.match_blocks:
            if length <= 0:
                continue

            orig_start, orig_end = resource.map_normalized_to_original_position(
                norm_start, length
            )

            if orig_start is None or orig_end is None:
                # As a safety net, try expanding the normalized window by one character
                # on either side before giving up entirely.
                expanded_start = max(norm_start - 1, 0)
                expanded_length = min(
                    length + 2, len(resource.normalized_text) - expanded_start
                )
                orig_start, orig_end = resource.map_normalized_to_original_position(
                    expanded_start, expanded_length
                )

            if orig_start is None or orig_end is None or orig_start >= orig_end:
                continue

            # Expand to word boundaries to avoid mid-word splits
            orig_start, orig_end = _expand_to_word_boundaries(
                resource.text, orig_start, orig_end
            )

            candidate_spans.append((orig_start, orig_end))

        if not candidate_spans:
            raise ValueError("Unable to map fuzzy alignment blocks to document spans")

        # Merge overlapping or adjacent spans to reduce fragmentation
        candidate_spans.sort()
        merged_spans: List[Tuple[int, int]] = []
        for start, end in candidate_spans:
            if not merged_spans:
                merged_spans.append((start, end))
                continue

            last_start, last_end = merged_spans[-1]
            if start <= last_end + 1:
                merged_spans[-1] = (last_start, max(last_end, end))
            else:
                merged_spans.append((start, end))

        is_disjoint = len(merged_spans) > 1

        # Build query text directly from the original document spans.
        if is_disjoint:
            segments = [resource.text[s:e].strip() for s, e in merged_spans]
            query_text = " ... ".join(seg for seg in segments if seg)
        else:
            span_start, span_end = merged_spans[0]
            query_text = resource.text[span_start:span_end]

        if not query_text.strip():
            raise ValueError("Fuzzy alignment produced empty quote text")

        return merged_spans, is_disjoint, query_text

    def _find_next_match(
        self, normalized_text: str, pattern: str, start: int = 0
    ) -> Optional[Tuple[int, int]]:
        """Find next occurrence of pattern in normalized text, return normalized span."""
        pos = normalized_text.find(pattern, start)
        if pos == -1:
            return None
        return (pos, pos + len(pattern))

    def _original_positions(
        self, resource: Resource, norm_spans: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Map normalized spans to original positions with word boundary adjustment."""
        original_spans = []
        for norm_start, norm_end in norm_spans:
            orig_start, orig_end = resource.map_normalized_to_original_position(
                norm_start, norm_end - norm_start
            )
            if orig_start is not None and orig_end is not None:
                # Expand to word boundaries to avoid mid-word splits
                orig_start, orig_end = _expand_to_word_boundaries(
                    resource.text, orig_start, orig_end
                )
                original_spans.append((orig_start, orig_end))
        return original_spans

    @field_validator("spans")
    @classmethod
    def validate_spans_within_text(cls, v, info: ValidationInfo):
        """Ensure all spans are within the resource text bounds and properly ordered"""
        if info.data and "resource" in info.data:
            resource = info.data["resource"]
            text_length = len(resource.text)

            # Validate each span
            for i, (start, end) in enumerate(v):
                if not (0 <= start < end <= text_length):
                    bound_msg = (
                        f"within text bounds (0-{text_length})"
                        if start < 0 or end > text_length
                        else "start < end"
                    )
                    raise ValueError(
                        f"Span {i + 1} ({start}-{end}) must be {bound_msg}"
                    )
        return v

    @property
    def count(self) -> int:
        """Number of occurrences found."""
        if self.is_disjoint:
            # For disjoint quotes, count how many segment groups we have
            segments_in_query = len(
                [
                    s.strip()
                    for s in re.split(r"\s*(?:\.{3,}|…)\s*", self.query_text)
                    if s.strip()
                ]
            )
            return len(self.spans) // segments_in_query if segments_in_query > 0 else 0
        else:
            # For continuous quotes, each span is one occurrence
            return len(self.spans)

    def get_quote_text(self, occurrence: int = 1) -> str:
        """
        Extract quote text from a specific occurrence.

        Args:
            occurrence: Which occurrence to get (1-based)

        Returns:
            Quote text from the specified occurrence, joining segments for disjoint quotes

        Raises:
            IndexError: If occurrence doesn't exist
        """
        if occurrence < 1 or occurrence > self.count:
            raise IndexError(
                f"Occurrence {occurrence} not found (have {self.count} occurrences)"
            )

        if self.is_disjoint:
            # For disjoint quotes, get segments for this occurrence
            segments_in_query = len(
                [
                    s.strip()
                    for s in re.split(r"\s*(?:\.{3,}|…)\s*", self.query_text)
                    if s.strip()
                ]
            )
            start_idx = (occurrence - 1) * segments_in_query
            end_idx = start_idx + segments_in_query
            occurrence_spans = self.spans[start_idx:end_idx]

            # Extract and join with ellipses
            segments = []
            for start, end in occurrence_spans:
                segments.append(self.resource.text[start:end])
            return " ... ".join(segments)
        else:
            # For continuous quotes, just get the single span
            start, end = self.spans[occurrence - 1]
            return self.resource.text[start:end]

    def get_all_quote_texts(self) -> List[str]:
        """
        Extract quote text from all occurrences.

        Returns:
            List of quote texts for all occurrences
        """
        return [self.get_quote_text(i + 1) for i in range(self.count)]

    def get_context(self, occurrence: int = 1, context_chars: int = 200) -> str:
        """
        Get surrounding context around a specific quote occurrence.

        Args:
            occurrence: Which occurrence to get context for (1-based)
            context_chars: Number of characters to include before/after quote

        Returns:
            Context text with quote highlighted, showing each segment for disjoint quotes

        Raises:
            IndexError: If occurrence doesn't exist
        """
        if occurrence < 1 or occurrence > self.count:
            raise IndexError(
                f"Occurrence {occurrence} not found (have {self.count} occurrences)"
            )

        if self.is_disjoint:
            # For disjoint quotes, get segments for this occurrence
            segments_in_query = len(
                [
                    s.strip()
                    for s in re.split(r"\s*(?:\.{3,}|…)\s*", self.query_text)
                    if s.strip()
                ]
            )
            start_idx = (occurrence - 1) * segments_in_query
            end_idx = start_idx + segments_in_query
            occurrence_spans = self.spans[start_idx:end_idx]
        else:
            # For continuous quotes, just get the single span for this occurrence
            occurrence_spans = [self.spans[occurrence - 1]]

        text = self.resource.text

        if len(occurrence_spans) == 1:
            # Single continuous segment
            start, end = occurrence_spans[0]
            context_start = max(0, start - context_chars)
            context_end = min(len(text), end + context_chars)
            before = text[context_start:start]
            quote = text[start:end]
            after = text[end:context_end]
            return f"{before}**{quote}**{after}"
        else:
            # Multiple segments - show context around each
            contexts = []
            for start, end in occurrence_spans:
                ctx_start = max(0, start - context_chars)
                ctx_end = min(len(text), end + context_chars)
                before = text[ctx_start:start]
                segment = text[start:end]
                after = text[end:ctx_end]
                contexts.append(f"{before}**{segment}**{after}")
            return "\n...\n".join(contexts)

    def __repr__(self) -> str:
        """Rich representation for REPL display."""
        # First line: ResourceId
        lines = [f"ResourceQuote for {self.resource.id!r}:"]

        # Following lines: Each quote with position
        for i in range(self.count):
            quote_text = self.get_quote_text(i + 1)
            if self.is_disjoint:
                # For disjoint quotes, show overall span range
                segments_per = len(
                    [
                        s.strip()
                        for s in re.split(r"\s*(?:\.{3,}|…)\s*", self.query_text)
                        if s.strip()
                    ]
                )
                start_idx = i * segments_per
                end_idx = start_idx + segments_per
                spans_for_occurrence = self.spans[start_idx:end_idx]
                first_start = spans_for_occurrence[0][0]
                last_end = spans_for_occurrence[-1][1]
                lines.append(f"  [{i + 1}] '{quote_text}' at {first_start}-{last_end}")
            else:
                start, end = self.spans[i]
                lines.append(f"  [{i + 1}] '{quote_text}' at {start}-{end}")

        return "\n".join(lines)

    @property
    def chunk_indices(self) -> List[int]:
        """
        Get chunk indices for all quote spans.

        Returns:
            List of unique chunk indices containing this quote, sorted
        """
        chunks = set()
        for start, end in self.spans:
            chunks.update(self.resource.get_chunks_for_span(start, end))
        return sorted(chunks)

    def get_chunk_contexts(self, chunk_padding: int = 50) -> List[str]:
        """
        Get the text context within each chunk containing this quote.

        Args:
            chunk_padding: Characters to include before/after quote within chunk

        Returns:
            List of context strings, one per chunk containing the quote
        """
        contexts = []
        for chunk_idx in self.chunk_indices:
            chunk_text = self.resource.get_chunk_text(chunk_idx)
            if chunk_text is None:
                continue

            chunk_start, chunk_end = self.resource.chunks[chunk_idx]

            # Find quote spans within this chunk
            quote_spans_in_chunk = []
            for start, end in self.spans:
                if chunk_start <= start < chunk_end or chunk_start < end <= chunk_end:
                    # Convert absolute positions to chunk-relative positions
                    rel_start = max(0, start - chunk_start)
                    rel_end = min(len(chunk_text), end - chunk_start)
                    quote_spans_in_chunk.append((rel_start, rel_end))

            if quote_spans_in_chunk:
                # Get context around the quote spans within the chunk
                first_span_start = min(span[0] for span in quote_spans_in_chunk)
                last_span_end = max(span[1] for span in quote_spans_in_chunk)

                ctx_start = max(0, first_span_start - chunk_padding)
                ctx_end = min(len(chunk_text), last_span_end + chunk_padding)

                context = chunk_text[ctx_start:ctx_end]
                contexts.append(context)

        return contexts

    def get_chunk_text_for_occurrence(self, occurrence: int = 1) -> Optional[str]:
        """
        Get the chunk text containing a specific occurrence of the quote.

        Args:
            occurrence: Which occurrence to get chunk for (1-based)

        Returns:
            Full text of the chunk containing this occurrence, or None if not found

        Raises:
            IndexError: If occurrence doesn't exist
        """
        if occurrence < 1 or occurrence > self.count:
            raise IndexError(
                f"Occurrence {occurrence} not found (have {self.count} occurrences)"
            )

        if self.is_disjoint:
            # For disjoint quotes, use the first segment's position
            segments_in_query = len(
                [
                    s.strip()
                    for s in re.split(r"\s*(?:\.{3,}|…)\s*", self.query_text)
                    if s.strip()
                ]
            )
            span_idx = (occurrence - 1) * segments_in_query
        else:
            # For continuous quotes
            span_idx = occurrence - 1

        start, _ = self.spans[span_idx]
        chunk_idx = self.resource.get_chunk_for_position(start)

        if chunk_idx is not None:
            return self.resource.get_chunk_text(chunk_idx)
        return None
