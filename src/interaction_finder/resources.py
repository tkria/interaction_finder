"""
Resource management models for document sources and quote tracking.

This module provides structured management of document resources and validates
that extracted information can be traced back to specific documents with
supporting quotes.
"""

import hashlib
from urllib.parse import urlparse, urlunparse
import re
import bisect
import unicodedata
from difflib import SequenceMatcher
from typing import List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, ValidationInfo


# Greek letter mappings for scientific text normalization (lowercase only)
GREEK_LETTER_MAP = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "ς": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}


def normalize_text_for_matching(text: str) -> str:
    """
    Normalize text for fuzzy quote matching.

    Converts to lowercase, handles Unicode normalization, converts Greek letters
    to ASCII equivalents, removes punctuation, and normalizes whitespace to make
    quote matching more robust against formatting differences.

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text suitable for comparison
    """
    # Apply Unicode normalization first
    unicode_text = unicodedata.normalize("NFD", text)
    unicode_text = "".join(c for c in unicode_text if unicodedata.category(c) != "Mn")

    normalized = []
    text_len = len(unicode_text)
    last_was_space = True

    for i, char in enumerate(unicode_text):
        char_lower = char.lower()

        # ASCII alphanumeric - fast path
        if char_lower.isascii() and char_lower.isalnum():
            normalized.append(char_lower)
            last_was_space = False

        # Greek letters
        elif 0x0370 <= ord(char) <= 0x03FF and char_lower in GREEK_LETTER_MAP:
            # Add space before if needed
            if normalized and normalized[-1].isalnum():
                normalized.append(" ")
            # Add Greek name
            normalized.extend(GREEK_LETTER_MAP[char_lower])
            # Add space after if needed
            if i + 1 < text_len and unicode_text[i + 1].isalnum():
                normalized.append(" ")
            last_was_space = False

        # Skip contractions and decimals
        elif (
            char == "'"
            and i > 0
            and i < text_len - 1
            and unicode_text[i - 1].isalnum()
            and unicode_text[i + 1].isalnum()
        ) or (
            char == "."
            and i > 0
            and i < text_len - 1
            and unicode_text[i - 1].isdigit()
            and unicode_text[i + 1].isdigit()
        ):
            continue

        # Other alphanumeric
        elif char_lower.isalnum():
            normalized.append(char_lower)
            last_was_space = False

        # Everything else becomes space
        elif not last_was_space:
            normalized.append(" ")
            last_was_space = True

    return "".join(normalized).strip()


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


class FuzzySuggestion(BaseModel):
    """
    Suggestion for LLM correction when fuzzy match is below auto-correct threshold.

    Provides the best matching segment from the document for LLM to consider
    when correcting the quote.
    """

    suggested_quote: str = Field(
        description="Suggested quote from document that best matches LLM output"
    )
    similarity: float = Field(
        description="Similarity score between LLM quote and suggestion (0.0-1.0)"
    )
    original_quote: str = Field(description="Original LLM quote for reference")


def fuzzy_match_quote(
    llm_quote: str,
    document_segment: str,
    threshold: float = 0.90,
) -> Optional[FuzzyMatchResult]:
    """
    Fuzzy match LLM-generated quote against document segment using difflib.

    Uses SequenceMatcher for similarity calculation and alignment-based extraction.
    Returns auto-corrected quote if similarity meets threshold, otherwise None.

    Args:
        llm_quote: Quote text from LLM (may be slightly paraphrased)
        document_segment: Segment of document to match against
        threshold: Minimum similarity for auto-correction (default: 0.90)

    Returns:
        FuzzyMatchResult with corrected quote and metadata, or None if below threshold

    Notes:
        - Uses normalized text for comparison (via normalize_text_for_matching)
        - Threshold of 0.90 is conservative to avoid false corrections
        - Very short quotes (<5 chars) are unreliable and return None
    """
    # Edge case: skip very short quotes (too unreliable)
    if len(llm_quote.strip()) < 5:
        return None

    # Normalize both for comparison
    normalized_llm = normalize_text_for_matching(llm_quote)
    normalized_doc = normalize_text_for_matching(document_segment)

    # Edge case: empty after normalization
    if not normalized_llm or not normalized_doc:
        return None

    # Calculate similarity using SequenceMatcher
    matcher = SequenceMatcher(None, normalized_llm, normalized_doc)
    similarity = matcher.ratio()

    # Return None if below threshold
    if similarity < threshold:
        return None

    # Extract aligned text from document using matching blocks
    corrected_quote = _auto_correct_quote_from_alignment(
        normalized_llm, normalized_doc, matcher
    )

    # Get match blocks for provenance
    match_blocks = matcher.get_matching_blocks()

    return FuzzyMatchResult(
        corrected_quote=corrected_quote,
        similarity=similarity,
        match_blocks=match_blocks,
    )


def _auto_correct_quote_from_alignment(
    normalized_llm: str,
    normalized_doc: str,
    matcher: SequenceMatcher,
) -> str:
    """
    Extract corrected quote from document using SequenceMatcher alignment.

    Builds corrected quote by extracting aligned portions from the document,
    ensuring the result actually exists in the document.

    Args:
        normalized_llm: Normalized LLM quote text
        normalized_doc: Normalized document text
        matcher: Pre-configured SequenceMatcher for the texts

    Returns:
        Corrected quote string extracted from document

    Notes:
        - Uses get_matching_blocks() to identify aligned segments
        - Extracts text from document at aligned positions
        - Preserves word boundaries and spacing
    """
    # Get matching blocks: (llm_pos, doc_pos, length) tuples
    blocks = matcher.get_matching_blocks()

    # Extract aligned segments from document
    segments = []
    for llm_pos, doc_pos, length in blocks:
        if length > 0:  # Skip dummy block at end
            segment = normalized_doc[doc_pos : doc_pos + length]
            segments.append(segment)

    # Join segments with single space
    corrected = " ".join(segments)

    # Clean up multiple spaces and strip
    corrected = " ".join(corrected.split())

    return corrected


def find_quote_with_fuzzy_matching(
    resource: "Resource",
    quote_text: str,
    auto_correct_threshold: float = 0.90,
    suggest_threshold: float = 0.75,
) -> Union["ResourceQuote", FuzzySuggestion, None]:
    """
    Find quote in resource with multi-strategy matching including fuzzy matching.

    Tries multiple strategies in order:
    1. Exact match via resource.quote()
    2. Normalized match (existing normalization)
    3. Shorthand expansion (e.g., "ISCA1,2" → "ISCA1" or "ISCA2")
    4. Fuzzy matching with auto-correction (≥90% similarity)
    5. Fuzzy matching with suggestion (75-90% similarity)

    Args:
        resource: Resource to search within
        quote_text: Quote text to find
        auto_correct_threshold: Similarity threshold for auto-correction (default: 0.90)
        suggest_threshold: Similarity threshold for suggestions (default: 0.75)

    Returns:
        - ResourceQuote: If exact or auto-corrected match found
        - FuzzySuggestion: If 75-90% similarity (for LLM correction)
        - None: If no match found (<75% similarity)

    Notes:
        - Fuzzy matching searches against entire normalized document text
        - High auto-correct threshold (90%) prevents false positives
        - Suggestion threshold (75%) provides helpful hints to LLM
    """
    # Strategy 1: Try exact match
    try:
        return resource.quote(quote_text)
    except ValueError:
        pass

    # Strategy 2: Try normalized match
    normalized_quote = normalize_text_for_matching(quote_text)
    try:
        return resource.quote(normalized_quote)
    except ValueError:
        pass

    # Strategy 3: Try shorthand expansion
    expanded_variants = expand_scientific_shorthand(quote_text)
    for variant in expanded_variants:
        try:
            return resource.quote(variant)
        except ValueError:
            continue

    # Strategy 4 & 5: Try fuzzy matching against entire document
    # Use resource.normalized_text as single candidate segment
    fuzzy_result = fuzzy_match_quote(
        quote_text,
        resource.normalized_text,
        threshold=suggest_threshold,  # Use lower threshold for suggestions
    )

    if fuzzy_result is None:
        return None

    # If similarity is high enough for auto-correction
    if fuzzy_result.similarity >= auto_correct_threshold:
        # Try to create ResourceQuote with corrected text
        try:
            return resource.quote(fuzzy_result.corrected_quote)
        except ValueError:
            # Fall through to suggestion if quote creation fails
            pass

    # Return suggestion for LLM correction (75-90% similarity)
    return FuzzySuggestion(
        suggested_quote=fuzzy_result.corrected_quote,
        similarity=fuzzy_result.similarity,
        original_quote=quote_text,
    )


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
            normalized_chunk = normalize_text_for_matching(chunk_text)
            normalized_search = normalize_text_for_matching(full_text[search_start:])

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


class ResourceId(BaseModel):
    """
    Lightweight identifier for a document resource.

    Automatically generates stable hash-based ID from URL and counter.
    """

    id: str = Field(description="Stable hash-based resource identifier")
    url: str = Field(description="Original URL of the resource")

    def __init__(self, url: str, counter: int, **data):
        """
        Create ResourceId with automatic ID generation.

        Args:
            url: Document URL
            counter: Sequential counter for this resource
        """
        # Generate stable ID using same scheme as cache hashing:
        # sha256(normalize_url(url)) first 4 bytes → 8 hex chars
        # Normalize URL inline (remove fragment) to align with cache hashing behavior
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
    position_offsets: List[Tuple[int, int]] = Field(
        default_factory=list,
        exclude=True,
        description="Binary-searchable list of (normalized_pos, original_pos) for position mapping",
    )
    chunks: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="List of (start, end) character positions for document chunks",
    )

    def __init__(
        self,
        id: ResourceId,
        title: str,
        text: str,
        chunks: Optional[List[Tuple[int, int]]] = None,
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
        """
        # Build normalized text with Greek letter support
        normalized_text, position_offsets = self._build_normalized_text_and_offsets(
            text
        )

        # Default chunks to entire document if not provided
        if chunks is None:
            chunks = [(0, len(text))]

        super().__init__(
            id=id,
            title=title,
            text=text,
            normalized_text=normalized_text,
            chunks=chunks,
            **data,
        )
        self.position_offsets = position_offsets

    @staticmethod
    def _build_normalized_text_and_offsets(
        original_text: str,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Build normalized text and position offset mapping simultaneously.

        Args:
            original_text: Original document text

        Returns:
            Tuple of (normalized_text, position_offsets) where position_offsets
            is a list of (normalized_pos, original_pos) tuples for binary search
        """
        # Apply Unicode normalization first
        unicode_text = unicodedata.normalize("NFD", original_text)
        unicode_text = "".join(
            c for c in unicode_text if unicodedata.category(c) != "Mn"
        )

        normalized = []
        position_offsets = []
        text_len = len(unicode_text)
        last_was_space = True  # Start as True to avoid leading spaces

        def _should_skip_char(char: str, pos: int) -> bool:
            """Check if character should be skipped (contractions, decimals)."""
            if pos == 0 or pos >= text_len - 1:
                return False
            prev_char, next_char = unicode_text[pos - 1], unicode_text[pos + 1]
            return (char == "'" and prev_char.isalnum() and next_char.isalnum()) or (
                char == "." and prev_char.isdigit() and next_char.isdigit()
            )

        i = 0
        while i < text_len:
            char = unicode_text[i]

            # Fast path for ASCII alphanumeric (most common case)
            if "a" <= char <= "z" or "0" <= char <= "9":
                position_offsets.append((len(normalized), i))
                normalized.append(char)
                last_was_space = False

            elif "A" <= char <= "Z":
                position_offsets.append((len(normalized), i))
                normalized.append(char.lower())
                last_was_space = False

            # Greek letters (Unicode range check first for performance)
            elif 0x0370 <= ord(char) <= 0x03FF:
                char_lower = char.lower()
                if char_lower in GREEK_LETTER_MAP:
                    # Add space before if needed
                    if normalized and normalized[-1].isalnum():
                        position_offsets.append((len(normalized), i))
                        normalized.append(" ")
                    # Add Greek name
                    position_offsets.append((len(normalized), i))
                    normalized.extend(GREEK_LETTER_MAP[char_lower])
                    # Add space after if needed
                    if i + 1 < text_len and unicode_text[i + 1].isalnum():
                        normalized.append(" ")
                    last_was_space = False
                else:
                    # Non-Greek unicode letter
                    if char.isalnum():
                        position_offsets.append((len(normalized), i))
                        normalized.append(char_lower)
                        last_was_space = False
                    elif not last_was_space:
                        position_offsets.append((len(normalized), i))
                        normalized.append(" ")
                        last_was_space = True

            # Skip contractions and decimal points
            elif _should_skip_char(char, i):
                pass

            # Other alphanumeric characters
            elif char.isalnum():
                position_offsets.append((len(normalized), i))
                normalized.append(char.lower())
                last_was_space = False

            # Convert everything else to single space
            elif not last_was_space:
                position_offsets.append((len(normalized), i))
                normalized.append(" ")
                last_was_space = True

            i += 1

        # Final result and position mapping
        normalized_text = "".join(normalized).strip()
        position_offsets.append((len(normalized_text), len(original_text)))

        return normalized_text, position_offsets

    def quote(self, text: str) -> "ResourceQuote":
        """
        Create a ResourceQuote by finding all occurrences of the given text in this resource.

        Args:
            text: Text to find and quote

        Returns:
            ResourceQuote with all occurrences

        Raises:
            ValueError: If quote text is not found in the resource
        """
        return ResourceQuote(self, text)

    def __repr__(self) -> str:
        """Informative representation for REPL display."""
        text_preview = (self.text[:50] + "...") if len(self.text) > 50 else self.text
        return (
            f"Resource(id='{self.id.id}', title='{self.title}', "
            f"text='{text_preview}', {len(self.text)} chars)"
        )

    def _find_original_position(self, normalized_pos: int) -> Optional[int]:
        """
        Find original text position for a normalized text position using binary search.

        Args:
            normalized_pos: Position in normalized text

        Returns:
            Original text position or None if mapping fails
        """
        idx = bisect.bisect_left(self.position_offsets, (normalized_pos, 0))
        if idx < len(self.position_offsets):
            # Check if we have exact match or need closest
            if (
                idx > 0
                and self.position_offsets[idx][0] != normalized_pos
                and self.position_offsets[idx - 1][0] <= normalized_pos
            ):
                idx -= 1
            return self.position_offsets[idx][1]
        return None

    def map_normalized_to_original_position(
        self, norm_start: int, norm_length: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Map character positions from normalized text back to original text using binary search.

        Uses the precomputed position offset list for efficient O(log n) lookups.

        Args:
            norm_start: Start position in normalized text
            norm_length: Length of span in normalized text

        Returns:
            Tuple of (original_start, original_end) or (None, None) if mapping fails
        """
        original_start = self._find_original_position(norm_start)
        original_end = self._find_original_position(norm_start + norm_length)
        return original_start, original_end

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


class ResourcePool(BaseModel):
    """
    Collection of document resources with separate ID registration and content storage.

    Supports registering resource IDs first, then loading content separately,
    allowing for flexible workflows where resource lists are known before content.
    """

    resource_map: dict[ResourceId, Optional[Resource]] = Field(default_factory=dict)

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
        for resource_id in self.resource_map.keys():
            if resource_id.url == url:
                raise ValueError(f"Resource with url {url} already exists")

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
    ) -> Resource:
        """
        Add content to a previously registered resource.

        Args:
            resource_id: Previously registered ResourceId
            title: Human-readable document title
            document_text: Full text content of the document
            chunks: Optional list of (start, end) positions for document chunks

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
            id=resource_id, title=title, text=document_text, chunks=chunks
        )
        self.resource_map[resource_id] = resource

        return resource

    def add(
        self,
        url: str,
        title: str,
        document_text: str,
        chunks: Optional[List[Tuple[int, int]]] = None,
    ) -> Resource:
        """
        Add a complete resource (register ID + content) in one step.

        Args:
            url: Document URL (must be unique)
            title: Human-readable document title
            document_text: Full text content of the document
            chunks: Optional list of (start, end) positions for document chunks

        Returns:
            Complete Resource

        Raises:
            ValueError: If URL already exists in pool
        """
        resource_id = self.register(url)
        return self.add_content(resource_id, title, document_text, chunks)

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

    def __init__(self, resource: Resource, text: str, **data):
        """
        Create ResourceQuote by finding all occurrences of text in resource.

        Uses normalized text comparison for robust matching against formatting
        differences while maintaining precise character spans in original text.

        Args:
            resource: Resource to search within
            text: Text quote to locate

        Raises:
            ValueError: If quote text is not found in the resource
        """
        # Allow direct construction if spans and query_text are provided
        if "spans" in data and "query_text" in data:
            # Set is_disjoint if not provided
            if "is_disjoint" not in data:
                data["is_disjoint"] = (
                    "..." in data["query_text"] or "…" in data["query_text"]
                )
            super().__init__(resource=resource, **data)
            return

        # Check if this is a disjoint quote (both ... and …)
        segments = re.split(r"\s*(?:\.{3,}|…)\s*", text)
        segments = [s.strip() for s in segments if s.strip()]
        is_disjoint = len(segments) > 1

        if is_disjoint:
            # Find all occurrences where segments appear in order
            normalized_text = resource.normalized_text
            normalized_segments = [normalize_text_for_matching(seg) for seg in segments]

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

            super().__init__(
                resource=resource,
                query_text=text,
                spans=spans,
                is_disjoint=True,
            )
        else:
            # Continuous quote - find all matches by looping
            normalized_text = resource.normalized_text
            normalized_pattern = normalize_text_for_matching(text)

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

            super().__init__(
                resource=resource,
                query_text=text,
                spans=spans,
                is_disjoint=False,
            )

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
        """Map normalized spans to original positions."""
        original_spans = []
        for norm_start, norm_end in norm_spans:
            orig_start, orig_end = resource.map_normalized_to_original_position(
                norm_start, norm_end - norm_start
            )
            if orig_start is not None and orig_end is not None:
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
