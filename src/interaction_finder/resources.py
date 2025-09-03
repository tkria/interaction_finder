"""
Resource management models for document sources and quote tracking.

This module provides structured management of document resources and validates
that extracted information can be traced back to specific documents with
supporting quotes.
"""

import hashlib
import re
import bisect
import unicodedata
from typing import List, Optional, Tuple, Dict, Any
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
        # Generate stable ID: counter_hash
        url_hash = hashlib.shake_128(url.encode()).hexdigest(4)
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
    quote matching and position translation.
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

    def __init__(self, id: ResourceId, title: str, text: str, **data):
        """
        Create Resource with automatic normalized text and position mapping.

        Args:
            id: ResourceId for the document
            title: Human-readable document title
            text: Full document text content
        """
        # Build normalized text with Greek letter support
        normalized_text, position_offsets = self._build_normalized_text_and_offsets(
            text
        )

        super().__init__(
            id=id,
            title=title,
            text=text,
            normalized_text=normalized_text,
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

    def quote(self, text: str) -> Optional["ResourceQuote"]:
        """
        Create a ResourceQuote by finding all occurrences of the given text in this resource.

        Args:
            text: Text to find and quote

        Returns:
            ResourceQuote with all occurrences if text found, None otherwise
        """
        try:
            return ResourceQuote(self, text)
        except ValueError:
            return None

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
        self, resource_id: ResourceId, title: str, document_text: str
    ) -> Resource:
        """
        Add content to a previously registered resource.

        Args:
            resource_id: Previously registered ResourceId
            title: Human-readable document title
            document_text: Full text content of the document

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
        resource = Resource(id=resource_id, title=title, text=document_text)
        self.resource_map[resource_id] = resource

        return resource

    def add(self, url: str, title: str, document_text: str) -> Resource:
        """
        Add a complete resource (register ID + content) in one step.

        Args:
            url: Document URL (must be unique)
            title: Human-readable document title
            document_text: Full text content of the document

        Returns:
            Complete Resource

        Raises:
            ValueError: If URL already exists in pool
        """
        resource_id = self.register(url)
        return self.add_content(resource_id, title, document_text)

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
        description="List of (start, end) character spans for all occurrences"
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
            super().__init__(resource=resource, **data)
            return

        # Search for all occurrences of the quote text
        normalized_quote = normalize_text_for_matching(text)
        normalized_text = resource.normalized_text

        # Find all occurrences in normalized text
        normalized_positions = []
        start_pos = 0
        while True:
            pos = normalized_text.find(normalized_quote, start_pos)
            if pos == -1:
                break
            normalized_positions.append(pos)
            start_pos = pos + 1

        if not normalized_positions:
            raise ValueError(f"Quote text not found in resource: {text!r}")

        # Map all occurrences back to original text positions
        spans = []
        for normalized_pos in normalized_positions:
            original_start, original_end = resource.map_normalized_to_original_position(
                normalized_pos,
                len(normalized_quote),
            )
            if original_start is not None and original_end is not None:
                spans.append((original_start, original_end))

        if not spans:
            raise ValueError(
                f"Could not map any occurrences back to original text: {text!r}"
            )

        super().__init__(resource=resource, query_text=text, spans=spans)

    @field_validator("spans")
    @classmethod
    def validate_spans_within_text(cls, v, info: ValidationInfo):
        """Ensure all spans are within the resource text bounds and properly ordered"""
        if info.data and "resource" in info.data:
            resource = info.data["resource"]
            text_length = len(resource.text)

            # Check all spans are within bounds and properly ordered
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
        return len(self.spans)

    def get_quote_text(self, occurrence: int = 1) -> str:
        """
        Extract quote text from a specific occurrence.

        Args:
            occurrence: Which occurrence to get (1-based)

        Returns:
            Quote text from the specified occurrence

        Raises:
            IndexError: If occurrence doesn't exist
        """
        if occurrence < 1 or occurrence > len(self.spans):
            raise IndexError(
                f"Occurrence {occurrence} not found (have {len(self.spans)} occurrences)"
            )

        start, end = self.spans[occurrence - 1]
        return self.resource.text[start:end]

    def get_all_quote_texts(self) -> List[str]:
        """
        Extract quote text from all occurrences.

        Returns:
            List of quote texts for all occurrences
        """
        return [self.resource.text[start:end] for start, end in self.spans]

    def get_context(self, occurrence: int = 1, context_chars: int = 200) -> str:
        """
        Get surrounding context around a specific quote occurrence.

        Args:
            occurrence: Which occurrence to get context for (1-based)
            context_chars: Number of characters to include before/after quote

        Returns:
            Context text with quote highlighted

        Raises:
            IndexError: If occurrence doesn't exist
        """
        if occurrence < 1 or occurrence > len(self.spans):
            raise IndexError(
                f"Occurrence {occurrence} not found (have {len(self.spans)} occurrences)"
            )

        start, end = self.spans[occurrence - 1]
        text = self.resource.text

        # Calculate context bounds
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)

        # Extract context with quote markers
        before = text[context_start:start]
        quote = text[start:end]
        after = text[end:context_end]

        return f"{before}**{quote}**{after}"

    def get_all_contexts(self, context_chars: int = 200) -> List[str]:
        """
        Get surrounding context for all occurrences.

        Args:
            context_chars: Number of characters to include before/after each quote

        Returns:
            List of context strings for all occurrences
        """
        return [self.get_context(i + 1, context_chars) for i in range(len(self.spans))]

    def validate_quote(self, expected_quote: str) -> bool:
        """
        Validate that the query matches the expected quote text.

        Uses normalized comparison to handle formatting differences.

        Args:
            expected_quote: Quote text to validate against

        Returns:
            True if query matches expected quote, False otherwise
        """
        normalized_actual = normalize_text_for_matching(self.query_text)
        normalized_expected = normalize_text_for_matching(expected_quote)

        return normalized_actual == normalized_expected

    def __repr__(self) -> str:
        """Rich representation for REPL display."""
        # First line: ResourceId
        lines = [f"ResourceQuote for {self.resource.id!r}:"]

        # Following lines: Each quote with position
        for i in range(self.count):
            start, end = self.spans[i]
            quote_text = self.get_quote_text(i + 1)
            lines.append(f"  [{i + 1}] '{quote_text}' at {start}-{end}")

        return "\n".join(lines)
