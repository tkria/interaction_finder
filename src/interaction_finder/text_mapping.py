"""Text position mapping between source and target coordinate spaces.

Provides efficient position translation and searching: query in source text space,
get results in target coordinate space. Useful for text transformations
(normalization, rendering, etc.) where character positions need to be tracked.
"""

import bisect
import re
import unicodedata
from typing import List, Optional, Pattern, Tuple, Union


class TextPositionMapper:
    """Map positions from source text space to target coordinate space.

    Enables efficient position lookup and text searching with automatic coordinate
    remapping. Source is the text you search/query in; target is the coordinate
    space you want results mapped to.

    Example:
        >>> original = "Hello α-world"
        >>> normalized = "hello alpha world"
        >>> offsets = [(0, 0), (6, 0), (11, -4)]  # delta changes at positions 6 and 11
        >>> mapper = TextPositionMapper(normalized, original, offsets)
        >>> mapper.targetpos(6)  # 'a' in "alpha" maps to position 6 in original
        6
        >>> mapper.find("alpha")  # Search in normalized, get original coords
        (6, 7)
    """

    def __init__(self, source: str, target: str, offsets: List[Tuple[int, int]]):
        """Initialize position mapper.

        Args:
            source: Text to query/search in
            target: Text whose coordinate space we map to
            offsets: List of (source_pos, delta) tuples, sorted by source_pos.
                    Delta = target_pos - source_pos at that point.
                    Only record entries where delta changes.
        """
        self.source = source
        self.target = target
        self._offsets = offsets

        if not offsets:
            raise ValueError("Offsets list cannot be empty")

        # Validate offsets are sorted by source position
        for i in range(len(offsets) - 1):
            if offsets[i][0] > offsets[i + 1][0]:
                raise ValueError(
                    f"Offsets must be sorted by source position, "
                    f"but offsets[{i}]={offsets[i]} > offsets[{i + 1}]={offsets[i + 1]}"
                )

    def targetpos(self, source_pos: int) -> int:
        """Map position in source to position in target.

        Uses binary search for O(log n) lookup in the precomputed offset list.
        The offset list contains (source_pos, delta) pairs, and intermediate
        positions maintain the same delta until the next offset entry.

        Args:
            source_pos: Character position in source text

        Returns:
            Corresponding position in target text

        Raises:
            ValueError: If source_pos is out of bounds
        """
        if source_pos < 0 or source_pos > len(self.source):
            raise ValueError(
                f"Position {source_pos} out of bounds for source text of length "
                f"{len(self.source)}"
            )

        if not self._offsets:
            raise RuntimeError("No position offsets available")

        # Binary search to find the offset entry at or before source_pos
        idx = bisect.bisect_right(self._offsets, (source_pos, float("inf"))) - 1

        # Get the delta and calculate target position
        _, delta = self._offsets[idx]
        return source_pos + delta

    def targetspan(self, start: int, end: int) -> Tuple[int, int]:
        """Map span in source to span in target.

        Args:
            start: Start position in source text
            end: End position in source text

        Returns:
            (target_start, target_end) tuple

        Raises:
            ValueError: If positions are out of bounds
        """
        return self.targetpos(start), self.targetpos(end)

    def find(self, pattern: Union[str, Pattern]) -> Optional[Tuple[int, int]]:
        """Search in source text, return first match in target coordinates.

        Args:
            pattern: String or compiled regex to search for

        Returns:
            (start, end) in target coordinates, or None if not found

        Example:
            >>> mapper.find("alpha")  # String search
            (6, 7)
            >>> mapper.find(re.compile(r"\\balpha\\b"))  # Regex search
            (6, 7)
        """
        if isinstance(pattern, str):
            # Simple string search
            idx = self.source.find(pattern)
            if idx == -1:
                return None
            source_start = idx
            source_end = idx + len(pattern)
        else:
            # Regex search
            match = pattern.search(self.source)
            if match is None:
                return None
            source_start = match.start()
            source_end = match.end()

        # Map to target coordinates
        return self.targetspan(source_start, source_end)

    def findall(self, pattern: Union[str, Pattern]) -> List[Tuple[int, int]]:
        """Search in source text, return all matches in target coordinates.

        Args:
            pattern: String or compiled regex to search for

        Returns:
            List of (start, end) tuples in target coordinates

        Example:
            >>> mapper.findall("l")
            [(2, 2), (3, 3), (9, 8), (12, 11)]
        """
        matches = []

        if isinstance(pattern, str):
            # Simple string search - find all occurrences
            pos = 0
            while True:
                idx = self.source.find(pattern, pos)
                if idx == -1:
                    break
                source_start = idx
                source_end = idx + len(pattern)
                matches.append(self.targetspan(source_start, source_end))
                pos = idx + 1
        else:
            # Regex search - use finditer
            for match in pattern.finditer(self.source):
                source_start = match.start()
                source_end = match.end()
                matches.append(self.targetspan(source_start, source_end))

        return matches


# Greek letter mapping for normalization
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
    "ς": "sigma",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}


def normalize_text_for_matching(text: str) -> str:
    """Normalize text for case-insensitive matching with Greek letter support.

    Converts to lowercase, expands Greek letters to ASCII names, removes
    punctuation (except contractions/decimals), and normalizes whitespace.

    Args:
        text: Text to normalize

    Returns:
        Normalized text suitable for fuzzy matching
    """
    # Apply Unicode normalization
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    normalized = []
    text_len = len(text)
    last_was_space = True

    def _should_skip_char(char: str, pos: int) -> bool:
        """Check if character should be skipped (contractions, decimals)."""
        if pos == 0 or pos >= text_len - 1:
            return False
        prev_char, next_char = text[pos - 1], text[pos + 1]
        return (char == "'" and prev_char.isalnum() and next_char.isalnum()) or (
            char == "." and prev_char.isdigit() and next_char.isdigit()
        )

    i = 0
    while i < text_len:
        char = text[i]

        # Fast path for ASCII alphanumeric
        if "a" <= char <= "z" or "0" <= char <= "9":
            normalized.append(char)
            last_was_space = False
        elif "A" <= char <= "Z":
            normalized.append(char.lower())
            last_was_space = False
        # Greek letters
        elif 0x0370 <= ord(char) <= 0x03FF:
            char_lower = char.lower()
            if char_lower in GREEK_LETTER_MAP:
                # Add space before if needed
                if normalized and normalized[-1].isalnum():
                    normalized.append(" ")
                # Add Greek name
                normalized.extend(GREEK_LETTER_MAP[char_lower])
                # Add space after if needed
                if i + 1 < text_len and text[i + 1].isalnum():
                    normalized.append(" ")
                last_was_space = False
            else:
                if char.isalnum():
                    normalized.append(char_lower)
                    last_was_space = False
                elif not last_was_space:
                    normalized.append(" ")
                    last_was_space = True
        # Skip contractions and decimal points
        elif _should_skip_char(char, i):
            pass
        # Other alphanumeric
        elif char.isalnum():
            normalized.append(char.lower())
            last_was_space = False
        # Convert everything else to single space
        elif not last_was_space:
            normalized.append(" ")
            last_was_space = True

        i += 1

    return "".join(normalized).strip()


class NormalizedTextMapper(TextPositionMapper):
    """Position mapper with automatic normalization for Greek letter support.

    Extends TextPositionMapper to automatically normalize search terms before
    searching, enabling Greek letter equivalence (α ↔ alpha) without requiring
    callers to manually normalize terms.

    The mapper stores normalized text as source and original text as target,
    with position offsets for efficient coordinate translation.
    """

    @classmethod
    def from_text(
        cls, original_text: str, offsets: Optional[List[Tuple[int, int]]] = None
    ) -> "NormalizedTextMapper":
        """Create mapper from original text with automatic normalization.

        Args:
            original_text: Original document text
            offsets: Optional pre-computed offsets. If not provided, will be
                    computed from scratch.

        Returns:
            NormalizedTextMapper instance

        Example:
            >>> mapper = NormalizedTextMapper.from_text("TGF-α receptor")
            >>> mapper.find("alpha")  # Automatically finds "α"
            (4, 5)
        """
        if offsets is None:
            normalized_text, offsets = cls._build_normalized_offsets(original_text)
        else:
            normalized_text = normalize_text_for_matching(original_text)

        return cls(source=normalized_text, target=original_text, offsets=offsets)

    @staticmethod
    def _build_normalized_offsets(
        original_text: str,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """Build normalized text and delta-based position offsets.

        Args:
            original_text: Original document text

        Returns:
            Tuple of (normalized_text, offsets) where offsets is a list of
            (normalized_pos, delta) tuples with delta = original_pos - normalized_pos
        """
        # Apply Unicode normalization
        unicode_text = unicodedata.normalize("NFD", original_text)
        unicode_text = "".join(
            c for c in unicode_text if unicodedata.category(c) != "Mn"
        )

        normalized = []
        position_offsets = []
        text_len = len(unicode_text)
        last_was_space = True
        last_delta: Optional[int] = None

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
            # Calculate delta: target (original) - source (normalized)
            current_delta = i - len(normalized)

            # Record delta change (sparse representation)
            if last_delta is None or current_delta != last_delta:
                position_offsets.append((len(normalized), current_delta))
                last_delta = current_delta

            # Fast path for ASCII alphanumeric
            if "a" <= char <= "z" or "0" <= char <= "9":
                normalized.append(char)
                last_was_space = False
            elif "A" <= char <= "Z":
                normalized.append(char.lower())
                last_was_space = False
            # Greek letters
            elif 0x0370 <= ord(char) <= 0x03FF:
                char_lower = char.lower()
                if char_lower in GREEK_LETTER_MAP:
                    # Add space before if needed
                    if normalized and normalized[-1].isalnum():
                        normalized.append(" ")
                    # Add Greek name
                    normalized.extend(GREEK_LETTER_MAP[char_lower])
                    # Add space after if needed
                    if i + 1 < text_len and unicode_text[i + 1].isalnum():
                        normalized.append(" ")
                    last_was_space = False
                else:
                    if char.isalnum():
                        normalized.append(char_lower)
                        last_was_space = False
                    elif not last_was_space:
                        normalized.append(" ")
                        last_was_space = True
            # Skip contractions and decimal points
            elif _should_skip_char(char, i):
                pass
            # Other alphanumeric
            elif char.isalnum():
                normalized.append(char.lower())
                last_was_space = False
            # Convert everything else to single space
            elif not last_was_space:
                normalized.append(" ")
                last_was_space = True

            i += 1

        normalized_text = "".join(normalized).strip()
        return normalized_text, position_offsets

    def find(self, pattern: Union[str, Pattern]) -> Optional[Tuple[int, int]]:
        """Search with automatic normalization for string patterns.

        String patterns are automatically normalized before searching, enabling
        Greek letter matching. Regex patterns are used as-is.

        Args:
            pattern: String or compiled regex to search for

        Returns:
            (start, end) in target (original) coordinates, or None if not found

        Example:
            >>> mapper = NormalizedTextMapper.from_text("TGF-α receptor")
            >>> mapper.find("TGF-alpha")  # Finds "TGF-α" automatically
            (0, 5)
        """
        if isinstance(pattern, str):
            # Auto-normalize string searches
            pattern = normalize_text_for_matching(pattern)
        # Regex patterns used as-is (must match normalized space)
        return super().find(pattern)

    def findall(self, pattern: Union[str, Pattern]) -> List[Tuple[int, int]]:
        """Search all with automatic normalization for string patterns.

        String patterns are automatically normalized before searching, enabling
        Greek letter matching. Regex patterns are used as-is.

        Args:
            pattern: String or compiled regex to search for

        Returns:
            List of (start, end) tuples in target (original) coordinates

        Example:
            >>> mapper = NormalizedTextMapper.from_text("α and β receptors")
            >>> mapper.findall("alpha")
            [(0, 1)]
            >>> mapper.findall("beta")
            [(6, 7)]
        """
        if isinstance(pattern, str):
            # Auto-normalize string searches
            pattern = normalize_text_for_matching(pattern)
        # Regex patterns used as-is (must match normalized space)
        return super().findall(pattern)
