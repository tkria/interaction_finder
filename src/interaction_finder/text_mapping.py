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
        >>> normalized = "hello a world"
        >>> offsets = [(0, 0), (6, 0), (8, 0)]  # delta changes at positions
        >>> mapper = TextPositionMapper(normalized, original, offsets)
        >>> mapper.targetpos(6)  # 'a' (from α) maps to position 6 in original
        6
        >>> mapper.find("a")  # Search in normalized, get original coords
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


# Greek letter mapping for normalization (to single ASCII letter)
GREEK_LETTER_MAP = {
    "α": "a",
    "β": "b",
    "γ": "g",
    "δ": "d",
    "ε": "e",
    "ζ": "z",
    "η": "h",
    "θ": "q",  # No direct equivalent, use q
    "ι": "i",
    "κ": "k",
    "λ": "l",
    "μ": "m",
    "ν": "n",
    "ξ": "x",
    "ο": "o",
    "π": "p",
    "ρ": "r",
    "ς": "s",
    "σ": "s",
    "τ": "t",
    "υ": "u",
    "φ": "f",
    "χ": "c",
    "ψ": "y",  # No direct equivalent, use y
    "ω": "w",
}
# Spelled-out Greek letter names to single ASCII letter (for word replacement)
GREEK_NAME_MAP = {
    "alpha": "a",
    "beta": "b",
    "gamma": "g",
    "delta": "d",
    "epsilon": "e",
    "zeta": "z",
    "eta": "h",
    "theta": "q",
    "iota": "i",
    "kappa": "k",
    "lambda": "l",
    "mu": "m",
    "nu": "n",
    "xi": "x",
    "omicron": "o",
    "pi": "p",
    "rho": "r",
    "sigma": "s",
    "tau": "t",
    "upsilon": "u",
    "phi": "f",
    "chi": "c",
    "psi": "y",
    "omega": "w",
}


class NormalizedTextMapper(TextPositionMapper):
    """Position mapper with automatic normalization for Greek letter support.

    Extends TextPositionMapper to automatically normalize search terms before
    searching, enabling Greek letter equivalence (α ↔ alpha) without requiring
    callers to manually normalize terms.

    The mapper stores normalized text as source and original text as target,
    with position offsets for efficient coordinate translation.
    """

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for case-insensitive matching with Greek letter support.

        Converts to lowercase, collapses Greek letters and spelled-out Greek names
        to single ASCII letters (α/alpha → a, β/beta → b), removes punctuation
        (except contractions/decimals), and normalizes whitespace.

        This is a lightweight method for when you only need normalized text
        without position mapping. For position tracking, use from_text() instead.

        Args:
            text: Text to normalize

        Returns:
            Normalized text suitable for fuzzy matching

        Example:
            >>> NormalizedTextMapper.normalize("TGF-α receptor")
            'tgfa receptor'
            >>> NormalizedTextMapper.normalize("TGF-alpha receptor")
            'tgfa receptor'
        """
        normalized_text, _ = NormalizedTextMapper._build_normalized_offsets(text)
        return normalized_text

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
            >>> mapper.find("a")  # Finds "α" (normalized to "a")
            (4, 5)
        """
        if offsets is None:
            normalized_text, offsets = cls._build_normalized_offsets(original_text)
        else:
            normalized_text = cls.normalize(original_text)

        return cls(source=normalized_text, target=original_text, offsets=offsets)

    @staticmethod
    def _build_normalized_offsets(
        original_text: str,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """Build normalized text and delta-based position offsets.

        Normalizes text by:
        - Lowercasing
        - Converting Greek letters (α, β, etc.) to single ASCII letters (a, b, etc.)
        - Converting spelled-out Greek names (alpha, beta) to single letters
        - Removing punctuation (except contractions/decimals)
        - Normalizing whitespace

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

        # Helper to check if character should be skipped (contractions, decimals)
        def _should_skip_char(char: str, pos: int) -> bool:
            if pos == 0 or pos >= text_len - 1:
                return False
            prev_char, next_char = unicode_text[pos - 1], unicode_text[pos + 1]
            return (char == "'" and prev_char.isalnum() and next_char.isalnum()) or (
                char == "." and prev_char.isdigit() and next_char.isdigit()
            )

        # Helper to check for spelled-out Greek letter name at position
        def _try_greek_word(pos: int) -> Optional[Tuple[str, int]]:
            """Check if a Greek letter name starts at pos. Returns (letter, length) or None.

            Matches Greek names when at a word boundary:
            - Word start + word end: "alpha" → "a", "TGF-alpha" → "tgf a"
            - Word start + digit after: "alpha1" → "a1", "alpha2beta1" → "a2b1"
            - Word end only: "TGFalpha" → "tgfa", "TNFalpha" → "tnfa"

            Does NOT match in the middle of words: "alphabet" stays "alphabet"
            (alpha at start but followed by letters "bet")
            """
            text_lower = unicode_text[pos : pos + 10].lower()
            for name, letter in GREEK_NAME_MAP.items():
                if text_lower.startswith(name):
                    end_pos = pos + len(name)
                    at_start = pos == 0 or not unicode_text[pos - 1].isalpha()
                    at_end = end_pos >= text_len or not unicode_text[end_pos].isalpha()
                    # Match if:
                    # 1. At word start AND (at word end OR followed by digit)
                    # 2. OR just at word end (e.g., "TGFalpha")
                    if at_start and at_end:
                        return (letter, len(name))
                    if at_end and not at_start:
                        # At end only (e.g., "TGFalpha") - still match
                        return (letter, len(name))
            return None

        # Helper to remove trailing space and fix offsets
        def _remove_trailing_space() -> None:
            nonlocal last_delta
            if normalized and normalized[-1] == " ":
                space_pos = len(normalized) - 1
                normalized.pop()
                # Remove offset entry if it was for the space position
                if position_offsets and position_offsets[-1][0] == space_pos:
                    position_offsets.pop()
                    # Reset last_delta so next iteration records fresh
                    last_delta = position_offsets[-1][1] if position_offsets else None

        i = 0
        while i < text_len:
            char = unicode_text[i]
            # Calculate delta: target (original) - source (normalized)
            current_delta = i - len(normalized)
            # Record delta change (sparse representation)
            if last_delta is None or current_delta != last_delta:
                position_offsets.append((len(normalized), current_delta))
                last_delta = current_delta
            # Check for spelled-out Greek letter name (alpha, beta, etc.)
            if char.isalpha():
                greek_match = _try_greek_word(i)
                if greek_match:
                    letter, length = greek_match
                    _remove_trailing_space()
                    normalized.append(letter)
                    last_was_space = False
                    i += length
                    continue
            # Fast path for ASCII alphanumeric
            if "a" <= char <= "z" or "0" <= char <= "9":
                normalized.append(char)
                last_was_space = False
            elif "A" <= char <= "Z":
                normalized.append(char.lower())
                last_was_space = False
            # Greek letters (symbols)
            elif 0x0370 <= ord(char) <= 0x03FF:
                char_lower = char.lower()
                if char_lower in GREEK_LETTER_MAP:
                    _remove_trailing_space()
                    normalized.append(GREEK_LETTER_MAP[char_lower])
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
            >>> mapper.find("TGF-alpha")  # Finds "TGF-α" (both normalize to "tgfa")
            (0, 5)
        """
        if isinstance(pattern, str):
            # Auto-normalize string searches
            pattern = self.normalize(pattern)
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
            >>> mapper.findall("alpha")  # "alpha" normalizes to "a", finds "α"
            [(0, 1)]
            >>> mapper.findall("beta")  # "beta" normalizes to "b", finds "β"
            [(6, 7)]
        """
        if isinstance(pattern, str):
            # Auto-normalize string searches
            pattern = self.normalize(pattern)
        # Regex patterns used as-is (must match normalized space)
        return super().findall(pattern)
