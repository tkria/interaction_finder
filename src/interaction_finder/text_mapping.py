"""Text position mapping between source and target coordinate spaces.

Provides efficient position translation and searching: query in source text space,
get results in target coordinate space. Useful for text transformations
(normalization, rendering, etc.) where character positions need to be tracked.
"""

import bisect
import re
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
