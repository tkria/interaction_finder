"""
Simplified ResourceQuote implementation demonstrating the cleaner approach.
This shows how we can eliminate most of the complexity while maintaining functionality.
"""

import re
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field


class ResourceQuote(BaseModel):
    """
    Simple ResourceQuote that handles both continuous and disjoint quotes.

    For continuous quotes: stores individual spans
    For disjoint quotes: stores all segments flattened, joins with " ... " in get_quote_text()
    """

    resource: "Resource" = Field(description="Source document")
    query_text: str = Field(description="Original query text")
    spans: List[Tuple[int, int]] = Field(description="All character spans")
    is_disjoint: bool = Field(description="True if contains ellipsis markers")

    def __init__(self, resource: "Resource", text: str, **data):
        """Create ResourceQuote by finding text in resource."""
        if "spans" in data:
            super().__init__(resource=resource, query_text=text, **data)
            return

        # Check if disjoint (contains ellipses)
        segments = re.split(r"\s*\.{3,}\s*", text)
        segments = [s.strip() for s in segments if s.strip()]
        is_disjoint = len(segments) > 1

        # Find matches using unified logic
        spans = self._find_matches(resource, segments, is_disjoint)
        if not spans:
            raise ValueError(f"Quote text not found in resource: {text!r}")

        super().__init__(
            resource=resource, query_text=text, spans=spans, is_disjoint=is_disjoint
        )

    def _find_matches(
        self, resource, segments: List[str], is_disjoint: bool
    ) -> List[Tuple[int, int]]:
        """Find all matches - unified logic for both continuous and disjoint."""
        if not is_disjoint:
            # Simple continuous matching
            return self._find_simple_matches(resource, segments[0])
        else:
            # Disjoint matching: find segments in sequence
            return self._find_sequential_segments(resource, segments)

    def _find_simple_matches(self, resource, text: str) -> List[Tuple[int, int]]:
        """Find all occurrences of continuous text."""
        from interaction_finder.resources import normalize_text_for_matching

        normalized_query = normalize_text_for_matching(text)
        normalized_text = resource.normalized_text

        spans = []
        start_pos = 0
        while True:
            pos = normalized_text.find(normalized_query, start_pos)
            if pos == -1:
                break

            # Map back to original positions
            original_start, original_end = resource.map_normalized_to_original_position(
                pos, len(normalized_query)
            )
            if original_start is not None and original_end is not None:
                spans.append((original_start, original_end))
            start_pos = pos + 1

        return spans

    def _find_sequential_segments(
        self, resource, segments: List[str]
    ) -> List[Tuple[int, int]]:
        """Find segments that appear in sequence."""
        from interaction_finder.resources import normalize_text_for_matching

        normalized_segments = [normalize_text_for_matching(seg) for seg in segments]
        normalized_text = resource.normalized_text

        all_spans = []

        # Find all occurrences of first segment
        first_positions = []
        start_pos = 0
        while True:
            pos = normalized_text.find(normalized_segments[0], start_pos)
            if pos == -1:
                break
            first_positions.append(pos)
            start_pos = pos + 1

        # For each first segment, try to find the rest in order
        for first_pos in first_positions:
            current_pos = first_pos
            occurrence_spans = []

            for i, seg in enumerate(normalized_segments):
                if i == 0:
                    seg_pos = first_pos
                else:
                    seg_pos = normalized_text.find(seg, current_pos)
                    if seg_pos == -1:
                        break

                # Map to original positions
                original_start, original_end = (
                    resource.map_normalized_to_original_position(seg_pos, len(seg))
                )
                if original_start is None or original_end is None:
                    break

                occurrence_spans.append((original_start, original_end))
                current_pos = seg_pos + len(seg)

            # Only add if we found all segments
            if len(occurrence_spans) == len(segments):
                all_spans.extend(occurrence_spans)

        return all_spans

    @property
    def count(self) -> int:
        """Number of occurrences."""
        if self.is_disjoint:
            segments_in_query = len(
                [
                    s.strip()
                    for s in re.split(r"\s*\.{3,}\s*", self.query_text)
                    if s.strip()
                ]
            )
            return len(self.spans) // segments_in_query if segments_in_query > 0 else 0
        else:
            return len(self.spans)

    def get_quote_text(self, occurrence: int = 1) -> str:
        """Get text for a specific occurrence."""
        if occurrence < 1 or occurrence > self.count:
            raise IndexError(
                f"Occurrence {occurrence} not found (have {self.count} occurrences)"
            )

        if self.is_disjoint:
            # Calculate which spans belong to this occurrence
            segments_per = len(
                [
                    s.strip()
                    for s in re.split(r"\s*\.{3,}\s*", self.query_text)
                    if s.strip()
                ]
            )
            start_idx = (occurrence - 1) * segments_per
            end_idx = start_idx + segments_per
            occurrence_spans = self.spans[start_idx:end_idx]

            # Join segments with ellipses
            segments = [
                self.resource.text[start:end] for start, end in occurrence_spans
            ]
            return " ... ".join(segments)
        else:
            # Simple continuous text
            start, end = self.spans[occurrence - 1]
            return self.resource.text[start:end]
