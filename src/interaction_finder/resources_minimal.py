"""
Minimal ResourceQuote implementation using core primitive approach.
"""

import re
from typing import List, Tuple, Optional
from pydantic import BaseModel, Field


def find_text_spans(resource, text: str) -> List[Tuple[int, int]]:
    """
    Core primitive: find all occurrences of normalized text in a resource.

    Args:
        resource: Resource with normalized_text and position mapping
        text: Text to search for

    Returns:
        List of (start, end) character spans in original text
    """
    from interaction_finder.resources import normalize_text_for_matching

    normalized_query = normalize_text_for_matching(text)
    normalized_text = resource.normalized_text

    spans = []
    start_pos = 0

    while True:
        pos = normalized_text.find(normalized_query, start_pos)
        if pos == -1:
            break

        # Map back to original text positions
        original_start, original_end = resource.map_normalized_to_original_position(
            pos, len(normalized_query)
        )
        if original_start is not None and original_end is not None:
            spans.append((original_start, original_end))

        start_pos = pos + 1

    return spans


class ResourceQuote(BaseModel):
    """Simple ResourceQuote built on the core primitive."""

    resource: "Resource" = Field(description="Source document")
    query_text: str = Field(description="Original query text")
    spans: List[Tuple[int, int]] = Field(description="Character spans")

    def __init__(self, resource: "Resource", text: str, **data):
        """Create ResourceQuote by finding text in resource."""
        if "spans" in data:
            super().__init__(resource=resource, query_text=text, **data)
            return

        # Split on ellipses to get segments (both ... and …)
        segments = re.split(r"\s*(?:\.{3,}|…)\s*", text)
        segments = [s.strip() for s in segments if s.strip()]

        if len(segments) == 1:
            # Continuous quote: use core primitive directly
            spans = find_text_spans(resource, segments[0])
        else:
            # Disjoint quote: find each segment and combine in sequence
            spans = []

            # Find first segment occurrences
            first_spans = find_text_spans(resource, segments[0])

            for first_start, first_end in first_spans:
                occurrence_spans = [(first_start, first_end)]
                search_from = first_end

                # Try to find remaining segments after this first segment
                valid_sequence = True
                for segment in segments[1:]:
                    # Find next segment that appears after search_from position
                    segment_spans = find_text_spans(resource, segment)
                    next_span = None

                    for span_start, span_end in segment_spans:
                        if span_start >= search_from:
                            next_span = (span_start, span_end)
                            break

                    if next_span is None:
                        valid_sequence = False
                        break

                    occurrence_spans.append(next_span)
                    search_from = next_span[1]

                # Add all spans from this valid sequence
                if valid_sequence:
                    spans.extend(occurrence_spans)

        if not spans:
            raise ValueError(f"Quote text not found in resource: {text!r}")

        super().__init__(resource=resource, query_text=text, spans=spans)

    @property
    def is_disjoint(self) -> bool:
        """True if quote contains ellipsis markers."""
        return "..." in self.query_text or "…" in self.query_text

    @property
    def segments_per_occurrence(self) -> int:
        """Number of segments per occurrence."""
        segments = re.split(r"\s*(?:\.{3,}|…)\s*", self.query_text)
        return len([s.strip() for s in segments if s.strip()])

    @property
    def count(self) -> int:
        """Number of occurrences found."""
        return len(self.spans) // self.segments_per_occurrence

    def get_quote_text(self, occurrence: int = 1) -> str:
        """Get text for a specific occurrence."""
        if occurrence < 1 or occurrence > self.count:
            raise IndexError(f"Occurrence {occurrence} not found")

        segments_per = self.segments_per_occurrence
        start_idx = (occurrence - 1) * segments_per
        end_idx = start_idx + segments_per
        occurrence_spans = self.spans[start_idx:end_idx]

        # Extract text segments
        segments = [self.resource.text[start:end] for start, end in occurrence_spans]

        # Join with ellipses if multiple segments
        return " ... ".join(segments) if len(segments) > 1 else segments[0]
