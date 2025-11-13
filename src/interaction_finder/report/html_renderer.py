"""HTML rendering with position tracking for report generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import BaseModel

from interaction_finder.resources import Resource, ResourceQuote


class MarkdownToHTMLRenderer:
    """Markdown renderer that preserves original character offsets."""

    def __init__(self, text: str):
        self.original_text = text
        self.html: str = ""
        self.position_offsets: list[tuple[int, int]] = []
        self._md = MarkdownIt("commonmark", {"html": False, "typographer": False})
        self._line_offsets = self._compute_line_offsets(text)
        self._builder = HTMLBuilder(text, self._line_offsets)

    def render(self) -> tuple[str, list[tuple[int, int]]]:
        """Render markdown to HTML while tracking source offsets."""
        tokens = self._md.parse(self.original_text)
        self._builder.reset()
        self._builder.render(tokens)
        self.html = self._builder.html
        self.position_offsets = self._builder.position_offsets
        return self.html, self.position_offsets

    def map_original_to_html_position(self, original_pos: int) -> int:
        """Map a position in the original markdown to rendered HTML."""
        if original_pos < 0 or original_pos > len(self.original_text):
            raise ValueError(
                f"Position {original_pos} out of bounds for text of length "
                f"{len(self.original_text)}"
            )
        if not self.position_offsets:
            raise RuntimeError("No position offsets available. Call render() first.")

        offsets = self.position_offsets
        if original_pos == 0:
            for html_pos, orig_pos in offsets:
                if orig_pos == 0:
                    return html_pos
            return 0
        if original_pos >= offsets[-1][1]:
            return offsets[-1][0]

        left, right = 0, len(offsets) - 1
        while left < right:
            mid = (left + right) // 2
            if offsets[mid][1] < original_pos:
                left = mid + 1
            else:
                right = mid
        if left > 0 and offsets[left][1] > original_pos:
            left -= 1
        return offsets[left][0]

    @staticmethod
    def _compute_line_offsets(text: str) -> list[int]:
        offsets = [0]
        for idx, char in enumerate(text):
            if char == "\n":
                offsets.append(idx + 1)
        offsets.append(len(text))
        return offsets


@dataclass
class InlineCursor:
    """Tracks progress through the source substring for an inline token."""

    source: str
    abs_start: int
    pos: int = 0

    def advance(self, length: int) -> int:
        start = self.abs_start + self.pos
        self.pos += max(length, 0)
        return start

    def find(self, needle: str) -> int:
        if not needle:
            return self.pos
        idx = self.source.find(needle, self.pos)
        if idx == -1:
            idx = self.pos
        return idx

    def consume_markup(self, markup: str) -> int:
        if not markup:
            return self.abs_start + self.pos
        if self.source.startswith(markup, self.pos):
            start = self.abs_start + self.pos
            self.pos += len(markup)
            return start
        idx = self.source.find(markup, self.pos)
        if idx == -1:
            start = self.abs_start + self.pos
            return start
        start = self.abs_start + idx
        self.pos = idx + len(markup)
        return start

    def consume_until(self, needle: str) -> int:
        idx = self.source.find(needle, self.pos)
        if idx == -1:
            idx = len(self.source)
        start = self.abs_start + self.pos
        self.pos = idx
        return start


class HTMLBuilder:
    """Incrementally builds HTML while tracking mapping back to original text."""

    def __init__(self, text: str, line_offsets: Sequence[int]):
        self.text = text
        self.line_offsets = line_offsets
        self.html_parts: list[str] = []
        self._html_len = 0
        self.position_offsets: list[tuple[int, int]] = []
        self._last_orig_pos = 0

    def reset(self) -> None:
        self.html_parts = []
        self._html_len = 0
        self.position_offsets = []
        self._last_orig_pos = 0

    @property
    def html(self) -> str:
        return "".join(self.html_parts)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _append_literal(self, text: str, orig_pos: int | None = None) -> None:
        if not text:
            return
        if orig_pos is not None:
            self.position_offsets.append((self._html_len, orig_pos))
        self.html_parts.append(text)
        self._html_len += len(text)

    def _append_text(self, text: str, orig_start: int) -> None:
        for idx, ch in enumerate(text):
            html_fragment = _escape_html_char(ch)
            self.position_offsets.append((self._html_len, orig_start + idx))
            self.html_parts.append(html_fragment)
            self._html_len += len(html_fragment)
            self._last_orig_pos = orig_start + idx + 1

    def _append_newline(self, orig_pos: int) -> None:
        self.position_offsets.append((self._html_len, orig_pos))
        self.html_parts.append("\n")
        self._html_len += 1
        self._last_orig_pos = orig_pos + 1

    # ------------------------------------------------------------------
    # Block rendering
    # ------------------------------------------------------------------

    def render(self, tokens: Sequence[Token]) -> None:
        stack: list[str] = []
        for token in tokens:
            if token.type == "inline":
                self._render_inline(token)
                continue

            if token.nesting == 1:
                stack.append(token.tag)
                attr_text = self._format_attrs(token.attrs)
                self._append_literal(f"<{token.tag}{attr_text}>")
                self._consume_block_markup(token)
            elif token.nesting == -1:
                if stack:
                    stack.pop()
                self._append_literal(f"</{token.tag}>")
                self._consume_block_markup(token)
            else:
                if token.type == "hr":
                    self._append_literal("<hr />")
                elif token.type == "code_block":
                    self._render_code_block(token)
                elif token.type == "fence":
                    self._render_fence(token)
                elif token.type == "html_block":
                    self._render_html_block(token)

        # Final mapping guard
        self.position_offsets.append((self._html_len, len(self.text)))

    def _format_attrs(self, attrs: list[tuple[str, str]] | None) -> str:
        if not attrs:
            return ""
        joined = " ".join(f'{name}="{_escape_html_attr(value)}"' for name, value in attrs)
        return f" {joined}" if joined else ""

    def _render_code_block(self, token: Token) -> None:
        start = self.line_offsets[token.map[0]] if token.map else 0
        end = self.line_offsets[token.map[1]] if token.map else start
        content = self.text[start:end]
        self._append_literal("<pre><code>")
        self._append_text(content, start)
        self._append_literal("</code></pre>", start)

    def _render_fence(self, token: Token) -> None:
        info = token.info.strip() if token.info else ""
        class_attr = f' class="language-{_escape_html_attr(info)}"' if info else ""
        start = self.line_offsets[token.map[0]] if token.map else 0
        content_start = start
        self._append_literal(f"<pre><code{class_attr}>")
        self._append_text(token.content, content_start)
        self._append_literal("</code></pre>", content_start + len(token.content))

    def _render_html_block(self, token: Token) -> None:
        start = self.line_offsets[token.map[0]] if token.map else 0
        self._append_text(token.content, start)

    def _consume_block_markup(self, token: Token) -> None:
        """Advance cursor past block-level markup characters (e.g. '# ', '- ')."""
        if not token.map or not token.markup:
            return
        start_line = token.map[0]
        abs_pos = self.line_offsets[start_line]
        markup = token.markup
        idx = self.text.find(markup, abs_pos)
        if idx != -1:
            self._last_orig_pos = idx + len(markup)

    # ------------------------------------------------------------------
    # Inline rendering
    # ------------------------------------------------------------------

    def _render_inline(self, token: Token) -> None:
        if not token.map:
            return
        start_line, end_line = token.map
        start = self.line_offsets[start_line]
        end = self.line_offsets[end_line]
        block_text = self.text[start:end]
        cursor = InlineCursor(block_text, start)
        for child in token.children or []:
            handler = getattr(self, f"_inline_{child.type}", self._inline_unknown)
            handler(child, cursor)

    def _inline_text(self, token: Token, cursor: InlineCursor) -> None:
        idx = cursor.find(token.content)
        abs_start = cursor.abs_start + idx
        self._append_text(token.content, abs_start)
        cursor.pos = idx + len(token.content)

    def _inline_code_inline(self, token: Token, cursor: InlineCursor) -> None:
        start_pos = cursor.consume_markup(token.markup or "`")
        self._append_literal("<code>", start_pos)
        idx = cursor.find(token.content)
        abs_start = cursor.abs_start + idx
        self._append_text(token.content, abs_start)
        cursor.pos = idx + len(token.content)
        end_pos = cursor.consume_markup(token.markup or "`")
        self._append_literal("</code>", end_pos)

    def _inline_strong_open(self, token: Token, cursor: InlineCursor) -> None:
        pos = cursor.consume_markup(token.markup or "**")
        self._append_literal("<strong>", pos)

    def _inline_strong_close(self, token: Token, cursor: InlineCursor) -> None:
        pos = cursor.consume_markup(token.markup or "**")
        self._append_literal("</strong>", pos)

    def _inline_em_open(self, token: Token, cursor: InlineCursor) -> None:
        pos = cursor.consume_markup(token.markup or "*")
        self._append_literal("<em>", pos)

    def _inline_em_close(self, token: Token, cursor: InlineCursor) -> None:
        pos = cursor.consume_markup(token.markup or "*")
        self._append_literal("</em>", pos)

    def _inline_softbreak(self, token: Token, cursor: InlineCursor) -> None:
        idx = cursor.find("\n")
        pos = cursor.abs_start + idx
        self._append_newline(pos)
        cursor.pos = idx + 1

    def _inline_hardbreak(self, token: Token, cursor: InlineCursor) -> None:
        space_idx = cursor.find("  ")
        cursor.pos = min(space_idx + 2, len(cursor.source))
        newline_idx = cursor.find("\n")
        pos = cursor.abs_start + newline_idx
        self._append_literal("<br />", pos)
        cursor.pos = newline_idx + 1

    def _inline_html_inline(self, token: Token, cursor: InlineCursor) -> None:
        idx = cursor.find(token.content)
        abs_start = cursor.abs_start + idx
        self._append_literal(token.content, abs_start)
        cursor.pos = idx + len(token.content)

    def _inline_link_open(self, token: Token, cursor: InlineCursor) -> None:
        pos = cursor.consume_markup("[")
        href = token.attrGet("href") or ""
        title = token.attrGet("title")
        attr = f' href="{_escape_html_attr(str(href))}"'
        if title:
            attr += f' title="{_escape_html_attr(str(title))}"'
        self._append_literal(f"<a{attr}>", pos)

    def _inline_link_close(self, token: Token, cursor: InlineCursor) -> None:
        pos = cursor.consume_markup("]")
        if cursor.source.startswith("(", cursor.pos):
            end = cursor.source.find(")", cursor.pos)
            if end != -1:
                cursor.pos = end + 1
        self._append_literal("</a>", pos)

    def _inline_unknown(self, token: Token, cursor: InlineCursor) -> None:
        # Fallback: treat content as literal text
        if token.content:
            self._inline_text(token, cursor)


def _escape_html_char(char: str) -> str:
    if char == "<":
        return "&lt;"
    if char == ">":
        return "&gt;"
    if char == "&":
        return "&amp;"
    if char == '"':
        return "&quot;"
    return char


def _escape_html_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    def map_original_to_html_position(self, original_pos: int) -> int:
        """Map position in original markdown to position in rendered HTML.

        Uses binary search for O(log n) lookup, following the pattern in
        Resource._find_original_position().

        Args:
            original_pos: Position in original markdown text

        Returns:
            Corresponding position in rendered HTML

        Raises:
            ValueError: If original_pos is out of bounds
        """
        if original_pos < 0 or original_pos > len(self.original_text):
            raise ValueError(
                f"Position {original_pos} out of bounds for text of length "
                f"{len(self.original_text)}"
            )

        if not self.position_offsets:
            raise RuntimeError("No position offsets available. Call render() first.")

        # Binary search for the mapping entry
        # We're searching in (html_pos, orig_pos) tuples, but want to find by orig_pos
        # So we need to search through the list looking at the second element

        # Handle edge cases
        if original_pos == 0 and len(self.position_offsets) > 0:
            # Find first offset entry for position 0
            for html_pos, orig_pos in self.position_offsets:
                if orig_pos == 0:
                    return html_pos
            return 0  # Fallback
        if original_pos >= self.position_offsets[-1][1]:
            return self.position_offsets[-1][0]

        # Binary search
        left, right = 0, len(self.position_offsets) - 1
        while left < right:
            mid = (left + right) // 2
            if self.position_offsets[mid][1] < original_pos:
                left = mid + 1
            else:
                right = mid

        # At this point, position_offsets[left][1] >= original_pos
        # We want the entry where orig_pos is between entries
        if left > 0 and self.position_offsets[left][1] > original_pos:
            left -= 1

        return self.position_offsets[left][0]


class QuoteMetadata(BaseModel):
    """Metadata for a quote span in rendered HTML."""

    span_id: str
    pair_indices: list[int]  # Which pair(s) reference this quote
    original_spans: list[tuple[int, int]]  # Spans in original text
    html_spans: list[tuple[int, int]]  # Spans in rendered HTML


class EntityMetadata(BaseModel):
    """Metadata for an entity mention in rendered HTML."""

    span_id: str
    name: str  # Canonical entity name
    kind: str  # Entity kind (gene, disease, etc.)
    aliases: list[str]
    pair_indices: list[int]  # Which pair(s) this entity belongs to


class PrerenderedDocument(BaseModel):
    """Pre-rendered document with annotated HTML and metadata."""

    doc_id: str
    html: str  # Fully rendered and annotated HTML
    quote_map: dict[str, QuoteMetadata]  # span_id -> quote info
    entity_map: dict[str, EntityMetadata]  # span_id -> entity info


class DocumentAnnotator:
    """Annotate rendered HTML with quote and entity spans.

    Takes HTML from MarkdownToHTMLRenderer and adds <span> tags to mark
    quotes and entity mentions, creating a fully annotated document for
    interactive display.
    """

    def __init__(
        self,
        resource: Resource,
        renderer: MarkdownToHTMLRenderer,
    ):
        """Initialize annotator.

        Args:
            resource: Resource containing original text and validation facilities
            renderer: Renderer that produced HTML (must have called render())
        """
        self.resource = resource
        self.renderer = renderer

        if not renderer.position_offsets:
            raise ValueError("Renderer must have called render() before annotation")

    def annotate(
        self,
        quotes: list[ResourceQuote],
        entities: dict[str, Any],  # pair_idx -> {entity1: ..., entity2: ...}
    ) -> PrerenderedDocument:
        """Annotate HTML with quote and entity spans.

        Args:
            quotes: List of validated ResourceQuote objects for this document
            entities: Dictionary mapping pair indices to entity information
                Expected format: {pair_idx: {"entity1": {...}, "entity2": {...}}}

        Returns:
            PrerenderedDocument with annotated HTML and metadata
        """
        # Generate document hash for unique IDs
        doc_hash = abs(hash(self.resource.id.url)) % 10000

        quote_map: dict[str, QuoteMetadata] = {}
        entity_map: dict[str, EntityMetadata] = {}

        # Step 1: Map all quotes to HTML coordinates
        quote_counter = 0
        html_quote_spans: list[
            tuple[int, int, str, list[int]]
        ] = []  # (start, end, span_id, pair_indices)

        for quote_idx, quote in enumerate(quotes):
            # Map quote spans from original to HTML
            html_spans = []
            for orig_start, orig_end in quote.spans:
                html_start = self.renderer.map_original_to_html_position(orig_start)
                html_end = self.renderer.map_original_to_html_position(orig_end)
                html_spans.append((html_start, html_end))

            # Generate quote span ID
            quote_id = f"doc-{doc_hash}-quote-{quote_counter}"
            quote_counter += 1

            # Record quote metadata
            quote_map[quote_id] = QuoteMetadata(
                span_id=quote_id,
                pair_indices=[],  # Will be populated later
                original_spans=quote.spans,
                html_spans=html_spans,
            )

            # Add to HTML spans list (use first and last span for overall range)
            if html_spans:
                html_quote_spans.append(
                    (
                        html_spans[0][0],
                        html_spans[-1][1],
                        quote_id,
                        [],  # pair_indices
                    )
                )

        # Step 2: Find entity mentions within quotes
        # First pass: collect all entity positions in ORIGINAL TEXT and which pairs reference them
        # Key: (start_pos, end_pos, entity_name, matched_term) in ORIGINAL text coordinates
        entity_position_map: dict[tuple[int, int, str, str], dict[str, Any]] = {}

        for pair_idx, pair_entities in entities.items():
            entity1 = pair_entities.get("entity1")
            entity2 = pair_entities.get("entity2")

            if not entity1 or not entity2:
                continue

            # Search for entity mentions within quote boundaries in ORIGINAL text
            for entity in [entity1, entity2]:
                entity_name = entity.get("name", "")
                entity_kind = entity.get("kind", "")
                entity_aliases = entity.get("aliases", [])

                if not entity_name:
                    continue

                # Search for entity name and aliases in ORIGINAL text
                search_terms = [entity_name] + entity_aliases

                for term in search_terms:
                    # Case-insensitive search in original text
                    original_text = self.resource.text.lower()
                    term_lower = term.lower()

                    pos = 0
                    while True:
                        pos = original_text.find(term_lower, pos)
                        if pos == -1:
                            break

                        # Check if this position is within any quote span (in original coordinates)
                        in_quote = False
                        for quote_meta in quote_map.values():
                            for q_start, q_end in quote_meta.original_spans:
                                if q_start <= pos < q_end:
                                    in_quote = True
                                    break
                            if in_quote:
                                break

                        if in_quote:
                            # Create key for this position in ORIGINAL coordinates (include matched term)
                            position_key = (pos, pos + len(term), entity_name, term)

                            if position_key not in entity_position_map:
                                entity_position_map[position_key] = {
                                    "name": entity_name,
                                    "kind": entity_kind,
                                    "aliases": entity_aliases,
                                    "pair_indices": [pair_idx],
                                    "matched_term": term,
                                }
                            else:
                                # Same position, add pair index if not already present
                                if (
                                    pair_idx
                                    not in entity_position_map[position_key][
                                        "pair_indices"
                                    ]
                                ):
                                    entity_position_map[position_key][
                                        "pair_indices"
                                    ].append(pair_idx)

                        pos += len(term)

        # Second pass: Map entity positions from original text to HTML coordinates
        # Then resolve overlapping entity spans in HTML coordinates
        entity_positions_html = []
        for (
            orig_start,
            orig_end,
            entity_name,
            matched_term,
        ), entity_data in entity_position_map.items():
            # Map positions from original text to HTML
            html_start = self.renderer.map_original_to_html_position(orig_start)
            html_end = self.renderer.map_original_to_html_position(orig_end)

            entity_positions_html.append(
                (html_start, html_end, entity_name, matched_term, entity_data)
            )

        # Sort by position, then by length (longer first for overlap resolution)
        entity_positions_html.sort(
            key=lambda x: (x[0], -(x[1] - x[0]))
        )  # Sort by start, then by length descending

        # Remove overlapping spans (keep the first/longest match at each position)
        non_overlapping_entities = []
        for start, end, entity_name, matched_term, entity_data in entity_positions_html:
            # Check if this span overlaps with any already selected span
            overlaps = False
            for existing_start, existing_end, _, _, _ in non_overlapping_entities:
                # Check for overlap
                if not (end <= existing_start or start >= existing_end):
                    overlaps = True
                    break

            if not overlaps:
                non_overlapping_entities.append(
                    (start, end, entity_name, matched_term, entity_data)
                )

        # Third pass: create entity spans from non-overlapping positions
        # Skip empty spans (where start == end from position mapping)
        entity_counter = 0
        html_entity_spans: list[
            tuple[int, int, str, str, str, list[int]]
        ] = []  # (start, end, span_id, entity_name, matched_term, pair_indices)

        for (
            start,
            end,
            entity_name,
            matched_term,
            entity_data,
        ) in non_overlapping_entities:
            # Skip empty spans (position mapping can collapse ranges to a single point)
            if start >= end:
                continue

            # Generate unique entity span ID
            entity_id = f"doc-{doc_hash}-entity-{entity_counter}"
            entity_counter += 1

            # Record entity metadata
            entity_map[entity_id] = EntityMetadata(
                span_id=entity_id,
                name=entity_data["name"],
                kind=entity_data["kind"],
                aliases=entity_data["aliases"],
                pair_indices=entity_data["pair_indices"],
            )

            # Add to HTML spans list
            html_entity_spans.append(
                (
                    start,
                    end,
                    entity_id,
                    entity_name,
                    matched_term,
                    entity_data["pair_indices"],
                )
            )

        # Step 3: Insert spans into HTML
        # Build entity span lookup: span_id -> (entity_name, matched_term)
        entity_span_info = {
            span_id: (entity_name, matched_term)
            for _, _, span_id, entity_name, matched_term, _ in html_entity_spans
        }

        # New approach: Use flat quote spans with CSS classes (no nesting)
        # Collect all quote boundaries to create regions
        quote_boundaries: list[
            tuple[int, bool, str]
        ] = []  # (position, is_start, quote_id)

        # Find HTML block closing tags to constrain span boundaries
        block_tags = [
            "</p>",
            "</h1>",
            "</h2>",
            "</h3>",
            "</h4>",
            "</h5>",
            "</h6>",
            "</div>",
            "</section>",
        ]
        block_boundaries = []
        for tag in block_tags:
            pos = 0
            while True:
                pos = self.renderer.html.find(tag, pos)
                if pos == -1:
                    break
                block_boundaries.append(pos)
                pos += len(tag)

        # Collect quote boundaries, adjusting end positions to respect block boundaries
        for start, end, span_id, _ in html_quote_spans:
            quote_boundaries.append((start, True, span_id))

            # Adjust end position to not go past block boundaries
            adjusted_end = end
            for boundary in block_boundaries:
                if start < boundary <= end:
                    adjusted_end = min(adjusted_end, boundary)

            quote_boundaries.append((adjusted_end, False, span_id))

        # Sort boundaries
        quote_boundaries.sort(
            key=lambda x: (x[0], not x[1])
        )  # Sort by position, starts before ends

        # Build quote regions - each region has a set of active quotes
        # Format: [(start_pos, end_pos, set of active quote_ids)]
        quote_regions: list[tuple[int, int, set[str]]] = []
        active_quotes: set[str] = set()
        prev_pos = 0

        for pos, is_start, quote_id in quote_boundaries:
            # If we have active quotes and position changed, save current region
            if active_quotes and pos > prev_pos:
                quote_regions.append((prev_pos, pos, active_quotes.copy()))

            # Update active quotes
            if is_start:
                active_quotes.add(quote_id)
            else:
                active_quotes.discard(quote_id)

            prev_pos = pos

        # Now build entity span positions (still using IDs)
        entity_span_positions: list[
            tuple[int, bool, str]
        ] = []  # (position, is_opening, span_id)

        for start, end, span_id, _, _, _ in html_entity_spans:
            entity_span_positions.append((start, True, span_id))
            entity_span_positions.append((end, False, span_id))

        entity_span_positions.sort(key=lambda x: (x[0], not x[1]))

        # Build annotated HTML using quote regions and entity spans
        if not quote_regions and not entity_span_positions:
            # No annotations, return as-is
            return PrerenderedDocument(
                doc_id=self.resource.id.id,
                html=self.renderer.html,
                quote_map=quote_map,
                entity_map=entity_map,
            )

        # Process HTML by quote regions, inserting entity spans within each region
        html_parts = []
        current_pos = 0

        consumed_boundary_events: set[tuple[int, str]] = set()

        # Group entity spans by their containing quote region
        for region_start, region_end, quote_ids in quote_regions:
            # Add any HTML before this region (unquoted)
            if region_start > current_pos:
                # Check for entity spans in the unquoted region
                unquoted_html = self._build_html_with_entities(
                    current_pos,
                    region_start,
                    entity_span_positions,
                    entity_span_info,
                    consumed_boundary_events,
                )
                html_parts.append(unquoted_html)

            # Open quote span with classes for all active quotes
            quote_classes = " ".join(sorted(quote_ids))
            html_parts.append(f'<span class="quote-span {quote_classes}">')

            # Add content with entity spans
            region_html = self._build_html_with_entities(
                region_start,
                region_end,
                entity_span_positions,
                entity_span_info,
                consumed_boundary_events,
            )
            html_parts.append(region_html)

            # Close quote span
            html_parts.append("</span>")

            current_pos = region_end

        # Add any remaining HTML after last quote region
        if current_pos < len(self.renderer.html):
            remaining_html = self._build_html_with_entities(
                current_pos,
                len(self.renderer.html),
                entity_span_positions,
                entity_span_info,
                consumed_boundary_events,
            )
            html_parts.append(remaining_html)

        annotated_html = "".join(html_parts)

        return PrerenderedDocument(
            doc_id=self.resource.id.id,
            html=annotated_html,
            quote_map=quote_map,
            entity_map=entity_map,
        )

    def _build_html_with_entities(
        self,
        start_pos: int,
        end_pos: int,
        entity_span_positions: list[tuple[int, bool, str]],
        entity_span_info: dict[str, tuple[str, str]],
        consumed_boundary_events: set[tuple[int, str]],
    ) -> str:
        """Build HTML for a region, inserting entity spans.

        Args:
            start_pos: Start position in HTML
            end_pos: End position in HTML
            entity_span_positions: List of (position, is_opening, span_id) tuples
            entity_span_info: Map of span_id -> (entity_name, matched_term)

        Returns:
            HTML string with entity spans inserted
        """
        # Filter entity positions to those within this region
        relevant_entities = []
        for pos, is_opening, span_id in entity_span_positions:
            boundary_key = (pos, span_id)
            if (
                not is_opening
                and boundary_key in consumed_boundary_events
                and pos == start_pos
            ):
                continue
            if start_pos <= pos < end_pos:
                relevant_entities.append((pos, is_opening, span_id))
            elif not is_opening and pos == end_pos:
                consumed_boundary_events.add(boundary_key)
                relevant_entities.append((pos, is_opening, span_id))

        if not relevant_entities:
            # No entities in this region
            return self.renderer.html[start_pos:end_pos]

        html_parts = []
        current_pos = start_pos

        for pos, is_opening, span_id in relevant_entities:
            # Add HTML before this entity span
            if pos > current_pos:
                html_parts.append(self.renderer.html[current_pos:pos])

            # Add entity span tag
            if is_opening:
                if span_id in entity_span_info:
                    entity_name, matched_term = entity_span_info[span_id]
                    use_abbr = len(matched_term) < len(entity_name)

                    if use_abbr:
                        # Escape entity name for HTML attribute
                        escaped_name = _escape_html(entity_name)
                        html_parts.append(
                            f'<span id="{span_id}" class="entity-span">'
                            f'<abbr title="{escaped_name}">'
                        )
                    else:
                        html_parts.append(f'<span id="{span_id}" class="entity-span">')
            else:
                # Closing tag
                if span_id in entity_span_info:
                    entity_name, matched_term = entity_span_info[span_id]
                    use_abbr = len(matched_term) < len(entity_name)
                    if use_abbr:
                        html_parts.append("</abbr></span>")
                    else:
                        html_parts.append("</span>")

            current_pos = pos

        # Add remaining HTML in this region
        if current_pos < end_pos:
            html_parts.append(self.renderer.html[current_pos:end_pos])

        return "".join(html_parts)


def _escape_html(text: str) -> str:
    """Escape HTML special characters.

    Args:
        text: Text to escape

    Returns:
        HTML-safe text
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
