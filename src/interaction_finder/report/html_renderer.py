"""HTML rendering with position tracking for report generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Sequence

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import BaseModel

from interaction_finder.resources import Resource, ResourceQuote
from interaction_finder.text_mapping import TextPositionMapper


@dataclass
class DocumentQuoteEntry:
    """Quote entry paired with the referencing pair indices."""

    quote: ResourceQuote
    pair_indices: set[int]


@dataclass
class HTMLTag:
    """Represents an HTML tag found in text."""

    tag_name: str  # e.g., "strong", "em", "a"
    is_opening: bool  # True for <tag>, False for </tag>
    start_pos: int  # Position where tag starts (at '<')
    end_pos: int  # Position where tag ends (after '>')
    is_self_closing: bool = False  # For <br />, <img />, etc.


@dataclass
class HTMLTagPair:
    """Represents a matched pair of opening/closing tags."""

    tag_name: str
    opening: HTMLTag
    closing: HTMLTag | None  # None for self-closing tags
    content_start: int  # Position after opening tag
    content_end: int  # Position before closing tag


class HTMLTagScanner:
    """Scans HTML to find and pair up all tags."""

    # Tags that are self-closing
    VOID_ELEMENTS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    # Inline tags that we care about for nesting validation
    INLINE_TAGS = {
        "a",
        "abbr",
        "b",
        "bdi",
        "bdo",
        "cite",
        "code",
        "data",
        "dfn",
        "em",
        "i",
        "kbd",
        "mark",
        "q",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "time",
        "u",
        "var",
    }

    def __init__(self, html: str):
        self.html = html
        self.tags: list[HTMLTag] = []
        self.tag_pairs: list[HTMLTagPair] = []

    def scan(self) -> list[HTMLTagPair]:
        """Scan HTML and return list of matched tag pairs.

        Returns:
            List of HTMLTagPair objects representing matched tags
        """
        self._find_all_tags()
        self._pair_tags()
        return self.tag_pairs

    def _find_all_tags(self) -> None:
        """Find all HTML tags in the text."""
        # Regex to match HTML tags
        # Matches: <tagname>, </tagname>, <tagname attr="value">, <tagname />
        tag_pattern = re.compile(
            r"<"  # Opening bracket
            r"(/?)?"  # Optional closing slash
            r"([a-zA-Z][a-zA-Z0-9]*)"  # Tag name
            r"(?:\s[^>]*)?"  # Optional attributes
            r"(/?)?"  # Optional self-closing slash
            r">"  # Closing bracket
        )

        for match in tag_pattern.finditer(self.html):
            is_closing = bool(match.group(1))
            tag_name = match.group(2).lower()
            is_self_closing = bool(match.group(3)) or tag_name in self.VOID_ELEMENTS

            tag = HTMLTag(
                tag_name=tag_name,
                is_opening=not is_closing,
                start_pos=match.start(),
                end_pos=match.end(),
                is_self_closing=is_self_closing,
            )
            self.tags.append(tag)

    def _pair_tags(self) -> None:
        """Pair up opening and closing tags."""
        # Stack-based matching for nested tags
        stack: list[HTMLTag] = []

        for tag in self.tags:
            if tag.is_self_closing:
                # Self-closing tag - create pair with no closing tag
                pair = HTMLTagPair(
                    tag_name=tag.tag_name,
                    opening=tag,
                    closing=None,
                    content_start=tag.end_pos,
                    content_end=tag.end_pos,
                )
                self.tag_pairs.append(pair)
            elif tag.is_opening:
                stack.append(tag)
            else:
                # Closing tag - find matching opening tag
                # Search backwards through stack for matching tag
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i].tag_name == tag.tag_name:
                        opening = stack.pop(i)
                        pair = HTMLTagPair(
                            tag_name=tag.tag_name,
                            opening=opening,
                            closing=tag,
                            content_start=opening.end_pos,
                            content_end=tag.start_pos,
                        )
                        self.tag_pairs.append(pair)
                        break


@dataclass
class SpanInsertionPlan:
    """Plan for where to insert opening/closing span tags."""

    open_pos: int  # Where to insert opening <span>
    close_pos: int  # Where to insert closing </span>
    original_open_pos: int  # Original desired position
    original_close_pos: int  # Original desired position
    adjusted: bool  # Whether positions were adjusted
    split_required: bool  # Whether span needs to be split
    split_segments: list[tuple[int, int]] | None = (
        None  # Split (start, end) segments if needed
    )


class HTMLTagValidator:
    """Validates and adjusts span insertion positions to avoid malformed HTML."""

    def __init__(self, html: str, tag_pairs: list[HTMLTagPair]):
        self.html = html
        self.tag_pairs = tag_pairs
        # Filter to only inline tags for performance
        self.inline_pairs = [
            p for p in tag_pairs if p.tag_name in HTMLTagScanner.INLINE_TAGS
        ]

    def validate_insertion(self, open_pos: int, close_pos: int) -> SpanInsertionPlan:
        """Validate span insertion positions and adjust if needed.

        Args:
            open_pos: Desired position for opening <span>
            close_pos: Desired position for closing </span>

        Returns:
            SpanInsertionPlan with validated/adjusted positions
        """
        # Check if positions would create invalid nesting
        conflicts = self._find_conflicts(open_pos, close_pos)

        if not conflicts:
            # No conflicts - positions are valid
            return SpanInsertionPlan(
                open_pos=open_pos,
                close_pos=close_pos,
                original_open_pos=open_pos,
                original_close_pos=close_pos,
                adjusted=False,
                split_required=False,
            )

        # Try to adjust positions to avoid conflicts
        adjusted_plan = self._try_adjust_positions(open_pos, close_pos, conflicts)
        if adjusted_plan:
            return adjusted_plan

        # Adjustment failed - compute split segments
        split_segments = self._compute_split_segments(open_pos, close_pos, conflicts)
        return SpanInsertionPlan(
            open_pos=open_pos,
            close_pos=close_pos,
            original_open_pos=open_pos,
            original_close_pos=close_pos,
            adjusted=False,
            split_required=True,
            split_segments=split_segments,
        )

    def _find_conflicts(self, open_pos: int, close_pos: int) -> list[HTMLTagPair]:
        """Find tags that would be invalidly split by span insertion."""
        return [
            tag
            for tag in self.inline_pairs
            if (tag.content_start <= open_pos < tag.content_end)
            != (tag.content_start < close_pos <= tag.content_end)
        ]

    def _try_adjust_positions(
        self, open_pos: int, close_pos: int, conflicts: list[HTMLTagPair]
    ) -> SpanInsertionPlan | None:
        """Adjust positions to avoid conflicts by moving boundaries outside tags."""
        # Separate conflicts affecting opening vs closing positions
        open_conflicts = [
            t for t in conflicts if t.content_start <= open_pos < t.content_end
        ]
        close_conflicts = [
            t for t in conflicts if t.content_start < close_pos <= t.content_end
        ]

        # Move boundaries outside conflicting tags
        adjusted_open = (
            min(t.opening.start_pos for t in open_conflicts)
            if open_conflicts
            else open_pos
        )
        adjusted_close = (
            max(
                t.closing.end_pos if t.closing else t.opening.end_pos
                for t in close_conflicts
            )
            if close_conflicts
            else close_pos
        )

        # Validate adjustment doesn't create new conflicts
        if self._find_conflicts(adjusted_open, adjusted_close):
            return None

        return SpanInsertionPlan(
            open_pos=adjusted_open,
            close_pos=adjusted_close,
            original_open_pos=open_pos,
            original_close_pos=close_pos,
            adjusted=True,
            split_required=False,
        )

    def _compute_split_segments(
        self, open_pos: int, close_pos: int, conflicts: list[HTMLTagPair]
    ) -> list[tuple[int, int]]:
        """Split span into segments that don't cross conflicting tag boundaries."""
        # Collect all relevant boundaries within our range
        boundaries = {open_pos, close_pos}
        for tag in conflicts:
            for pos in [
                tag.opening.start_pos,
                tag.opening.end_pos,
                tag.content_start,
                tag.content_end,
                tag.closing.start_pos if tag.closing else None,
                tag.closing.end_pos if tag.closing else None,
            ]:
                if pos and open_pos < pos < close_pos:
                    boundaries.add(pos)

        # Test each potential segment for validity
        sorted_boundaries = sorted(boundaries)
        segments = []
        for i in range(len(sorted_boundaries) - 1):
            start, end = sorted_boundaries[i], sorted_boundaries[i + 1]
            # Valid if no conflicts for this segment
            if not self._find_conflicts(start, end):
                # Merge with previous if adjacent
                if segments and segments[-1][1] == start:
                    segments[-1] = (segments[-1][0], end)
                else:
                    segments.append((start, end))

        return segments or [(open_pos, close_pos)]


class MarkdownToHTMLRenderer:
    """Markdown renderer that preserves original character offsets."""

    def __init__(self, text: str):
        self.original_text = text
        self.html: str = ""
        self._position_mapper: TextPositionMapper | None = None
        self._md = MarkdownIt(
            "commonmark", {"html": False, "typographer": False}
        ).enable("table")
        self._line_offsets = self._compute_line_offsets(text)
        self._builder = HTMLBuilder(text, self._line_offsets)

    def render(self) -> TextPositionMapper:
        """Render markdown to HTML and return position mapper.

        Returns:
            TextPositionMapper for mapping original markdown positions to HTML
        """
        tokens = self._md.parse(self.original_text)
        self._builder.reset()
        self._builder.render(tokens)
        self.html = self._builder.html

        # Create position mapper for original -> HTML mapping
        # Offsets are (original_pos, delta) where delta = HTML - original
        # So: source=original, target=HTML
        offsets = self._builder.position_offsets
        if not offsets:
            offsets = [(0, 0)]

        self._position_mapper = TextPositionMapper(
            source=self.original_text, target=self.html, offsets=offsets
        )
        return self._position_mapper

    def map_original_to_html_position(self, original_pos: int) -> int:
        """Map a position in the original markdown to rendered HTML."""
        if self._position_mapper is None:
            raise RuntimeError("Position mapper not available. Call render() first.")
        return self._position_mapper.targetpos(original_pos)

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
        self._last_delta: int | None = None
        # Track cursors by line to handle table cells on the same row
        self._line_cursors: dict[tuple[int, int], InlineCursor] = {}

    def reset(self) -> None:
        self.html_parts = []
        self._html_len = 0
        self.position_offsets = []
        self._last_orig_pos = 0
        self._last_delta = None
        self._line_cursors = {}

    @property
    def html(self) -> str:
        return "".join(self.html_parts)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _record_position(self, orig_pos: int) -> None:
        """Record position mapping using delta encoding.

        Maps original markdown position to HTML position.
        Only records when delta changes (sparse representation).
        Delta = target_pos - source_pos (HTML - original).

        Offsets are stored as (source_pos, delta) where source=original, target=HTML.
        """
        current_delta = self._html_len - orig_pos
        if self._last_delta is None or current_delta != self._last_delta:
            self.position_offsets.append((orig_pos, current_delta))
            self._last_delta = current_delta

    def _append_literal(self, text: str, orig_pos: int | None = None) -> None:
        if not text:
            return
        if orig_pos is not None:
            self._record_position(orig_pos)
        self.html_parts.append(text)
        self._html_len += len(text)

    def _append_text(self, text: str, orig_start: int) -> None:
        for idx, ch in enumerate(text):
            html_fragment = _escape_html_char(ch)
            self._record_position(orig_start + idx)
            self.html_parts.append(html_fragment)
            self._html_len += len(html_fragment)
            self._last_orig_pos = orig_start + idx + 1

    def _append_newline(self, orig_pos: int) -> None:
        self._record_position(orig_pos)
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

    def _format_attrs(self, attrs: Any) -> str:
        if not attrs:
            return ""
        if isinstance(attrs, Mapping):
            attr_iter = attrs.items()
        else:
            attr_iter = attrs

        normalized: list[tuple[str, Any]] = []
        for item in attr_iter:
            if isinstance(item, (list, tuple)):
                if not item:
                    continue
                name = str(item[0])
                value = item[1] if len(item) > 1 else ""
            else:
                name = str(item)
                value = ""
            normalized.append((name, value))

        if not normalized:
            return ""

        parts: list[str] = []
        for name, value in normalized:
            if value is False:
                continue
            if value is None or value is True:
                parts.append(name)
            else:
                parts.append(f'{name}="{_escape_html_attr(str(value))}"')

        return f" {' '.join(parts)}" if parts else ""

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

        # Reuse cursor for the same line range (important for table cells)
        # This ensures that when multiple cells share the same row, each cell
        # continues from where the previous cell left off
        line_key = (start_line, end_line)
        if line_key in self._line_cursors:
            cursor = self._line_cursors[line_key]
        else:
            block_text = self.text[start:end]
            cursor = InlineCursor(block_text, start)
            self._line_cursors[line_key] = cursor

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

        if renderer._position_mapper is None:
            raise ValueError("Renderer must have called render() before annotation")

    def annotate(
        self,
        doc_idx: int,
        quotes: list[DocumentQuoteEntry],
        entities: dict[str, Any],  # pair_idx -> {entity1: ..., entity2: ...}
    ) -> PrerenderedDocument:
        """Annotate HTML with quote and entity spans using numeric doc index.

        Args:
            doc_idx: Sequential document index for ID generation
            quotes: List of validated ResourceQuote objects for this document
            entities: Dictionary mapping pair indices to entity information
                Expected format: {pair_idx: {"entity1": {...}, "entity2": {...}}}

        Returns:
            PrerenderedDocument with annotated HTML and metadata
        """

        quote_map: dict[str, QuoteMetadata] = {}
        entity_map: dict[str, EntityMetadata] = {}

        # Step 1: Map all quotes to HTML coordinates
        html_quote_spans: list[
            tuple[int, int, str, list[int]]
        ] = []  # (start, end, span_id, pair_indices)

        for quote_idx, quote_entry in enumerate(quotes):
            quote = quote_entry.quote
            # Map quote spans from original to HTML
            html_spans = []
            for orig_start, orig_end in quote.spans:
                html_start = self.renderer.map_original_to_html_position(orig_start)
                html_end = self.renderer.map_original_to_html_position(orig_end)
                html_spans.append((html_start, html_end))

            # Generate quote span ID using doc_idx
            quote_id = f"doc-{doc_idx}-quote-{quote_idx}"

            # Record quote metadata
            quote_map[quote_id] = QuoteMetadata(
                span_id=quote_id,
                pair_indices=sorted(quote_entry.pair_indices),
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
        # Use normalized search via Resource._position_mapper for Greek letter support
        # Key: (start_pos, end_pos, entity_name, matched_term) in ORIGINAL text coordinates
        entity_position_map: dict[tuple[int, int, str, str], dict[str, Any]] = {}

        for pair_idx, pair_entities in entities.items():
            entity1 = pair_entities.get("entity1")
            entity2 = pair_entities.get("entity2")

            if not entity1 or not entity2:
                continue

            # Search for entity mentions within quote boundaries
            for entity in [entity1, entity2]:
                entity_name = entity.get("name", "")
                entity_kind = entity.get("kind", "")
                entity_aliases = entity.get("aliases", [])

                if not entity_name:
                    continue

                # Search for entity name and aliases in NORMALIZED text
                # This handles Greek letters (α ↔ alpha) automatically
                search_terms = [entity_name] + entity_aliases

                for term in search_terms:
                    # Use Resource's NormalizedTextMapper to search
                    # Automatically normalizes search term (Greek letters, punctuation, etc.)
                    # Returns positions in original text coordinates
                    matches = self.resource._position_mapper.findall(term)

                    for orig_start, orig_end in matches:
                        # Check if this position is within any quote span (in original coordinates)
                        in_quote = False
                        containing_quote_ids = []
                        for quote_id, quote_meta in quote_map.items():
                            for q_start, q_end in quote_meta.original_spans:
                                if q_start <= orig_start < q_end:
                                    in_quote = True
                                    containing_quote_ids.append(quote_id)
                                    break

                        if in_quote:
                            # Extract the actual matched text from original document
                            actual_matched_text = self.resource.text[
                                orig_start:orig_end
                            ]

                            # Create key for this position in ORIGINAL coordinates
                            position_key = (orig_start, orig_end, entity_name, term)

                            if position_key not in entity_position_map:
                                entity_position_map[position_key] = {
                                    "name": entity_name,
                                    "kind": entity_kind,
                                    "aliases": entity_aliases,
                                    "pair_indices": [pair_idx],
                                    "matched_term": actual_matched_text,
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

        # Second pass: Map entity positions from original text to HTML coordinates
        # Then resolve overlapping entity spans in HTML coordinates
        entity_positions_html = []
        for (
            (
                orig_start,
                orig_end,
                entity_name,
                _search_term,  # Ignore - this is the search term, not the actual matched text
            ),
            entity_data,
        ) in entity_position_map.items():
            # Map positions from original text to HTML
            html_start = self.renderer.map_original_to_html_position(orig_start)
            html_end = self.renderer.map_original_to_html_position(orig_end)

            # Use the actual matched text from entity_data
            actual_matched_text = entity_data["matched_term"]

            entity_positions_html.append(
                (html_start, html_end, entity_name, actual_matched_text, entity_data)
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

            # Generate unique entity span ID using doc_idx
            entity_id = f"doc-{doc_idx}-entity-{entity_counter}"
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
        # Build entity span lookup: span_id -> (entity_name, matched_term, pair_indices)
        entity_span_info = {
            span_id: (entity_name, matched_term, pair_indices)
            for _, _, span_id, entity_name, matched_term, pair_indices in html_entity_spans
        }

        # Scan HTML for tag structure to validate span insertion positions
        scanner = HTMLTagScanner(self.renderer.html)
        tag_pairs = scanner.scan()
        validator = HTMLTagValidator(self.renderer.html, tag_pairs)

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

        # Collect quote boundaries, validating and adjusting positions to avoid malformed HTML
        for start, end, span_id, _ in html_quote_spans:
            # First adjust end position to respect block boundaries
            adjusted_end = end
            for boundary in block_boundaries:
                if start < boundary <= end:
                    adjusted_end = min(adjusted_end, boundary)

            # Validate span insertion positions to avoid splitting inline tags
            plan = validator.validate_insertion(start, adjusted_end)

            # Handle the validation result
            if plan.split_required and plan.split_segments:
                # Span needs to be split into multiple segments
                # Each segment gets its own start/end boundary with the same quote ID
                for seg_start, seg_end in plan.split_segments:
                    quote_boundaries.append((seg_start, True, span_id))
                    quote_boundaries.append((seg_end, False, span_id))
            else:
                # Use adjusted positions (or original if no adjustment needed)
                final_start = plan.open_pos
                final_end = plan.close_pos
                quote_boundaries.append((final_start, True, span_id))
                quote_boundaries.append((final_end, False, span_id))

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
        annotated_html = ""
        if not quote_regions and not entity_span_positions:
            # No annotations, use HTML as-is
            annotated_html = self.renderer.html
        else:
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

                # Open quote span with classes for all active quotes and data-pairs attribute
                quote_classes = " ".join(sorted(quote_ids))
                # Collect all pair indices for quotes in this region
                region_pair_indices = set()
                for quote_id in quote_ids:
                    quote_meta = quote_map.get(quote_id)
                    if quote_meta:
                        region_pair_indices.update(quote_meta.pair_indices)
                # Format as space-separated list of numeric indices
                pairs_attr = " ".join(str(idx) for idx in sorted(region_pair_indices))
                html_parts.append(
                    f'<span class="quote-span {quote_classes}" data-pairs="{pairs_attr}">'
                )

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

        # Build document links (URL badge and DOI link)
        url_badge = ""
        if self.resource.id.url:
            escaped_url = _escape_html(self.resource.id.url)
            url_badge = f"""
                <a class="document-url-badge" href="{escaped_url}" target="_blank" rel="noreferrer noopener">
                    <span aria-hidden="true">&#128279;</span>
                    <span>View original</span>
                </a>"""

        doi_link = ""
        if self.resource.doi:
            escaped_doi = _escape_html(self.resource.doi)
            doi_link = f"""
                <a class="document-doi-link" href="https://doi.org/{escaped_doi}" target="_blank" rel="noreferrer noopener">
                    DOI: {escaped_doi}
                </a>"""

        # Prepend links if present
        links_html = ""
        if url_badge or doi_link:
            links_html = f"""
            <div class="document-links">{url_badge}{doi_link}
            </div>"""

        # Build list of all pairs that reference this document
        all_pair_indices = set()
        for pair_idx in entities.keys():
            all_pair_indices.add(pair_idx)
        pairs_attr = " ".join(str(idx) for idx in sorted(all_pair_indices))

        # Wrap document text in container with doc_idx and pair indices
        full_html = (
            f"{links_html}"
            f'<div id="doc-{doc_idx}" class="document-text" data-pairs="{pairs_attr}">'
            f"{annotated_html}"
            f"</div>"
        )

        return PrerenderedDocument(
            doc_id=self.resource.id.id,
            html=full_html,
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
                    entity_name, matched_term, pair_indices = entity_span_info[span_id]
                    use_abbr = len(matched_term) < len(entity_name)

                    # Format pair indices as space-separated list
                    pairs_attr = " ".join(str(idx) for idx in sorted(pair_indices))
                    escaped_name = _escape_html(entity_name)

                    if use_abbr:
                        html_parts.append(
                            f'<span id="{span_id}" class="entity-span" '
                            f'data-entity="{escaped_name}" data-pairs="{pairs_attr}">'
                            f'<abbr title="{escaped_name}">'
                        )
                    else:
                        html_parts.append(
                            f'<span id="{span_id}" class="entity-span" '
                            f'data-entity="{escaped_name}" data-pairs="{pairs_attr}">'
                        )
            else:
                # Closing tag
                if span_id in entity_span_info:
                    entity_name, matched_term, pair_indices = entity_span_info[span_id]
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
