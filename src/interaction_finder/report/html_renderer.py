"""HTML rendering with position tracking for report generation.

This module provides facilities to convert markdown documents to HTML while
maintaining precise coordinate mappings between original and rendered text.
This enables accurate placement of quote and entity annotations in the final HTML.

The rendering approach follows patterns established in resources.py:
- Single-pass character-by-character processing
- Simultaneous output generation and position offset tracking
- Binary search for efficient coordinate lookup
"""

import re
from typing import Any

from pydantic import BaseModel

from interaction_finder.resources import Resource, ResourceQuote


class MarkdownToHTMLRenderer:
    """Convert markdown to HTML while tracking position transformations.

    Processes markdown character-by-character, building HTML output and
    maintaining a coordinate mapping that allows translation of positions
    from the original markdown text to the rendered HTML.

    Follows the pattern established in Resource._build_normalized_text_and_offsets()
    for simultaneous processing and position tracking.
    """

    def __init__(self, text: str):
        """Initialize renderer with markdown text.

        Args:
            text: Markdown text to render
        """
        self.original_text = text
        self.html: str = ""
        self.position_offsets: list[tuple[int, int]] = []

    def render(self) -> tuple[str, list[tuple[int, int]]]:
        """Render markdown to HTML with position tracking.

        Uses regex-based approach: find all markdown patterns, compute replacements,
        track offset shifts, and build position mapping.

        Returns:
            Tuple of (html_string, position_offsets)
            position_offsets maps HTML positions to original text positions:
            [(html_pos, original_pos), ...]
        """
        text = self.original_text

        # Collect all transformations to apply
        # Format: (start, end, replacement_text, description)
        transformations: list[tuple[int, int, str, str]] = []

        # Find headers (### heading, ## heading, # heading)
        # Match at line start only
        for match in re.finditer(r"^(#{1,3})\s+(.+?)$", text, re.MULTILINE):
            level = len(match.group(1))
            content = match.group(2)
            # Escape HTML first
            content = _escape_html(content)
            # Process inline formatting within header content (after escaping)
            # Handle links first
            content = re.sub(
                r"\[([^\]]+)\]\(([^\)]+)\)",
                lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
                content,
            )
            # Handle bold
            content = re.sub(r"\*\*([^\*]+)\*\*", r"<strong>\1</strong>", content)
            # Handle italic
            content = re.sub(r"(?<!\*)\*([^\*]+)\*(?!\*)", r"<em>\1</em>", content)
            content = re.sub(r"_([^_]+)_", r"<em>\1</em>", content)
            # Handle code
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            # Replace with <hN>content</hN>
            replacement = f"<h{level}>{content}</h{level}>"
            transformations.append(
                (match.start(), match.end(), replacement, f"header-{level}")
            )

        # Find bold (**text**)
        # Allow content with single asterisks but not double asterisks
        for match in re.finditer(r"\*\*(.+?)\*\*", text):
            content = match.group(1)
            # Skip if contains ** (would be nested bold, which we don't support)
            if "**" in content:
                continue
            # Process inline formatting within bold
            content_escaped = _escape_html(content)
            # Handle links
            content_escaped = re.sub(
                r"\[([^\]]+)\]\(([^\)]+)\)",
                lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
                content_escaped,
            )
            content_escaped = re.sub(r"\*([^\*]+)\*", r"<em>\1</em>", content_escaped)
            content_escaped = re.sub(r"_([^_]+)_", r"<em>\1</em>", content_escaped)
            content_escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", content_escaped)
            replacement = f"<strong>{content_escaped}</strong>"
            transformations.append((match.start(), match.end(), replacement, "bold"))

        # Find italic (*text* or _text_) - but not if part of **
        for match in re.finditer(r"(?<!\*)\*([^\*]+)\*(?!\*)", text):
            content = match.group(1)
            # Process inline formatting within italic
            content = _escape_html(content)
            content = re.sub(r"\*\*([^\*]+)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            replacement = f"<em>{content}</em>"
            transformations.append((match.start(), match.end(), replacement, "italic"))

        for match in re.finditer(r"_([^_]+)_", text):
            content = match.group(1)
            content = _escape_html(content)
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            replacement = f"<em>{content}</em>"
            transformations.append(
                (match.start(), match.end(), replacement, "italic_underscore")
            )

        # Find inline code (`code`)
        for match in re.finditer(r"`([^`]+)`", text):
            content = match.group(1)
            replacement = f"<code>{content}</code>"  # No escaping in code
            transformations.append((match.start(), match.end(), replacement, "code"))

        # Find links ([text](url))
        for match in re.finditer(r"\[([^\]]+)\]\(([^\)]+)\)", text):
            link_text = match.group(1)
            link_url = match.group(2)
            replacement = (
                f'<a href="{_escape_html(link_url)}">{_escape_html(link_text)}</a>'
            )
            transformations.append((match.start(), match.end(), replacement, "link"))

        # Find list items (- item, * item, 1. item) at line start
        for match in re.finditer(r"^[-\*]\s+(.+?)$", text, re.MULTILINE):
            content = match.group(1)
            # Process inline formatting within list items
            content = _escape_html(content)
            content = re.sub(r"\*\*([^\*]+)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"(?<!\*)\*([^\*]+)\*(?!\*)", r"<em>\1</em>", content)
            content = re.sub(r"_([^_]+)_", r"<em>\1</em>", content)
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            replacement = f"<li>{content}</li>"
            transformations.append(
                (match.start(), match.end(), replacement, "list_unordered")
            )

        for match in re.finditer(r"^\d+\.\s+(.+?)$", text, re.MULTILINE):
            content = match.group(1)
            # Process inline formatting within list items
            content = _escape_html(content)
            content = re.sub(r"\*\*([^\*]+)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"(?<!\*)\*([^\*]+)\*(?!\*)", r"<em>\1</em>", content)
            content = re.sub(r"_([^_]+)_", r"<em>\1</em>", content)
            content = re.sub(r"`([^`]+)`", r"<code>\1</code>", content)
            replacement = f"<li>{content}</li>"
            transformations.append(
                (match.start(), match.end(), replacement, "list_ordered")
            )

        # Sort transformations by position (apply in order)
        # If overlapping, keep first one
        transformations.sort(
            key=lambda x: (x[0], -x[1])
        )  # Sort by start, then by length (desc)

        # Remove overlapping transformations (keep first match)
        filtered_transformations = []
        last_end = -1
        for trans in transformations:
            if trans[0] >= last_end:
                filtered_transformations.append(trans)
                last_end = trans[1]

        # Build HTML and position offsets by applying transformations
        position_offsets: list[tuple[int, int]] = []
        html_parts: list[str] = []

        current_pos = 0

        for trans_start, trans_end, replacement, desc in filtered_transformations:
            # Copy text before this transformation (with HTML escaping)
            before_text = text[current_pos:trans_start]
            for i, char in enumerate(before_text):
                orig_pos = current_pos + i
                html_pos = len("".join(html_parts))
                position_offsets.append((html_pos, orig_pos))

                # Escape HTML
                if char == "<":
                    html_parts.append("&lt;")
                elif char == ">":
                    html_parts.append("&gt;")
                elif char == "&":
                    html_parts.append("&amp;")
                elif char == '"':
                    html_parts.append("&quot;")
                else:
                    html_parts.append(char)

            # Record position at start of transformed region
            html_pos_before = len("".join(html_parts))
            position_offsets.append((html_pos_before, trans_start))

            # Add replacement
            html_parts.append(replacement)

            # Move past the transformed region
            current_pos = trans_end

        # Copy remaining text after last transformation
        remaining_text = text[current_pos:]
        for i, char in enumerate(remaining_text):
            orig_pos = current_pos + i
            html_pos = len("".join(html_parts))
            position_offsets.append((html_pos, orig_pos))

            # Escape HTML
            if char == "<":
                html_parts.append("&lt;")
            elif char == ">":
                html_parts.append("&gt;")
            elif char == "&":
                html_parts.append("&amp;")
            elif char == '"':
                html_parts.append("&quot;")
            else:
                html_parts.append(char)

        html = "".join(html_parts)

        # Handle paragraphs by splitting on \n\n and wrapping each in <p> tags
        # Track position shifts as we add paragraph tags
        if "\n\n" in html:
            # Split into paragraphs
            paragraphs = html.split("\n\n")
            new_html_parts = []
            new_offsets = []

            current_html_pos = 0
            current_shift = 0

            for i, para in enumerate(paragraphs):
                if not para.strip():
                    current_html_pos += 2  # Skip \n\n
                    continue

                # Add opening <p>
                new_html_parts.append("<p>")
                current_shift += 3

                # Update offsets for this paragraph
                para_end = current_html_pos + len(para)
                for html_pos, orig_pos in position_offsets:
                    if current_html_pos <= html_pos < para_end:
                        new_offsets.append((html_pos + current_shift, orig_pos))

                # Add paragraph content
                new_html_parts.append(para)

                # Add closing </p>
                new_html_parts.append("</p>")
                current_shift += 4

                if i < len(paragraphs) - 1:
                    # Account for the \n\n separator we're skipping
                    current_html_pos += len(para) + 2
                else:
                    current_html_pos += len(para)

            html = "".join(new_html_parts)
            position_offsets = new_offsets
        else:
            # Single paragraph - simple wrap
            offset_shift = 3  # Length of "<p>"
            position_offsets = [
                (html_pos + offset_shift, orig_pos)
                for html_pos, orig_pos in position_offsets
            ]
            html = "<p>" + html + "</p>"

        # Final position mapping
        position_offsets.append((len(html), len(text)))

        self.html = html
        self.position_offsets = position_offsets

        return html, position_offsets

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
        # First pass: collect all entity positions and which pairs reference them
        # Key: (start_pos, end_pos, entity_name, matched_term)
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

                # Search for entity name and aliases in HTML
                search_terms = [entity_name] + entity_aliases

                for term in search_terms:
                    # Simple case-insensitive search in HTML
                    html = self.renderer.html.lower()
                    term_lower = term.lower()

                    pos = 0
                    while True:
                        pos = html.find(term_lower, pos)
                        if pos == -1:
                            break

                        # Check if this position is within any quote span
                        in_quote = False
                        for q_start, q_end, _, _ in html_quote_spans:
                            if q_start <= pos < q_end:
                                in_quote = True
                                break

                        if in_quote:
                            # Create key for this position (include matched term)
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

        # Second pass: resolve overlapping entity spans
        # Convert to list and sort by position, then by length (longer first for overlap resolution)
        entity_positions = [
            (start, end, entity_name, matched_term, entity_data)
            for (
                start,
                end,
                entity_name,
                matched_term,
            ), entity_data in entity_position_map.items()
        ]
        entity_positions.sort(
            key=lambda x: (x[0], -(x[1] - x[0]))
        )  # Sort by start, then by length descending

        # Remove overlapping spans (keep the first/longest match at each position)
        non_overlapping_entities = []
        for start, end, entity_name, matched_term, entity_data in entity_positions:
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

        # Collect all span positions (quotes and entities)
        # Format: (position, is_opening, span_id, span_type)
        span_positions: list[tuple[int, bool, str, str]] = []

        for start, end, span_id, _ in html_quote_spans:
            span_positions.append((start, True, span_id, "quote"))
            span_positions.append((end, False, span_id, "quote"))

        for start, end, span_id, _, _, _ in html_entity_spans:
            span_positions.append((start, True, span_id, "entity"))
            span_positions.append((end, False, span_id, "entity"))

        # Sort by position, then by type (opening before closing at same position)
        span_positions.sort(key=lambda x: (x[0], not x[1]))

        # Build annotated HTML
        if not span_positions:
            # No annotations, return as-is
            return PrerenderedDocument(
                doc_id=self.resource.id.id,
                html=self.renderer.html,
                quote_map=quote_map,
                entity_map=entity_map,
            )

        html_parts = []
        current_pos = 0

        for pos, is_opening, span_id, span_type in span_positions:
            # Add HTML before this span
            if pos > current_pos:
                html_parts.append(self.renderer.html[current_pos:pos])

            # Add span tag
            if is_opening:
                if span_type == "quote":
                    html_parts.append(f'<span id="{span_id}" class="quote-span">')
                else:  # entity
                    # Check if we should add <abbr> tag for shortened forms
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
                if span_type == "entity" and span_id in entity_span_info:
                    entity_name, matched_term = entity_span_info[span_id]
                    use_abbr = len(matched_term) < len(entity_name)
                    if use_abbr:
                        html_parts.append("</abbr></span>")
                    else:
                        html_parts.append("</span>")
                else:
                    html_parts.append("</span>")

            current_pos = pos

        # Add remaining HTML
        if current_pos < len(self.renderer.html):
            html_parts.append(self.renderer.html[current_pos:])

        annotated_html = "".join(html_parts)

        return PrerenderedDocument(
            doc_id=self.resource.id.id,
            html=annotated_html,
            quote_map=quote_map,
            entity_map=entity_map,
        )


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
