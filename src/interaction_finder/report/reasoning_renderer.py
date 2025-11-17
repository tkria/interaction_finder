"""Reasoning sidebar template rendering.

Pre-renders reasoning panel HTML templates for all pairs and assessments,
with entity highlighting and navigation elements.
"""

import re
from typing import Any


class EntityHighlighter:
    """Highlights entity mentions in plain text with HTML spans.

    Uses position-based matching (like document body) rather than iterative
    regex replacement to avoid nested HTML in attributes.
    """

    def __init__(
        self,
        entity1_terms: list[str],
        entity2_terms: list[str],
        entity1_name: str,
        entity2_name: str,
    ):
        """Initialize highlighter with entity search terms.

        Args:
            entity1_terms: Entity 1 name + aliases
            entity2_terms: Entity 2 name + aliases
            entity1_name: Canonical name for entity 1 (for title attribute)
            entity2_name: Canonical name for entity 2 (for title attribute)
        """
        self.entity1_terms = entity1_terms
        self.entity2_terms = entity2_terms
        self.entity1_name = entity1_name
        self.entity2_name = entity2_name

    def highlight(self, text: str) -> str:
        """Highlight entity mentions with HTML spans.

        Uses position-based approach to avoid nested HTML:
        1. Find all match positions in plain text
        2. Deduplicate overlapping matches (longest first)
        3. Insert spans in reverse order (before HTML escaping)
        4. Escape HTML of final result

        Args:
            text: Plain text to highlight

        Returns:
            HTML string with entity mentions wrapped in
            <span class="entity-highlight entity1/entity2">
        """
        # Step 1: Find all matches with positions
        matches = self._find_all_matches(text)

        # Step 2: Deduplicate overlapping matches
        non_overlapping = self._remove_overlaps(matches)

        # Step 3: Insert spans in reverse order (preserves positions)
        # Do this BEFORE HTML escaping so positions stay valid
        result_text = text
        for start, end, entity_type, canonical, matched_text in reversed(
            non_overlapping
        ):
            span = (
                f'<span class="entity-highlight {entity_type}" '
                f'title="{_escape_html(canonical)}">{matched_text}</span>'
            )
            result_text = result_text[:start] + span + result_text[end:]

        # Step 4: Escape HTML OUTSIDE of spans we just inserted
        # Split by our inserted tags and escape only the text portions
        return self._escape_text_outside_spans(result_text)

    def _escape_text_outside_spans(self, html: str) -> str:
        """Escape HTML in text portions, leaving our span tags intact.

        Args:
            html: HTML string with entity-highlight spans

        Returns:
            HTML with text escaped but span tags preserved
        """
        # Split by our span tags
        parts = []
        current_pos = 0

        # Pattern to match our inserted spans
        import re

        span_pattern = re.compile(
            r'<span class="entity-highlight (?:entity1|entity2)" title="[^"]*">.*?</span>'
        )

        for match in span_pattern.finditer(html):
            # Escape text before this span
            if match.start() > current_pos:
                parts.append(_escape_html(html[current_pos : match.start()]))

            # Keep span as-is (it's our generated HTML)
            parts.append(match.group(0))
            current_pos = match.end()

        # Escape remaining text
        if current_pos < len(html):
            parts.append(_escape_html(html[current_pos:]))

        return "".join(parts)

    def _find_all_matches(self, text: str) -> list[tuple[int, int, str, str, str]]:
        """Find all entity matches with their positions.

        Args:
            text: Plain text to search (before HTML escaping)

        Returns:
            List of (start, end, entity_type, canonical_name, matched_text)
            sorted by position, then by length (longest first)
        """
        matches = []

        # Build term mapping: lowercase term -> (entity_type, canonical_name)
        term_map: dict[str, tuple[str, str]] = {}
        for term in self.entity1_terms:
            term_map[term.lower()] = ("entity1", self.entity1_name)
        for term in self.entity2_terms:
            term_map[term.lower()] = ("entity2", self.entity2_name)

        # Sort by length (longest first) for better overlap resolution
        all_terms = sorted(
            self.entity1_terms + self.entity2_terms,
            key=len,
            reverse=True,
        )

        # Find all matches
        for term in all_terms:
            entity_type, canonical = term_map[term.lower()]
            pattern = re.compile(re.escape(term), re.IGNORECASE)

            for match in pattern.finditer(text):
                matches.append(
                    (
                        match.start(),
                        match.end(),
                        entity_type,
                        canonical,
                        match.group(0),  # Actual matched text (preserves case)
                    )
                )

        # Sort by position, then by length (longest first at same position)
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        return matches

    def _remove_overlaps(
        self, matches: list[tuple[int, int, str, str, str]]
    ) -> list[tuple[int, int, str, str, str]]:
        """Remove overlapping matches, keeping longest at each position.

        Args:
            matches: Sorted list of (start, end, entity_type, canonical, text)

        Returns:
            Non-overlapping matches
        """
        non_overlapping = []

        for match in matches:
            start, end = match[0], match[1]

            # Check if this overlaps with any already-selected match
            overlaps = any(
                not (end <= existing[1] or start >= existing[0])
                for existing in non_overlapping
            )

            if not overlaps:
                non_overlapping.append(match)

        return non_overlapping


class ReasoningTemplateRenderer:
    """Renders reasoning panel templates for a single pair."""

    def __init__(
        self, pair: dict[str, Any], pair_idx: int, doc_idx_map: dict[str, int]
    ):
        """Initialize renderer for a specific pair.

        Args:
            pair: Pair data dictionary from prepare_report_data
            pair_idx: Index of this pair in the full pairs array
            doc_idx_map: Mapping of resource_id -> doc_idx
        """
        self.pair = pair
        self.pair_idx = pair_idx
        self.doc_idx_map = doc_idx_map

        # Build entity search terms
        self.entity1_terms = [pair["entity1"]["name"]] + pair["entity1"]["aliases"]
        self.entity2_terms = [pair["entity2"]["name"]] + pair["entity2"]["aliases"]

        # Create highlighter
        self.highlighter = EntityHighlighter(
            self.entity1_terms,
            self.entity2_terms,
            pair["entity1"]["name"],
            pair["entity2"]["name"],
        )

    def render_overall_template(self) -> str:
        """Render overall pair reasoning template.

        Returns:
            Complete HTML for reasoning panel (overall assessment)
        """
        pair = self.pair

        # Highlight entities in reasoning text
        highlighted_reasoning = self.highlighter.highlight(pair["reasoning"])

        # Build entity aliases sections
        entity1_aliases_html = _render_aliases(pair["entity1"]["aliases"])
        entity2_aliases_html = _render_aliases(pair["entity2"]["aliases"])

        return f"""
    <div class="reasoning-panel">
        <div class="pair-header">
            <div class="pair-header-entities">
                <span>{_escape_html(pair["entity1"]["name"])}</span>
                <span>{_escape_html(pair["entity2"]["name"])}</span>
            </div>
            <div class="pair-header-relation">{_escape_html(pair["relationship"])}</div>
        </div>
        <div class="reasoning-title">Overall Assessment</div>
        <div class="reasoning-content">
            <p><strong>Confidence:</strong> <span class="confidence-badge confidence-{_escape_html(pair["confidence"])}">{_escape_html(pair["confidence"])}</span></p>
            <p>{highlighted_reasoning}</p>
        </div>
        <div class="entity-details-section">
            <div class="reasoning-title">Entity Details</div>
            <div class="entity-detail-item">
                <strong>{_escape_html(pair["entity1"]["name"])}</strong> ({_escape_html(pair["entity1"]["kind"])})
                {entity1_aliases_html}
            </div>
            <div class="entity-detail-item">
                <strong>{_escape_html(pair["entity2"]["name"])}</strong> ({_escape_html(pair["entity2"]["kind"])})
                {entity2_aliases_html}
            </div>
        </div>
    </div>"""

    def render_assessment_template(
        self,
        assess: dict[str, Any],
        assess_idx: int,
        all_pairs: list[dict[str, Any]],
    ) -> str:
        """Render per-document assessment reasoning template.

        Args:
            assess: Assessment data dictionary
            assess_idx: Index of this assessment in sorted assessments list
            all_pairs: Full list of pairs (for finding other pairs using same doc)

        Returns:
            Complete HTML for reasoning panel (document assessment)
        """
        pair = self.pair

        # Get doc_idx from resource_id
        doc_idx = self.doc_idx_map.get(assess["resource_id"])
        if doc_idx is None:
            doc_idx = 0  # Fallback

        # Highlight entities in assessment reasoning
        highlighted_reasoning = self.highlighter.highlight(assess["reasoning"])

        # Render quote navigation with doc_idx for quote IDs
        quote_nav_html = self._render_quote_navigation(assess["quotes"], doc_idx)

        # Render other pairs navigation
        other_pairs_html = self._render_other_pairs_navigation(
            assess["resource_id"], all_pairs
        )

        return f"""
    <div class="reasoning-panel">
        <div class="pair-header">
            <div class="pair-header-entities">
                <span>{_escape_html(pair["entity1"]["name"])}</span>
                <span>{_escape_html(pair["entity2"]["name"])}</span>
            </div>
            <div class="pair-header-relation">{_escape_html(pair["relationship"])}</div>
        </div>
        <div class="reasoning-title">Document Assessment</div>
        <div class="reasoning-content">
            <p><strong>Document:</strong> {_escape_html(assess["title"])}</p>
            <p><strong>Relationship:</strong> {_escape_html(assess["relationship"])}</p>
            <p><strong>Confidence:</strong> <span class="confidence-badge confidence-{_escape_html(assess["confidence"])}">{_escape_html(assess["confidence"])}</span></p>
            <p>{highlighted_reasoning}</p>
        </div>
        {quote_nav_html}
        {other_pairs_html}
    </div>"""

    def _render_quote_navigation(
        self, quotes: list[dict[str, Any]], doc_idx: int
    ) -> str:
        """Render quote navigation list.

        Args:
            quotes: List of quote dictionaries from assessment
            doc_idx: Document index for generating quote IDs

        Returns:
            HTML for quote navigation section (empty string if no quotes)
        """
        if not quotes:
            return ""

        items = []
        for idx, quote in enumerate(quotes):
            quote_text = quote["text"]
            preview = quote_text[:60]
            if len(quote_text) > 60:
                preview += "..."

            # Generate quote ID for direct navigation
            quote_id = f"doc-{doc_idx}-quote-{idx}"

            items.append(
                f"""
                <li class="quote-nav-item" onclick="scrollToQuote('{quote_id}')" title="{_escape_html(quote_text)}">
                    <span class="quote-number">{idx + 1}</span>
                    <span class="quote-preview">{_escape_html(preview)}</span>
                </li>"""
            )

        items_html = "".join(items)

        return f"""
        <div class="quote-navigation">
            <div class="quote-nav-title">Jump to Quotes ({len(quotes)})</div>
            <ul class="quote-nav-list">
                {items_html}
            </ul>
        </div>"""

    def _render_other_pairs_navigation(
        self, doc_id: str, all_pairs: list[dict[str, Any]]
    ) -> str:
        """Render navigation to other pairs using this document.

        Args:
            doc_id: Resource ID of current document
            all_pairs: Full list of pairs

        Returns:
            HTML for other pairs navigation section (empty string if none)
        """
        other_pairs = []

        for idx, pair in enumerate(all_pairs):
            # Skip current pair
            if idx == self.pair_idx:
                continue

            # Check if this pair uses the current document
            if any(a["resource_id"] == doc_id for a in pair["assessments"]):
                other_pairs.append((idx, pair))

        if not other_pairs:
            return ""

        items = []
        for pair_idx, pair in other_pairs:
            items.append(
                f"""
                <li class="quote-nav-item" onclick="selectPairAndDocument({pair_idx}, '{_escape_html(doc_id)}')">
                    <span class="quote-preview">
                        {_escape_html(pair["entity1"]["name"])} {_escape_html(pair["entity2"]["name"])}
                    </span>
                </li>"""
            )

        items_html = "".join(items)

        return f"""
        <div class="quote-navigation">
            <div class="quote-nav-title">Other Pairs ({len(other_pairs)})</div>
            <ul class="quote-nav-list">
                {items_html}
            </ul>
        </div>"""


def render_all_reasoning_templates(
    pairs: list[dict[str, Any]],
    doc_idx_map: dict[str, int],
) -> dict[str, dict[str, str]]:
    """Render all reasoning templates for all pairs.

    Args:
        pairs: List of pair dictionaries (with pre-sorted assessments)
        doc_idx_map: Mapping of resource_id -> doc_idx for template ID generation

    Returns:
        Nested dict mapping pair_idx -> template_type -> HTML
        Example: {"0": {"overall": "<div>...</div>", "doc-3": "<div>...</div>"}}
    """
    templates: dict[str, dict[str, str]] = {}

    for pair_idx, pair in enumerate(pairs):
        renderer = ReasoningTemplateRenderer(pair, pair_idx, doc_idx_map)

        pair_templates: dict[str, str] = {}

        # Render overall assessment template
        pair_templates["overall"] = renderer.render_overall_template()

        # Render per-assessment templates using doc_idx as key
        for assess_idx, assess in enumerate(pair["assessments"]):
            doc_idx = doc_idx_map.get(assess["resource_id"])
            if doc_idx is not None:
                pair_templates[f"doc-{doc_idx}"] = renderer.render_assessment_template(
                    assess, assess_idx, pairs
                )

        templates[str(pair_idx)] = pair_templates

    return templates


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
        .replace("'", "&#39;")
    )


def _render_aliases(aliases: list[str]) -> str:
    """Render entity aliases section.

    Args:
        aliases: List of alias strings

    Returns:
        HTML for aliases (empty string if no aliases)
    """
    if not aliases:
        return ""

    aliases_text = ", ".join(aliases)
    return f"""
                <div class="alias-tooltip">Aliases: {_escape_html(aliases_text)}</div>"""
