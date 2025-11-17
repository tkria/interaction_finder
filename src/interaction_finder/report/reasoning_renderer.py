"""Reasoning sidebar template rendering.

Pre-renders reasoning panel HTML templates for all pairs and assessments,
with entity highlighting and navigation elements.
"""

import re
from typing import Any


class EntityHighlighter:
    """Highlights entity mentions in plain text with HTML spans.

    Uses regex-based search for simplicity (no position mapping needed for
    reasoning text, unlike document annotation which requires precise tracking).
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

        Args:
            text: Plain text to highlight

        Returns:
            HTML string with entity mentions wrapped in
            <span class="entity-highlight entity1/entity2">
        """
        # Escape HTML first to prevent injection
        text = _escape_html(text)

        # Build term mapping: lowercase term -> (entity_type, canonical_name)
        term_map: dict[str, tuple[str, str]] = {}
        for term in self.entity1_terms:
            term_map[term.lower()] = ("entity1", self.entity1_name)
        for term in self.entity2_terms:
            term_map[term.lower()] = ("entity2", self.entity2_name)

        # Sort by length (longest first) to avoid partial matches
        # Example: "BRCA1" should match before "BRCA"
        all_terms = sorted(
            self.entity1_terms + self.entity2_terms,
            key=len,
            reverse=True,
        )

        # Replace each term with highlighted version
        # Case-insensitive matching
        for term in all_terms:
            entity_type, canonical = term_map[term.lower()]
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            replacement = (
                f'<span class="entity-highlight {entity_type}" '
                f'title="{_escape_html(canonical)}">\\g<0></span>'
            )
            text = pattern.sub(replacement, text)

        return text


class ReasoningTemplateRenderer:
    """Renders reasoning panel templates for a single pair."""

    def __init__(self, pair: dict[str, Any], pair_idx: int):
        """Initialize renderer for a specific pair.

        Args:
            pair: Pair data dictionary from prepare_report_data
            pair_idx: Index of this pair in the full pairs array
        """
        self.pair = pair
        self.pair_idx = pair_idx

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

        # Highlight entities in assessment reasoning
        highlighted_reasoning = self.highlighter.highlight(assess["reasoning"])

        # Render quote navigation
        quote_nav_html = self._render_quote_navigation(assess["quotes"])

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

    def _render_quote_navigation(self, quotes: list[dict[str, Any]]) -> str:
        """Render quote navigation list.

        Args:
            quotes: List of quote dictionaries from assessment

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

            items.append(
                f"""
                <li class="quote-nav-item" onclick="scrollToQuote({idx})" title="{_escape_html(quote_text)}">
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
) -> dict[str, dict[str, str]]:
    """Render all reasoning templates for all pairs.

    Args:
        pairs: List of pair dictionaries (with pre-sorted assessments)

    Returns:
        Nested dict mapping pair_idx -> template_type -> HTML
        Example: {"0": {"overall": "<div>...</div>", "assess_0": "<div>...</div>"}}
    """
    templates: dict[str, dict[str, str]] = {}

    for pair_idx, pair in enumerate(pairs):
        renderer = ReasoningTemplateRenderer(pair, pair_idx)

        pair_templates: dict[str, str] = {}

        # Render overall assessment template
        pair_templates["overall"] = renderer.render_overall_template()

        # Render per-assessment templates
        for assess_idx, assess in enumerate(pair["assessments"]):
            pair_templates[f"assess_{assess_idx}"] = (
                renderer.render_assessment_template(assess, assess_idx, pairs)
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
