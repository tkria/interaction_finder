"""Reasoning sidebar template rendering.

Pre-renders reasoning panel HTML templates for all pairs and assessments,
with entity highlighting and navigation elements.
"""

import re
from typing import Any


def _quote_key_for_id(quote: dict[str, Any]) -> tuple:
    """Generate lookup key for quote ID mapping.

    Must match the key format used in data_prep._quote_key_for_id.
    """
    return (
        tuple(tuple(span) for span in quote["spans"]),
        quote["text"],
        quote.get("fuzzy_corrected", False),
    )


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

        Uses position-based approach to avoid nested/overlapping spans:
        1. Find all match positions in plain text
        2. Deduplicate overlapping matches (longest first)
        3. Escape HTML characters in original text
        4. Insert spans in reverse order to preserve positions

        Args:
            text: Plain text to highlight

        Returns:
            HTML string with entity mentions wrapped in
            <span class="entity-highlight entity1/entity2">
        """
        # Step 1: Find all matches with positions in ORIGINAL text
        matches = self._find_all_matches(text)

        # Step 2: Deduplicate overlapping matches (keep longest)
        non_overlapping = self._remove_overlaps(matches)

        # Step 3: Escape HTML in original text
        escaped_text = _escape_html(text)

        # Step 4: Insert spans in reverse order
        # Important: We use positions from original text, but insert into escaped text
        # Since escaping can change positions, we need to track the offset
        result_parts = []
        prev_end = len(text)

        # Process matches in reverse order (end to start)
        for start, end, entity_type, canonical, matched_text in reversed(
            non_overlapping
        ):
            # Escape the matched text and canonical name
            escaped_matched = _escape_html(matched_text)
            escaped_canonical = _escape_html(canonical)

            # Add text after this match (already escaped)
            result_parts.insert(0, _escape_html(text[end:prev_end]))

            # Add highlighted span
            span = (
                f'<span class="entity-highlight {entity_type}" '
                f'title="{escaped_canonical}">{escaped_matched}</span>'
            )
            result_parts.insert(0, span)

            prev_end = start

        # Add text before first match
        result_parts.insert(0, _escape_html(text[:prev_end]))

        return "".join(result_parts)

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
                not (end <= existing[0] or start >= existing[1])
                for existing in non_overlapping
            )

            if not overlaps:
                non_overlapping.append(match)

        return non_overlapping


class ReasoningTemplateRenderer:
    """Renders reasoning panel templates for a single pair."""

    def __init__(
        self,
        pair: dict[str, Any],
        pair_idx: int,
        quote_id_map: dict[tuple[int, tuple], str],
    ):
        """Initialize renderer for a specific pair.

        Args:
            pair: Pair data dictionary from prepare_report_data
            pair_idx: Index of this pair in the full pairs array
            quote_id_map: Mapping of (doc_idx, quote_key) -> quote_id
        """
        self.pair = pair
        self.pair_idx = pair_idx
        self.quote_id_map = quote_id_map
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
            <span class="pair-entity">{_escape_html(pair["entity1"]["name"])}</span>
            <span class="pair-relation">{_escape_html(pair["relationship"])}</span>
            <span class="pair-entity">{_escape_html(pair["entity2"]["name"])}</span>
        </div>
        <div class="reasoning-title">Overall Assessment</div>
        <div class="reasoning-content">
            <strong>Confidence:</strong> <span class="confidence-{_escape_html(pair["confidence"])}">{_escape_html(pair["confidence"])}</span>
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

    def render_document_group_template(
        self,
        doc_group: dict[str, Any],
        all_pairs: list[dict[str, Any]],
    ) -> str:
        """Render reasoning template for document group (consolidates multiple assessments).

        Args:
            doc_group: Document group with assessments list
            all_pairs: Full list of pairs (for cross-document navigation)

        Returns:
            HTML for reasoning panel showing all assessments from this document
        """
        assessments = doc_group["assessments"]
        doc_idx = doc_group["doc_idx"]

        # Helper to format polarity badge
        polarity_map = {
            "supporting": "S",
            "refuting": "R",
            "neutral": "N",
            "irrelevant": "I",
        }
        # Source descriptors for non-direct assessments
        source_descriptors = {"sweep": "co-mention sweep"}

        def render_source_line(assess: dict[str, Any]) -> str:
            """Render source indicator for non-direct assessments."""
            source = assess.get("source", "direct")
            if source == "direct":
                return ""
            descriptor = source_descriptors.get(source, source)
            return (
                f'<div class="assessment-source">From {_escape_html(descriptor)}</div>'
            )

        def render_assessment(idx: int, assess: dict[str, Any]) -> str:
            polarity = assess.get("polarity", "")
            confidence = assess.get("confidence", "low")
            source_line = render_source_line(assess)
            return f"""
        <div class="assessment-section open">
            <div class="assessment-header" onclick="this.parentElement.classList.toggle('open')">
                <span class="assessment-label"><span class="assess-num-badge">{idx + 1}</span>&nbsp;{_escape_html(assess["relationship"])}</span>
                <span class="pc-chip polarity-{_escape_html(polarity)} confidence-{_escape_html(confidence)}">
                    <span>{_escape_html(polarity_map.get(polarity, polarity))}</span>
                    <span>{_escape_html(confidence)}</span>
                </span>
            </div>
            <div class="assessment-content">
                {source_line}<p>{self.highlighter.highlight(assess["reasoning"])}</p>
            </div>
        </div>"""

        def render_single_assessment(assess: dict[str, Any]) -> str:
            """Render single assessment in a box without polarity/confidence badge."""
            source_line = render_source_line(assess)
            return f"""
        <div class="assessment-section open">
            <div class="assessment-header" onclick="this.parentElement.classList.toggle('open')">
                <span class="assessment-label">{_escape_html(assess["relationship"])}</span>
            </div>
            <div class="assessment-content">
                {source_line}<p>{self.highlighter.highlight(assess["reasoning"])}</p>
            </div>
        </div>"""

        # Collect and deduplicate quotes, tracking which assessments use each
        quote_assessments: dict[tuple, list[int]] = {}  # quote_key -> [assess_idx, ...]
        all_quotes: list[dict[str, Any]] = []
        for assess_idx, assess in enumerate(assessments):
            for quote in assess["quotes"]:
                key = _quote_key_for_id(quote)
                if key not in quote_assessments:
                    quote_assessments[key] = []
                    all_quotes.append(quote)
                quote_assessments[key].append(assess_idx + 1)  # 1-indexed for display
        if len(assessments) == 1:
            # Single assessment: render without box wrapper
            assessments_html = render_single_assessment(assessments[0])
        else:
            # Multiple assessments: render each in a section box
            assessments_html = "".join(
                render_assessment(i, a) for i, a in enumerate(assessments)
            )

        # Get title from first assessment
        title = assessments[0].get("title", "Untitled")
        count_text = (
            f"{len(assessments)} assessment{'s' if len(assessments) > 1 else ''}"
        )

        return f"""
    <div class="reasoning-panel">
        <div class="pair-header">
            <span class="pair-entity">{_escape_html(self.pair["entity1"]["name"])}</span>
            <span class="pair-relation">{_escape_html(self.pair["relationship"])}</span>
            <span class="pair-entity">{_escape_html(self.pair["entity2"]["name"])}</span>
        </div>
        <div class="reasoning-doc-title">{_escape_html(title)}</div>
        <div class="reasoning-subtitle">{count_text}</div>
        {assessments_html}
        {self._render_quote_navigation(all_quotes, doc_idx, quote_assessments, len(assessments) > 1)}
        {self._render_other_pairs_navigation(doc_idx, all_pairs)}
    </div>"""

    def _render_quote_navigation(
        self,
        quotes: list[dict[str, Any]],
        doc_idx: int,
        quote_assessments: dict[tuple, list[int]] | None = None,
        show_assessment_badges: bool = False,
    ) -> str:
        """Render quote navigation list.

        Args:
            quotes: List of quote dictionaries from assessment
            doc_idx: Document index for quote ID lookup
            quote_assessments: Mapping of quote_key -> list of assessment indices (1-indexed)
            show_assessment_badges: Whether to show which assessments use each quote

        Returns:
            HTML for quote navigation section (empty string if no quotes)
        """
        if not quotes:
            return ""
        items = []
        for idx, quote in enumerate(quotes):
            quote_text = quote["text"]
            preview = quote_text[:60] + ("..." if len(quote_text) > 60 else "")
            # Look up quote ID from map
            quote_key = _quote_key_for_id(quote)
            quote_id = self.quote_id_map.get((doc_idx, quote_key), "")
            if not quote_id:
                continue  # Skip quotes not found in map
            # Build assessment badges if multiple assessments
            badges_html = ""
            if show_assessment_badges and quote_assessments:
                assess_indices = quote_assessments.get(quote_key, [])
                if assess_indices:
                    badges = "".join(
                        f'<span class="assess-num-badge">{i}</span>'
                        for i in assess_indices
                    )
                    badges_html = f'<div class="quote-assess-badges">{badges}</div>'
            items.append(
                f"""
                <li class="quote-nav-item" onclick="scrollToQuote('{quote_id}')" title="{_escape_html(quote_text)}">
                    <div class="quote-nav-left">
                        <div class="quote-number">{idx + 1}</div>{badges_html}
                    </div>
                    <div class="quote-preview">{_escape_html(preview)}</div>
                </li>"""
            )
        if not items:
            return ""
        return f"""
        <div class="quote-navigation">
            <div class="quote-nav-title">Jump to Quotes ({len(items)})</div>
            <ul class="quote-nav-list">
                {"".join(items)}
            </ul>
        </div>"""

    def _render_other_pairs_navigation(
        self, doc_idx: int, all_pairs: list[dict[str, Any]]
    ) -> str:
        """Render navigation to other pairs using this document.

        Args:
            doc_idx: Document index of current document
            all_pairs: Full list of pairs

        Returns:
            HTML for other pairs navigation section (empty string if none)
        """
        other_pairs = []

        for idx, pair in enumerate(all_pairs):
            # Skip current pair
            if idx == self.pair_idx:
                continue

            # Check if this pair uses the current document (by doc_idx)
            if any(a["doc_idx"] == doc_idx for a in pair["assessments"]):
                other_pairs.append((idx, pair))

        if not other_pairs:
            return ""

        # Check if all other pairs share a common entity with current pair
        current_e1 = self.pair["entity1"]["name"]
        current_e2 = self.pair["entity2"]["name"]
        # Determine which entity (if any) is shared by ALL other pairs
        all_share_e1 = all(
            p["entity1"]["name"] == current_e1 or p["entity2"]["name"] == current_e1
            for _, p in other_pairs
        )
        all_share_e2 = all(
            p["entity1"]["name"] == current_e2 or p["entity2"]["name"] == current_e2
            for _, p in other_pairs
        )
        # Fade the shared entity, highlight the varying one
        fade_entity = None
        if all_share_e1 and not all_share_e2:
            fade_entity = current_e1
        elif all_share_e2 and not all_share_e1:
            fade_entity = current_e2

        items = []
        for pair_idx, pair in other_pairs:
            e1_name = pair["entity1"]["name"]
            e2_name = pair["entity2"]["name"]
            if fade_entity:
                e1_class = (
                    "other-pair-faded"
                    if e1_name == fade_entity
                    else "other-pair-highlight"
                )
                e2_class = (
                    "other-pair-faded"
                    if e2_name == fade_entity
                    else "other-pair-highlight"
                )
                pair_html = (
                    f'<span class="{e1_class}">{_escape_html(e1_name)}</span> '
                    f'<span class="{e2_class}">{_escape_html(e2_name)}</span>'
                )
            else:
                pair_html = f"{_escape_html(e1_name)} {_escape_html(e2_name)}"
            items.append(
                f'<li class="quote-nav-item" onclick="selectPairAndDocument({pair_idx}, {doc_idx})">{pair_html}</li>'
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
    quote_id_map: dict[tuple[int, tuple], str],
) -> dict[str, dict[str, str]]:
    """Render all reasoning templates for all pairs.

    Args:
        pairs: List of pair dictionaries (with document_groups)
        quote_id_map: Mapping of (doc_idx, quote_key) -> quote_id

    Returns:
        Nested dict mapping pair_idx -> template_type -> HTML
        Example: {"0": {"overall": "<div>...</div>", "doc-3": "<div>...</div>"}}
    """
    templates: dict[str, dict[str, str]] = {}
    for pair_idx, pair in enumerate(pairs):
        renderer = ReasoningTemplateRenderer(pair, pair_idx, quote_id_map)

        pair_templates: dict[str, str] = {}

        # Render overall assessment template
        pair_templates["overall"] = renderer.render_overall_template()

        # Render per-document-group templates using doc_idx as key
        # Each template may now contain multiple assessments from the same document
        for doc_group in pair["document_groups"]:
            doc_idx = doc_group["doc_idx"]
            pair_templates[f"doc-{doc_idx}"] = renderer.render_document_group_template(
                doc_group, pairs
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
