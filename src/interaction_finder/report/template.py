"""HTML template for report generation.

Renders self-contained HTML reports with minimal data-attributes.
No JSON embedding - all data queryable from HTML structure.
"""

from typing import Any

from interaction_finder.report.assets import get_css, get_js
from interaction_finder.version import format_version_tooltip


def _render_pair_cards(pairs: list[dict[str, Any]]) -> str:
    """Render pair cards as HTML.

    Args:
        pairs: List of pair data dictionaries

    Returns:
        HTML string for all pair cards
    """
    # Determine if we should show entity kinds
    # Show kinds if: (1) more than 2 distinct kinds, OR (2) any self-pairs exist
    all_kinds = set()
    has_self_pair = False
    for pair in pairs:
        all_kinds.add(pair["entity1"]["kind"])
        all_kinds.add(pair["entity2"]["kind"])
        if pair["entity1"]["kind"] == pair["entity2"]["kind"]:
            has_self_pair = True
    show_kinds = len(all_kinds) > 2 or has_self_pair

    cards = []

    for idx, pair in enumerate(pairs):
        # Build entity aliases (comma-separated for data-attribute)
        entity1_aliases_data = ",".join(pair["entity1"]["aliases"])
        entity2_aliases_data = ",".join(pair["entity2"]["aliases"])
        entity1_aliases_display = ", ".join(pair["entity1"]["aliases"])
        entity2_aliases_display = ", ".join(pair["entity2"]["aliases"])

        # Build document groups data - simplified structure for client-side rendering
        import json

        doc_groups_data = json.dumps(
            [
                {
                    "doc_idx": g["doc_idx"],
                    "total_quotes": g["total_quotes"],
                    "relationships": g["relationships"],
                    "assessment_count": len(g["assessments"]),
                    # First assessment's polarity/evidence for badge display
                    "polarity": g["assessments"][0].get("polarity"),
                    "overall": g["assessments"][0].get("overall"),
                }
                for g in pair["document_groups"]
            ]
        )

        doc_indices = " ".join(str(g["doc_idx"]) for g in pair["document_groups"])

        # Build card classes
        card_classes = ["pair-card"]
        if not pair["accepted"]:
            card_classes.append("rejected")
        if pair.get("contentious"):
            card_classes.append("contentious")

        # Build kinds/relationship row
        if show_kinds:
            kinds_html = f"""
            <div class="pair-kinds">
                <span>{_escape_html(pair["entity1"]["kind"])}</span>
                <span class="relationship-label">{_format_relationship(pair["relationship"])}</span>
                <span>{_escape_html(pair["entity2"]["kind"])}</span>
            </div>"""
        else:
            kinds_html = f"""
            <div class="pair-relationship-only">
                <span class="relationship-label">{_format_relationship(pair["relationship"])}</span>
            </div>"""

        # Build summary line: doc count, polarity badges, quote count, evidence level
        doc_count = pair["doc_count"]
        quote_count = pair["quote_count"]
        overall = pair.get("overall", 1)
        label = pair.get("label", "None")

        # Build compact polarity pill with counts and tooltips
        polarity_summary = pair.get("polarity_summary", {})
        polarity_labels = {
            "positive": "Positive relationships",
            "negative": "Negating relationships",
            "neutral": "Neutral relationships",
        }
        polarity_counts = []
        for polarity in ("positive", "negative", "neutral"):
            info = polarity_summary.get(polarity)
            if info and info.get("count", 0) > 0:
                count = info["count"]
                label = polarity_labels[polarity]
                polarity_counts.append(
                    f'<span class="polarity-count polarity-{polarity}" '
                    f'data-tooltip="{label}">{count}</span>'
                )
        polarity_html = (
            f'<span class="polarity-pill">{"".join(polarity_counts)}</span>'
            if polarity_counts
            else ""
        )

        summary_html = f"""
            <div class="pair-summary">
                <span class="pair-counts">{doc_count}d {quote_count}q</span>{polarity_html}<span class="evidence-badge evidence-{overall}" data-tooltip="Evidence level {overall}/9">{_escape_html(label.upper())}</span>
            </div>"""

        # Build complete card with minimal data-attributes
        card_html = f"""
        <div id="pair-{idx}" class="{" ".join(card_classes)}"
             data-e1="{_escape_html(pair["entity1"]["name"])}"
             data-e1a="{_escape_html(entity1_aliases_data)}"
             data-e2="{_escape_html(pair["entity2"]["name"])}"
             data-e2a="{_escape_html(entity2_aliases_data)}"
             data-rel="{_escape_html(pair["relationship"])}"
             data-accepted="{str(pair["accepted"]).lower()}"
             data-docs="{doc_indices}"
             data-contentious="{str(bool(pair.get("contentious"))).lower()}"
             data-doc-groups='{doc_groups_data}'>
            <div class="pair-entities">
                <span title="{_escape_html(entity1_aliases_display)}" data-kind="{_escape_html(pair["entity1"]["kind"])}">
                    {_escape_html(pair["entity1"]["name"])}
                </span>
                <span title="{_escape_html(entity2_aliases_display)}" data-kind="{_escape_html(pair["entity2"]["kind"])}">
                    {_escape_html(pair["entity2"]["name"])}
                </span>
            </div>{kinds_html}{summary_html}
        </div>"""

        cards.append(card_html)

    return "\n".join(cards)


# HTML template with Jinja2 placeholders
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>

    <!-- Pico CSS from CDN -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.colors.min.css">

    <!-- Custom CSS -->
    <style>
{{ css }}
    </style>
</head>
<body>
    <!-- Header -->
    <header>
        <div class="header-content">
            <div class="header-info">
                <h1 class="header-title">{{ title }}</h1>
                <div class="header-stats">
                    <span class="stat-item">
                        <span class="stat-label">Pairs:</span>
                        <span data-stat="pairs">{{ metadata.total_pairs }}</span>
                    </span>
                    {% for kind, count in metadata.entity_stats.items() %}
                    <span class="stat-item">
                        <span class="stat-label">{{ kind }}:</span>
                        <span data-stat="entity-kind-{{ kind }}">{{ count }}</span>
                    </span>
                    {% endfor %}
                    <span class="stat-item">
                        <span class="stat-label">Documents:</span>
                        <span data-stat="documents">{{ metadata.resource_count }}</span>
                    </span>
                    {{ metadata.version_stat }}
                </div>
            </div>
            <div class="header-controls-wrapper">
                <div class="header-controls">
                    <div class="search-wrapper">
                        <input type="text"
                               id="search-input"
                               placeholder="Search entities or relationships..."
                               aria-label="Search">
                        <button type="button"
                                id="search-clear"
                                class="search-clear-btn"
                                aria-label="Clear search"
                                tabindex="-1">×</button>
                    </div>
                    <label>
                        <input type="checkbox" id="show-rejected" role="switch">
                        Show rejected
                    </label>
                </div>
            </div>
        </div>
    </header>

    <!-- Left Sidebar: Pair List -->
    <aside id="sidebar">
{{ pair_cards }}
    </aside>

    <!-- Main Content: Document Accordions -->
    <main id="content">
        <div class="content-placeholder">Select a pair to view documents</div>
    </main>

    <!-- Right Sidebar: Reasoning -->
    <aside id="rightbar">
        <div class="reasoning-panel">
            <p>Select a pair to view reasoning</p>
        </div>
    </aside>

    <!-- Document Templates (pre-rendered HTML) -->
    <div id="document-templates" style="display: none;">
{{ document_templates }}
    </div>

    <!-- Reasoning Templates (pre-rendered HTML) -->
    <div id="reasoning-templates" style="display: none;">
{{ reasoning_templates }}
    </div>

    <!-- JavaScript (no embedded JSON data) -->
    <script>
{{ js }}
    </script>
</body>
</html>
"""


def render_template(
    pairs: list[dict[str, Any]],
    document_html: dict[int, str],
    reasoning_templates: dict[str, dict[str, str]],
    indexed_docs: list[tuple[int, Any]],
    topic: str,
    title: str | None = None,
    version: str | None = None,
) -> str:
    """Render HTML report from prepared data.

    Args:
        pairs: List of pair data dicts with doc indices
        document_html: Mapping of doc_idx -> pre-rendered HTML string
        reasoning_templates: Nested dict pair_idx -> template_type -> HTML
        indexed_docs: List of (doc_idx, resource) tuples for metadata
        topic: Report topic for header
        title: Optional report title (defaults to "Extraction Report: {topic}")
        version: Optional version string to display in header

    Returns:
        Complete HTML document as string
    """
    if title is None:
        title = f"Extraction Report: {topic}"

    # Simple template rendering without Jinja2 dependency
    # Use string replacement for placeholders
    html = HTML_TEMPLATE

    # Build pair cards HTML
    pair_cards_html = _render_pair_cards(pairs)

    # Build doc_idx -> resource map for quick lookup
    doc_resources = {idx: resource for idx, resource in indexed_docs}

    # Build document templates HTML with metadata
    doc_templates_parts = []
    for doc_idx, doc_html in document_html.items():
        # Get resource metadata
        resource = doc_resources.get(doc_idx)
        if resource:
            title_text = resource.title or "Untitled"
            date_html = (
                f'<span class="document-date">{_escape_html(resource.publication_date)}</span>'
                if resource.publication_date
                else ""
            )
        else:
            title_text = "Untitled"
            date_html = ""

        # Wrap document HTML with metadata header
        # The JavaScript will extract this metadata when cloning the template
        # Note: doc_html already contains #doc-{doc_idx} on the .document-text element
        template_content = f"""<div class="document-template-wrapper">
                <div class="document-title">{_escape_html(title_text)}{date_html}</div>
                <div class="document-body">{doc_html}</div>
            </div>"""

        doc_templates_parts.append(
            f'        <template id="doc-template-{doc_idx}">\n'
            f"            {template_content}\n"
            f"        </template>"
        )
    document_templates_html = "\n".join(doc_templates_parts)

    # Build reasoning templates HTML
    reasoning_templates_parts = []
    for pair_idx_str, templates in reasoning_templates.items():
        for template_type, template_html in templates.items():
            template_id = f"reasoning-pair-{pair_idx_str}-{template_type}"
            reasoning_templates_parts.append(
                f'        <template id="{template_id}">\n'
                f"            {template_html}\n"
                f"        </template>"
            )
    reasoning_templates_html = "\n".join(reasoning_templates_parts)

    # Compute statistics from pairs (no pre-computed JSON)
    total_pairs = len(pairs)
    # Count unique documents from all assessments
    unique_docs = set()
    for pair in pairs:
        for assess in pair["assessments"]:
            unique_docs.add(assess["doc_idx"])
    resource_count = len(unique_docs)

    # Build version stat HTML (only if version provided)
    version_stat_html = ""
    if version:
        display_text, tooltip = format_version_tooltip(version)
        tooltip_attr = f' data-tooltip="{_escape_html(tooltip)}"' if tooltip else ""
        version_stat_html = f"""<span class="stat-item">
                        <span class="stat-label">Version:</span>
                        <span{tooltip_attr}>{_escape_html(display_text)}</span>
                    </span>"""
    # Replace placeholders
    replacements = {
        "{{ title }}": _escape_html(title),
        "{{ metadata.topic }}": _escape_html(topic),
        "{{ metadata.total_pairs }}": str(total_pairs),
        "{{ metadata.resource_count }}": str(resource_count),
        "{{ metadata.version_stat }}": version_stat_html,
        "{{ css }}": get_css(),
        "{{ js }}": get_js(),
        "{{ document_templates }}": document_templates_html,
        "{{ reasoning_templates }}": reasoning_templates_html,
        "{{ pair_cards }}": pair_cards_html,
    }

    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    # Handle entity stats loop (compute from pairs)
    entity_kinds: dict[str, set[str]] = {}
    for pair in pairs:
        kind1 = pair["entity1"]["kind"]
        kind2 = pair["entity2"]["kind"]
        if kind1 not in entity_kinds:
            entity_kinds[kind1] = set()
        if kind2 not in entity_kinds:
            entity_kinds[kind2] = set()
        entity_kinds[kind1].add(pair["entity1"]["name"])
        entity_kinds[kind2].add(pair["entity2"]["name"])

    entity_stats_html = ""
    for kind, names in entity_kinds.items():
        entity_stats_html += f"""
                <span class="stat-item">
                    <span class="stat-label">{_escape_html(kind)}:</span>
                    <span>{len(names)}</span>
                </span>"""

    # Replace the Jinja2 loop with rendered HTML
    loop_start = "{% for kind, count in metadata.entity_stats.items() %}"
    loop_end = "{% endfor %}"
    start_idx = html.find(loop_start)
    end_idx = html.find(loop_end)

    if start_idx != -1 and end_idx != -1:
        # Find the content between loop tags (the template)
        html = html[:start_idx] + entity_stats_html + html[end_idx + len(loop_end) :]

    return html


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


def _format_relationship(rel: str) -> str:
    """Format relationship for display (escape HTML and replace underscores)."""
    return _escape_html(rel.replace("_", " "))
