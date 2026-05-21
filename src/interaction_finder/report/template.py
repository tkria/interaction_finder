"""HTML template for report generation.

Renders self-contained HTML reports with minimal data-attributes.
No JSON embedding - all data queryable from HTML structure.
"""

from itertools import groupby
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
                    "label": g["assessments"][0].get("label"),
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
        polarity_tooltips = {
            "positive": "Positive relationships",
            "negative": "Negating relationships",
            "neutral": "Neutral relationships",
        }
        polarity_counts = []
        for polarity in ("positive", "negative", "neutral"):
            info = polarity_summary.get(polarity)
            if info and info.get("count", 0) > 0:
                count = info["count"]
                tooltip = polarity_tooltips[polarity]
                polarity_counts.append(
                    f'<span class="polarity-count polarity-{polarity}" '
                    f'data-tooltip="{tooltip}">{count}</span>'
                )
        polarity_html = (
            f'<span class="polarity-pill">{"".join(polarity_counts)}</span>'
            if polarity_counts
            else ""
        )

        # Build evidence tooltip if there are varying evidence levels
        evidence_levels = pair.get("evidence_levels", [])
        unique_levels = set(evidence_levels)
        evidence_tooltip = ""
        if len(unique_levels) > 1:
            # Group by level, sort descending, format as "4×Strong, 1×Good"
            sorted_levels = sorted(evidence_levels, key=lambda x: -x[0])
            parts = [
                f"{len(list(g))}×{lbl}"
                for (_lvl, lbl), g in groupby(sorted_levels, key=lambda x: x)
            ]
            evidence_tooltip = f' data-tooltip="{_escape_html(", ".join(parts))}"'

        summary_html = f"""
            <div class="pair-summary">
                <span class="pair-counts">{doc_count}d {quote_count}q</span>{polarity_html}<span class="evidence-badge evidence-{overall}"{evidence_tooltip}>{_escape_html(label.upper())}</span>
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
             data-overall="{pair["overall"]}"
             data-doc-count="{pair["doc_count"]}"
             data-quote-count="{pair["quote_count"]}"
             data-relevance="{pair.get("topic_relevance", 3)}"
             data-rank-score="{pair.get("rank_sum_score", 0)}"
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
<body data-topic="{{ metadata.topic }}">
    <!-- Header -->
    <header>
        <div class="header-content">
            <div class="header-info">
                <h1 class="header-title" aria-busy="true">{{ title }}</h1>
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
                    <button type="button"
                            id="filter-toggle"
                            class="outline secondary icon-btn"
                            aria-expanded="false"
                            aria-controls="filter-panel"
                            aria-label="Toggle filter options">
                        <svg class="icon-btn-svg" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 15.5A3.5 3.5 0 0 1 8.5 12 3.5 3.5 0 0 1 12 8.5a3.5 3.5 0 0 1 3.5 3.5 3.5 3.5 0 0 1-3.5 3.5m7.43-2.53c.04-.32.07-.64.07-.97 0-.33-.03-.66-.07-1l2.11-1.63c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.31-.61-.22l-2.49 1c-.52-.39-1.06-.73-1.69-.98l-.37-2.65A.506.506 0 0 0 14 2h-4c-.25 0-.46.18-.5.42l-.37 2.65c-.63.25-1.17.59-1.69.98l-2.49-1c-.22-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64L4.57 11c-.04.34-.07.67-.07 1 0 .33.03.65.07.97l-2.11 1.66c-.19.15-.25.42-.12.64l2 3.46c.12.22.39.3.61.22l2.49-1.01c.52.4 1.06.74 1.69.99l.37 2.65c.04.24.25.42.5.42h4c.25 0 .46-.18.5-.42l.37-2.65c.63-.26 1.17-.59 1.69-.99l2.49 1.01c.22.08.49 0 .61-.22l2-3.46c.12-.22.07-.49-.12-.64l-2.11-1.66z"/>
                        </svg>
                        <span class="filter-active-dot"></span>
                    </button>
                    <button type="button"
                            id="export-toggle"
                            class="icon-btn icon-btn-borderless"
                            aria-label="Open export dialog">
                        <svg class="icon-btn-svg" viewBox="0 0 448 512" xmlns="http://www.w3.org/2000/svg">
                            <path d="M246.6 9.4c-12.5-12.5-32.8-12.5-45.3 0l-128 128c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0L192 109.3 192 320c0 17.7 14.3 32 32 32s32-14.3 32-32l0-210.7 73.4 73.4c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3l-128-128zM64 352c0-17.7-14.3-32-32-32S0 334.3 0 352l0 64c0 53 43 96 96 96l256 0c53 0 96-43 96-96l0-64c0-17.7-14.3-32-32-32s-32 14.3-32 32l0 64c0 17.7-14.3 32-32 32L96 448c-17.7 0-32-14.3-32-32l0-64z"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
        <div id="filter-panel" class="filter-panel" hidden>
            <div class="filter-group">
                <label for="sort-field">Sort:</label>
                <select id="sort-field">
                    <option value="default" selected>Default</option>
                    <option value="evidence">Evidence</option>
                    <option value="relevance">Relevance</option>
                    <option value="docs">Documents</option>
                    <option value="quotes">Quotes</option>
                    <option value="entity">Entity name</option>
                </select>
                <button type="button" id="sort-dir-toggle" class="outline contrast" data-dir="desc" aria-label="Toggle sort direction">↓</button>
            </div>
            <div class="filter-group">
                <label for="evidence-min">Evidence:</label>
                <select id="evidence-min">
                    <option value="0" selected>Any</option>
                    <option value="1">1: None</option>
                    <option value="2">2: Minimal</option>
                    <option value="3">3: Tenuous</option>
                    <option value="4">4: Weak</option>
                    <option value="5">5: Limited</option>
                    <option value="6">6: Moderate</option>
                    <option value="7">7: Good</option>
                    <option value="8">8: Strong</option>
                    <option value="9">9: Robust</option>
                </select>
                <span class="filter-range-sep">–</span>
                <select id="evidence-max">
                    <option value="10" selected>Any</option>
                    <option value="9">9: Robust</option>
                    <option value="8">8: Strong</option>
                    <option value="7">7: Good</option>
                    <option value="6">6: Moderate</option>
                    <option value="5">5: Limited</option>
                    <option value="4">4: Weak</option>
                    <option value="3">3: Tenuous</option>
                    <option value="2">2: Minimal</option>
                    <option value="1">1: None</option>
                </select>
            </div>
            <div class="filter-group">
                <label for="relevance-min">Relevance:</label>
                <select id="relevance-min">
                    <option value="0" selected>Any</option>
                    <option value="1">1: Tangential</option>
                    <option value="2">2: Peripheral</option>
                    <option value="3">3: Related</option>
                    <option value="4">4: Significant</option>
                    <option value="5">5: Central</option>
                </select>
                <span class="filter-range-sep">–</span>
                <select id="relevance-max">
                    <option value="6" selected>Any</option>
                    <option value="5">5: Central</option>
                    <option value="4">4: Significant</option>
                    <option value="3">3: Related</option>
                    <option value="2">2: Peripheral</option>
                    <option value="1">1: Tangential</option>
                </select>
            </div>
            <label class="filter-checkbox">
                <input type="checkbox" id="show-rejected">
                Show rejected
            </label>
            <label class="filter-checkbox">
                <input type="checkbox" id="contentious-only">
                Contentious only
            </label>
            <button type="button"
                    id="filter-clear"
                    class="icon-btn icon-btn-borderless filter-clear-btn"
                    aria-label="Clear all filters"
                    hidden>
                <svg class="icon-btn-svg" viewBox="0 0 576 512" xmlns="http://www.w3.org/2000/svg">
                    <path d="M32 64C19.1 64 7.4 71.8 2.4 83.8S.2 109.5 9.4 118.6L192 301.3 192 416c0 8.5 3.4 16.6 9.4 22.6l64 64c2.5 2.5 5.3 4.5 8.3 6-21.2-30.9-33.6-68.3-33.6-108.6 0-99.4 75.5-181.1 172.3-191l90.4-90.4c9.2-9.2 11.9-22.9 6.9-34.9S492.9 64 480 64L32 64zM432 544a144 144 0 1 0 0-288 144 144 0 1 0 0 288zm59.3-180.7l-36.7 36.7 36.7 36.7c6.2 6.2 6.2 16.4 0 22.6s-16.4 6.2-22.6 0l-36.7-36.7-36.7 36.7c-6.2 6.2-16.4 6.2-22.6 0s-6.2-16.4 0-22.6l36.7-36.7-36.7-36.7c-6.2-6.2-6.2-16.4 0-22.6s16.4-6.2 22.6 0l36.7 36.7 36.7-36.7c6.2-6.2 16.4-6.2 22.6 0s6.2 16.4 0 22.6z"/>
                </svg>
            </button>
        </div>
    </header>

    <!-- Left Sidebar: Pair List -->
    <aside id="sidebar">
{{ pair_cards }}
    </aside>

    <!-- Resizer between sidebar and content -->
    <div class="col-resizer" id="resizer-left" role="separator" aria-orientation="vertical" aria-label="Resize pair list column" tabindex="0"></div>

    <!-- Main Content: Document Accordions -->
    <main id="content">
        <div class="content-placeholder" aria-busy="true">Loading report…</div>
    </main>

    <!-- Resizer between content and rightbar -->
    <div class="col-resizer" id="resizer-right" role="separator" aria-orientation="vertical" aria-label="Resize reasoning column" tabindex="0"></div>

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

    <!-- Export Dialog -->
    <dialog id="export-dialog" aria-labelledby="export-dialog-title">
        <article class="export-modal">
            <header>
                <button aria-label="Close" rel="prev"></button>
                <p><strong id="export-dialog-title">Export</strong></p>
            </header>
            <fieldset id="export-mode-fieldset">
                <legend>Export</legend>
                <!-- populated at runtime: 'Pairs' + one radio per detected kind -->
            </fieldset>
            <fieldset id="export-filters-fieldset">
                <label>
                    <input type="checkbox" role="switch" id="export-apply-filters" checked>
                    Apply current filters
                </label>
            </fieldset>
            <fieldset>
                <legend>Format</legend>
                <input type="radio" id="export-fmt-csv" name="export-format" value="csv" checked>
                <label for="export-fmt-csv">CSV</label>
                <input type="radio" id="export-fmt-tsv" name="export-format" value="tsv">
                <label for="export-fmt-tsv">TSV</label>
                <input type="radio" id="export-fmt-json" name="export-format" value="json">
                <label for="export-fmt-json">JSON</label>
            </fieldset>
            <fieldset id="export-columns-pairs">
                <legend>Attributes</legend>
                <div class="export-col-grid" data-section="standard"></div>
                <small class="export-json-hint">Available when JSON format selected</small>
                <div class="export-col-grid" data-section="json"></div>
            </fieldset>
            <fieldset id="export-columns-entities">
                <label>
                    <input type="checkbox" id="export-entity-include-aliases">
                    Include aliases
                </label>
            </fieldset>
            <div class="export-preview-section">
                <strong>Preview</strong>
                <div class="export-preview-wrap">
                    <table id="export-preview-table"></table>
                </div>
                <small id="export-row-count"></small>
            </div>
            <footer>
                <button type="button" class="secondary" id="export-copy-btn">Copy to clipboard</button>
                <button type="button" id="export-download-btn">Download</button>
            </footer>
        </article>
    </dialog>

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
