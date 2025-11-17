"""HTML template for report generation.

Provides a Jinja2 template for rendering self-contained HTML reports
with embedded data, CSS, and JavaScript.
"""

import json
from typing import Any

from interaction_finder.report.assets import get_css, get_js


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
        # Build entity aliases tooltip
        entity1_aliases = ", ".join(pair["entity1"]["aliases"])
        entity2_aliases = ", ".join(pair["entity2"]["aliases"])

        # Build card classes
        card_classes = ["pair-card"]
        if not pair["accepted"]:
            card_classes.append("rejected")

        # Build kinds/relationship row
        if show_kinds:
            kinds_html = f"""
            <div class="pair-kinds">
                <span class="entity-kind left">{_escape_html(pair["entity1"]["kind"])}</span>
                <span class="relationship-label">{_escape_html(pair["relationship"])}</span>
                <span class="entity-kind right">{_escape_html(pair["entity2"]["kind"])}</span>
            </div>"""
        else:
            kinds_html = f"""
            <div class="pair-relationship-only">
                <span class="relationship-label">{_escape_html(pair["relationship"])}</span>
            </div>"""

        # Build variants section if present
        variants_html = ""
        if "variants" in pair and pair["variants"]:
            variant_items = []
            for variant in pair["variants"]:
                variant_items.append(f"""
                        <div class="variant-item">
                            <span class="variant-relationship">{_escape_html(variant["relationship"])}</span>
                            <span class="confidence-badge variant-badge confidence-{_escape_html(variant["confidence"])}">{_escape_html(variant["confidence"])}</span>
                        </div>""")
            variants_html = f"""
                <div class="pair-variants">
                    {"".join(variant_items)}
                </div>"""

        # Build complete card
        card_html = f"""
        <div class="{" ".join(card_classes)}" data-pair-idx="{idx}">
            <div class="pair-entities">
                <span class="entity-name left"
                      title="{_escape_html(entity1_aliases)}">
                    {_escape_html(pair["entity1"]["name"])}
                </span>
                <span class="entity-name right"
                      title="{_escape_html(entity2_aliases)}">
                    {_escape_html(pair["entity2"]["name"])}
                </span>
            </div>{kinds_html}
            <div class="pair-meta">
                <span class="pair-counts">{pair["doc_count"]} docs, {pair["quote_count"]} quotes</span>
                <span class="confidence-badge confidence-{_escape_html(pair["confidence"])}">{_escape_html(pair["confidence"])}</span>
            </div>{variants_html}
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
                        <span class="stat-label">Topic:</span>
                        <span>{{ metadata.topic }}</span>
                    </span>
                    <span class="stat-item">
                        <span class="stat-label">Pairs:</span>
                        <span>{{ metadata.total_pairs }}</span>
                    </span>
                    {% for kind, count in metadata.entity_stats.items() %}
                    <span class="stat-item">
                        <span class="stat-label">{{ kind }}:</span>
                        <span>{{ count }}</span>
                    </span>
                    {% endfor %}
                    <span class="stat-item">
                        <span class="stat-label">Documents:</span>
                        <span>{{ metadata.resource_count }}</span>
                    </span>
                </div>
            </div>
            <div class="header-controls-wrapper">
                <div class="header-controls">
                    <input type="search"
                           id="search-input"
                           placeholder="Search entities or relationships..."
                           aria-label="Search"
                           style="margin: 0;">
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

    <!-- Embedded Data -->
    <script>
        window.REPORT_DATA = {{ data_json }};
    </script>

    <!-- JavaScript -->
    <script>
{{ js }}
    </script>
</body>
</html>
"""


def render_template(
    data: dict[str, Any],
    document_html: dict[str, str],
    reasoning_templates: dict[str, dict[str, str]],
    title: str = "Extraction Report",
) -> str:
    """Render HTML report from prepared data.

    Args:
        data: Prepared report data from prepare_report_data() (JSON-serializable)
        document_html: Mapping of doc_id -> pre-rendered HTML string
        reasoning_templates: Nested dict pair_idx -> template_type -> HTML
        title: Report title

    Returns:
        Complete HTML document as string
    """
    # Simple template rendering without Jinja2 dependency
    # Use string replacement for placeholders
    html = HTML_TEMPLATE

    # Build pair cards HTML
    pair_cards_html = _render_pair_cards(data["pairs"])

    # Build document templates HTML
    doc_templates_parts = []
    for doc_id, doc_html in document_html.items():
        # Wrap each document's HTML in a <template> tag with unique ID
        # Note: doc_html already contains <div class="document-links"> and <div class="document-text">
        escaped_doc_id = _escape_html(doc_id)
        doc_templates_parts.append(
            f'        <template id="doc-template-{escaped_doc_id}">\n'
            f"            {doc_html}\n"
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

    # Replace placeholders
    replacements = {
        "{{ title }}": _escape_html(title),
        "{{ metadata.topic }}": _escape_html(data["metadata"]["topic"]),
        "{{ metadata.total_pairs }}": str(data["metadata"]["total_pairs"]),
        "{{ metadata.resource_count }}": str(data["metadata"]["resource_count"]),
        "{{ css }}": get_css(),
        "{{ js }}": get_js(),
        "{{ data_json }}": json.dumps(data, ensure_ascii=False, indent=2),
        "{{ document_templates }}": document_templates_html,
        "{{ reasoning_templates }}": reasoning_templates_html,
        "{{ pair_cards }}": pair_cards_html,
    }

    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    # Handle entity stats loop
    entity_stats_html = ""
    for kind, count in data["metadata"]["entity_stats"].items():
        entity_stats_html += f"""
                <span class="stat-item">
                    <span class="stat-label">{_escape_html(kind)}:</span>
                    <span>{count}</span>
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
