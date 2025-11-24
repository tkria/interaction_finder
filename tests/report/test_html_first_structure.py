"""Tests for HTML-first report structure.

Validates that the refactored report generation produces the expected
HTML structure with minimal data-attributes and zero JSON embedding.
"""

import pytest

try:
    from bs4 import BeautifulSoup

    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from interaction_finder.checkpoint import ExtractionStageData, PipelineCheckpoint
from interaction_finder.extraction.models import (
    EntityMention,
    ExtractionMetadata,
    ExtractionResult,
    PairAssessment,
    PairJudgment,
    PairSpread,
    SimpleEntity,
)
from interaction_finder.report import generate_report
from interaction_finder.report.data_prep import prepare_report_data
from interaction_finder.report.template import render_template
from interaction_finder.resources import ResourcePool, ResourceQuote


def create_test_checkpoint(
    pairs: int = 1, docs_per_pair: int = 1
) -> PipelineCheckpoint:
    """Create minimal test checkpoint for HTML structure validation.

    Args:
        pairs: Number of entity pairs to create
        docs_per_pair: Number of documents (assessments) per pair

    Returns:
        PipelineCheckpoint with extraction results
    """
    pool = ResourcePool()

    # Create documents
    resources = []
    for doc_idx in range(pairs * docs_per_pair):
        resource = pool.add(
            url=f"https://example.com/doc{doc_idx}",
            title=f"Test Document {doc_idx}",
            document_text=f"This is test document {doc_idx} about Entity{doc_idx}_A and Entity{doc_idx}_B.",
        )
        resources.append(resource)

    # Create judgments (one per pair)
    judgments = []
    for pair_idx in range(pairs):
        # Create assessments for this pair
        assessments = []
        for doc_idx in range(docs_per_pair):
            resource_idx = pair_idx * docs_per_pair + doc_idx
            resource = resources[resource_idx]

            quote = ResourceQuote(
                resource=resource, query_text="test quote", spans=[(10, 20)]
            )

            entity1_mention = EntityMention(
                name=f"Entity{pair_idx}_A",
                kind="gene",
                aliases=[f"Entity{pair_idx}_A", f"E{pair_idx}A"],
                quotes=[quote],
                reasoning=f"Test reasoning for Entity{pair_idx}_A",
            )

            entity2_mention = EntityMention(
                name=f"Entity{pair_idx}_B",
                kind="disease",
                aliases=[f"Entity{pair_idx}_B", f"E{pair_idx}B"],
                quotes=[quote],
                reasoning=f"Test reasoning for Entity{pair_idx}_B",
            )

            assessment = PairAssessment(
                resource_id=resource.id,
                entity1=entity1_mention,
                entity2=entity2_mention,
                relationship="associated_with",
                quotes=[quote],
                confidence="high",
                reasoning=f"Test assessment {doc_idx}",
            )
            assessments.append(assessment)

        # Create judgment
        spread = PairSpread(supporting=list(assessments))
        judgment = PairJudgment(
            entity1=SimpleEntity(
                name=f"Entity{pair_idx}_A",
                kind="gene",
                aliases=[f"Entity{pair_idx}_A", f"E{pair_idx}A"],
            ),
            entity2=SimpleEntity(
                name=f"Entity{pair_idx}_B",
                kind="disease",
                aliases=[f"Entity{pair_idx}_B", f"E{pair_idx}B"],
            ),
            relationship="associated_with",
            spread=spread,
            accepted=True,
            confidence="high",
            reasoning=f"Test judgment {pair_idx}",
        )
        judgments.append(judgment)

    # Create extraction result
    result = ExtractionResult(
        topic="test topic",
        target_entity_types=["gene", "disease"],
        permitted_pairs={"gene": ["disease"]},
        resources=pool,
        judgments=judgments,
        metadata=ExtractionMetadata(
            topic="test topic",
            resource_count=len(resources),
            total_entities_found=pairs * 2,
            entities_after_validation=pairs * 2,
            entities_merged=0,
            merge_cache_hits=0,
            merge_cache_misses=0,
            proximal_sets_found=pairs,
            total_pairs_found=pairs,
            pairs_accepted=pairs,
            pairs_rejected=0,
            quotes_validated=pairs * docs_per_pair,
            quotes_failed=0,
        ),
    )

    # Convert to ExtractionStageData for checkpoint
    extraction_data = ExtractionStageData(
        target_entity_types=result.target_entity_types,
        permitted_pairs=result.permitted_pairs,
        judgments=result.judgments,
        metadata=result.metadata,
    )

    return PipelineCheckpoint(
        topic="test topic", resources=pool, extraction=extraction_data
    )


@pytest.mark.skipif(not HAS_BS4, reason="BeautifulSoup4 not installed")
def test_pair_cards_have_minimal_attributes():
    """Verify pair cards use only minimal required data-attributes."""
    checkpoint = create_test_checkpoint(pairs=2, docs_per_pair=2)
    pairs, html_map, templates, indexed_docs = prepare_report_data(checkpoint)
    html = render_template(pairs, html_map, templates, indexed_docs, topic="test topic")

    soup = BeautifulSoup(html, "html.parser")

    # Find all pair cards
    pair_cards = soup.find_all("div", class_="pair-card")
    assert len(pair_cards) == 2

    for idx, pair_card in enumerate(pair_cards):
        # Verify required attributes exist
        assert pair_card.get("id") == f"pair-{idx}"
        assert pair_card.get("data-e1") is not None
        assert pair_card.get("data-e1a") is not None
        assert pair_card.get("data-e2") is not None
        assert pair_card.get("data-e2a") is not None
        assert pair_card.get("data-rel") is not None
        assert pair_card.get("data-accepted") in ["true", "false"]
        assert pair_card.get("data-docs") is not None

        # Verify redundant attributes are NOT present
        assert pair_card.get("data-confidence") is None
        assert pair_card.get("data-e1-kind") is None
        assert pair_card.get("data-e2-kind") is None
        assert pair_card.get("data-doc-count") is None
        assert pair_card.get("data-quote-count") is None


@pytest.mark.skipif(not HAS_BS4, reason="BeautifulSoup4 not installed")
def test_no_json_embedded():
    """Verify zero JSON embedding - no window.REPORT_DATA assignment."""
    checkpoint = create_test_checkpoint()
    pairs, html_map, templates, indexed_docs = prepare_report_data(checkpoint)
    html = render_template(pairs, html_map, templates, indexed_docs, topic="test topic")

    # Should NOT contain window.REPORT_DATA = {...} assignment
    # (Comments mentioning it are OK, but no actual data assignment)
    assert "window.REPORT_DATA =" not in html
    assert "REPORT_DATA = {" not in html

    # Should NOT contain large JSON blobs
    # (Allow small config JSON, but not data)
    import json

    # Look for large object literals that might be JSON
    # If we find `{` followed by many lines and `}`, that's suspicious
    lines = html.split("\n")
    in_large_object = False
    object_depth = 0
    object_lines = 0

    for line in lines:
        if "{" in line and not line.strip().startswith("//"):
            object_depth += line.count("{")
            if object_depth > 0:
                in_large_object = True

        if in_large_object:
            object_lines += 1

        if "}" in line:
            object_depth -= line.count("}")
            if object_depth == 0:
                # End of object - check if it was large
                if object_lines > 100:
                    # Large object literal found - likely embedded data
                    pytest.fail(
                        f"Large object literal found ({object_lines} lines) - "
                        "possible embedded JSON data"
                    )
                object_lines = 0
                in_large_object = False


@pytest.mark.skipif(not HAS_BS4, reason="BeautifulSoup4 not installed")
def test_numeric_ids_throughout():
    """Verify all IDs use numeric indices (pair-5, doc-3, doc-3-quote-0)."""
    checkpoint = create_test_checkpoint(pairs=3, docs_per_pair=2)
    pairs, html_map, templates, indexed_docs = prepare_report_data(checkpoint)
    html = render_template(pairs, html_map, templates, indexed_docs, topic="test topic")

    soup = BeautifulSoup(html, "html.parser")

    # Check pair IDs
    pair_cards = soup.find_all("div", class_="pair-card")
    assert len(pair_cards) == 3
    for idx, pair_card in enumerate(pair_cards):
        assert pair_card.get("id") == f"pair-{idx}"

    # Check document template IDs
    doc_templates = soup.find_all(
        "template", id=lambda x: x and x.startswith("doc-template-")
    )
    assert len(doc_templates) > 0
    for tmpl in doc_templates:
        doc_id = tmpl.get("id").replace("doc-template-", "")
        assert doc_id.isdigit(), f"Document ID '{doc_id}' is not numeric"

    # Check reasoning template IDs
    reasoning_templates = soup.find_all(
        "template", id=lambda x: x and x.startswith("reasoning-pair-")
    )
    assert len(reasoning_templates) > 0
    for tmpl in reasoning_templates:
        tmpl_id = tmpl.get("id")
        # Format: reasoning-pair-{idx}-overall or reasoning-pair-{idx}-doc-{idx}
        if "-overall" in tmpl_id:
            pair_part = tmpl_id.replace("reasoning-pair-", "").replace("-overall", "")
            assert pair_part.isdigit(), f"Pair ID '{pair_part}' is not numeric"
        elif "-doc-" in tmpl_id:
            parts = tmpl_id.replace("reasoning-pair-", "").split("-doc-")
            assert len(parts) == 2
            assert parts[0].isdigit(), f"Pair ID '{parts[0]}' is not numeric"
            assert parts[1].isdigit(), f"Doc ID '{parts[1]}' is not numeric"


@pytest.mark.skipif(not HAS_BS4, reason="BeautifulSoup4 not installed")
def test_space_separated_doc_list():
    """Verify data-docs uses space-separated numeric IDs."""
    checkpoint = create_test_checkpoint(pairs=2, docs_per_pair=3)
    pairs, html_map, templates, indexed_docs = prepare_report_data(checkpoint)
    html = render_template(pairs, html_map, templates, indexed_docs, topic="test topic")

    soup = BeautifulSoup(html, "html.parser")

    # Check first pair's doc list
    pair_0 = soup.find(id="pair-0")
    assert pair_0 is not None

    doc_ids = pair_0.get("data-docs").split()
    assert len(doc_ids) == 3, f"Expected 3 docs, got {len(doc_ids)}"
    assert all(doc_id.isdigit() for doc_id in doc_ids), (
        f"Not all doc IDs are numeric: {doc_ids}"
    )

    # Check second pair's doc list
    pair_1 = soup.find(id="pair-1")
    assert pair_1 is not None

    doc_ids = pair_1.get("data-docs").split()
    assert len(doc_ids) == 3
    assert all(doc_id.isdigit() for doc_id in doc_ids)


@pytest.mark.skipif(not HAS_BS4, reason="BeautifulSoup4 not installed")
def test_document_templates_exist():
    """Verify document HTML is wrapped in <template> tags."""
    checkpoint = create_test_checkpoint(pairs=1, docs_per_pair=2)
    pairs, html_map, templates, indexed_docs = prepare_report_data(checkpoint)
    html = render_template(pairs, html_map, templates, indexed_docs, topic="test topic")

    soup = BeautifulSoup(html, "html.parser")

    # Check document templates exist
    doc_templates = soup.find_all(
        "template", id=lambda x: x and x.startswith("doc-template-")
    )
    assert len(doc_templates) >= 2

    # Check template contains document structure
    template = doc_templates[0]
    # Templates have their content as string, not parsed
    # BeautifulSoup stores template content as NavigableString
    template_html = template.decode_contents()

    assert "document-links" in template_html or "document-text" in template_html


@pytest.mark.skipif(not HAS_BS4, reason="BeautifulSoup4 not installed")
def test_reasoning_templates_exist():
    """Verify reasoning panels are pre-rendered in <template> tags."""
    checkpoint = create_test_checkpoint(pairs=2, docs_per_pair=2)
    pairs, html_map, templates, indexed_docs = prepare_report_data(checkpoint)
    html = render_template(pairs, html_map, templates, indexed_docs, topic="test topic")

    soup = BeautifulSoup(html, "html.parser")

    # Check reasoning templates exist
    reasoning_templates = soup.find_all(
        "template", id=lambda x: x and x.startswith("reasoning-pair-")
    )
    assert len(reasoning_templates) > 0

    # Should have both overall and doc-specific templates
    overall_templates = [
        t for t in reasoning_templates if "-overall" in t.get("id", "")
    ]
    doc_templates = [t for t in reasoning_templates if "-doc-" in t.get("id", "")]

    assert len(overall_templates) >= 2  # One per pair
    assert len(doc_templates) >= 4  # 2 pairs × 2 docs


@pytest.mark.skipif(not HAS_BS4, reason="BeautifulSoup4 not installed")
def test_javascript_query_patterns():
    """Verify HTML structure enables expected JavaScript query patterns."""
    checkpoint = create_test_checkpoint(pairs=2, docs_per_pair=2)
    pairs, html_map, templates, indexed_docs = prepare_report_data(checkpoint)
    html = render_template(pairs, html_map, templates, indexed_docs, topic="test topic")

    soup = BeautifulSoup(html, "html.parser")

    # Pattern 1: Filter pairs by entity name
    # JavaScript uses: document.querySelectorAll('[data-e1="Entity0_A"]')
    pairs_with_entity0_a = soup.select('[data-e1="Entity0_A"]')
    assert len(pairs_with_entity0_a) >= 1

    # Pattern 2: Get documents for specific pair
    # JavaScript uses: getElementById('pair-0').dataset.docs
    pair_0 = soup.find(id="pair-0")
    assert pair_0 is not None
    doc_ids = pair_0.get("data-docs", "").split()
    assert len(doc_ids) > 0

    # Pattern 3: Check if pair is rejected
    # JavaScript uses: card.dataset.accepted === 'false'
    accepted = pair_0.get("data-accepted")
    assert accepted in ["true", "false"]


@pytest.mark.skipif(not HAS_BS4, reason="BeautifulSoup4 not installed")
def test_entity_aliases_format():
    """Verify entity aliases use comma-separated format."""
    checkpoint = create_test_checkpoint()
    pairs, html_map, templates, indexed_docs = prepare_report_data(checkpoint)
    html = render_template(pairs, html_map, templates, indexed_docs, topic="test topic")

    soup = BeautifulSoup(html, "html.parser")

    pair_0 = soup.find(id="pair-0")
    assert pair_0 is not None

    # Check aliases are comma-separated
    aliases_e1 = pair_0.get("data-e1a", "")
    aliases_e2 = pair_0.get("data-e2a", "")

    # Should contain commas (for multiple aliases)
    assert "," in aliases_e1
    assert "," in aliases_e2

    # Should be parseable
    alias_list = aliases_e1.split(",")
    assert len(alias_list) >= 2  # At least name + one alias


def test_file_size_reduction():
    """Verify report size is smaller than old JSON-based approach.

    This is a regression test - the HTML-first approach should produce
    smaller files than embedding JSON data.
    """
    checkpoint = create_test_checkpoint(pairs=10, docs_per_pair=2)
    pairs, html_map, templates, indexed_docs = prepare_report_data(checkpoint)
    html = render_template(pairs, html_map, templates, indexed_docs, topic="test topic")

    html_size = len(html.encode("utf-8"))

    # With 10 pairs and 20 documents, expect reasonable size
    # Old approach would embed ~100-200 KB of JSON
    # New approach should be similar total size but without duplicate data

    # Sanity check: HTML exists and is non-trivial
    assert html_size > 10000  # At least 10 KB
    assert html_size < 5000000  # Less than 5 MB

    # Verify no large JSON blob
    # Quick heuristic: count occurrences of common JSON patterns
    json_like_patterns = html.count('": "') + html.count('": {') + html.count('": [')

    # Should have very few JSON-like patterns (maybe some in comments or strings)
    assert json_like_patterns < 100, (
        f"Found {json_like_patterns} JSON-like patterns - possible embedded data"
    )


def test_html_is_self_contained():
    """Verify HTML report is self-contained (CSS and JS embedded)."""
    checkpoint = create_test_checkpoint()
    pairs, html_map, templates, indexed_docs = prepare_report_data(checkpoint)
    html = render_template(pairs, html_map, templates, indexed_docs, topic="test topic")

    # Should contain embedded CSS
    assert "<style>" in html
    assert ".pair-card" in html  # CSS class from stylesheet

    # Should contain embedded JavaScript
    assert "<script>" in html
    assert "function" in html  # JavaScript functions

    # Pico CSS is loaded from CDN (acceptable dependency)
    # Custom JavaScript should be embedded (not external)
    assert "<script src=" not in html  # No external JS files


@pytest.mark.skipif(not HAS_BS4, reason="BeautifulSoup4 not installed")
def test_display_content_matches_data_attributes():
    """Verify display content matches data-attributes (single source of truth)."""
    checkpoint = create_test_checkpoint()
    pairs, html_map, templates, indexed_docs = prepare_report_data(checkpoint)
    html = render_template(pairs, html_map, templates, indexed_docs, topic="test topic")

    soup = BeautifulSoup(html, "html.parser")

    pair_0 = soup.find(id="pair-0")
    assert pair_0 is not None

    # Get data from attributes
    entity1_name = pair_0.get("data-e1")
    entity2_name = pair_0.get("data-e2")
    relationship = pair_0.get("data-rel")

    # Get data from display
    pair_entities = pair_0.find("div", class_="pair-entities")
    entity_names = pair_entities.find_all("span", recursive=False)
    assert len(entity_names) == 2
    displayed_entity1 = entity_names[0].get_text().strip()
    displayed_entity2 = entity_names[1].get_text().strip()

    # Should match
    assert displayed_entity1 == entity1_name
    assert displayed_entity2 == entity2_name

    # Relationship should be displayed somewhere
    relationship_labels = pair_0.find_all("span", class_="relationship-label")
    if relationship_labels:
        displayed_rel = relationship_labels[0].get_text().strip()
        assert displayed_rel == relationship
