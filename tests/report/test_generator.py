"""Tests for report generator."""

import json
from pathlib import Path

import pytest

from interaction_finder.extraction.models import (
    ExtractionMetadata,
    ExtractionResult,
    PairAssessment,
    PairJudgment,
    SimpleEntity,
)
from interaction_finder.report import generate_report
from interaction_finder.resources import (
    Resource,
    ResourceId,
    ResourcePool,
    ResourceQuote,
)


def create_minimal_extraction_result() -> ExtractionResult:
    """Create minimal valid extraction result for testing."""
    pool = ResourcePool()
    resource = pool.add(
        url="https://example.com/doc", title="Test Doc", document_text="Test text"
    )

    quote = ResourceQuote(resource=resource, query_text="test", spans=[(0, 4)])

    from interaction_finder.extraction.models import EntityMention

    entity_mention = EntityMention(
        name="TestEntity",
        kind="type",
        aliases=["TestEntity"],
        quotes=[quote],
        reasoning="test",
    )

    assessment = PairAssessment(
        resource_id=resource.id,
        entity1=entity_mention,
        entity2=entity_mention,
        relationship="test",
        quotes=[quote],
        confidence="high",
        reasoning="test",
    )

    judgment = PairJudgment(
        entity1=SimpleEntity(name="Entity1", kind="type1", aliases=["Entity1"]),
        entity2=SimpleEntity(name="Entity2", kind="type2", aliases=["Entity2"]),
        relationship="associated_with",
        assessments=[assessment],
        accepted=True,
        confidence="high",
        reasoning="test reasoning",
    )

    return ExtractionResult(
        topic="test topic",
        target_entity_types=["type1", "type2"],
        permitted_pairs=[("type1", "type2")],
        resources=pool,
        judgments=[judgment],
        metadata=ExtractionMetadata(
            resource_count=1,
            total_entities_found=2,
            entities_after_validation=2,
            entities_merged=0,
            merge_cache_hits=0,
            merge_cache_misses=0,
            proximal_sets_found=1,
            total_pairs_found=1,
            pairs_accepted=1,
            pairs_rejected=0,
            quotes_validated=1,
            quotes_failed=0,
        ),
    )


def append_judgment(
    result: ExtractionResult,
    name_suffix: str,
    accepted: bool,
    confidence: str,
) -> None:
    """Append a new judgment to an existing ExtractionResult for testing."""
    resource = result.resources.resources[0]
    quote = ResourceQuote(resource=resource, query_text=name_suffix, spans=[(0, 4)])

    from interaction_finder.extraction.models import EntityMention

    entity1_name = f"{name_suffix}_A"
    entity2_name = f"{name_suffix}_B"

    entity1_mention = EntityMention(
        name=entity1_name,
        kind="type1",
        aliases=[entity1_name],
        quotes=[quote],
        reasoning="test",
    )
    entity2_mention = EntityMention(
        name=entity2_name,
        kind="type2",
        aliases=[entity2_name],
        quotes=[quote],
        reasoning="test",
    )

    assessment = PairAssessment(
        resource_id=resource.id,
        entity1=entity1_mention,
        entity2=entity2_mention,
        relationship="related_to",
        quotes=[quote],
        confidence=confidence,
        reasoning="test",
    )

    judgment = PairJudgment(
        entity1=SimpleEntity(name=entity1_name, kind="type1", aliases=[entity1_name]),
        entity2=SimpleEntity(name=entity2_name, kind="type2", aliases=[entity2_name]),
        relationship="related_to",
        assessments=[assessment],
        accepted=accepted,
        confidence=confidence,
        reasoning="test",
    )

    result.judgments.append(judgment)
    result.metadata.total_pairs_found += 1
    if accepted:
        result.metadata.pairs_accepted += 1
    else:
        result.metadata.pairs_rejected += 1


def test_generate_report_basic(tmp_path):
    """Test basic report generation."""
    result = create_minimal_extraction_result()
    output = tmp_path / "report.html"

    generated_path = generate_report(result, output)

    # Verify file was created
    assert generated_path.exists()
    assert generated_path == output

    # Verify it's valid HTML
    html = output.read_text()
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "</html>" in html

    # Verify Pico CSS is loaded
    assert "pico" in html.lower()

    # Verify data is embedded
    assert "window.REPORT_DATA" in html

    # Verify title
    assert "test topic" in html


def test_generate_report_with_custom_title(tmp_path):
    """Test report generation with custom title."""
    result = create_minimal_extraction_result()
    output = tmp_path / "report.html"

    generate_report(result, output, title="Custom Report Title")

    html = output.read_text()
    assert "Custom Report Title" in html


def test_generate_report_creates_parent_directories(tmp_path):
    """Test that parent directories are created if they don't exist."""
    result = create_minimal_extraction_result()
    output = tmp_path / "nested" / "dir" / "report.html"

    generate_report(result, output)

    assert output.exists()
    assert output.parent.exists()


def test_generate_report_includes_rejected(tmp_path):
    """Test report generation with rejected pairs included."""
    result = create_minimal_extraction_result()

    # Add a rejected judgment
    resource = result.resources.resources[0]
    quote = ResourceQuote(resource=resource, query_text="Test", spans=[(0, 4)])

    from interaction_finder.extraction.models import EntityMention

    entity_mention = EntityMention(
        name="E", kind="t", aliases=["E"], quotes=[quote], reasoning="test"
    )

    assessment = PairAssessment(
        resource_id=resource.id,
        entity1=entity_mention,
        entity2=entity_mention,
        relationship="test",
        quotes=[quote],
        confidence="low",
        reasoning="weak evidence",
    )

    rejected_judgment = PairJudgment(
        entity1=SimpleEntity(name="E3", kind="type3", aliases=["E3"]),
        entity2=SimpleEntity(name="E4", kind="type4", aliases=["E4"]),
        relationship="test",
        assessments=[assessment],
        accepted=False,
        confidence="low",
        reasoning="insufficient evidence",
    )

    result.judgments.append(rejected_judgment)
    result.metadata.total_pairs_found = 2
    result.metadata.pairs_rejected = 1

    output = tmp_path / "report.html"
    generate_report(result, output)

    html = output.read_text()

    # Parse embedded data
    import re

    match = re.search(r"window\.REPORT_DATA = ({.*?});", html, re.DOTALL)
    assert match is not None

    data = json.loads(match.group(1))

    # Verify both accepted and rejected are included
    assert len(data["pairs"]) == 2
    assert sum(1 for p in data["pairs"] if p["accepted"]) == 1
    assert sum(1 for p in data["pairs"] if not p["accepted"]) == 1


def test_generate_report_plain_format(tmp_path):
    """Test that plain format outputs tuple lines."""
    result = create_minimal_extraction_result()
    output = tmp_path / "report.txt"

    generate_report(result, output, format="plain")

    content = output.read_text().strip().splitlines()
    assert content == ["Entity1, associated_with, Entity2"]


def test_generate_report_plain_kind_format(tmp_path):
    """Test that plain:KIND format outputs unique entities of that kind."""
    result = create_minimal_extraction_result()
    output = tmp_path / "genes.txt"

    generate_report(result, output, format="plain:TYPE1")

    content = output.read_text().strip().splitlines()
    assert content == ["Entity1"]


def test_generate_report_plain_defaults_to_accepted(tmp_path):
    """Plain format should include only accepted pairs by default."""
    result = create_minimal_extraction_result()
    append_judgment(result, "rejected_pair", accepted=False, confidence="low")

    output = tmp_path / "plain.txt"
    generate_report(result, output, format="plain")

    content = output.read_text().strip().splitlines()
    assert content == ["Entity1, associated_with, Entity2"]


def test_generate_report_plain_accepts_any_filter(tmp_path):
    """Accepted:any should include both accepted and rejected pairs."""
    result = create_minimal_extraction_result()
    append_judgment(result, "rejected_pair", accepted=False, confidence="low")

    output = tmp_path / "plain_any.txt"
    generate_report(result, output, format="plain", filters={"accepted": "any"})

    content = output.read_text().strip().splitlines()
    assert len(content) == 2
    assert any("Entity1, associated_with, Entity2" == line for line in content)
    assert any(
        "rejected_pair_A, related_to, rejected_pair_B" == line for line in content
    )


def test_generate_report_plain_confidence_filter(tmp_path):
    """Confidence filters should be honored for plain format."""
    result = create_minimal_extraction_result()
    append_judgment(result, "medium_pair", accepted=True, confidence="medium")
    append_judgment(result, "low_pair", accepted=True, confidence="low")

    output = tmp_path / "plain_conf.txt"
    generate_report(
        result,
        output,
        format="plain",
        filters={"confidence": "medium+"},
    )

    content = output.read_text().strip().splitlines()
    assert len(content) == 2
    assert any("Entity1, associated_with, Entity2" == line for line in content)
    assert any("medium_pair_A, related_to, medium_pair_B" == line for line in content)
    assert all("low_pair" not in line for line in content)


def test_generate_report_html_confidence_filter(tmp_path):
    """Confidence filters should also apply to HTML output."""
    result = create_minimal_extraction_result()
    append_judgment(result, "low_pair", accepted=True, confidence="low")

    output = tmp_path / "report_filtered.html"
    generate_report(result, output, format="html", filters={"confidence": "high"})

    html = output.read_text()
    import re

    match = re.search(r"window\.REPORT_DATA = ({.*?});", html, re.DOTALL)
    assert match is not None
    data = json.loads(match.group(1))

    assert len(data["pairs"]) == 1
    pair = data["pairs"][0]
    assert pair["entity1"]["name"] == "Entity1"


def test_generate_report_filters_no_match(tmp_path):
    """If filters remove all judgments, raise ValueError."""
    result = create_minimal_extraction_result()

    with pytest.raises(ValueError, match="No judgments matched"):
        generate_report(
            result,
            tmp_path / "plain_nomatch.txt",
            format="plain",
            filters={"confidence": "low"},
        )


def test_generate_report_plain_stdout(capsys):
    """Plain format writes tuples to stdout when output is '-'."""
    result = create_minimal_extraction_result()

    output_path = generate_report(result, "-", format="plain")

    captured = capsys.readouterr()
    assert captured.out.strip() == "Entity1, associated_with, Entity2"
    assert str(output_path) == "-"


def test_generate_report_html_stdout(capsys):
    """HTML format streams full document to stdout when requested."""
    result = create_minimal_extraction_result()

    output_path = generate_report(result, "-", format="html")

    captured = capsys.readouterr()
    assert captured.out.startswith("<!DOCTYPE html>")
    assert str(output_path) == "-"


def test_generate_report_rejects_empty_result(tmp_path):
    """Test that empty extraction results raise ValueError."""
    pool = ResourcePool()
    result = ExtractionResult(
        resources=pool,
        judgments=[],
        metadata=ExtractionMetadata(
            topic="empty",
            resource_count=0,
            total_entities_found=0,
            entities_after_validation=0,
            entities_merged=0,
            merge_cache_hits=0,
            merge_cache_misses=0,
            proximal_sets_found=0,
            total_pairs_found=0,
            pairs_accepted=0,
            pairs_rejected=0,
            quotes_validated=0,
            quotes_failed=0,
        ),
    )

    output = tmp_path / "report.html"

    with pytest.raises(ValueError, match="no judgments"):
        generate_report(result, output)


def test_generate_report_embedded_data_is_valid_json(tmp_path):
    """Test that embedded data is valid JSON and contains expected structure."""
    result = create_minimal_extraction_result()
    output = tmp_path / "report.html"

    generate_report(result, output)

    html = output.read_text()

    # Extract embedded JSON
    import re

    match = re.search(r"window\.REPORT_DATA = ({.*?});", html, re.DOTALL)
    assert match is not None

    # Parse JSON
    data = json.loads(match.group(1))

    # Verify structure
    assert "metadata" in data
    assert "pairs" in data
    assert "documents" in data
    assert "entity_index" in data

    # Verify metadata
    assert data["metadata"]["topic"] == "test topic"
    assert data["metadata"]["total_pairs"] == 1

    # Verify pairs
    assert len(data["pairs"]) == 1
    pair = data["pairs"][0]
    assert "entity1" in pair
    assert "entity2" in pair
    assert "relationship" in pair
    assert "confidence" in pair
    assert "assessments" in pair


def test_generate_report_css_and_js_embedded(tmp_path):
    """Test that CSS and JavaScript are embedded in output."""
    result = create_minimal_extraction_result()
    output = tmp_path / "report.html"

    generate_report(result, output)

    html = output.read_text()

    # Verify CSS is embedded
    assert "<style>" in html
    assert ".pair-card" in html
    assert "grid-area:" in html

    # Verify JS is embedded
    assert "<script>" in html
    assert "function initReport" in html
    assert "renderPairList" in html
    assert "renderContent" in html


def test_generate_report_overwrites_existing_file(tmp_path):
    """Test that existing files are overwritten."""
    result = create_minimal_extraction_result()
    output = tmp_path / "report.html"

    # Write initial content
    output.write_text("old content")

    # Generate report
    generate_report(result, output)

    # Verify old content is replaced
    html = output.read_text()
    assert "old content" not in html
    assert "<!DOCTYPE html>" in html
