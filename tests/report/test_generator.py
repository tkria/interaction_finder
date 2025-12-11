"""Tests for report generation (high-level API)."""

from pathlib import Path

import pytest

from interaction_finder.checkpoint import ExtractionStageData, PipelineCheckpoint
from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    ExtractionMetadata,
    PairAssessment,
    PairJudgment,
    PairSpread,
    SimpleEntity,
)
from interaction_finder.report import generate_report
from interaction_finder.resources import ResourcePool, ResourceQuote
from tests.extraction.conftest import make_evidence


def create_minimal_checkpoint() -> PipelineCheckpoint:
    """Create minimal checkpoint for testing."""
    pool = ResourcePool()
    resource = pool.add(
        url="https://example.com/doc",
        title="Test Doc",
        document_text="Test text about BRCA1 and cancer.",
    )

    quote = ResourceQuote(resource=resource, query_text="test", spans=[(0, 4)])

    entity_mention = EntityMention(
        name="BRCA1",
        kind="gene",
        aliases=["BRCA1"],
        quotes=[quote],
        reasoning="test",
    )

    entity_ref = EntityRef(canonical="BRCA1", mentions=[entity_mention])

    assessment = PairAssessment(
        topic_relevance=3,
        resource_id=resource.id,
        entity1=entity_ref,
        entity2=entity_ref,
        relationship="test",
        quotes=[quote],
        evidence=make_evidence(8),
        reasoning="test",
    )

    spread = PairSpread(positive=[assessment])
    judgment = PairJudgment(
        topic_relevance=3,
        entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
        entity2=SimpleEntity(name="Cancer", kind="disease", aliases=["Cancer"]),
        relationship="associated_with",
        spread=spread,
        accepted=True,
        evidence=make_evidence(8),
        decision_confidence=0.9,
        reasoning="test reasoning",
    )

    extraction_data = ExtractionStageData(
        target_entity_types=["gene", "disease"],
        permitted_pairs={"gene": ["disease"]},
        judgments=[judgment],
        metadata=ExtractionMetadata(
            topic="test topic",
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

    return PipelineCheckpoint(
        topic="test topic", resources=pool, extraction=extraction_data
    )


def test_generate_report_html_basic(tmp_path):
    """Test basic HTML report generation."""
    checkpoint = create_minimal_checkpoint()
    output_path = tmp_path / "report.html"

    generate_report(checkpoint, output_path, format="html")

    assert output_path.exists()
    content = output_path.read_text()

    # Verify basic structure
    assert "<!DOCTYPE html>" in content
    assert "<html" in content
    assert "test topic" in content
    assert "BRCA1" in content
    assert "Cancer" in content


def test_generate_report_with_custom_title(tmp_path):
    """Test report generation with custom title."""
    checkpoint = create_minimal_checkpoint()
    output_path = tmp_path / "report.html"

    generate_report(checkpoint, output_path, format="html", title="Custom Title")

    content = output_path.read_text()
    assert "Custom Title" in content


def test_generate_report_creates_parent_directories(tmp_path):
    """Test that parent directories are created if they don't exist."""
    checkpoint = create_minimal_checkpoint()
    output_path = tmp_path / "subdir" / "nested" / "report.html"

    generate_report(checkpoint, output_path, format="html")

    assert output_path.exists()


def test_generate_report_plain_format(tmp_path):
    """Test plain text report generation."""
    checkpoint = create_minimal_checkpoint()
    output_path = tmp_path / "report.txt"

    generate_report(checkpoint, output_path, format="plain")

    assert output_path.exists()
    content = output_path.read_text()

    # Verify basic content
    assert "BRCA1" in content
    assert "Cancer" in content
    assert "associated_with" in content


def test_generate_report_html_has_embedded_css(tmp_path):
    """Test that HTML report has embedded CSS."""
    checkpoint = create_minimal_checkpoint()
    output_path = tmp_path / "report.html"

    generate_report(checkpoint, output_path, format="html")

    content = output_path.read_text()
    assert "<style>" in content
    assert ".pair-card" in content  # CSS class


def test_generate_report_html_has_embedded_js(tmp_path):
    """Test that HTML report has embedded JavaScript."""
    checkpoint = create_minimal_checkpoint()
    output_path = tmp_path / "report.html"

    generate_report(checkpoint, output_path, format="html")

    content = output_path.read_text()
    assert "<script>" in content
    assert "function" in content  # JavaScript functions


def test_generate_report_overwrites_existing_file(tmp_path):
    """Test that generating a report overwrites existing file."""
    checkpoint = create_minimal_checkpoint()
    output_path = tmp_path / "report.html"

    # Create initial file
    output_path.write_text("old content")

    # Generate report (should overwrite)
    generate_report(checkpoint, output_path, format="html")

    content = output_path.read_text()
    assert "old content" not in content
    assert "<!DOCTYPE html>" in content


def test_generate_report_stdout(capsys):
    """Test report generation to stdout."""
    checkpoint = create_minimal_checkpoint()

    generate_report(checkpoint, "-", format="plain")

    captured = capsys.readouterr()
    assert "BRCA1" in captured.out
    assert "Cancer" in captured.out


def test_generate_report_rejects_empty_checkpoint():
    """Test that generating report fails for checkpoint without extraction."""
    checkpoint = PipelineCheckpoint(
        topic="test",
        resources=ResourcePool(),
    )

    with pytest.raises((ValueError, AttributeError)):
        generate_report(checkpoint, "-", format="plain")
