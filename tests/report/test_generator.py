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


def test_generate_report_stats_format(tmp_path):
    """Test stats format report generation."""
    checkpoint = create_minimal_checkpoint()
    output_path = tmp_path / "report.stats.txt"

    generate_report(checkpoint, output_path, format="stats")

    assert output_path.exists()
    content = output_path.read_text()

    # Verify topic and extraction stats
    assert "test topic" in content
    assert "Extraction Stage" in content
    assert "Documents processed: 1" in content
    assert "Entities found: 2" in content
    assert "Pairs" in content
    assert "Total pairs found: 1" in content
    assert "Accepted: 1" in content
    assert "Rejected: 0" in content


def test_generate_report_stats_stdout(capsys):
    """Test stats format to stdout (non-TTY falls back to plain text)."""
    checkpoint = create_minimal_checkpoint()

    generate_report(checkpoint, "-", format="stats")

    captured = capsys.readouterr()
    assert "test topic" in captured.out
    assert "Extraction Stage" in captured.out
    assert "Documents processed: 1" in captured.out


def test_generate_report_stats_without_extraction():
    """Test stats format works without extraction data (shows available stages)."""
    from interaction_finder.checkpoint import SearchStageData
    from interaction_finder.search.models import SearchResult

    pool = ResourcePool()
    checkpoint = PipelineCheckpoint(
        topic="test topic",
        resources=pool,
        search=SearchStageData(
            results=[
                SearchResult(
                    url="https://example.com",
                    title="Test",
                    snippet="A snippet",
                    rank=1,
                )
            ],
            queries=["query1", "query2"],
            query_results={"query1": ["https://example.com"]},
            keyphrases=["keyword1"],
            rounds_completed=2,
        ),
    )

    # Should not raise - stats format doesn't require extraction
    from io import StringIO
    import sys

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        generate_report(checkpoint, "-", format="stats")
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    assert "Search Stage" in output
    assert "Rounds completed: 2" in output
    assert "Queries executed: 2" in output
    assert "Results selected: 1" in output
    # Should NOT have extraction since it wasn't present
    assert "Extraction Stage" not in output


def test_generate_report_stats_entities_by_kind(tmp_path):
    """Test that stats includes entity breakdown by kind."""
    checkpoint = create_minimal_checkpoint()
    output_path = tmp_path / "report.stats.txt"

    generate_report(checkpoint, output_path, format="stats")

    content = output_path.read_text()
    assert "Entities by Kind" in content
    assert "gene:" in content or "disease:" in content


def test_generate_report_stats_relationship_types(tmp_path):
    """Test that stats includes relationship type counts."""
    checkpoint = create_minimal_checkpoint()
    output_path = tmp_path / "report.stats.txt"

    generate_report(checkpoint, output_path, format="stats")

    content = output_path.read_text()
    assert "Relationship Types" in content
    assert "associated_with: 1" in content


def test_generate_report_stats_with_entity_filter(tmp_path):
    """Test stats format with entity kind filter updates all counts."""
    from interaction_finder.checkpoint import ExtractionStageData
    from interaction_finder.extraction.models import (
        ExtractionMetadata,
        PairAssessment,
        PairJudgment,
        PairSpread,
        SimpleEntity,
    )
    from interaction_finder.resources import ResourcePool, ResourceQuote

    pool = ResourcePool()
    # Create two resources via pool.add()
    res1 = pool.add(
        url="https://example.com/doc1",
        title="Doc 1",
        document_text="BRCA1 is associated with breast cancer.",
    )
    res2 = pool.add(
        url="https://example.com/doc2",
        title="Doc 2",
        document_text="TP53 causes tumor suppression.",
    )

    # Create two judgments referencing different resources
    judgment1 = PairJudgment(
        entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=[]),
        entity2=SimpleEntity(name="breast cancer", kind="disease", aliases=[]),
        relationship="associated_with",
        spread=PairSpread(
            positive=[
                PairAssessment(
                    resource_id=res1.id,
                    entity1=EntityRef(canonical="BRCA1", mentions=[]),
                    entity2=EntityRef(canonical="breast cancer", mentions=[]),
                    relationship="associated_with",
                    quotes=[
                        ResourceQuote(resource=res1, query_text="BRCA1", spans=[(0, 5)])
                    ],
                    evidence=make_evidence(7),
                    topic_relevance=4,
                    reasoning="test",
                    source="direct",
                )
            ]
        ),
        accepted=True,
        evidence=make_evidence(7),
        topic_relevance=4,
        decision_confidence=0.9,
        reasoning="test",
    )
    judgment2 = PairJudgment(
        entity1=SimpleEntity(name="TP53", kind="gene", aliases=[]),
        entity2=SimpleEntity(name="tumor", kind="phenotype", aliases=[]),
        relationship="causes",
        spread=PairSpread(
            positive=[
                PairAssessment(
                    resource_id=res2.id,
                    entity1=EntityRef(canonical="TP53", mentions=[]),
                    entity2=EntityRef(canonical="tumor", mentions=[]),
                    relationship="causes",
                    quotes=[
                        ResourceQuote(resource=res2, query_text="TP53", spans=[(0, 4)])
                    ],
                    evidence=make_evidence(6),
                    topic_relevance=3,
                    reasoning="test",
                    source="direct",
                )
            ]
        ),
        accepted=True,
        evidence=make_evidence(6),
        topic_relevance=3,
        decision_confidence=0.8,
        reasoning="test",
    )

    checkpoint = PipelineCheckpoint(
        topic="test topic",
        resources=pool,
        extraction=ExtractionStageData(
            target_entity_types=["gene", "disease", "phenotype"],
            permitted_pairs={"gene": ["disease", "phenotype"]},
            judgments=[judgment1, judgment2],
            metadata=ExtractionMetadata(
                topic="test topic",
                resource_count=2,
                total_entities_found=4,
                entities_after_validation=4,
                entities_merged=0,
                merge_cache_hits=0,
                merge_cache_misses=0,
                proximal_sets_found=0,
                total_pairs_found=2,
                pairs_accepted=2,
                pairs_rejected=0,
                quotes_validated=2,
                quotes_failed=0,
            ),
        ),
    )

    output_path = tmp_path / "report.stats.txt"
    # Filter to only BRCA1-related pairs
    generate_report(checkpoint, output_path, format="stats", filters={"gene": "BRCA1"})

    content = output_path.read_text()
    # Should have filtered down to 1 document, 1 pair
    assert "Documents processed: 1" in content
    assert "Total pairs found: 1" in content
    assert "Accepted: 1" in content
    assert "Total in pool: 1" in content


def test_filter_checkpoint_prunes_resources():
    """Test that filter_checkpoint removes unreferenced resources."""
    from interaction_finder.report.generator import (
        filter_checkpoint,
        _normalize_filters,
    )
    from interaction_finder.checkpoint import ExtractionStageData
    from interaction_finder.extraction.models import (
        ExtractionMetadata,
        PairAssessment,
        PairJudgment,
        PairSpread,
        SimpleEntity,
    )
    from interaction_finder.resources import ResourcePool, ResourceQuote

    pool = ResourcePool()
    res1 = pool.add(url="https://example.com/1", title="Doc 1", document_text="text 1")
    res2 = pool.add(url="https://example.com/2", title="Doc 2", document_text="text 2")

    # Judgment 1 references res1 only
    judgment1 = PairJudgment(
        entity1=SimpleEntity(name="A", kind="gene", aliases=[]),
        entity2=SimpleEntity(name="B", kind="disease", aliases=[]),
        relationship="causes",
        spread=PairSpread(
            positive=[
                PairAssessment(
                    resource_id=res1.id,
                    entity1=EntityRef(canonical="A", mentions=[]),
                    entity2=EntityRef(canonical="B", mentions=[]),
                    relationship="causes",
                    quotes=[
                        ResourceQuote(resource=res1, query_text="A", spans=[(0, 1)])
                    ],
                    evidence=make_evidence(5),
                    topic_relevance=3,
                    reasoning="test",
                    source="direct",
                )
            ]
        ),
        accepted=True,
        evidence=make_evidence(5),
        topic_relevance=3,
        decision_confidence=0.8,
        reasoning="test",
    )
    # Judgment 2 references res2 only
    judgment2 = PairJudgment(
        entity1=SimpleEntity(name="C", kind="protein", aliases=[]),
        entity2=SimpleEntity(name="D", kind="disease", aliases=[]),
        relationship="inhibits",
        spread=PairSpread(
            positive=[
                PairAssessment(
                    resource_id=res2.id,
                    entity1=EntityRef(canonical="C", mentions=[]),
                    entity2=EntityRef(canonical="D", mentions=[]),
                    relationship="inhibits",
                    quotes=[
                        ResourceQuote(resource=res2, query_text="C", spans=[(0, 1)])
                    ],
                    evidence=make_evidence(4),
                    topic_relevance=2,
                    reasoning="test",
                    source="direct",
                )
            ]
        ),
        accepted=True,
        evidence=make_evidence(4),
        topic_relevance=2,
        decision_confidence=0.7,
        reasoning="test",
    )

    checkpoint = PipelineCheckpoint(
        topic="test",
        resources=pool,
        extraction=ExtractionStageData(
            target_entity_types=["gene", "protein", "disease"],
            permitted_pairs={"gene": ["disease"], "protein": ["disease"]},
            judgments=[judgment1, judgment2],
            metadata=ExtractionMetadata(
                topic="test",
                resource_count=2,
                total_entities_found=4,
                entities_after_validation=4,
                entities_merged=0,
                merge_cache_hits=0,
                merge_cache_misses=0,
                proximal_sets_found=0,
                total_pairs_found=2,
                pairs_accepted=2,
                pairs_rejected=0,
                quotes_validated=2,
                quotes_failed=0,
            ),
        ),
    )

    # Filter to only gene:A (judgment1)
    filters = _normalize_filters({"gene": "A"}, {"gene", "protein", "disease"})
    filtered = filter_checkpoint(checkpoint, filters)

    # Should only have 1 resource now
    assert len(filtered.resources.resource_map) == 1
    assert filtered.resources.get("https://example.com/1") is not None
    assert filtered.resources.get("https://example.com/2") is None
    # Should only have 1 judgment
    assert len(filtered.extraction.judgments) == 1
    assert filtered.extraction.judgments[0].entity1.name == "A"
    # Metadata should be recalculated
    assert filtered.extraction.metadata.resource_count == 1
    assert filtered.extraction.metadata.total_pairs_found == 1
