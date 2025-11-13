"""Tests for report data preparation."""

import pytest

from interaction_finder.extraction.models import (
    ExtractionMetadata,
    ExtractionResult,
    PairAssessment,
    PairJudgment,
    SimpleEntity,
)
from interaction_finder.report.data_prep import prepare_report_data
from interaction_finder.resources import (
    Resource,
    ResourceId,
    ResourcePool,
    ResourceQuote,
)


def create_test_entity_mention(
    name: str, kind: str, aliases: list[str], quotes: list[ResourceQuote]
):
    """Helper to create EntityMention for testing."""
    from interaction_finder.extraction.models import EntityMention

    return EntityMention(
        name=name,
        kind=kind,
        aliases=aliases,
        quotes=quotes,
        reasoning=f"Test reasoning for {name}",
    )


def test_prepare_report_data_basic():
    """Test basic data preparation with simple extraction result."""
    # Create test resource
    pool = ResourcePool()
    resource = pool.add(
        url="https://example.com/doc1",
        title="Test Document",
        document_text="This is a test document about BRCA1 and breast cancer.",
    )

    # Create test quote
    quote = ResourceQuote(
        resource=resource,
        query_text="BRCA1 is associated with breast cancer",
        spans=[(10, 50)],
    )

    # Create entity mentions
    entity1_mention = create_test_entity_mention(
        name="BRCA1",
        kind="gene",
        aliases=["BRCA1", "BRCA-1"],
        quotes=[quote],
    )
    entity2_mention = create_test_entity_mention(
        name="breast cancer",
        kind="disease",
        aliases=["breast cancer", "breast carcinoma"],
        quotes=[quote],
    )

    # Create assessment
    assessment = PairAssessment(
        resource_id=resource.id,
        entity1=entity1_mention,
        entity2=entity2_mention,
        relationship="associated_with",
        quotes=[quote],
        confidence="high",
        reasoning="Strong evidence of association",
    )

    # Create judgment
    judgment = PairJudgment(
        entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1", "BRCA-1"]),
        entity2=SimpleEntity(
            name="breast cancer",
            kind="disease",
            aliases=["breast cancer", "breast carcinoma"],
        ),
        relationship="associated_with",
        assessments=[assessment],
        accepted=True,
        confidence="high",
        reasoning="Consistent evidence across documents",
    )

    # Create extraction result
    result = ExtractionResult(
        resources=pool,
        judgments=[judgment],
        metadata=ExtractionMetadata(
            topic="cancer genetics",
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

    # Prepare report data
    data = prepare_report_data(result, include_rejected=False)

    # Verify structure
    assert "metadata" in data
    assert "pairs" in data
    assert "documents" in data
    assert "entity_index" in data

    # Verify metadata
    assert data["metadata"]["topic"] == "cancer genetics"
    assert data["metadata"]["total_pairs"] == 1
    assert data["metadata"]["accepted_pairs"] == 1
    assert data["metadata"]["rejected_pairs"] == 0
    assert "gene" in data["metadata"]["entity_stats"]
    assert "disease" in data["metadata"]["entity_stats"]

    # Verify pairs
    assert len(data["pairs"]) == 1
    pair = data["pairs"][0]
    assert pair["entity1"]["name"] == "BRCA1"
    assert pair["entity2"]["name"] == "breast cancer"
    assert pair["relationship"] == "associated_with"
    assert pair["confidence"] == "high"
    assert pair["accepted"] is True
    assert pair["doc_count"] == 1
    assert pair["quote_count"] == 1

    # Verify assessments
    assert len(pair["assessments"]) == 1
    assess = pair["assessments"][0]
    assert assess["title"] == "Test Document"
    assert assess["confidence"] == "high"
    assert len(assess["quotes"]) == 1

    # Verify entity index
    assert "BRCA1" in data["entity_index"]
    assert "breast cancer" in data["entity_index"]
    assert 0 in data["entity_index"]["BRCA1"]
    assert 0 in data["entity_index"]["breast cancer"]


def test_prepare_report_data_filters_rejected():
    """Test that rejected pairs are filtered by default."""
    # Create minimal test data
    pool = ResourcePool()
    resource = pool.add(
        url="https://example.com/doc", title="Doc", document_text="Text"
    )

    quote = ResourceQuote(resource=resource, query_text="test", spans=[(0, 4)])
    entity_mention = create_test_entity_mention("E1", "type1", ["E1"], [quote])

    assessment = PairAssessment(
        resource_id=resource.id,
        entity1=entity_mention,
        entity2=entity_mention,
        relationship="test",
        quotes=[quote],
        confidence="low",
        reasoning="test",
    )

    # Create accepted and rejected judgments
    accepted = PairJudgment(
        entity1=SimpleEntity(name="E1", kind="type1", aliases=["E1"]),
        entity2=SimpleEntity(name="E2", kind="type2", aliases=["E2"]),
        relationship="test",
        assessments=[assessment],
        accepted=True,
        confidence="high",
        reasoning="accepted",
    )

    rejected = PairJudgment(
        entity1=SimpleEntity(name="E3", kind="type1", aliases=["E3"]),
        entity2=SimpleEntity(name="E4", kind="type2", aliases=["E4"]),
        relationship="test",
        assessments=[assessment],
        accepted=False,
        confidence="low",
        reasoning="rejected",
    )

    result = ExtractionResult(
        resources=pool,
        judgments=[accepted, rejected],
        metadata=ExtractionMetadata(
            topic="test",
            resource_count=1,
            total_entities_found=4,
            entities_after_validation=4,
            entities_merged=0,
            merge_cache_hits=0,
            merge_cache_misses=0,
            proximal_sets_found=0,
            total_pairs_found=2,
            pairs_accepted=1,
            pairs_rejected=1,
            quotes_validated=1,
            quotes_failed=0,
        ),
    )

    # Test without rejected
    data = prepare_report_data(result, include_rejected=False)
    assert len(data["pairs"]) == 1
    assert data["pairs"][0]["accepted"] is True

    # Test with rejected
    data = prepare_report_data(result, include_rejected=True)
    assert len(data["pairs"]) == 2
    assert sum(1 for p in data["pairs"] if p["accepted"]) == 1
    assert sum(1 for p in data["pairs"] if not p["accepted"]) == 1


def test_prepare_report_data_multiple_assessments():
    """Test data preparation with multiple documents per pair."""
    # Create two resources
    pool = ResourcePool()
    resource1 = pool.add(
        url="https://example.com/doc1", title="Doc 1", document_text="Text 1"
    )
    resource2 = pool.add(
        url="https://example.com/doc2", title="Doc 2", document_text="Text 2"
    )

    # Create quotes for each resource
    quote1 = ResourceQuote(resource=resource1, query_text="test1", spans=[(0, 5)])
    quote2 = ResourceQuote(resource=resource2, query_text="test2", spans=[(0, 5)])

    entity1 = create_test_entity_mention("E1", "type1", ["E1"], [quote1])
    entity2 = create_test_entity_mention("E2", "type2", ["E2"], [quote1])

    # Create assessments for each resource
    assessment1 = PairAssessment(
        resource_id=resource1.id,
        entity1=entity1,
        entity2=entity2,
        relationship="associated_with",
        quotes=[quote1],
        confidence="high",
        reasoning="Evidence in doc 1",
    )

    assessment2 = PairAssessment(
        resource_id=resource2.id,
        entity1=entity1,
        entity2=entity2,
        relationship="associated_with",
        quotes=[quote2],
        confidence="medium",
        reasoning="Evidence in doc 2",
    )

    judgment = PairJudgment(
        entity1=SimpleEntity(name="E1", kind="type1", aliases=["E1"]),
        entity2=SimpleEntity(name="E2", kind="type2", aliases=["E2"]),
        relationship="associated_with",
        assessments=[assessment1, assessment2],
        accepted=True,
        confidence="high",
        reasoning="Consistent across documents",
    )

    result = ExtractionResult(
        resources=pool,
        judgments=[judgment],
        metadata=ExtractionMetadata(
            topic="test",
            resource_count=2,
            total_entities_found=2,
            entities_after_validation=2,
            entities_merged=0,
            merge_cache_hits=0,
            merge_cache_misses=0,
            proximal_sets_found=1,
            total_pairs_found=1,
            pairs_accepted=1,
            pairs_rejected=0,
            quotes_validated=2,
            quotes_failed=0,
        ),
    )

    data = prepare_report_data(result)

    # Verify multiple assessments
    assert len(data["pairs"]) == 1
    pair = data["pairs"][0]
    assert pair["doc_count"] == 2
    assert pair["quote_count"] == 2
    assert len(pair["assessments"]) == 2

    # Verify both documents in document index
    assert len(data["documents"]) == 2


def test_prepare_report_data_entity_index():
    """Test that entity index correctly maps entities to pairs."""
    # Create test data with entities appearing in multiple pairs
    pool = ResourcePool()
    resource = pool.add(
        url="https://example.com/doc", title="Doc", document_text="Text"
    )

    quote = ResourceQuote(resource=resource, query_text="test", spans=[(0, 4)])
    entity_mention = create_test_entity_mention("Entity", "type", ["Entity"], [quote])

    assessment = PairAssessment(
        resource_id=resource.id,
        entity1=entity_mention,
        entity2=entity_mention,
        relationship="test",
        quotes=[quote],
        confidence="high",
        reasoning="test",
    )

    # Create three judgments where "A" appears in all of them
    judgment1 = PairJudgment(
        entity1=SimpleEntity(name="A", kind="type1", aliases=["A"]),
        entity2=SimpleEntity(name="B", kind="type2", aliases=["B"]),
        relationship="test",
        assessments=[assessment],
        accepted=True,
        confidence="high",
        reasoning="test",
    )

    judgment2 = PairJudgment(
        entity1=SimpleEntity(name="A", kind="type1", aliases=["A"]),
        entity2=SimpleEntity(name="C", kind="type3", aliases=["C"]),
        relationship="test",
        assessments=[assessment],
        accepted=True,
        confidence="high",
        reasoning="test",
    )

    judgment3 = PairJudgment(
        entity1=SimpleEntity(name="D", kind="type4", aliases=["D"]),
        entity2=SimpleEntity(name="E", kind="type5", aliases=["E"]),
        relationship="test",
        assessments=[assessment],
        accepted=True,
        confidence="high",
        reasoning="test",
    )

    result = ExtractionResult(
        resources=pool,
        judgments=[judgment1, judgment2, judgment3],
        metadata=ExtractionMetadata(
            topic="test",
            resource_count=1,
            total_entities_found=5,
            entities_after_validation=5,
            entities_merged=0,
            merge_cache_hits=0,
            merge_cache_misses=0,
            proximal_sets_found=0,
            total_pairs_found=3,
            pairs_accepted=3,
            pairs_rejected=0,
            quotes_validated=1,
            quotes_failed=0,
        ),
    )

    data = prepare_report_data(result)

    # Verify entity index
    assert "A" in data["entity_index"]
    assert len(data["entity_index"]["A"]) == 2  # Appears in pairs 0 and 1
    assert 0 in data["entity_index"]["A"]
    assert 1 in data["entity_index"]["A"]

    assert "B" in data["entity_index"]
    assert len(data["entity_index"]["B"]) == 1
    assert 0 in data["entity_index"]["B"]

    assert "D" in data["entity_index"]
    assert len(data["entity_index"]["D"]) == 1
    assert 2 in data["entity_index"]["D"]


def test_prepare_report_data_empty_result():
    """Test handling of extraction result with no judgments."""
    pool = ResourcePool()
    result = ExtractionResult(
        resources=pool,
        judgments=[],
        metadata=ExtractionMetadata(
            topic="empty test",
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

    data = prepare_report_data(result)

    assert data["metadata"]["total_pairs"] == 0
    assert len(data["pairs"]) == 0
    assert len(data["documents"]) == 0
    assert len(data["entity_index"]) == 0
