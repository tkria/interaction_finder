"""Tests for extraction data models.

Tests Pydantic validation, field constraints, and model structure.
"""

import pytest
from pydantic import ValidationError

from interaction_finder.extraction.models import (
    EntityExtractionOut,
    EntityInfo,
    EntityMergeDecision,
    EntityMergeDecisions,
    ExtractionMetadata,
    ExtractionResult,
    PairJudgment,
    ProximalPairExtraction,
    ProximalPairInfo,
    SimpleEntity,
)
from interaction_finder.resources import ResourcePool


class TestEntityInfo:
    """Tests for EntityInfo model."""

    def test_valid_entity_info(self):
        """Test creating valid EntityInfo."""
        info = EntityInfo(
            kind="gene",
            name="BRCA1",
            aliases=["BRCA1", "BRCA-1"],
            quotes=["BRCA1 is associated with breast cancer."],
            reasoning="BRCA1 is mentioned in context of breast cancer susceptibility.",
        )
        assert info.kind == "gene"
        assert info.name == "BRCA1"
        assert len(info.aliases) == 2
        assert len(info.quotes) == 1
        assert len(info.reasoning) >= 20

    def test_requires_aliases(self):
        """Test that aliases must be non-empty."""
        with pytest.raises(ValidationError):
            EntityInfo(
                kind="gene",
                name="BRCA1",
                aliases=[],
                quotes=["quote"],
                reasoning="This should fail due to empty aliases.",
            )

    def test_requires_quotes(self):
        """Test that quotes must be non-empty."""
        with pytest.raises(ValidationError):
            EntityInfo(
                kind="gene",
                name="BRCA1",
                aliases=["BRCA1"],
                quotes=[],
                reasoning="This should fail due to empty quotes.",
            )

    def test_requires_reasoning_min_length(self):
        """Test that reasoning must be at least 20 characters."""
        with pytest.raises(ValidationError):
            EntityInfo(
                kind="gene",
                name="BRCA1",
                aliases=["BRCA1"],
                quotes=["quote"],
                reasoning="Too short",
            )

    def test_accepts_type_alias(self):
        """Test that 'type' alias works for kind field.

        This ensures LLM responses using 'type' are accepted.
        """
        info = EntityInfo(
            type="gene",
            name="BRCA1",
            aliases=["BRCA1"],
            quotes=["BRCA1 is a tumor suppressor gene."],
            reasoning="BRCA1 is a well-known tumor suppressor gene involved in DNA repair.",
        )
        assert info.kind == "gene"

    def test_accepts_kind_field_name(self):
        """Test that 'kind' field name works directly."""
        info = EntityInfo(
            kind="disease",
            name="breast cancer",
            aliases=["breast cancer", "mammary carcinoma"],
            quotes=["The patient has breast cancer."],
            reasoning="Breast cancer is the primary disease mentioned in this context.",
        )
        assert info.kind == "disease"


class TestEntityExtractionOut:
    """Tests for EntityExtractionOut model."""

    def test_valid_extraction_output(self):
        """Test creating valid EntityExtractionOut."""
        output = EntityExtractionOut(
            entities=[
                EntityInfo(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=["BRCA1 is associated with breast cancer."],
                    reasoning="BRCA1 is a key gene in breast cancer susceptibility.",
                )
            ]
        )
        assert len(output.entities) == 1
        assert output.entities[0].name == "BRCA1"

    def test_empty_entities_list(self):
        """Test that empty entities list is valid when no relevant entities found."""
        output = EntityExtractionOut(entities=[])
        assert output.entities == []

    def test_multiple_entities(self):
        """Test extraction output with multiple entities."""
        output = EntityExtractionOut(
            entities=[
                EntityInfo(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=["quote1"],
                    reasoning="BRCA1 is a tumor suppressor gene.",
                ),
                EntityInfo(
                    kind="disease",
                    name="breast cancer",
                    aliases=["breast cancer"],
                    quotes=["quote2"],
                    reasoning="Breast cancer is the primary disease discussed.",
                ),
            ]
        )
        assert len(output.entities) == 2
        assert output.entities[0].name == "BRCA1"
        assert output.entities[1].name == "breast cancer"


class TestEntityMergeDecision:
    """Tests for EntityMergeDecision model."""

    def test_valid_merge_decision(self):
        """Test creating valid merge decision."""
        decision = EntityMergeDecision(
            parent_entity="BRCA1",
            child_entity="BRCA",
            should_merge=True,
            reasoning="BRCA is commonly used shorthand for BRCA1 in this context.",
        )
        assert decision.parent_entity == "BRCA1"
        assert decision.child_entity == "BRCA"
        assert decision.should_merge is True

    def test_requires_reasoning_min_length(self):
        """Test that reasoning must be at least 20 characters."""
        with pytest.raises(ValidationError):
            EntityMergeDecision(
                parent_entity="BRCA1",
                child_entity="BRCA",
                should_merge=True,
                reasoning="Too short",
            )


class TestProximalPairInfo:
    """Tests for ProximalPairInfo model."""

    def test_valid_proximal_pair(self):
        """Test creating valid ProximalPairInfo."""
        pair = ProximalPairInfo(
            entity1="BRCA1",
            entity2="breast cancer",
            relationship_types=["associated_with", "causes"],
            supporting_quotes=["BRCA1 is associated with breast cancer."],
        )
        assert pair.entity1 == "BRCA1"
        assert pair.entity2 == "breast cancer"
        assert len(pair.relationship_types) == 2

    def test_requires_relationship_types(self):
        """Test that relationship_types must be non-empty."""
        with pytest.raises(ValidationError):
            ProximalPairInfo(
                entity1="BRCA1",
                entity2="breast cancer",
                relationship_types=[],
                supporting_quotes=["quote"],
            )

    def test_requires_supporting_quotes(self):
        """Test that supporting_quotes must be non-empty."""
        with pytest.raises(ValidationError):
            ProximalPairInfo(
                entity1="BRCA1",
                entity2="breast cancer",
                relationship_types=["associated_with"],
                supporting_quotes=[],
            )


class TestProximalPairExtraction:
    """Tests for ProximalPairExtraction model."""

    def test_valid_extraction(self):
        """Test creating valid ProximalPairExtraction."""
        output = ProximalPairExtraction(
            pairs=[
                ProximalPairInfo(
                    entity1="BRCA1",
                    entity2="breast cancer",
                    relationship_types=["associated_with"],
                    supporting_quotes=["quote"],
                )
            ],
            reasoning="Found clear association between BRCA1 and breast cancer.",
        )
        assert len(output.pairs) == 1
        assert len(output.reasoning) >= 20

    def test_empty_pairs_with_valid_reasoning(self):
        """Test that empty pairs list is valid when no associations found."""
        output = ProximalPairExtraction(
            pairs=[],
            reasoning="No relevant associations between entities were found in the text.",
        )
        assert output.pairs == []
        assert len(output.reasoning) >= 20


class TestSimpleEntity:
    """Tests for SimpleEntity model."""

    def test_valid_simple_entity(self):
        """Test creating valid SimpleEntity."""
        entity = SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1", "BRCA-1"])
        assert entity.name == "BRCA1"
        assert entity.kind == "gene"
        assert len(entity.aliases) == 2


class TestPairJudgment:
    """Tests for PairJudgment model."""

    def test_valid_judgment_accepted(self):
        """Test creating valid accepted judgment."""
        judgment = PairJudgment(
            entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
            entity2=SimpleEntity(
                name="breast cancer", kind="disease", aliases=["breast cancer"]
            ),
            relationship="associated_with",
            assessments=[],
            accepted=True,
            confidence="high",
            reasoning="Multiple strong sources with consistent evidence support acceptance.",
        )
        assert judgment.accepted is True
        assert judgment.confidence == "high"
        assert judgment.entity1.name == "BRCA1"
        assert judgment.entity2.name == "breast cancer"

    def test_valid_judgment_rejected(self):
        """Test creating valid rejected judgment."""
        judgment = PairJudgment(
            entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
            entity2=SimpleEntity(
                name="breast cancer", kind="disease", aliases=["breast cancer"]
            ),
            relationship="associated_with",
            assessments=[],
            accepted=False,
            confidence="low",
            reasoning="Evidence is weak or contradictory, leading to rejection.",
        )
        assert judgment.accepted is False
        assert judgment.confidence == "low"


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_valid_result(self):
        """Test creating valid ExtractionResult."""
        pool = ResourcePool()

        result = ExtractionResult(
            topic="test topic",
            target_entity_types=["gene", "disease"],
            permitted_pairs={"gene": ["disease"], "disease": ["gene"]},
            resources=pool,
            judgments=[
                PairJudgment(
                    entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
                    entity2=SimpleEntity(
                        name="breast cancer", kind="disease", aliases=["breast cancer"]
                    ),
                    relationship="associated_with",
                    assessments=[],
                    accepted=True,
                    confidence="high",
                    reasoning="Strong evidence from multiple sources.",
                )
            ],
            metadata=ExtractionMetadata(
                topic="BRCA1 and breast cancer",
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

        assert result.topic == "test topic"
        assert result.target_entity_types == ["gene", "disease"]
        assert len(result.judgments) == 1
        assert result.metadata.pairs_accepted == 1

    def test_empty_result(self):
        """Test creating empty ExtractionResult."""
        pool = ResourcePool()
        result = ExtractionResult(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs={"gene": ["gene"]},
            resources=pool,
            judgments=[],
            metadata=ExtractionMetadata(
                topic="test",
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

        assert len(result.judgments) == 0
        assert result.metadata.pairs_accepted == 0
