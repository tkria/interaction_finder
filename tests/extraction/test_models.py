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
    PairJudgment,
    PairSpread,
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
            spread=PairSpread(),
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
            spread=PairSpread(),
            accepted=False,
            confidence="low",
            reasoning="Evidence is weak or contradictory, leading to rejection.",
        )
        assert judgment.accepted is False
        assert judgment.confidence == "low"


class TestPairSpread:
    """Tests for PairSpread model."""

    def test_empty_spread(self):
        """Test creating empty PairSpread."""
        spread = PairSpread()
        assert len(spread.supporting) == 0
        assert len(spread.refuting) == 0
        assert len(spread.neutral) == 0
        assert len(spread.irrelevant) == 0

    def test_spread_with_supporting_only(self):
        """Test PairSpread with only supporting assessments."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com", title="Test", document_text="Test"
        )

        from interaction_finder.extraction.models import EntityMention, PairAssessment

        entity1 = EntityMention(
            kind="gene", name="BRCA1", aliases=["BRCA1"], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="disease",
            name="cancer",
            aliases=["cancer"],
            quotes=[],
            reasoning="test",
        )

        assessment = PairAssessment(
            resource_id=resource.id,
            entity1=entity1,
            entity2=entity2,
            relationship="increases_risk_of",
            quotes=[],
            confidence="high",
            reasoning="test",
        )

        spread = PairSpread(supporting=[assessment])
        assert len(spread.supporting) == 1
        assert len(spread.refuting) == 0
        assert len(spread.neutral) == 0
        assert len(spread.irrelevant) == 0

    def test_spread_serialization(self):
        """Test that PairSpread can be serialized and deserialized."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com", title="Test", document_text="Test"
        )

        from interaction_finder.extraction.models import EntityMention, PairAssessment

        entity1 = EntityMention(
            kind="gene", name="BRCA1", aliases=["BRCA1"], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="disease",
            name="cancer",
            aliases=["cancer"],
            quotes=[],
            reasoning="test",
        )

        assessment1 = PairAssessment(
            resource_id=resource.id,
            entity1=entity1,
            entity2=entity2,
            relationship="increases_risk_of",
            quotes=[],
            confidence="high",
            reasoning="supporting evidence",
        )

        assessment2 = PairAssessment(
            resource_id=resource.id,
            entity1=entity1,
            entity2=entity2,
            relationship="protects_against",
            quotes=[],
            confidence="medium",
            reasoning="refuting evidence",
        )

        spread = PairSpread(supporting=[assessment1], refuting=[assessment2])

        # Serialize to dict
        spread_dict = spread.model_dump()
        assert "supporting" in spread_dict
        assert "refuting" in spread_dict
        assert len(spread_dict["supporting"]) == 1
        assert len(spread_dict["refuting"]) == 1

        # Deserialize from dict
        spread_restored = PairSpread.model_validate(spread_dict)
        assert len(spread_restored.supporting) == 1
        assert len(spread_restored.refuting) == 1
        assert spread_restored.supporting[0].relationship == "increases_risk_of"
        assert spread_restored.refuting[0].relationship == "protects_against"

    def test_spread_with_mixed_polarities(self):
        """Test PairSpread with assessments in multiple categories."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com", title="Test", document_text="Test"
        )

        from interaction_finder.extraction.models import EntityMention, PairAssessment

        entity1 = EntityMention(
            kind="gene", name="BRCA1", aliases=["BRCA1"], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="disease",
            name="cancer",
            aliases=["cancer"],
            quotes=[],
            reasoning="test",
        )

        assessments = {
            "supporting": PairAssessment(
                resource_id=resource.id,
                entity1=entity1,
                entity2=entity2,
                relationship="increases_risk_of",
                quotes=[],
                confidence="high",
                reasoning="supporting",
            ),
            "refuting": PairAssessment(
                resource_id=resource.id,
                entity1=entity1,
                entity2=entity2,
                relationship="protects_against",
                quotes=[],
                confidence="high",
                reasoning="refuting",
            ),
            "neutral": PairAssessment(
                resource_id=resource.id,
                entity1=entity1,
                entity2=entity2,
                relationship="regulates",
                quotes=[],
                confidence="medium",
                reasoning="neutral",
            ),
            "irrelevant": PairAssessment(
                resource_id=resource.id,
                entity1=entity1,
                entity2=entity2,
                relationship="spatial_colocalization",
                quotes=[],
                confidence="low",
                reasoning="irrelevant",
            ),
        }

        spread = PairSpread(
            supporting=[assessments["supporting"]],
            refuting=[assessments["refuting"]],
            neutral=[assessments["neutral"]],
            irrelevant=[assessments["irrelevant"]],
        )

        # All categories should have exactly one assessment
        assert len(spread.supporting) == 1
        assert len(spread.refuting) == 1
        assert len(spread.neutral) == 1
        assert len(spread.irrelevant) == 1

        # Verify relationships are correct
        assert spread.supporting[0].relationship == "increases_risk_of"
        assert spread.refuting[0].relationship == "protects_against"
        assert spread.neutral[0].relationship == "regulates"
        assert spread.irrelevant[0].relationship == "spatial_colocalization"


class TestPairJudgmentSerialization:
    """Tests for PairJudgment serialization with new spread field."""

    def test_judgment_with_spread_serialization(self):
        """Test that PairJudgment with PairSpread serializes correctly."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com", title="Test", document_text="Test"
        )

        from interaction_finder.extraction.models import EntityMention, PairAssessment

        entity1 = EntityMention(
            kind="gene", name="BRCA1", aliases=["BRCA1"], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="disease",
            name="cancer",
            aliases=["cancer"],
            quotes=[],
            reasoning="test",
        )

        assessment = PairAssessment(
            resource_id=resource.id,
            entity1=entity1,
            entity2=entity2,
            relationship="increases_risk_of",
            quotes=[],
            confidence="high",
            reasoning="test",
        )

        spread = PairSpread(supporting=[assessment])

        judgment = PairJudgment(
            entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
            entity2=SimpleEntity(name="cancer", kind="disease", aliases=["cancer"]),
            relationship="increases_risk_of",
            spread=spread,
            accepted=True,
            confidence="high",
            reasoning="Strong supporting evidence",
        )

        # Serialize
        judgment_dict = judgment.model_dump()
        assert "spread" in judgment_dict
        assert "supporting" in judgment_dict["spread"]
        assert len(judgment_dict["spread"]["supporting"]) == 1

        # Deserialize
        judgment_restored = PairJudgment.model_validate(judgment_dict)
        assert judgment_restored.accepted is True
        assert len(judgment_restored.spread.supporting) == 1
        assert (
            judgment_restored.spread.supporting[0].relationship == "increases_risk_of"
        )

    def test_judgment_with_contentious_spread(self):
        """Test judgment with both supporting and refuting evidence."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com", title="Test", document_text="Test"
        )

        from interaction_finder.extraction.models import EntityMention, PairAssessment

        entity1 = EntityMention(
            kind="gene", name="BRCA1", aliases=["BRCA1"], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="disease",
            name="cancer",
            aliases=["cancer"],
            quotes=[],
            reasoning="test",
        )

        supporting_assessment = PairAssessment(
            resource_id=resource.id,
            entity1=entity1,
            entity2=entity2,
            relationship="increases_risk_of",
            quotes=[],
            confidence="high",
            reasoning="supporting",
        )

        refuting_assessment = PairAssessment(
            resource_id=resource.id,
            entity1=entity1,
            entity2=entity2,
            relationship="protects_against",
            quotes=[],
            confidence="medium",
            reasoning="refuting",
        )

        spread = PairSpread(
            supporting=[supporting_assessment], refuting=[refuting_assessment]
        )

        judgment = PairJudgment(
            entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
            entity2=SimpleEntity(name="cancer", kind="disease", aliases=["cancer"]),
            relationship="increases_risk_of",
            spread=spread,
            accepted=True,
            confidence="medium",
            reasoning="Mixed evidence, supporting evidence stronger",
        )

        # Verify contentious pair structure
        assert len(judgment.spread.supporting) == 1
        assert len(judgment.spread.refuting) == 1

        # Serialize and deserialize
        judgment_dict = judgment.model_dump()
        judgment_restored = PairJudgment.model_validate(judgment_dict)

        assert len(judgment_restored.spread.supporting) == 1
        assert len(judgment_restored.spread.refuting) == 1
