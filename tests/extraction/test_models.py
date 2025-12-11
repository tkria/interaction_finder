"""Tests for extraction data models.

Tests Pydantic validation, field constraints, and model structure.
"""

import pytest
from pydantic import ValidationError

from interaction_finder.extraction.models import (
    EntityConsolidationDecision,
    EntityConsolidationDecisions,
    EntityExtractionOut,
    EntityInfo,
    EntityMention,
    ExtractionMetadata,
    ExtractionResult,
    PairAssessment,
    PairJudgment,
    PairSpread,
    ProximalPairExtraction,
    ProximalPairInfo,
    SimpleEntity,
)
from interaction_finder.resources import ResourcePool, ResourceQuote
from tests.extraction.conftest import make_evidence


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


class TestEntityConsolidationDecision:
    """Tests for EntityConsolidationDecision model."""

    def test_valid_merge_decision(self):
        """Test creating valid merge decision (rename=None)."""
        decision = EntityConsolidationDecision(
            pair_id=1,
            confirm_token="xK7m",
            rename=None,
            reasoning="BRCA is commonly used shorthand for BRCA1 in this context.",
        )
        assert decision.pair_id == 1
        assert decision.confirm_token == "xK7m"
        assert decision.rename is None

    def test_valid_rename_decision(self):
        """Test creating valid rename decision with target."""
        decision = EntityConsolidationDecision(
            pair_id=1,
            confirm_token="xK7m",
            rename="APAH",
            reasoning="The verbose name should be simplified to its standard abbreviation.",
        )
        assert decision.rename == "APAH"

    def test_rename_rejects_empty_string(self):
        """Test that rename cannot be empty string (must be None or non-empty)."""
        with pytest.raises(ValidationError):
            EntityConsolidationDecision(
                pair_id=1,
                confirm_token="xK7m",
                rename="",
                reasoning="The verbose name should be simplified to its standard abbreviation.",
            )
        with pytest.raises(ValidationError):
            EntityConsolidationDecision(
                pair_id=1,
                confirm_token="xK7m",
                rename="   ",  # Whitespace only
                reasoning="The verbose name should be simplified to its standard abbreviation.",
            )

    def test_requires_reasoning_min_length(self):
        """Test that reasoning must be at least 20 characters."""
        with pytest.raises(ValidationError):
            EntityConsolidationDecision(
                pair_id=1,
                confirm_token="xK7m",
                rename=None,
                reasoning="Too short",
            )

    def test_requires_token_exact_length(self):
        """Test that confirm_token must be exactly 4 characters."""
        with pytest.raises(ValidationError):
            EntityConsolidationDecision(
                pair_id=1,
                confirm_token="abc",  # Too short
                rename=None,
                reasoning="BRCA is commonly used shorthand for BRCA1.",
            )
        with pytest.raises(ValidationError):
            EntityConsolidationDecision(
                pair_id=1,
                confirm_token="abcde",  # Too long
                rename=None,
                reasoning="BRCA is commonly used shorthand for BRCA1.",
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
            topic_relevance=3,
            entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
            entity2=SimpleEntity(
                name="breast cancer", kind="disease", aliases=["breast cancer"]
            ),
            relationship="associated_with",
            spread=PairSpread(),
            accepted=True,
            evidence=make_evidence(8),
            decision_confidence=0.9,
            reasoning="Multiple strong sources with consistent evidence support acceptance.",
        )
        assert judgment.accepted is True
        assert judgment.evidence.overall == 8
        assert judgment.entity1.name == "BRCA1"
        assert judgment.entity2.name == "breast cancer"

    def test_valid_judgment_rejected(self):
        """Test creating valid rejected judgment."""
        judgment = PairJudgment(
            topic_relevance=3,
            entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
            entity2=SimpleEntity(
                name="breast cancer", kind="disease", aliases=["breast cancer"]
            ),
            relationship="associated_with",
            spread=PairSpread(),
            accepted=False,
            evidence=make_evidence(3),
            decision_confidence=0.85,
            reasoning="Evidence is weak or contradictory, leading to rejection.",
        )
        assert judgment.accepted is False
        assert judgment.evidence.overall == 3

    def test_assessments_property_flattens_spread(self):
        """Test backward-compatible assessments property."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com",
            title="Doc",
            document_text="Doc text",
        )

        entity1 = EntityMention(
            kind="gene", name="BRCA1", aliases=["BRCA1"], quotes=[], reasoning="test"
        )
        entity2 = EntityMention(
            kind="disease",
            name="Cancer",
            aliases=["Cancer"],
            quotes=[],
            reasoning="test",
        )

        from interaction_finder.extraction.models import EntityRef

        supporting = PairAssessment(
            topic_relevance=3,
            resource_id=resource.id,
            entity1=EntityRef(canonical=entity1.name, mentions=[entity1]),
            entity2=EntityRef(canonical=entity2.name, mentions=[entity2]),
            relationship="increases_risk_of",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="support",
        )
        refuting = PairAssessment(
            topic_relevance=3,
            resource_id=resource.id,
            entity1=EntityRef(canonical=entity1.name, mentions=[entity1]),
            entity2=EntityRef(canonical=entity2.name, mentions=[entity2]),
            relationship="protects_against",
            quotes=[],
            evidence=make_evidence(6),
            reasoning="refute",
        )

        spread = PairSpread(positive=[supporting], negative=[refuting])
        judgment = PairJudgment(
            topic_relevance=3,
            entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
            entity2=SimpleEntity(name="Cancer", kind="disease", aliases=["Cancer"]),
            relationship="increases_risk_of",
            spread=spread,
            accepted=True,
            evidence=make_evidence(8),
            decision_confidence=0.85,
            reasoning="test",
        )

        flattened = judgment.assessments
        assert len(flattened) == 2
        assert flattened[0].relationship == "increases_risk_of"
        assert flattened[1].relationship == "protects_against"


class TestPairSpread:
    """Tests for PairSpread model."""

    def test_empty_spread(self):
        """Test creating empty PairSpread."""
        spread = PairSpread()
        assert len(spread.positive) == 0
        assert len(spread.negative) == 0
        assert len(spread.neutral) == 0
        assert len(spread.irrelevant) == 0

    def test_spread_with_supporting_only(self):
        """Test PairSpread with only supporting assessments."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com", title="Test", document_text="Test"
        )

        from interaction_finder.extraction.models import (
            EntityMention,
            EntityRef,
            PairAssessment,
        )

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
            topic_relevance=3,
            resource_id=resource.id,
            entity1=EntityRef(canonical=entity1.name, mentions=[entity1]),
            entity2=EntityRef(canonical=entity2.name, mentions=[entity2]),
            relationship="increases_risk_of",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="test",
        )

        spread = PairSpread(positive=[assessment])
        assert len(spread.positive) == 1
        assert len(spread.negative) == 0
        assert len(spread.neutral) == 0
        assert len(spread.irrelevant) == 0

    def test_spread_serialization(self):
        """Test that PairSpread can be serialized and deserialized."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com", title="Test", document_text="Test"
        )

        from interaction_finder.extraction.models import (
            EntityMention,
            EntityRef,
            PairAssessment,
        )

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
            topic_relevance=3,
            resource_id=resource.id,
            entity1=EntityRef(canonical=entity1.name, mentions=[entity1]),
            entity2=EntityRef(canonical=entity2.name, mentions=[entity2]),
            relationship="increases_risk_of",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="supporting evidence",
        )

        assessment2 = PairAssessment(
            topic_relevance=3,
            resource_id=resource.id,
            entity1=EntityRef(canonical=entity1.name, mentions=[entity1]),
            entity2=EntityRef(canonical=entity2.name, mentions=[entity2]),
            relationship="protects_against",
            quotes=[],
            evidence=make_evidence(6),
            reasoning="refuting evidence",
        )

        spread = PairSpread(positive=[assessment1], negative=[assessment2])

        # Serialize to dict
        spread_dict = spread.model_dump()
        assert "positive" in spread_dict
        assert "negative" in spread_dict
        assert len(spread_dict["positive"]) == 1
        assert len(spread_dict["negative"]) == 1

        # Deserialize from dict
        spread_restored = PairSpread.model_validate(spread_dict)
        assert len(spread_restored.positive) == 1
        assert len(spread_restored.negative) == 1
        assert spread_restored.positive[0].relationship == "increases_risk_of"
        assert spread_restored.negative[0].relationship == "protects_against"

    def test_spread_with_mixed_polarities(self):
        """Test PairSpread with assessments in multiple categories."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com", title="Test", document_text="Test"
        )

        from interaction_finder.extraction.models import (
            EntityMention,
            EntityRef,
            PairAssessment,
        )

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
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])

        assessments = {
            "supporting": PairAssessment(
                topic_relevance=3,
                resource_id=resource.id,
                entity1=entity1_ref,
                entity2=entity2_ref,
                relationship="increases_risk_of",
                quotes=[],
                evidence=make_evidence(8),
                reasoning="supporting",
            ),
            "refuting": PairAssessment(
                topic_relevance=3,
                resource_id=resource.id,
                entity1=entity1_ref,
                entity2=entity2_ref,
                relationship="protects_against",
                quotes=[],
                evidence=make_evidence(8),
                reasoning="refuting",
            ),
            "neutral": PairAssessment(
                topic_relevance=3,
                resource_id=resource.id,
                entity1=entity1_ref,
                entity2=entity2_ref,
                relationship="regulates",
                quotes=[],
                evidence=make_evidence(6),
                reasoning="neutral",
            ),
            "irrelevant": PairAssessment(
                topic_relevance=3,
                resource_id=resource.id,
                entity1=entity1_ref,
                entity2=entity2_ref,
                relationship="spatial_colocalization",
                quotes=[],
                evidence=make_evidence(3),
                reasoning="irrelevant",
            ),
        }

        spread = PairSpread(
            positive=[assessments["supporting"]],
            negative=[assessments["refuting"]],
            neutral=[assessments["neutral"]],
            irrelevant=[assessments["irrelevant"]],
        )

        # All categories should have exactly one assessment
        assert len(spread.positive) == 1
        assert len(spread.negative) == 1
        assert len(spread.neutral) == 1
        assert len(spread.irrelevant) == 1

        # Verify relationships are correct
        assert spread.positive[0].relationship == "increases_risk_of"
        assert spread.negative[0].relationship == "protects_against"
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

        from interaction_finder.extraction.models import (
            EntityMention,
            EntityRef,
            PairAssessment,
        )

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
            topic_relevance=3,
            resource_id=resource.id,
            entity1=EntityRef(canonical=entity1.name, mentions=[entity1]),
            entity2=EntityRef(canonical=entity2.name, mentions=[entity2]),
            relationship="increases_risk_of",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="test",
        )

        spread = PairSpread(positive=[assessment])

        judgment = PairJudgment(
            topic_relevance=3,
            entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
            entity2=SimpleEntity(name="cancer", kind="disease", aliases=["cancer"]),
            relationship="increases_risk_of",
            spread=spread,
            accepted=True,
            evidence=make_evidence(8),
            decision_confidence=0.9,
            reasoning="Strong supporting evidence",
        )
        # Serialize
        judgment_dict = judgment.model_dump()
        assert "spread" in judgment_dict
        assert "positive" in judgment_dict["spread"]
        assert len(judgment_dict["spread"]["positive"]) == 1

        # Deserialize
        judgment_restored = PairJudgment.model_validate(judgment_dict)
        assert judgment_restored.accepted is True
        assert len(judgment_restored.spread.positive) == 1
        assert judgment_restored.spread.positive[0].relationship == "increases_risk_of"

    def test_judgment_with_contentious_spread(self):
        """Test judgment with both supporting and refuting evidence."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com", title="Test", document_text="Test"
        )

        from interaction_finder.extraction.models import (
            EntityMention,
            EntityRef,
            PairAssessment,
        )

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
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])

        supporting_assessment = PairAssessment(
            topic_relevance=3,
            resource_id=resource.id,
            entity1=entity1_ref,
            entity2=entity2_ref,
            relationship="increases_risk_of",
            quotes=[],
            evidence=make_evidence(8),
            reasoning="supporting",
        )

        refuting_assessment = PairAssessment(
            topic_relevance=3,
            resource_id=resource.id,
            entity1=entity1_ref,
            entity2=entity2_ref,
            relationship="protects_against",
            quotes=[],
            evidence=make_evidence(6),
            reasoning="refuting",
        )

        spread = PairSpread(
            positive=[supporting_assessment], negative=[refuting_assessment]
        )
        judgment = PairJudgment(
            topic_relevance=3,
            entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
            entity2=SimpleEntity(name="cancer", kind="disease", aliases=["cancer"]),
            relationship="increases_risk_of",
            spread=spread,
            accepted=True,
            evidence=make_evidence(6),
            decision_confidence=0.7,
            reasoning="Mixed evidence, supporting evidence stronger",
        )

        # Verify contentious pair structure
        assert len(judgment.spread.positive) == 1
        assert len(judgment.spread.negative) == 1

        # Serialize and deserialize
        judgment_dict = judgment.model_dump()
        judgment_restored = PairJudgment.model_validate(judgment_dict)

        assert len(judgment_restored.spread.positive) == 1
        assert len(judgment_restored.spread.negative) == 1


class TestExtractionResultRehydration:
    """Tests for ExtractionResult serialization/deserialization."""

    def test_entities_dict_uses_resource_url_not_full_resource(self):
        """Verify entities dict serializes ResourceQuote with resource_url, not full Resource."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com/doc",
            title="Doc",
            document_text="Doc text with BRCA1 gene mentioned multiple times.",
        )

        def make_quote(text: str) -> ResourceQuote:
            return ResourceQuote.model_construct(
                resource=resource,
                query_text=text,
                spans=[(0, len(text))],
                is_disjoint=False,
                fuzzy_corrected=False,
                original_query=None,
                fuzzy_similarity=None,
            )

        from interaction_finder.extraction.models import EntityRef

        entity1 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["BRCA1"],
            quotes=[make_quote("BRCA1")],
            reasoning="test entity 1",
        )
        entity2 = EntityMention(
            kind="disease",
            name="Cancer",
            aliases=["Cancer"],
            quotes=[make_quote("Cancer")],
            reasoning="test entity 2",
        )

        assessment = PairAssessment(
            topic_relevance=3,
            resource_id=resource.id,
            entity1=EntityRef(canonical=entity1.name, mentions=[entity1]),
            entity2=EntityRef(canonical=entity2.name, mentions=[entity2]),
            relationship="increases_risk_of",
            quotes=[make_quote("pair quote")],
            evidence=make_evidence(8),
            reasoning="supporting evidence",
        )

        judgment = PairJudgment(
            topic_relevance=3,
            entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
            entity2=SimpleEntity(name="Cancer", kind="disease", aliases=["Cancer"]),
            relationship="increases_risk_of",
            spread=PairSpread(positive=[assessment]),
            accepted=True,
            evidence=make_evidence(8),
            decision_confidence=0.9,
            reasoning="Strong evidence",
        )
        metadata = ExtractionMetadata(
            topic="Topic",
            resource_count=1,
            total_entities_found=2,
            entities_after_validation=2,
            entities_merged=0,
            merge_cache_hits=0,
            merge_cache_misses=0,
            proximal_sets_found=0,
            total_pairs_found=1,
            pairs_accepted=1,
            pairs_rejected=0,
            quotes_validated=3,
            quotes_failed=0,
        )

        # Build entities dict (normally done by ConsolidateEntitiesNode)
        entities = {
            "BRCA1": EntityRef(canonical="BRCA1", mentions=[entity1]),
            "Cancer": EntityRef(canonical="Cancer", mentions=[entity2]),
        }

        result = ExtractionResult(
            topic="Topic",
            target_entity_types=["gene", "disease"],
            permitted_pairs={"gene": ["disease"], "disease": ["gene"]},
            resources=pool,
            judgments=[judgment],
            metadata=metadata,
            entities=entities,
        )

        # Serialize to JSON
        serialized = result.model_dump(mode="json")

        # Check that entities dict exists
        assert "entities" in serialized
        assert "BRCA1" in serialized["entities"]
        assert "Cancer" in serialized["entities"]

        # Check that entity mentions have quotes with resource_url (not full resource)
        brca1_entity = serialized["entities"]["BRCA1"]
        assert "mentions" in brca1_entity
        assert len(brca1_entity["mentions"]) == 1

        first_mention = brca1_entity["mentions"][0]
        assert "quotes" in first_mention
        assert len(first_mention["quotes"]) == 1

        first_quote = first_mention["quotes"][0]
        # The bug: this would have "resource" with full Resource object
        # The fix: should have "resource_url" with just the URL string
        assert "resource_url" in first_quote, (
            "Quote should have resource_url, not resource"
        )
        assert "resource" not in first_quote, (
            "Quote should not have full resource object"
        )
        assert first_quote["resource_url"] == resource.id.url

    def test_rehydrate_spread_based_judgments(self):
        """Ensure rehydration injects Resource objects for spread assessments."""
        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com/doc",
            title="Doc",
            document_text="Doc text content",
        )

        def make_quote(text: str) -> ResourceQuote:
            return ResourceQuote.model_construct(
                resource=resource,
                query_text=text,
                spans=[(0, len(text))],
                is_disjoint=False,
                fuzzy_corrected=False,
                original_query=None,
                fuzzy_similarity=None,
            )

        from interaction_finder.extraction.models import EntityRef

        entity1 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["BRCA1"],
            quotes=[make_quote("BRCA1")],
            reasoning="test entity 1",
        )
        entity2 = EntityMention(
            kind="disease",
            name="Cancer",
            aliases=["Cancer"],
            quotes=[make_quote("Cancer")],
            reasoning="test entity 2",
        )

        assessment = PairAssessment(
            topic_relevance=3,
            resource_id=resource.id,
            entity1=EntityRef(canonical=entity1.name, mentions=[entity1]),
            entity2=EntityRef(canonical=entity2.name, mentions=[entity2]),
            relationship="increases_risk_of",
            quotes=[make_quote("pair quote")],
            evidence=make_evidence(8),
            reasoning="supporting evidence",
        )

        judgement = PairJudgment(
            topic_relevance=3,
            entity1=SimpleEntity(name="BRCA1", kind="gene", aliases=["BRCA1"]),
            entity2=SimpleEntity(name="Cancer", kind="disease", aliases=["Cancer"]),
            relationship="increases_risk_of",
            spread=PairSpread(positive=[assessment]),
            accepted=True,
            evidence=make_evidence(8),
            decision_confidence=0.9,
            reasoning="Strong evidence",
        )
        metadata = ExtractionMetadata(
            topic="Topic",
            resource_count=1,
            total_entities_found=1,
            entities_after_validation=1,
            entities_merged=0,
            merge_cache_hits=0,
            merge_cache_misses=0,
            proximal_sets_found=0,
            total_pairs_found=1,
            pairs_accepted=1,
            pairs_rejected=0,
            quotes_validated=1,
            quotes_failed=0,
        )

        # Build entities dict (normally done by ConsolidateEntitiesNode)
        entities = {
            "BRCA1": EntityRef(canonical="BRCA1", mentions=[entity1]),
            "Cancer": EntityRef(canonical="Cancer", mentions=[entity2]),
        }

        result = ExtractionResult(
            topic="Topic",
            target_entity_types=["gene", "disease"],
            permitted_pairs={"gene": ["disease"], "disease": ["gene"]},
            resources=pool,
            judgments=[judgement],
            metadata=metadata,
            entities=entities,
        )

        serialized = result.model_dump(mode="json")
        restored = ExtractionResult.model_validate(serialized)

        restored_assessment = restored.judgments[0].spread.positive[0]
        quote = restored_assessment.quotes[0]
        assert isinstance(quote.resource, type(resource))
        assert quote.resource.id == resource.id

        entity_quote = restored_assessment.entity1.mentions[0].quotes[0]
        assert entity_quote.resource.id == resource.id
