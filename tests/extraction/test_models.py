"""Tests for extraction data models.

Tests Pydantic validation, field constraints, and model structure.
"""

import pytest
from pydantic import ValidationError

from interaction_finder.extraction.models import (
    EntityEvidenceAssessment,
    EntityExtractionOut,
    EntityInfo,
    ExtractionMetadata,
    ExtractionResult,
    FinalJudgmentOut,
    PairEvidenceAssessment,
    PairExtractionOut,
    PairInfo,
    PairWithProvenance,
)
from interaction_finder.resources import ResourcePool


class TestEntityInfo:
    """Tests for EntityInfo model."""

    def test_valid_entity_info(self):
        """Test creating valid EntityInfo."""
        info = EntityInfo(
            entity_type="gene",
            verbatim_names=["BRCA1", "BRCA-1"],
            supporting_quotes=["BRCA1 is associated with breast cancer."],
        )
        assert info.entity_type == "gene"
        assert len(info.verbatim_names) == 2
        assert len(info.supporting_quotes) == 1

    def test_requires_verbatim_names(self):
        """Test that verbatim_names must be non-empty."""
        with pytest.raises(ValidationError):
            EntityInfo(
                entity_type="gene", verbatim_names=[], supporting_quotes=["quote"]
            )

    def test_requires_supporting_quotes(self):
        """Test that supporting_quotes must be non-empty."""
        with pytest.raises(ValidationError):
            EntityInfo(
                entity_type="gene", verbatim_names=["BRCA1"], supporting_quotes=[]
            )


class TestEntityExtractionOut:
    """Tests for EntityExtractionOut model."""

    def test_valid_extraction_output(self):
        """Test creating valid EntityExtractionOut."""
        output = EntityExtractionOut(
            entities={
                "BRCA1": EntityInfo(
                    entity_type="gene",
                    verbatim_names=["BRCA1"],
                    supporting_quotes=["quote"],
                )
            },
            reasoning="Found BRCA1 mentioned in context of breast cancer.",
        )
        assert "BRCA1" in output.entities
        assert len(output.reasoning) >= 20

    def test_requires_reasoning_min_length(self):
        """Test that reasoning must be at least 20 characters."""
        with pytest.raises(ValidationError):
            EntityExtractionOut(entities={}, reasoning="Too short")


class TestPairInfo:
    """Tests for PairInfo model."""

    def test_valid_pair_info(self):
        """Test creating valid PairInfo."""
        pair = PairInfo(
            entity1="BRCA1",
            entity2="breast cancer",
            relationship_type="associated_with",
            supporting_quotes=["BRCA1 is associated with breast cancer."],
        )
        assert pair.entity1 == "BRCA1"
        assert pair.entity2 == "breast cancer"
        assert pair.relationship_type == "associated_with"

    def test_requires_supporting_quotes(self):
        """Test that supporting_quotes must be non-empty."""
        with pytest.raises(ValidationError):
            PairInfo(
                entity1="BRCA1",
                entity2="breast cancer",
                relationship_type="associated_with",
                supporting_quotes=[],
            )


class TestPairExtractionOut:
    """Tests for PairExtractionOut model."""

    def test_valid_pair_extraction(self):
        """Test creating valid PairExtractionOut."""
        output = PairExtractionOut(
            pairs=[
                PairInfo(
                    entity1="BRCA1",
                    entity2="breast cancer",
                    relationship_type="associated_with",
                    supporting_quotes=["quote"],
                )
            ],
            reasoning="Found clear association between BRCA1 and breast cancer.",
        )
        assert len(output.pairs) == 1
        assert len(output.reasoning) >= 20


class TestEntityEvidenceAssessment:
    """Tests for EntityEvidenceAssessment model."""

    def test_valid_assessment(self):
        """Test creating valid assessment."""
        assessment = EntityEvidenceAssessment(
            strength="strong",
            rationale="Multiple independent sources support this entity.",
            supporting_quote_ids=[0, 1, 2],
        )
        assert assessment.strength == "strong"
        assert len(assessment.rationale) >= 30

    def test_strength_must_be_valid(self):
        """Test that strength must be one of: none, weak, strong."""
        with pytest.raises(ValidationError):
            EntityEvidenceAssessment(
                strength="invalid",  # type: ignore
                rationale="This should fail due to invalid strength.",
                supporting_quote_ids=[],
            )

    def test_requires_rationale_min_length(self):
        """Test that rationale must be at least 30 characters."""
        with pytest.raises(ValidationError):
            EntityEvidenceAssessment(
                strength="weak", rationale="Too short", supporting_quote_ids=[]
            )


class TestPairEvidenceAssessment:
    """Tests for PairEvidenceAssessment model."""

    def test_valid_assessment(self):
        """Test creating valid assessment."""
        assessment = PairEvidenceAssessment(
            strength="strong",
            rationale="Clear experimental evidence supports this association.",
            supporting_quote_ids=[0, 1],
        )
        assert assessment.strength == "strong"
        assert len(assessment.rationale) >= 30


class TestFinalJudgmentOut:
    """Tests for FinalJudgmentOut model."""

    def test_valid_judgment_accepted(self):
        """Test creating valid accepted judgment."""
        judgment = FinalJudgmentOut(
            accepted=True,
            confidence="high",
            rationale="Multiple strong sources with consistent evidence support acceptance.",
        )
        assert judgment.accepted is True
        assert judgment.confidence == "high"
        assert len(judgment.rationale) >= 50

    def test_valid_judgment_rejected(self):
        """Test creating valid rejected judgment."""
        judgment = FinalJudgmentOut(
            accepted=False,
            confidence="medium",
            rationale="Evidence is weak or contradictory, leading to rejection despite some support.",
        )
        assert judgment.accepted is False
        assert judgment.confidence == "medium"

    def test_requires_rationale_min_length(self):
        """Test that rationale must be at least 50 characters."""
        with pytest.raises(ValidationError):
            FinalJudgmentOut(accepted=True, confidence="high", rationale="Too short")


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_valid_result(self):
        """Test creating valid ExtractionResult."""
        # Create minimal resource pool and quote
        pool = ResourcePool()
        resource = pool.add(
            url="https://example.com/paper",
            title="Test Paper",
            document_text="BRCA1 is associated with breast cancer.",
        )
        quote = resource.quote("BRCA1 is associated with breast cancer.")

        result = ExtractionResult(
            accepted_pairs=[
                PairWithProvenance(
                    entity1="BRCA1",
                    entity2="breast cancer",
                    relationship_type="associated_with",
                    entity1_type="gene",
                    entity2_type="disease",
                    all_quotes=[quote],
                    assessments=[],
                    final_judgment={
                        "accepted": True,
                        "confidence": "high",
                        "rationale": "Strong evidence",
                    },
                )
            ],
            metadata=ExtractionMetadata(
                topic="BRCA1 and breast cancer",
                resource_count=1,
                total_entities_found=2,
                total_pairs_found=1,
                pairs_accepted=1,
                pairs_rejected=0,
                quotes_validated=1,
                quotes_failed=0,
            ),
        )

        assert len(result.accepted_pairs) == 1
        assert result.metadata.pairs_accepted == 1

    def test_empty_result(self):
        """Test creating empty ExtractionResult."""
        result = ExtractionResult(
            accepted_pairs=[],
            metadata=ExtractionMetadata(
                topic="test",
                resource_count=0,
                total_entities_found=0,
                total_pairs_found=0,
                pairs_accepted=0,
                pairs_rejected=0,
                quotes_validated=0,
                quotes_failed=0,
            ),
        )

        assert len(result.accepted_pairs) == 0
        assert result.metadata.pairs_accepted == 0
