"""Tests for paper quality assessment models and functionality.

Tests the PaperQualityAssessment model validation, score computation,
quality tier classification, serialization, and the combined DocumentAnalysisOut model.
"""

import pytest
from pydantic import ValidationError

from interaction_finder.extraction.models import (
    DocumentAnalysisOut,
    EntityInfo,
    PaperQualityAssessment,
    QualityDimensionScore,
)


class TestQualityDimensionScore:
    """Tests for QualityDimensionScore model."""

    def test_valid_dimension_score(self):
        """Test creating valid dimension score."""
        score = QualityDimensionScore(
            score=2,
            justification="Methods described adequately with minor gaps.",
        )
        assert score.score == 2
        assert len(score.justification) >= 20

    def test_score_boundary_values(self):
        """Test all valid score values (0-4)."""
        for s in [0, 1, 2, 3, 4]:
            score = QualityDimensionScore(
                score=s,
                justification="This is a valid justification text for testing.",
            )
            assert score.score == s

    def test_invalid_score_value(self):
        """Test that scores outside 0-4 are rejected."""
        with pytest.raises(ValidationError):
            QualityDimensionScore(
                score=5,
                justification="This should fail due to invalid score value.",
            )
        with pytest.raises(ValidationError):
            QualityDimensionScore(
                score=-1,
                justification="This should fail due to negative score value.",
            )

    def test_justification_min_length(self):
        """Test that justification must be at least 20 characters."""
        with pytest.raises(ValidationError):
            QualityDimensionScore(
                score=2,
                justification="Too short here",
            )


class TestPaperQualityAssessment:
    """Tests for PaperQualityAssessment model."""

    def _make_dimension(self, score: int) -> QualityDimensionScore:
        """Helper to create dimension score with valid justification."""
        return QualityDimensionScore(
            score=score,
            justification=f"Score {score}: This is a valid justification for testing.",
        )

    def test_valid_assessment_all_zeros(self):
        """Test assessment with minimum scores (all zeros)."""
        assessment = PaperQualityAssessment(
            method_clarity=self._make_dimension(0),
            data_provenance=self._make_dimension(0),
            statistical_rigour=self._make_dimension(0),
            internal_consistency=self._make_dimension(0),
            plausibility=self._make_dimension(0),
            reproducibility_signals=self._make_dimension(0),
            integrity_indicators=self._make_dimension(0),
        )
        assert assessment.overall_score == 0

    def test_valid_assessment_all_fours(self):
        """Test assessment with maximum scores (all fours)."""
        assessment = PaperQualityAssessment(
            method_clarity=self._make_dimension(4),
            data_provenance=self._make_dimension(4),
            statistical_rigour=self._make_dimension(4),
            internal_consistency=self._make_dimension(4),
            plausibility=self._make_dimension(4),
            reproducibility_signals=self._make_dimension(4),
            integrity_indicators=self._make_dimension(4),
        )
        assert assessment.overall_score == 28

    def test_valid_assessment_mixed_scores(self):
        """Test assessment with mixed scores."""
        assessment = PaperQualityAssessment(
            method_clarity=self._make_dimension(2),
            data_provenance=self._make_dimension(2),
            statistical_rigour=self._make_dimension(1),
            internal_consistency=self._make_dimension(3),
            plausibility=self._make_dimension(2),
            reproducibility_signals=self._make_dimension(1),
            integrity_indicators=self._make_dimension(2),
        )
        assert assessment.overall_score == 13

    def test_serialization_roundtrip(self):
        """Test that assessment can be serialized and deserialized."""
        assessment = PaperQualityAssessment(
            method_clarity=self._make_dimension(2),
            data_provenance=self._make_dimension(2),
            statistical_rigour=self._make_dimension(1),
            internal_consistency=self._make_dimension(3),
            plausibility=self._make_dimension(2),
            reproducibility_signals=self._make_dimension(1),
            integrity_indicators=self._make_dimension(2),
        )
        # Serialize to dict
        data = assessment.model_dump()
        assert "method_clarity" in data
        assert data["method_clarity"]["score"] == 2
        # Deserialize from dict
        restored = PaperQualityAssessment.model_validate(data)
        assert restored.overall_score == assessment.overall_score

    def test_json_serialization(self):
        """Test that assessment serializes to JSON correctly."""
        assessment = PaperQualityAssessment(
            method_clarity=self._make_dimension(2),
            data_provenance=self._make_dimension(2),
            statistical_rigour=self._make_dimension(2),
            internal_consistency=self._make_dimension(2),
            plausibility=self._make_dimension(2),
            reproducibility_signals=self._make_dimension(2),
            integrity_indicators=self._make_dimension(2),
        )
        json_data = assessment.model_dump(mode="json")
        # JSON serialization should produce plain dicts
        assert isinstance(json_data, dict)
        assert isinstance(json_data["method_clarity"], dict)
        assert json_data["method_clarity"]["score"] == 2
        # overall_score is a computed properties, not in JSON
        # They are recomputed when the model is validated
        restored = PaperQualityAssessment.model_validate(json_data)
        assert restored.overall_score == 14


class TestPaperQualityInExtractionResult:
    """Tests for paper_quality field in ExtractionResult."""

    def test_extraction_result_with_paper_quality(self):
        """Test that ExtractionResult includes paper_quality field."""
        from interaction_finder.extraction.models import (
            ExtractionMetadata,
            ExtractionResult,
        )
        from interaction_finder.resources import ResourcePool

        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com/paper",
            title="Test Paper",
            document_text="Test content",
        )

        def make_dim(score: int) -> QualityDimensionScore:
            return QualityDimensionScore(
                score=score,
                justification=f"Score {score} justification for testing purposes.",
            )

        assessment = PaperQualityAssessment(
            method_clarity=make_dim(2),
            data_provenance=make_dim(2),
            statistical_rigour=make_dim(2),
            internal_consistency=make_dim(2),
            plausibility=make_dim(2),
            reproducibility_signals=make_dim(2),
            integrity_indicators=make_dim(2),
        )
        metadata = ExtractionMetadata(
            topic="Test",
            resource_count=1,
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
        )
        result = ExtractionResult(
            topic="Test",
            target_entity_types=["gene"],
            permitted_pairs={"gene": ["gene"]},
            resources=pool,
            judgments=[],
            metadata=metadata,
            paper_quality={resource.id.url: assessment},
        )
        assert resource.id.url in result.paper_quality
        assert result.paper_quality[resource.id.url].overall_score == 14

    def test_extraction_result_paper_quality_serialization(self):
        """Test that paper_quality serializes and deserializes correctly."""
        from interaction_finder.extraction.models import (
            ExtractionMetadata,
            ExtractionResult,
        )
        from interaction_finder.resources import ResourcePool

        pool = ResourcePool()
        resource = pool.add(
            url="http://example.com/paper",
            title="Test Paper",
            document_text="Test content",
        )

        def make_dim(score: int) -> QualityDimensionScore:
            return QualityDimensionScore(
                score=score,
                justification=f"Score {score} justification for testing purposes.",
            )

        assessment = PaperQualityAssessment(
            method_clarity=make_dim(3),
            data_provenance=make_dim(2),
            statistical_rigour=make_dim(2),
            internal_consistency=make_dim(3),
            plausibility=make_dim(2),
            reproducibility_signals=make_dim(1),
            integrity_indicators=make_dim(3),
        )
        metadata = ExtractionMetadata(
            topic="Test",
            resource_count=1,
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
        )
        result = ExtractionResult(
            topic="Test",
            target_entity_types=["gene"],
            permitted_pairs={"gene": ["gene"]},
            resources=pool,
            judgments=[],
            metadata=metadata,
            paper_quality={resource.id.url: assessment},
        )
        # Serialize to JSON
        json_data = result.model_dump(mode="json")
        assert "paper_quality" in json_data
        assert resource.id.url in json_data["paper_quality"]
        # Deserialize
        restored = ExtractionResult.model_validate(json_data)
        assert resource.id.url in restored.paper_quality
        restored_assessment = restored.paper_quality[resource.id.url]
        assert restored_assessment.overall_score == 16


class TestDocumentAnalysisOut:
    """Tests for combined DocumentAnalysisOut model."""

    def _make_dimension(self, score: int) -> QualityDimensionScore:
        """Helper to create dimension score with valid justification."""
        return QualityDimensionScore(
            score=score,
            justification=f"Score {score}: This is a valid justification for testing.",
        )

    def _make_quality_assessment(self, base_score: int = 2) -> PaperQualityAssessment:
        """Helper to create a paper quality assessment."""
        return PaperQualityAssessment(
            method_clarity=self._make_dimension(base_score),
            data_provenance=self._make_dimension(base_score),
            statistical_rigour=self._make_dimension(base_score),
            internal_consistency=self._make_dimension(base_score),
            plausibility=self._make_dimension(base_score),
            reproducibility_signals=self._make_dimension(base_score),
            integrity_indicators=self._make_dimension(base_score),
        )

    def _make_entity(self, name: str, kind: str = "gene") -> EntityInfo:
        """Helper to create an entity."""
        return EntityInfo(
            kind=kind,
            name=name,
            aliases=[name.lower()],
            quotes=[f"The {name} gene is important."],
            reasoning=f"Extracted {name} because it is relevant to the topic.",
        )

    def test_combined_output_with_entities(self):
        """Test DocumentAnalysisOut with quality assessment and entities."""
        quality = self._make_quality_assessment(2)
        entities = [
            self._make_entity("BRCA1"),
            self._make_entity("TP53"),
        ]
        output = DocumentAnalysisOut(
            paper_quality=quality,
            entities=entities,
        )
        assert output.paper_quality.overall_score == 14
        assert len(output.entities) == 2
        assert output.entities[0].name == "BRCA1"

    def test_combined_output_with_empty_entities(self):
        """Test DocumentAnalysisOut with quality assessment but no entities."""
        quality = self._make_quality_assessment(1)
        output = DocumentAnalysisOut(
            paper_quality=quality,
            entities=[],
        )
        assert output.paper_quality.overall_score == 7
        assert len(output.entities) == 0

    def test_combined_output_field_order(self):
        """Test that paper_quality comes before entities in field order."""
        # This tests the intentional field ordering for LLM output
        fields = list(DocumentAnalysisOut.model_fields.keys())
        assert fields.index("paper_quality") < fields.index("entities")

    def test_combined_output_serialization(self):
        """Test that DocumentAnalysisOut serializes and deserializes correctly."""
        quality = self._make_quality_assessment(4)
        entities = [self._make_entity("EGFR", "gene")]
        output = DocumentAnalysisOut(
            paper_quality=quality,
            entities=entities,
        )
        # Serialize to dict
        data = output.model_dump()
        assert "paper_quality" in data
        assert "entities" in data
        assert data["paper_quality"]["method_clarity"]["score"] == 4
        assert data["entities"][0]["name"] == "EGFR"
        # Deserialize
        restored = DocumentAnalysisOut.model_validate(data)
        assert restored.paper_quality.overall_score == 28
        assert len(restored.entities) == 1
        assert restored.entities[0].name == "EGFR"

    def test_combined_output_json_serialization(self):
        """Test JSON serialization of DocumentAnalysisOut."""
        quality = self._make_quality_assessment(2)
        entities = [
            self._make_entity("KRAS", "gene"),
            self._make_entity("lung cancer", "disease"),
        ]
        output = DocumentAnalysisOut(
            paper_quality=quality,
            entities=entities,
        )
        json_data = output.model_dump(mode="json")
        assert isinstance(json_data, dict)
        assert isinstance(json_data["paper_quality"], dict)
        assert isinstance(json_data["entities"], list)
        # Verify roundtrip
        restored = DocumentAnalysisOut.model_validate(json_data)
        assert (
            restored.paper_quality.overall_score == output.paper_quality.overall_score
        )
        assert len(restored.entities) == len(output.entities)
