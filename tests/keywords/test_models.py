"""Tests for Pydantic output models."""

import pytest
from pydantic import ValidationError

from interaction_finder.keywords.models import (
    BridgingTermsOut,
    DocumentSummaryOut,
    KeywordEvaluationOut,
    QueryExpansionOut,
    ReflectionOut,
    ResultSelectionOut,
)
from interaction_finder.keywords.extractors import ScoredKeyword


class TestQueryExpansionOut:
    """Test QueryExpansionOut model."""

    def test_valid_model(self):
        """Test creating valid QueryExpansionOut."""
        model = QueryExpansionOut(
            queries=["query 1", "query 2"],
            reasoning="These queries target review articles effectively",
        )
        assert len(model.queries) == 2
        assert len(model.reasoning) >= 20

    def test_rejects_short_reasoning(self):
        """Test model rejects too-short reasoning."""
        with pytest.raises(ValidationError):
            QueryExpansionOut(queries=["query"], reasoning="short")


class TestResultSelectionOut:
    """Test ResultSelectionOut model."""

    def test_valid_model(self):
        """Test creating valid ResultSelectionOut."""
        model = ResultSelectionOut(
            selected_indices=[0, 2, 5],
            reasoning="These results appear to be review articles",
        )
        assert model.selected_indices == [0, 2, 5]
        assert len(model.reasoning) >= 20


class TestKeywordEvaluationOut:
    """Test KeywordEvaluationOut model."""

    def test_valid_model(self):
        """Test creating valid KeywordEvaluationOut."""
        model = KeywordEvaluationOut(
            bridging_terms=["term1", "term2", "term3"],
            reasoning="These terms are useful because they connect concepts",
        )
        assert len(model.bridging_terms) == 3


class TestDocumentSummaryOut:
    """Test DocumentSummaryOut model."""

    def test_valid_model(self):
        """Test creating valid DocumentSummaryOut."""
        model = DocumentSummaryOut(
            summary="This document discusses machine learning approaches to problem X.",
            related_areas=["deep learning", "neural networks"],
            bridging_terms=["embedding", "representation"],
            coverage_contribution="Adds perspective on neural approaches",
        )
        assert 50 <= len(model.summary) <= 500
        assert len(model.related_areas) == 2

    def test_rejects_too_short_summary(self):
        """Test model rejects summary that's too short."""
        with pytest.raises(ValidationError):
            DocumentSummaryOut(
                summary="Too short",
                related_areas=[],
                bridging_terms=[],
                coverage_contribution="Some contribution",
            )


class TestReflectionOut:
    """Test ReflectionOut model."""

    def test_valid_continue_decision(self):
        """Test creating valid ReflectionOut with continue decision."""
        model = ReflectionOut(
            decision="continue",
            reasoning="Coverage is incomplete, missing X and Y areas",
            new_search_angles=["angle1", "angle2"],
        )
        assert model.decision == "continue"
        assert len(model.new_search_angles) > 0

    def test_valid_stop_decision(self):
        """Test creating valid ReflectionOut with stop decision."""
        model = ReflectionOut(
            decision="stop",
            reasoning="Coverage is sufficient across major areas",
            new_search_angles=[],
        )
        assert model.decision == "stop"
        assert len(model.new_search_angles) == 0

    def test_rejects_invalid_decision(self):
        """Test model rejects invalid decision values."""
        with pytest.raises(ValidationError):
            ReflectionOut(
                decision="maybe",  # type: ignore
                reasoning="Not sure what to do",
                new_search_angles=[],
            )


class TestBridgingTermsOut:
    """Test BridgingTermsOut model."""

    def test_valid_model(self):
        """Test creating valid BridgingTermsOut."""
        model = BridgingTermsOut(
            terms=["term1", "term2", "term3"],
            total_documents_processed=5,
            rounds_completed=2,
            coverage_assessment="Found comprehensive coverage of topic with 3 bridging terms",
        )
        assert len(model.terms) == 3
        assert model.total_documents_processed == 5
        assert model.rounds_completed == 2

    def test_rejects_negative_counts(self):
        """Test model rejects negative document counts."""
        with pytest.raises(ValidationError):
            BridgingTermsOut(
                terms=[],
                total_documents_processed=-1,
                rounds_completed=1,
                coverage_assessment="Assessment text here",
            )

    def test_rejects_zero_rounds(self):
        """Test model rejects zero rounds completed."""
        with pytest.raises(ValidationError):
            BridgingTermsOut(
                terms=[],
                total_documents_processed=0,
                rounds_completed=0,
                coverage_assessment="Assessment text here",
            )
