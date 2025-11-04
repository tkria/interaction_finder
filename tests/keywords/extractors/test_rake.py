"""Tests for RAKE keyword extractor."""

import pytest

from interaction_finder.keywords.extractors import RAKEExtractor, ScoredKeyword


class TestRAKEExtractor:
    """Test suite for RAKE extractor."""

    def test_initialization_defaults(self):
        """Test RAKE initializes with correct defaults."""
        extractor = RAKEExtractor()
        assert extractor.name == "rake"
        assert extractor.min_length == 1
        assert extractor.max_length == 4

    def test_initialization_custom_params(self):
        """Test RAKE initialization with custom parameters."""
        extractor = RAKEExtractor(min_length=2, max_length=3)
        assert extractor.min_length == 2
        assert extractor.max_length == 3

    def test_initialization_invalid_min_length(self):
        """Test RAKE rejects invalid min_length."""
        with pytest.raises(ValueError, match="min_length must be >= 1"):
            RAKEExtractor(min_length=0)

    def test_initialization_invalid_max_length(self):
        """Test RAKE rejects max_length < min_length."""
        with pytest.raises(ValueError, match="max_length.*must be >= min_length"):
            RAKEExtractor(min_length=3, max_length=2)

    def test_extract_basic_text(self):
        """Test RAKE extracts keywords from simple text."""
        extractor = RAKEExtractor()
        text = (
            "Pulmonary arterial hypertension is a progressive disease. "
            "Treatment options include endothelin receptor antagonists and "
            "phosphodiesterase inhibitors."
        )
        keywords = extractor.extract(text, max_keywords=5)
        # Check structure
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert all(isinstance(kw, ScoredKeyword) for kw in keywords)
        # Check scores are normalized
        assert all(0 <= kw.score <= 1 for kw in keywords)
        # Check keywords are sorted by score descending
        scores = [kw.score for kw in keywords]
        assert scores == sorted(scores, reverse=True)
        # Check some expected medical terms appear
        keywords_text = [kw.keyword.lower() for kw in keywords]
        assert any("pulmonary" in kw or "hypertension" in kw for kw in keywords_text)

    def test_extract_empty_text_raises(self):
        """Test RAKE raises on empty text."""
        extractor = RAKEExtractor()
        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("")
        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("   ")

    def test_extract_invalid_max_keywords(self):
        """Test RAKE raises on invalid max_keywords."""
        extractor = RAKEExtractor()
        with pytest.raises(ValueError, match="max_keywords must be >= 1"):
            extractor.extract("some text", max_keywords=0)

    def test_extract_single_word_text(self):
        """Test RAKE handles single-word text."""
        extractor = RAKEExtractor()
        keywords = extractor.extract("hypertension")
        # Should return at least the word itself
        assert len(keywords) >= 1
        assert keywords[0].keyword == "hypertension"

    def test_extract_respects_max_keywords(self):
        """Test RAKE respects max_keywords limit."""
        extractor = RAKEExtractor()
        text = " ".join([f"keyword{i} is important" for i in range(20)])
        keywords = extractor.extract(text, max_keywords=5)
        assert len(keywords) <= 5

    def test_extract_phrase_length_constraints(self):
        """Test RAKE respects min/max phrase length."""
        extractor = RAKEExtractor(min_length=2, max_length=2)
        text = (
            "Single word phrases like therapy or treatment. "
            "Two word phrases like arterial hypertension. "
            "Three word phrases like pulmonary arterial hypertension."
        )
        keywords = extractor.extract(text, max_keywords=10)
        # All keywords should be exactly 2 words
        assert all(len(kw.keyword.split()) == 2 for kw in keywords)

    def test_score_normalization(self):
        """Test RAKE scores are properly normalized to [0, 1]."""
        extractor = RAKEExtractor()
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "Deep learning is a subset of machine learning. "
            "Neural networks are used in deep learning."
        )
        keywords = extractor.extract(text, max_keywords=10)
        # Highest score should be close to 1.0
        assert keywords[0].score == 1.0
        # All scores should be in [0, 1]
        assert all(0 <= kw.score <= 1 for kw in keywords)

    def test_healthy_always_true(self):
        """Test RAKE is always healthy (no external dependencies)."""
        extractor = RAKEExtractor()
        assert extractor.healthy() is True
