"""Tests for YAKE keyword extractor."""

import pytest

from interaction_finder.keywords.extractors import ScoredKeyword, YAKEExtractor


class TestYAKEExtractor:
    """Test suite for YAKE extractor."""

    def test_initialization_defaults(self):
        """Test YAKE initializes with correct defaults."""
        extractor = YAKEExtractor()
        assert extractor.name == "yake"
        assert extractor.n_grams == 3
        assert extractor.deduplication_threshold == 0.9
        assert extractor.window_size == 1

    def test_initialization_custom_params(self):
        """Test YAKE initialization with custom parameters."""
        extractor = YAKEExtractor(n_grams=2, deduplication_threshold=0.8, window_size=2)
        assert extractor.n_grams == 2
        assert extractor.deduplication_threshold == 0.8
        assert extractor.window_size == 2

    def test_initialization_invalid_n_grams(self):
        """Test YAKE rejects invalid n_grams."""
        with pytest.raises(ValueError, match="n_grams must be in"):
            YAKEExtractor(n_grams=0)
        with pytest.raises(ValueError, match="n_grams must be in"):
            YAKEExtractor(n_grams=6)

    def test_initialization_invalid_deduplication_threshold(self):
        """Test YAKE rejects invalid deduplication_threshold."""
        with pytest.raises(ValueError, match="deduplication_threshold must be in"):
            YAKEExtractor(deduplication_threshold=-0.1)
        with pytest.raises(ValueError, match="deduplication_threshold must be in"):
            YAKEExtractor(deduplication_threshold=1.1)

    def test_initialization_invalid_window_size(self):
        """Test YAKE rejects invalid window_size."""
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            YAKEExtractor(window_size=0)

    def test_extract_basic_text(self):
        """Test YAKE extracts keywords from simple text."""
        extractor = YAKEExtractor()
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
        # Check scores are normalized and in valid range
        assert all(0 < kw.score <= 1 for kw in keywords)
        # Check keywords are sorted by score descending
        scores = [kw.score for kw in keywords]
        assert scores == sorted(scores, reverse=True)

    def test_extract_empty_text_raises(self):
        """Test YAKE raises on empty text."""
        extractor = YAKEExtractor()
        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("")
        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("   ")

    def test_extract_invalid_max_keywords(self):
        """Test YAKE raises on invalid max_keywords."""
        extractor = YAKEExtractor()
        with pytest.raises(ValueError, match="max_keywords must be >= 1"):
            extractor.extract("some text", max_keywords=0)

    def test_extract_single_word_text(self):
        """Test YAKE handles single-word text."""
        extractor = YAKEExtractor()
        keywords = extractor.extract("hypertension")
        # Should return at least one result
        assert len(keywords) >= 1

    def test_extract_respects_max_keywords(self):
        """Test YAKE respects max_keywords limit."""
        extractor = YAKEExtractor()
        text = " ".join([f"keyword{i} is important" for i in range(20)])
        keywords = extractor.extract(text, max_keywords=5)
        assert len(keywords) <= 5

    def test_extract_n_gram_control(self):
        """Test YAKE respects n_grams parameter."""
        text = (
            "Single word. Two word phrase. Three word key phrase. "
            "Four word long key phrase."
        )
        # Extract with max n-grams = 2
        extractor = YAKEExtractor(n_grams=2)
        keywords = extractor.extract(text, max_keywords=10)
        # All keywords should be at most 2 words
        assert all(len(kw.keyword.split()) <= 2 for kw in keywords)

    def test_score_conversion(self):
        """Test YAKE scores are properly converted to higher=better."""
        extractor = YAKEExtractor()
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "Deep learning is a subset of machine learning. "
            "Neural networks are used in deep learning algorithms."
        )
        keywords = extractor.extract(text, max_keywords=10)
        # All scores should be in (0, 1]
        assert all(0 < kw.score <= 1 for kw in keywords)
        # Scores should be sorted descending
        scores = [kw.score for kw in keywords]
        assert scores == sorted(scores, reverse=True)

    def test_healthy_always_true(self):
        """Test YAKE is always healthy (no external dependencies)."""
        extractor = YAKEExtractor()
        assert extractor.healthy() is True
