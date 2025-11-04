"""Tests for TF-IDF keyword extractor."""

import pytest

from interaction_finder.keywords.extractors import ScoredKeyword, TFIDFExtractor


class TestTFIDFExtractor:
    """Test suite for TF-IDF extractor."""

    def test_initialization_defaults(self):
        """Test TF-IDF initializes with correct defaults."""
        extractor = TFIDFExtractor()
        assert extractor.name == "tfidf"
        assert extractor.max_features == 50
        assert extractor.ngram_range == (1, 3)
        assert extractor.min_df == 1

    def test_initialization_custom_params(self):
        """Test TF-IDF initialization with custom parameters."""
        extractor = TFIDFExtractor(max_features=100, ngram_range=(1, 2), min_df=2)
        assert extractor.max_features == 100
        assert extractor.ngram_range == (1, 2)
        assert extractor.min_df == 2

    def test_initialization_invalid_max_features(self):
        """Test TF-IDF rejects invalid max_features."""
        with pytest.raises(ValueError, match="max_features must be >= 1"):
            TFIDFExtractor(max_features=0)

    def test_initialization_invalid_ngram_range(self):
        """Test TF-IDF rejects invalid ngram_range."""
        with pytest.raises(ValueError, match="ngram_range must be"):
            TFIDFExtractor(ngram_range=(1,))
        with pytest.raises(ValueError, match="Invalid ngram_range"):
            TFIDFExtractor(ngram_range=(0, 2))
        with pytest.raises(ValueError, match="Invalid ngram_range"):
            TFIDFExtractor(ngram_range=(3, 2))

    def test_initialization_invalid_min_df(self):
        """Test TF-IDF rejects invalid min_df."""
        with pytest.raises(ValueError, match="min_df must be >= 1"):
            TFIDFExtractor(min_df=0)

    def test_extract_basic_text(self):
        """Test TF-IDF extracts keywords from simple text."""
        extractor = TFIDFExtractor()
        text = (
            "Pulmonary arterial hypertension is a progressive disease. "
            "Treatment options include endothelin receptor antagonists. "
            "Phosphodiesterase inhibitors are another treatment option. "
            "Research continues on new treatment approaches."
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
        # Highest score should be 1.0 (normalized)
        if keywords:
            assert keywords[0].score == 1.0

    def test_extract_empty_text_raises(self):
        """Test TF-IDF raises on empty text."""
        extractor = TFIDFExtractor()
        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("")
        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("   ")

    def test_extract_invalid_max_keywords(self):
        """Test TF-IDF raises on invalid max_keywords."""
        extractor = TFIDFExtractor()
        with pytest.raises(ValueError, match="max_keywords must be >= 1"):
            extractor.extract("some text", max_keywords=0)

    def test_extract_single_sentence(self):
        """Test TF-IDF handles single sentence text."""
        extractor = TFIDFExtractor()
        text = "Machine learning is important for artificial intelligence research."
        keywords = extractor.extract(text, max_keywords=5)
        # Should handle gracefully and return some keywords
        assert isinstance(keywords, list)

    def test_extract_respects_max_keywords(self):
        """Test TF-IDF respects max_keywords limit."""
        extractor = TFIDFExtractor()
        text = ". ".join([f"Sentence {i} contains keyword{i}" for i in range(20)])
        keywords = extractor.extract(text, max_keywords=5)
        assert len(keywords) <= 5

    def test_extract_all_stop_words_returns_empty(self):
        """Test TF-IDF returns empty list when text is all stop words."""
        extractor = TFIDFExtractor()
        text = "the a an and or but is are was were"
        keywords = extractor.extract(text, max_keywords=5)
        # Should return empty or very few keywords
        assert len(keywords) <= 1

    def test_extract_identifies_repeated_terms(self):
        """Test TF-IDF identifies terms that appear frequently."""
        extractor = TFIDFExtractor()
        text = (
            "Treatment is important. Treatment options vary. "
            "Effective treatment requires research. New treatment approaches "
            "are being developed. Treatment guidelines help doctors."
        )
        keywords = extractor.extract(text, max_keywords=10)
        # "treatment" should appear as a high-scoring keyword
        keywords_text = [kw.keyword.lower() for kw in keywords]
        assert any("treatment" in kw for kw in keywords_text)

    def test_healthy_always_true(self):
        """Test TF-IDF is always healthy (no external dependencies)."""
        extractor = TFIDFExtractor()
        assert extractor.healthy() is True
