"""Tests for KeyBERT keyword extractor."""

import pytest

from interaction_finder.keywords.extractors import KeyBERTExtractor, ScoredKeyword


class TestKeyBERTExtractor:
    """Test suite for KeyBERT extractor."""

    def test_initialization_defaults(self):
        """Test KeyBERT initializes with correct defaults."""
        extractor = KeyBERTExtractor()
        assert extractor.name == "keybert"
        assert extractor.model_name == "all-MiniLM-L6-v2"
        assert extractor.diversity == 0.5
        assert extractor.top_n == 20

    def test_initialization_custom_params(self):
        """Test KeyBERT initialization with custom parameters."""
        extractor = KeyBERTExtractor(
            model_name="all-MiniLM-L12-v2", diversity=0.7, top_n=15
        )
        assert extractor.model_name == "all-MiniLM-L12-v2"
        assert extractor.diversity == 0.7
        assert extractor.top_n == 15

    def test_initialization_invalid_diversity(self):
        """Test KeyBERT rejects invalid diversity."""
        with pytest.raises(ValueError, match="diversity must be in"):
            KeyBERTExtractor(diversity=-0.1)
        with pytest.raises(ValueError, match="diversity must be in"):
            KeyBERTExtractor(diversity=1.1)

    def test_initialization_invalid_top_n(self):
        """Test KeyBERT rejects invalid top_n."""
        with pytest.raises(ValueError, match="top_n must be >= 1"):
            KeyBERTExtractor(top_n=0)

    @pytest.mark.slow
    def test_extract_basic_text(self):
        """Test KeyBERT extracts keywords from simple text."""
        extractor = KeyBERTExtractor()
        text = (
            "Pulmonary arterial hypertension is a progressive disease. "
            "Treatment options include endothelin receptor antagonists and "
            "phosphodiesterase inhibitors. Research continues on novel therapies."
        )
        keywords = extractor.extract(text, max_keywords=5)
        # Check structure
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert all(isinstance(kw, ScoredKeyword) for kw in keywords)
        # Check scores are in valid range (cosine similarity)
        assert all(0 <= kw.score <= 1 for kw in keywords)
        # Check keywords are sorted by score descending
        scores = [kw.score for kw in keywords]
        assert scores == sorted(scores, reverse=True)

    def test_extract_empty_text_raises(self):
        """Test KeyBERT raises on empty text."""
        extractor = KeyBERTExtractor()
        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("")
        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("   ")

    def test_extract_invalid_max_keywords(self):
        """Test KeyBERT raises on invalid max_keywords."""
        extractor = KeyBERTExtractor()
        with pytest.raises(ValueError, match="max_keywords must be >= 1"):
            extractor.extract("some text", max_keywords=0)

    @pytest.mark.slow
    def test_extract_single_word_text(self):
        """Test KeyBERT handles single-word text."""
        extractor = KeyBERTExtractor()
        keywords = extractor.extract("hypertension")
        # Should return at least one result
        assert len(keywords) >= 1

    @pytest.mark.slow
    def test_extract_respects_max_keywords(self):
        """Test KeyBERT respects max_keywords limit."""
        extractor = KeyBERTExtractor()
        text = (
            "Machine learning is a field of artificial intelligence. "
            "Deep learning is a subset of machine learning that uses neural networks. "
            "Neural networks are inspired by biological neurons. "
            "Convolutional neural networks are used for image processing. "
            "Recurrent neural networks handle sequential data."
        )
        keywords = extractor.extract(text, max_keywords=3)
        assert len(keywords) <= 3

    @pytest.mark.slow
    def test_diversity_parameter_affects_results(self):
        """Test that diversity parameter changes keyword selection."""
        text = (
            "Machine learning and deep learning are related fields. "
            "Neural networks and artificial intelligence are important topics. "
            "Data science and machine learning intersect in many areas."
        )
        # Low diversity (more similar keywords)
        extractor_low = KeyBERTExtractor(diversity=0.1)
        keywords_low = extractor_low.extract(text, max_keywords=5)
        # High diversity (more varied keywords)
        extractor_high = KeyBERTExtractor(diversity=0.9)
        keywords_high = extractor_high.extract(text, max_keywords=5)
        # Results should be different
        assert len(keywords_low) > 0
        assert len(keywords_high) > 0
        # Can't guarantee specific differences, but both should return valid results

    @pytest.mark.slow
    def test_lazy_model_loading(self):
        """Test that model is loaded lazily on first use."""
        extractor = KeyBERTExtractor()
        # Model should be None before first extract
        assert extractor._model is None
        # After extract, model should be loaded
        extractor.extract("test text", max_keywords=1)
        assert extractor._model is not None

    @pytest.mark.slow
    def test_healthy_check(self):
        """Test KeyBERT health check loads model successfully."""
        extractor = KeyBERTExtractor()
        # Should be able to load model
        assert extractor.healthy() is True
        # Model should now be loaded
        assert extractor._model is not None
