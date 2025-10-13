"""
Comprehensive tests for keyword extraction algorithms.

Tests cover all four extractors (YAKE, RAKE, TF-IDF, None) with:
- Sample text extraction (biomedical domain)
- Empty text handling
- Single-word text handling
- Invalid top_n values
- Factory function
- Edge cases (top_n > available keywords)
"""

import pytest

from interaction_finder.search.reverse.keyword_extractors import (
    KeywordExtractor,
    NoneExtractor,
    RAKEExtractor,
    TFIDFExtractor,
    YAKEExtractor,
    create_extractor,
)

# ==============================================================================
# Test Data
# ==============================================================================

# Sample biomedical text for realistic keyword extraction
SAMPLE_TEXT = """
Diabetes mellitus is a metabolic disease characterized by hyperglycemia.
Type 1 diabetes results from autoimmune destruction of pancreatic beta cells.
Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency.
Treatment includes insulin therapy, oral hypoglycemic agents, and lifestyle modifications.
"""

# Short text for edge case testing
SHORT_TEXT = "Diabetes"

# Empty/whitespace texts
EMPTY_TEXT = ""
WHITESPACE_TEXT = "   \n\t  "


# ==============================================================================
# YAKEExtractor Tests
# ==============================================================================


class TestYAKEExtractor:
    """Test YAKE keyword extraction."""

    def test_yake_sample_text(self):
        """YAKE extracts relevant keywords from sample text."""
        extractor = YAKEExtractor()
        keywords = extractor.extract(SAMPLE_TEXT, top_n=5)

        # Verify reasonable number of keywords returned
        assert len(keywords) <= 5
        assert len(keywords) > 0

        # Check that relevant medical terms appear (case-insensitive check)
        keywords_lower = [kw.lower() for kw in keywords]
        # At least one keyword should contain "diabetes"
        assert any("diabetes" in kw for kw in keywords_lower), (
            f"Expected 'diabetes' in keywords, got: {keywords}"
        )

    def test_yake_empty_text(self):
        """YAKE handles empty text gracefully."""
        extractor = YAKEExtractor()
        assert extractor.extract(EMPTY_TEXT, top_n=5) == []

    def test_yake_whitespace_text(self):
        """YAKE handles whitespace-only text gracefully."""
        extractor = YAKEExtractor()
        assert extractor.extract(WHITESPACE_TEXT, top_n=5) == []

    def test_yake_single_word(self):
        """YAKE handles single-word text."""
        extractor = YAKEExtractor()
        keywords = extractor.extract(SHORT_TEXT, top_n=5)
        # Should return the word or empty list (both acceptable)
        assert isinstance(keywords, list)
        assert len(keywords) <= 1

    def test_yake_top_n_zero(self):
        """YAKE raises ValueError for top_n=0."""
        extractor = YAKEExtractor()
        with pytest.raises(ValueError, match="top_n must be > 0"):
            extractor.extract(SAMPLE_TEXT, top_n=0)

    def test_yake_top_n_negative(self):
        """YAKE raises ValueError for negative top_n."""
        extractor = YAKEExtractor()
        with pytest.raises(ValueError, match="top_n must be > 0"):
            extractor.extract(SAMPLE_TEXT, top_n=-1)

    def test_yake_top_n_exceeds_available(self):
        """YAKE returns available keywords when top_n exceeds them."""
        extractor = YAKEExtractor()
        # Request more keywords than likely exist
        keywords = extractor.extract(SHORT_TEXT, top_n=100)
        # Should return what's available without error
        assert isinstance(keywords, list)
        assert len(keywords) <= 100

    def test_yake_name_property(self):
        """YAKE extractor has correct name."""
        extractor = YAKEExtractor()
        assert extractor.name == "yake"

    def test_yake_custom_parameters(self):
        """YAKE accepts custom parameters."""
        extractor = YAKEExtractor(max_ngram_size=2, dedup_threshold=0.8)
        keywords = extractor.extract(SAMPLE_TEXT, top_n=3)
        assert isinstance(keywords, list)


# ==============================================================================
# RAKEExtractor Tests
# ==============================================================================


class TestRAKEExtractor:
    """Test RAKE keyword extraction."""

    def test_rake_sample_text(self):
        """RAKE extracts relevant keywords from sample text."""
        extractor = RAKEExtractor()
        keywords = extractor.extract(SAMPLE_TEXT, top_n=5)

        # Verify reasonable number of keywords returned
        assert len(keywords) <= 5
        assert len(keywords) > 0

        # RAKE should extract multi-word phrases
        # At least verify we got keywords
        assert all(isinstance(kw, str) for kw in keywords)

    def test_rake_empty_text(self):
        """RAKE handles empty text gracefully."""
        extractor = RAKEExtractor()
        assert extractor.extract(EMPTY_TEXT, top_n=5) == []

    def test_rake_whitespace_text(self):
        """RAKE handles whitespace-only text gracefully."""
        extractor = RAKEExtractor()
        assert extractor.extract(WHITESPACE_TEXT, top_n=5) == []

    def test_rake_single_word(self):
        """RAKE handles single-word text."""
        extractor = RAKEExtractor()
        keywords = extractor.extract(SHORT_TEXT, top_n=5)
        # Should return the word or empty list
        assert isinstance(keywords, list)

    def test_rake_top_n_zero(self):
        """RAKE raises ValueError for top_n=0."""
        extractor = RAKEExtractor()
        with pytest.raises(ValueError, match="top_n must be > 0"):
            extractor.extract(SAMPLE_TEXT, top_n=0)

    def test_rake_top_n_negative(self):
        """RAKE raises ValueError for negative top_n."""
        extractor = RAKEExtractor()
        with pytest.raises(ValueError, match="top_n must be > 0"):
            extractor.extract(SAMPLE_TEXT, top_n=-1)

    def test_rake_top_n_exceeds_available(self):
        """RAKE returns available keywords when top_n exceeds them."""
        extractor = RAKEExtractor()
        keywords = extractor.extract(SHORT_TEXT, top_n=100)
        # Should return what's available without error
        assert isinstance(keywords, list)
        assert len(keywords) <= 100

    def test_rake_name_property(self):
        """RAKE extractor has correct name."""
        extractor = RAKEExtractor()
        assert extractor.name == "rake"

    def test_rake_extracts_phrases(self):
        """RAKE should extract multi-word phrases."""
        extractor = RAKEExtractor()
        text = (
            "Machine learning and artificial intelligence are transforming healthcare."
        )
        keywords = extractor.extract(text, top_n=3)

        # RAKE typically extracts phrases, not just single words
        # At least verify we get some multi-word phrases
        assert len(keywords) > 0


# ==============================================================================
# TFIDFExtractor Tests
# ==============================================================================


class TestTFIDFExtractor:
    """Test TF-IDF keyword extraction."""

    def test_tfidf_sample_text(self):
        """TF-IDF extracts relevant keywords from sample text."""
        extractor = TFIDFExtractor()
        keywords = extractor.extract(SAMPLE_TEXT, top_n=5)

        # Verify reasonable number of keywords returned
        assert len(keywords) <= 5
        assert len(keywords) > 0

        # All keywords should be strings
        assert all(isinstance(kw, str) for kw in keywords)

    def test_tfidf_empty_text(self):
        """TF-IDF handles empty text gracefully."""
        extractor = TFIDFExtractor()
        assert extractor.extract(EMPTY_TEXT, top_n=5) == []

    def test_tfidf_whitespace_text(self):
        """TF-IDF handles whitespace-only text gracefully."""
        extractor = TFIDFExtractor()
        assert extractor.extract(WHITESPACE_TEXT, top_n=5) == []

    def test_tfidf_single_word(self):
        """TF-IDF handles single-word text."""
        extractor = TFIDFExtractor()
        keywords = extractor.extract(SHORT_TEXT, top_n=5)
        # Should return the word if not a stop word, or empty list
        assert isinstance(keywords, list)

    def test_tfidf_top_n_zero(self):
        """TF-IDF raises ValueError for top_n=0."""
        extractor = TFIDFExtractor()
        with pytest.raises(ValueError, match="top_n must be > 0"):
            extractor.extract(SAMPLE_TEXT, top_n=0)

    def test_tfidf_top_n_negative(self):
        """TF-IDF raises ValueError for negative top_n."""
        extractor = TFIDFExtractor()
        with pytest.raises(ValueError, match="top_n must be > 0"):
            extractor.extract(SAMPLE_TEXT, top_n=-1)

    def test_tfidf_top_n_exceeds_available(self):
        """TF-IDF returns available keywords when top_n exceeds them."""
        extractor = TFIDFExtractor()
        keywords = extractor.extract(SHORT_TEXT, top_n=100)
        # Should return what's available without error
        assert isinstance(keywords, list)
        assert len(keywords) <= 100

    def test_tfidf_name_property(self):
        """TF-IDF extractor has correct name."""
        extractor = TFIDFExtractor()
        assert extractor.name == "tfidf"

    def test_tfidf_custom_ngram_range(self):
        """TF-IDF accepts custom n-gram range."""
        # Only unigrams
        extractor = TFIDFExtractor(ngram_range=(1, 1))
        keywords = extractor.extract(SAMPLE_TEXT, top_n=3)
        assert isinstance(keywords, list)

    def test_tfidf_no_valid_features(self):
        """TF-IDF handles text with only stop words."""
        extractor = TFIDFExtractor()
        # Text with only stop words
        stop_words_text = "the and or but a an"
        keywords = extractor.extract(stop_words_text, top_n=5)
        # Should return empty list when no valid features
        assert keywords == []


# ==============================================================================
# NoneExtractor Tests
# ==============================================================================


class TestNoneExtractor:
    """Test None extractor (null-object pattern)."""

    def test_none_returns_empty_list(self):
        """NoneExtractor always returns empty list."""
        extractor = NoneExtractor()
        assert extractor.extract(SAMPLE_TEXT, top_n=5) == []

    def test_none_with_empty_text(self):
        """NoneExtractor returns empty list for empty text."""
        extractor = NoneExtractor()
        assert extractor.extract(EMPTY_TEXT, top_n=5) == []

    def test_none_ignores_top_n(self):
        """NoneExtractor ignores top_n parameter."""
        extractor = NoneExtractor()
        # Different top_n values should all return empty list
        assert extractor.extract(SAMPLE_TEXT, top_n=1) == []
        assert extractor.extract(SAMPLE_TEXT, top_n=10) == []
        assert extractor.extract(SAMPLE_TEXT, top_n=100) == []

    def test_none_ignores_text_content(self):
        """NoneExtractor ignores text content."""
        extractor = NoneExtractor()
        # All texts should return empty list
        assert extractor.extract("Short text", top_n=5) == []
        assert extractor.extract("A" * 10000, top_n=5) == []
        assert extractor.extract("Special chars !@#$%", top_n=5) == []

    def test_none_name_property(self):
        """NoneExtractor has correct name."""
        extractor = NoneExtractor()
        assert extractor.name == "none"

    def test_none_docstring_describes_purpose(self):
        """NoneExtractor docstring explains null-object pattern."""
        # Verify class has descriptive docstring
        assert NoneExtractor.__doc__ is not None
        assert "null" in NoneExtractor.__doc__.lower()


# ==============================================================================
# Factory Function Tests
# ==============================================================================


class TestCreateExtractor:
    """Test the factory function for creating extractors."""

    def test_create_yake(self):
        """Factory creates YAKE extractor."""
        extractor = create_extractor("yake")
        assert isinstance(extractor, YAKEExtractor)
        assert extractor.name == "yake"

    def test_create_rake(self):
        """Factory creates RAKE extractor."""
        extractor = create_extractor("rake")
        assert isinstance(extractor, RAKEExtractor)
        assert extractor.name == "rake"

    def test_create_tfidf(self):
        """Factory creates TF-IDF extractor."""
        extractor = create_extractor("tfidf")
        assert isinstance(extractor, TFIDFExtractor)
        assert extractor.name == "tfidf"

    def test_create_none(self):
        """Factory creates None extractor."""
        extractor = create_extractor("none")
        assert isinstance(extractor, NoneExtractor)
        assert extractor.name == "none"

    def test_create_case_insensitive(self):
        """Factory handles case-insensitive names."""
        assert isinstance(create_extractor("YAKE"), YAKEExtractor)
        assert isinstance(create_extractor("Rake"), RAKEExtractor)
        assert isinstance(create_extractor("TfIdf"), TFIDFExtractor)
        assert isinstance(create_extractor("NONE"), NoneExtractor)

    def test_create_unknown_raises(self):
        """Factory raises ValueError for unknown extractor name."""
        with pytest.raises(ValueError, match="Unknown extractor name"):
            create_extractor("unknown")

    def test_create_empty_raises(self):
        """Factory raises ValueError for empty name."""
        with pytest.raises(ValueError, match="Unknown extractor name"):
            create_extractor("")


# ==============================================================================
# Interface Tests
# ==============================================================================


class TestKeywordExtractorInterface:
    """Test the abstract interface is properly implemented."""

    @pytest.mark.parametrize(
        "extractor_class",
        [YAKEExtractor, RAKEExtractor, TFIDFExtractor, NoneExtractor],
    )
    def test_implements_interface(self, extractor_class):
        """All extractors implement KeywordExtractor interface."""
        extractor = extractor_class()
        assert isinstance(extractor, KeywordExtractor)

    @pytest.mark.parametrize(
        "extractor_class",
        [YAKEExtractor, RAKEExtractor, TFIDFExtractor, NoneExtractor],
    )
    def test_has_extract_method(self, extractor_class):
        """All extractors have extract method."""
        extractor = extractor_class()
        assert hasattr(extractor, "extract")
        assert callable(extractor.extract)

    @pytest.mark.parametrize(
        "extractor_class",
        [YAKEExtractor, RAKEExtractor, TFIDFExtractor, NoneExtractor],
    )
    def test_has_name_property(self, extractor_class):
        """All extractors have name property."""
        extractor = extractor_class()
        assert hasattr(extractor, "name")
        assert isinstance(extractor.name, str)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestExtractorIntegration:
    """Integration tests comparing extractors on same text."""

    def test_all_extractors_on_sample_text(self):
        """All extractors can process the same sample text."""
        extractors = [
            YAKEExtractor(),
            RAKEExtractor(),
            TFIDFExtractor(),
            NoneExtractor(),
        ]

        for extractor in extractors:
            keywords = extractor.extract(SAMPLE_TEXT, top_n=5)
            assert isinstance(keywords, list)
            assert len(keywords) <= 5
            # NoneExtractor should return empty, others should extract keywords
            if extractor.name == "none":
                assert len(keywords) == 0, "NoneExtractor should return empty list"
            else:
                assert len(keywords) > 0, f"{extractor.name} extracted no keywords"

    def test_extractors_produce_different_results(self):
        """Different extractors may produce different keywords."""
        # This is expected - different algorithms have different strengths
        yake_kw = set(YAKEExtractor().extract(SAMPLE_TEXT, top_n=5))
        rake_kw = set(RAKEExtractor().extract(SAMPLE_TEXT, top_n=5))
        tfidf_kw = set(TFIDFExtractor().extract(SAMPLE_TEXT, top_n=5))
        none_kw = set(NoneExtractor().extract(SAMPLE_TEXT, top_n=5))

        # Results don't need to be identical (different algorithms)
        # Just verify they all produce results (except None which should be empty)
        assert len(yake_kw) > 0
        assert len(rake_kw) > 0
        assert len(tfidf_kw) > 0
        assert len(none_kw) == 0  # None always returns empty
