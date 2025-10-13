"""
Test suite for alignment-based quote validation utilities.

Tests the main interface functions that replace the old bag-of-words approach
with sequence alignment-based validation.
"""

import pytest
from unittest.mock import Mock, patch

from src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities import (
    compute_alignment_similarity,
    analyze_quote_alignment,
    find_best_alignment_segments,
    compute_bag_of_words_similarity,
    detect_split_quote,
    find_longest_matching_subquote,
    get_alignment_details,
    classify_quote_error,
    get_correction_suggestions,
    is_quote_high_quality,
)
from src.interaction_finder.extraction_graph_v2.quote_validation.alignment import (
    AlignmentResult,
    ErrorType,
    MatchingBlock,
    CorrectionSuggestion,
)


class TestAlignmentSimilarity:
    """Test alignment-based similarity computation."""

    def test_compute_alignment_similarity_exact_match(self, sample_gene_document):
        """Test similarity for exact quote matches."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 is a tumor suppressor gene"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities._default_aligner"
        ) as mock_aligner:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 6, 0, 6, 6)],
                gaps=[],
                similarity_ratio=1.0,
                contiguous=True,
                error_type=ErrorType.NOT_FOUND,
                quality_metrics={},
                corrections=[],
            )
            mock_aligner.align_quote_to_resource.return_value = mock_result

            similarity = compute_alignment_similarity(quote_text, resource)
            assert similarity == 1.0
            mock_aligner.align_quote_to_resource.assert_called_once_with(
                quote_text, resource
            )

    def test_compute_alignment_similarity_partial_match(self, sample_gene_document):
        """Test similarity for partial quote matches."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor mutations"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities._default_aligner"
        ) as mock_aligner:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 3, 0, 3, 3)],
                gaps=[],
                similarity_ratio=0.75,
                contiguous=True,
                error_type=ErrorType.DELETION,
                quality_metrics={},
                corrections=[],
            )
            mock_aligner.align_quote_to_resource.return_value = mock_result

            similarity = compute_alignment_similarity(quote_text, resource)
            assert similarity == 0.75

    def test_compute_alignment_similarity_no_match(self, sample_gene_document):
        """Test similarity for quotes with no matches."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "completely unrelated content"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities._default_aligner"
        ) as mock_aligner:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[],
                gaps=[],
                similarity_ratio=0.0,
                contiguous=False,
                error_type=ErrorType.NOT_FOUND,
                quality_metrics={},
                corrections=[],
            )
            mock_aligner.align_quote_to_resource.return_value = mock_result

            similarity = compute_alignment_similarity(quote_text, resource)
            assert similarity == 0.0


class TestQuoteAlignment:
    """Test comprehensive quote alignment analysis."""

    def test_analyze_quote_alignment_returns_result(self, sample_gene_document):
        """Test that analyze_quote_alignment returns AlignmentResult."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 is a tumor suppressor"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities._default_aligner"
        ) as mock_aligner:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 5, 0, 5, 5)],
                gaps=[],
                similarity_ratio=1.0,
                contiguous=True,
                error_type=ErrorType.NOT_FOUND,
                quality_metrics={},
                corrections=[],
            )
            mock_aligner.align_quote_to_resource.return_value = mock_result

            result = analyze_quote_alignment(quote_text, resource)
            assert isinstance(result, AlignmentResult)
            assert result.similarity_ratio == 1.0
            assert result.error_type == ErrorType.NOT_FOUND
            mock_aligner.align_quote_to_resource.assert_called_once_with(
                quote_text, resource
            )


class TestAlignmentSegments:
    """Test finding best matching segments using alignment."""

    def test_find_best_alignment_segments_with_corrections(self, sample_gene_document):
        """Test finding segments when corrections are available."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor gene mutations"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities._default_aligner"
        ) as mock_aligner:
            corrections = [
                CorrectionSuggestion(
                    text="BRCA1 is a tumor suppressor gene",
                    confidence=0.9,
                    alignment_score=0.95,
                    contiguity_score=1.0,
                    boundary_quality=0.8,
                    matching_blocks=[],
                    explanation="Added missing word 'is'",
                ),
                CorrectionSuggestion(
                    text="Mutations in BRCA1 are associated",
                    confidence=0.7,
                    alignment_score=0.8,
                    contiguity_score=0.9,
                    boundary_quality=0.7,
                    matching_blocks=[],
                    explanation="Different sentence segment",
                ),
            ]

            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 3, 0, 3, 3)],
                gaps=[],
                similarity_ratio=0.6,
                contiguous=True,
                error_type=ErrorType.DELETION,
                quality_metrics={},
                corrections=corrections,
            )
            mock_aligner.align_quote_to_resource.return_value = mock_result

            segments = find_best_alignment_segments(quote_text, resource)
            assert len(segments) == 2
            assert "BRCA1 is a tumor suppressor gene" in segments
            assert "Mutations in BRCA1 are associated" in segments
            # Should not contain duplicates
            assert len(set(segments)) == len(segments)

    def test_find_best_alignment_segments_from_blocks(self, sample_gene_document):
        """Test finding segments from matching blocks when no corrections."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor"
        doc_words = sample_gene_document.split()

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities._default_aligner"
        ) as mock_aligner:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=doc_words,
                matching_blocks=[
                    MatchingBlock(
                        0, 3, 0, 3, 3
                    ),  # "BRCA1 is a" -> "BRCA1 tumor suppressor"
                    MatchingBlock(1, 3, 5, 7, 2),  # Another block
                ],
                gaps=[],
                similarity_ratio=0.8,
                contiguous=False,
                error_type=ErrorType.REORDERING,
                quality_metrics={},
                corrections=[],
            )
            mock_aligner.align_quote_to_resource.return_value = mock_result

            segments = find_best_alignment_segments(quote_text, resource)
            assert len(segments) == 2  # Should extract from both blocks
            # Each segment should be meaningful length
            for segment in segments:
                assert len(segment.split()) >= 2

    def test_find_best_alignment_segments_empty_result(self, sample_gene_document):
        """Test behavior when no segments can be found."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "completely unrelated content"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities._default_aligner"
        ) as mock_aligner:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[],
                gaps=[],
                similarity_ratio=0.0,
                contiguous=False,
                error_type=ErrorType.NOT_FOUND,
                quality_metrics={},
                corrections=[],
            )
            mock_aligner.align_quote_to_resource.return_value = mock_result

            segments = find_best_alignment_segments(quote_text, resource)
            assert segments == []


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_compute_bag_of_words_similarity_exact_match(self):
        """Test bag-of-words compatibility with exact matches."""
        text1 = "BRCA1 is a tumor suppressor gene"
        text2 = "BRCA1 is a tumor suppressor gene"

        similarity = compute_bag_of_words_similarity(text1, text2)
        assert similarity == 1.0

    def test_compute_bag_of_words_similarity_partial_match(self):
        """Test bag-of-words compatibility with partial matches."""
        text1 = "BRCA1 tumor suppressor"
        text2 = "BRCA1 is a tumor suppressor gene"

        similarity = compute_bag_of_words_similarity(text1, text2)
        assert 0.5 < similarity < 1.0  # Should have reasonable similarity

    def test_compute_bag_of_words_similarity_empty_strings(self):
        """Test bag-of-words compatibility with empty strings."""
        assert compute_bag_of_words_similarity("", "test") == 0.0
        assert compute_bag_of_words_similarity("test", "") == 0.0
        assert compute_bag_of_words_similarity("", "") == 0.0
        assert compute_bag_of_words_similarity("  ", "test") == 0.0

    def test_detect_split_quote_with_split(self, sample_gene_document):
        """Test split quote detection compatibility."""
        resource = Mock()
        resource.text = sample_gene_document
        resource.quote = Mock(side_effect=lambda text: Mock())  # Mock quote creation

        quote_text = "BRCA1 tumor suppressor mutations cancer"
        doc_words = sample_gene_document.split()

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities.analyze_quote_alignment"
        ) as mock_analyze:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=doc_words,
                matching_blocks=[
                    MatchingBlock(0, 2, 0, 2, 2),  # "BRCA1 tumor"
                    MatchingBlock(2, 4, 10, 12, 2),  # "mutations cancer"
                ],
                gaps=[],
                similarity_ratio=0.6,
                contiguous=False,
                error_type=ErrorType.SPLIT_QUOTE,
                quality_metrics={},
                corrections=[],
            )
            mock_analyze.return_value = mock_result

            result = detect_split_quote(quote_text, resource)
            assert result is not None
            assert "prefix" in result
            assert "suffix" in result
            assert "alignment_result" in result
            assert isinstance(result["alignment_result"], AlignmentResult)

    def test_detect_split_quote_no_split(self, sample_gene_document):
        """Test split quote detection when no split detected."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 is a tumor suppressor gene"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities.analyze_quote_alignment"
        ) as mock_analyze:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 6, 0, 6, 6)],
                gaps=[],
                similarity_ratio=1.0,
                contiguous=True,
                error_type=ErrorType.NOT_FOUND,
                quality_metrics={},
                corrections=[],
            )
            mock_analyze.return_value = mock_result

            result = detect_split_quote(quote_text, resource)
            assert result is None

    def test_find_longest_matching_subquote_compatibility(self, sample_gene_document):
        """Test longest matching subquote compatibility."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor gene mutations"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities.find_best_alignment_segments"
        ) as mock_segments:
            mock_segments.return_value = [
                "BRCA1 is a tumor suppressor gene",
                "mutations in BRCA1",
            ]

            result = find_longest_matching_subquote(quote_text, resource)
            assert result == "BRCA1 is a tumor suppressor gene"  # Should return longest

    def test_find_longest_matching_subquote_no_match(self, sample_gene_document):
        """Test longest matching subquote when no matches found."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "completely unrelated content"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities.find_best_alignment_segments"
        ) as mock_segments:
            mock_segments.return_value = []

            result = find_longest_matching_subquote(quote_text, resource)
            assert result is None


class TestEnhancedUtilities:
    """Test enhanced utilities that leverage full alignment power."""

    def test_get_alignment_details(self, sample_gene_document):
        """Test detailed alignment information extraction."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor gene"

        corrections = [
            CorrectionSuggestion(
                text="BRCA1 is a tumor suppressor gene",
                confidence=0.9,
                alignment_score=0.95,
                contiguity_score=1.0,
                boundary_quality=0.8,
                matching_blocks=[],
                explanation="Added missing word",
            )
        ]

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities.analyze_quote_alignment"
        ) as mock_analyze:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 4, 0, 4, 4)],
                gaps=[],
                similarity_ratio=0.8,
                contiguous=True,
                error_type=ErrorType.DELETION,
                quality_metrics={"word_density": 0.85, "boundary_clarity": 0.9},
                corrections=corrections,
            )
            mock_analyze.return_value = mock_result

            details = get_alignment_details(quote_text, resource)

            assert details["similarity_ratio"] == 0.8
            assert details["error_type"] == "deletion"
            assert details["is_contiguous"] is True
            assert (
                details["coverage_ratio"] == 1.0
            )  # 4 matching words out of 4 quote words
            assert details["total_matching_words"] == 4
            assert details["num_matching_blocks"] == 1
            assert details["num_gaps"] == 0
            assert details["quality_metrics"] == {
                "word_density": 0.85,
                "boundary_clarity": 0.9,
            }
            assert (
                details["is_high_quality"] is False
            )  # similarity_ratio 0.8 < 0.85 threshold
            assert details["is_likely_split"] is False  # only 1 block and contiguous
            assert details["corrections_available"] == 1
            assert (
                details["best_correction"] == "BRCA1 is a tumor suppressor gene"
            )  # Correction text not cleaned by get_alignment_details

    def test_classify_quote_error(self, sample_gene_document):
        """Test quote error classification."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor mutations"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities.analyze_quote_alignment"
        ) as mock_analyze:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 3, 0, 3, 3)],
                gaps=[],
                similarity_ratio=0.7,
                contiguous=True,
                error_type=ErrorType.REORDERING,
                quality_metrics={},
                corrections=[],
            )
            mock_analyze.return_value = mock_result

            error_type = classify_quote_error(quote_text, resource)
            assert error_type == "reordering"

    def test_get_correction_suggestions(self, sample_gene_document):
        """Test getting ranked correction suggestions."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor mutations"

        corrections = [
            CorrectionSuggestion(
                text="BRCA1 is a tumor suppressor gene",
                confidence=0.9,
                alignment_score=0.95,
                contiguity_score=1.0,
                boundary_quality=0.8,
                matching_blocks=[],
                explanation="Added missing words for better match",
            ),
            CorrectionSuggestion(
                text="Mutations in BRCA1 are associated",
                confidence=0.7,
                alignment_score=0.8,
                contiguity_score=0.9,
                boundary_quality=0.7,
                matching_blocks=[],
                explanation="Alternative sentence segment",
            ),
            CorrectionSuggestion(
                text="BRCA1 mutations are found",
                confidence=0.6,
                alignment_score=0.75,
                contiguity_score=0.8,
                boundary_quality=0.6,
                matching_blocks=[],
                explanation="Partial match with reordering",
            ),
        ]

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities.analyze_quote_alignment"
        ) as mock_analyze:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 3, 0, 3, 3)],
                gaps=[],
                similarity_ratio=0.6,
                contiguous=True,
                error_type=ErrorType.DELETION,
                quality_metrics={},
                corrections=corrections,
            )
            mock_analyze.return_value = mock_result

            suggestions = get_correction_suggestions(
                quote_text, resource, max_suggestions=2
            )

            assert len(suggestions) == 2
            assert suggestions[0]["text"] == "BRCA1 is a tumor suppressor gene"
            assert suggestions[0]["confidence"] == 0.9
            assert suggestions[0]["alignment_score"] == 0.95
            assert (
                suggestions[0]["explanation"] == "Added missing words for better match"
            )

            assert suggestions[1]["text"] == "Mutations in BRCA1 are associated"
            assert suggestions[1]["confidence"] == 0.7

    def test_is_quote_high_quality(self, sample_gene_document):
        """Test high quality quote detection."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 is a tumor suppressor gene"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities.analyze_quote_alignment"
        ) as mock_analyze:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 6, 0, 6, 6)],
                gaps=[],
                similarity_ratio=1.0,
                contiguous=True,
                error_type=ErrorType.NOT_FOUND,
                quality_metrics={},
                corrections=[],
            )
            mock_analyze.return_value = mock_result

            is_high_quality = is_quote_high_quality(quote_text, resource)
            assert is_high_quality is True

        # Test low quality case
        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities.analyze_quote_alignment"
        ) as mock_analyze:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 2, 0, 2, 2)],
                gaps=[],
                similarity_ratio=0.3,
                contiguous=True,
                error_type=ErrorType.DELETION,
                quality_metrics={},
                corrections=[],
            )
            mock_analyze.return_value = mock_result

            is_high_quality = is_quote_high_quality(quote_text, resource)
            assert is_high_quality is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_quote_text(self, sample_gene_document):
        """Test behavior with empty quote text."""
        resource = Mock()
        resource.text = sample_gene_document

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities._default_aligner"
        ) as mock_aligner:
            mock_result = AlignmentResult(
                quote_words=[],
                doc_words=sample_gene_document.split(),
                matching_blocks=[],
                gaps=[],
                similarity_ratio=0.0,
                contiguous=False,
                error_type=ErrorType.NOT_FOUND,
                quality_metrics={},
                corrections=[],
            )
            mock_aligner.align_quote_to_resource.return_value = mock_result

            similarity = compute_alignment_similarity("", resource)
            assert similarity == 0.0

    def test_empty_resource_text(self):
        """Test behavior with empty resource text."""
        resource = Mock()
        resource.text = ""

        quote_text = "BRCA1 is a tumor suppressor"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities._default_aligner"
        ) as mock_aligner:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=[],
                matching_blocks=[],
                gaps=[],
                similarity_ratio=0.0,
                contiguous=False,
                error_type=ErrorType.NOT_FOUND,
                quality_metrics={},
                corrections=[],
            )
            mock_aligner.align_quote_to_resource.return_value = mock_result

            similarity = compute_alignment_similarity(quote_text, resource)
            assert similarity == 0.0

    def test_very_long_quote_text(self, sample_gene_document):
        """Test behavior with very long quote text."""
        resource = Mock()
        resource.text = sample_gene_document

        # Create a very long quote by repeating content
        quote_text = "BRCA1 is a tumor suppressor gene " * 100

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities._default_aligner"
        ) as mock_aligner:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 6, 0, 6, 6)],
                gaps=[],
                similarity_ratio=0.1,  # Lower due to repetition
                contiguous=True,
                error_type=ErrorType.INSERTION,
                quality_metrics={},
                corrections=[],
            )
            mock_aligner.align_quote_to_resource.return_value = mock_result

            similarity = compute_alignment_similarity(quote_text, resource)
            assert similarity == 0.1

    def test_unicode_and_special_characters(self, sample_gene_document):
        """Test behavior with unicode and special characters."""
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 is a 'tumor suppressor' gene—important for DNA repair"

        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.alignment_utilities._default_aligner"
        ) as mock_aligner:
            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 5, 0, 5, 5)],
                gaps=[],
                similarity_ratio=0.5,
                contiguous=True,
                error_type=ErrorType.INSERTION,
                quality_metrics={},
                corrections=[],
            )
            mock_aligner.align_quote_to_resource.return_value = mock_result

            similarity = compute_alignment_similarity(quote_text, resource)
            assert similarity == 0.5
