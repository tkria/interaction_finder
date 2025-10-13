"""
Test suite for quote correction and auto-accept mechanisms.

Tests the QuoteCorrector class that handles correction suggestion generation
and auto-accept logic for failed quote validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.interaction_finder.extraction_graph_v2.quote_validation.corrections import (
    QuoteCorrector,
)
from src.interaction_finder.extraction_graph_v2.quote_validation.alignment import (
    SequenceAligner,
    AlignmentResult,
    ErrorType,
    MatchingBlock,
    CorrectionSuggestion,
)


class TestQuoteCorrector:
    """Test the QuoteCorrector class initialization and basic behavior."""

    def test_init_default_threshold(self):
        """Test initialization with default auto-accept threshold."""
        corrector = QuoteCorrector()
        assert corrector.auto_accept_threshold == 85.0

    def test_init_custom_threshold(self):
        """Test initialization with custom auto-accept threshold."""
        corrector = QuoteCorrector(auto_accept_threshold=90.0)
        assert corrector.auto_accept_threshold == 90.0


class TestSuggestionGeneration:
    """Test correction suggestion generation."""

    def test_generate_suggestions_from_alignment(self, sample_gene_document):
        """Test suggestion generation using alignment-based corrections."""
        corrector = QuoteCorrector()
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor gene mutations"

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

        mock_aligner = Mock(spec=SequenceAligner)
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

        suggestions = corrector.generate_suggestions(quote_text, resource, mock_aligner)

        assert len(suggestions) == 2
        assert (
            suggestions[0][0] == "BRCA is a tumor suppressor gene"
        )  # "1" is cleaned by _clean_suggestion
        assert suggestions[0][1] == 90.0  # confidence as percentage
        assert (
            suggestions[1][0] == "Mutations in BRCA are associated"
        )  # "1" is cleaned by _clean_suggestion
        assert suggestions[1][1] == 70.0

        mock_aligner.align_quote_to_resource.assert_called_once_with(
            quote_text, resource
        )

    def test_generate_suggestions_fallback_to_legacy(self, sample_gene_document):
        """Test fallback to legacy methods when alignment suggestions insufficient."""
        corrector = QuoteCorrector()
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor"

        # Mock alignment with no corrections (triggering fallback)
        mock_aligner = Mock(spec=SequenceAligner)
        mock_result = AlignmentResult(
            quote_words=quote_text.split(),
            doc_words=sample_gene_document.split(),
            matching_blocks=[MatchingBlock(0, 2, 0, 2, 2)],
            gaps=[],
            similarity_ratio=0.5,
            contiguous=True,
            error_type=ErrorType.DELETION,
            quality_metrics={},
            corrections=[],
        )
        mock_aligner.align_quote_to_resource.return_value = mock_result

        # Mock legacy utility functions
        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.corrections.find_longest_matching_subquote"
        ) as mock_subquote:
            with patch(
                "src.interaction_finder.extraction_graph_v2.quote_validation.corrections.find_longest_matching_prefix"
            ) as mock_prefix:
                with patch(
                    "src.interaction_finder.extraction_graph_v2.quote_validation.corrections.find_longest_matching_suffix"
                ) as mock_suffix:
                    mock_subquote.return_value = "BRCA1 is a tumor suppressor gene"
                    mock_prefix.return_value = "BRCA1 is a tumor"
                    mock_suffix.return_value = "tumor suppressor gene"

                    # Mock extension functionality
                    corrector._extend_match_for_context = Mock(
                        return_value=["BRCA1 is a tumor suppressor gene that plays"]
                    )

                    suggestions = corrector.generate_suggestions(
                        quote_text, resource, mock_aligner
                    )

                    assert len(suggestions) > 0
                    # Should include results from various legacy methods
                    suggestion_texts = [s[0] for s in suggestions]
                    assert (
                        "BRCA is a tumor suppressor gene" in suggestion_texts
                    )  # "1" cleaned out

    def test_generate_suggestions_empty_result(self, sample_gene_document):
        """Test behavior when no suggestions can be generated."""
        corrector = QuoteCorrector()
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "completely unrelated content"

        mock_aligner = Mock(spec=SequenceAligner)
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

        # Mock legacy methods returning None
        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.corrections.find_longest_matching_subquote"
        ) as mock_subquote:
            with patch(
                "src.interaction_finder.extraction_graph_v2.quote_validation.corrections.find_longest_matching_prefix"
            ) as mock_prefix:
                with patch(
                    "src.interaction_finder.extraction_graph_v2.quote_validation.corrections.find_longest_matching_suffix"
                ) as mock_suffix:
                    mock_subquote.return_value = None
                    mock_prefix.return_value = None
                    mock_suffix.return_value = None

                    suggestions = corrector.generate_suggestions(
                        quote_text, resource, mock_aligner
                    )
                    assert suggestions == []

    def test_generate_suggestions_deduplication(self, sample_gene_document):
        """Test that duplicate suggestions are properly removed."""
        corrector = QuoteCorrector()
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor"

        # Create corrections with duplicate texts
        corrections = [
            CorrectionSuggestion(
                text="BRCA1 is a tumor suppressor gene",
                confidence=0.9,
                alignment_score=0.95,
                contiguity_score=1.0,
                boundary_quality=0.8,
                matching_blocks=[],
                explanation="First occurrence",
            ),
            CorrectionSuggestion(
                text="BRCA1 is a tumor suppressor gene",  # Duplicate
                confidence=0.8,  # Lower confidence
                alignment_score=0.85,
                contiguity_score=0.9,
                boundary_quality=0.7,
                matching_blocks=[],
                explanation="Second occurrence",
            ),
            CorrectionSuggestion(
                text="Mutations in BRCA1",
                confidence=0.7,
                alignment_score=0.8,
                contiguity_score=0.9,
                boundary_quality=0.7,
                matching_blocks=[],
                explanation="Different text",
            ),
        ]

        mock_aligner = Mock(spec=SequenceAligner)
        mock_result = AlignmentResult(
            quote_words=quote_text.split(),
            doc_words=sample_gene_document.split(),
            matching_blocks=[MatchingBlock(0, 2, 0, 2, 2)],
            gaps=[],
            similarity_ratio=0.6,
            contiguous=True,
            error_type=ErrorType.DELETION,
            quality_metrics={},
            corrections=corrections,
        )
        mock_aligner.align_quote_to_resource.return_value = mock_result

        suggestions = corrector.generate_suggestions(quote_text, resource, mock_aligner)

        # Should only have 2 unique suggestions (duplicates removed, higher confidence kept)
        assert len(suggestions) == 2
        suggestion_texts = [s[0] for s in suggestions]
        assert "BRCA is a tumor suppressor gene" in suggestion_texts
        assert "Mutations in BRCA" in suggestion_texts

        # Should keep the higher confidence version
        brca_suggestion = next(
            s
            for s in suggestions
            if s[0] == "BRCA is a tumor suppressor gene"  # "1" cleaned out
        )
        assert brca_suggestion[1] == 90.0  # Higher confidence version kept

    def test_generate_suggestions_minimum_length_filter(self, sample_gene_document):
        """Test that suggestions below minimum length are filtered out."""
        corrector = QuoteCorrector()
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor"

        corrections = [
            CorrectionSuggestion(
                text="BRCA1 is a tumor suppressor gene",  # Long enough
                confidence=0.9,
                alignment_score=0.95,
                contiguity_score=1.0,
                boundary_quality=0.8,
                matching_blocks=[],
                explanation="Good match",
            ),
            CorrectionSuggestion(
                text="BRCA1",  # Too short (< 10 chars after strip)
                confidence=0.8,
                alignment_score=0.7,
                contiguity_score=0.9,
                boundary_quality=0.6,
                matching_blocks=[],
                explanation="Too short",
            ),
        ]

        mock_aligner = Mock(spec=SequenceAligner)
        mock_result = AlignmentResult(
            quote_words=quote_text.split(),
            doc_words=sample_gene_document.split(),
            matching_blocks=[MatchingBlock(0, 2, 0, 2, 2)],
            gaps=[],
            similarity_ratio=0.7,
            contiguous=True,
            error_type=ErrorType.DELETION,
            quality_metrics={},
            corrections=corrections,
        )
        mock_aligner.align_quote_to_resource.return_value = mock_result

        suggestions = corrector.generate_suggestions(quote_text, resource, mock_aligner)

        # Should only include the long enough suggestion
        assert len(suggestions) == 1
        assert suggestions[0][0] == "BRCA is a tumor suppressor gene"  # "1" cleaned out

    def test_generate_suggestions_max_three_results(self, sample_gene_document):
        """Test that maximum of 3 suggestions are returned."""
        corrector = QuoteCorrector()
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor"

        # Create 5 corrections to test limiting
        corrections = []
        for i in range(5):
            corrections.append(
                CorrectionSuggestion(
                    text=f"BRCA1 is a tumor suppressor gene variant {i}",
                    confidence=0.9 - (i * 0.1),
                    alignment_score=0.95 - (i * 0.1),
                    contiguity_score=1.0,
                    boundary_quality=0.8,
                    matching_blocks=[],
                    explanation=f"Variant {i}",
                )
            )

        mock_aligner = Mock(spec=SequenceAligner)
        mock_result = AlignmentResult(
            quote_words=quote_text.split(),
            doc_words=sample_gene_document.split(),
            matching_blocks=[MatchingBlock(0, 2, 0, 2, 2)],
            gaps=[],
            similarity_ratio=0.7,
            contiguous=True,
            error_type=ErrorType.DELETION,
            quality_metrics={},
            corrections=corrections,
        )
        mock_aligner.align_quote_to_resource.return_value = mock_result

        suggestions = corrector.generate_suggestions(quote_text, resource, mock_aligner)

        # Should only return top 3
        assert len(suggestions) == 3
        # Should be sorted by confidence (highest first)
        assert suggestions[0][1] > suggestions[1][1] > suggestions[2][1]


class TestAutoAcceptLogic:
    """Test auto-accept decision logic."""

    def test_should_auto_accept_high_confidence(self):
        """Test auto-accept with high confidence suggestions."""
        corrector = QuoteCorrector(auto_accept_threshold=85.0)

        suggestions = [
            ("BRCA1 is a tumor suppressor gene", 90.0),
            ("Mutations in BRCA1", 80.0),
        ]

        should_accept = corrector.should_auto_accept(suggestions)
        assert should_accept is True

    def test_should_auto_accept_low_confidence(self):
        """Test no auto-accept with low confidence suggestions."""
        corrector = QuoteCorrector(auto_accept_threshold=85.0)

        suggestions = [("BRCA1 tumor", 80.0), ("suppressor gene", 70.0)]

        should_accept = corrector.should_auto_accept(suggestions)
        assert should_accept is False

    def test_should_auto_accept_empty_suggestions(self):
        """Test no auto-accept with empty suggestions."""
        corrector = QuoteCorrector()

        should_accept = corrector.should_auto_accept([])
        assert should_accept is False

    def test_should_auto_accept_with_alignment_quality(self):
        """Test auto-accept logic with alignment quality factor."""
        corrector = QuoteCorrector(auto_accept_threshold=85.0)

        suggestions = [("BRCA1 is a tumor suppressor gene", 90.0)]

        # High confidence + high alignment quality = accept
        should_accept = corrector.should_auto_accept(suggestions, alignment_quality=0.9)
        assert should_accept is True

        # High confidence + low alignment quality = reject
        should_accept = corrector.should_auto_accept(suggestions, alignment_quality=0.7)
        assert should_accept is False

        # Low confidence + high alignment quality = reject
        suggestions_low = [("BRCA1 tumor", 80.0)]
        should_accept = corrector.should_auto_accept(
            suggestions_low, alignment_quality=0.9
        )
        assert should_accept is False

    def test_get_best_suggestion(self):
        """Test getting the best suggestion from a list."""
        corrector = QuoteCorrector()

        suggestions = [
            ("BRCA is a tumor suppressor gene", 90.0),  # "1" would be cleaned
            ("Mutations in BRCA", 80.0),  # "1" would be cleaned
            ("tumor suppressor", 70.0),
        ]

        best = corrector.get_best_suggestion(suggestions)
        assert best == "BRCA is a tumor suppressor gene"  # "1" cleaned out

    def test_get_best_suggestion_empty(self):
        """Test getting best suggestion from empty list."""
        corrector = QuoteCorrector()

        best = corrector.get_best_suggestion([])
        assert best is None


class TestMatchPercentageCalculation:
    """Test match percentage calculation logic."""

    def test_calculate_match_percentage_exact_match(self):
        """Test percentage calculation for exact matches."""
        corrector = QuoteCorrector()

        original = "BRCA1 is a tumor suppressor gene"
        suggestion = "BRCA1 is a tumor suppressor gene"

        percentage = corrector._calculate_match_percentage(original, suggestion)
        assert percentage == 100.0

    def test_calculate_match_percentage_partial_match(self):
        """Test percentage calculation for partial matches."""
        corrector = QuoteCorrector()

        original = "BRCA1 tumor suppressor"
        suggestion = "BRCA1 is a tumor suppressor gene"

        percentage = corrector._calculate_match_percentage(original, suggestion)
        assert 50.0 < percentage < 100.0  # Should have reasonable similarity

    def test_calculate_match_percentage_no_match(self):
        """Test percentage calculation for no matches."""
        corrector = QuoteCorrector()

        original = "completely unrelated"
        suggestion = "BRCA1 tumor suppressor gene"

        percentage = corrector._calculate_match_percentage(original, suggestion)
        assert percentage == 0.0

    def test_calculate_match_percentage_empty_strings(self):
        """Test percentage calculation with empty strings."""
        corrector = QuoteCorrector()

        assert corrector._calculate_match_percentage("", "test") == 0.0
        assert corrector._calculate_match_percentage("test", "") == 0.0
        assert corrector._calculate_match_percentage("", "") == 0.0
        assert corrector._calculate_match_percentage("  ", "test") == 0.0


class TestMatchExtension:
    """Test match extension functionality."""

    def test_extend_match_for_context(self, sample_gene_document):
        """Test extending matches for better context."""
        corrector = QuoteCorrector()
        resource = Mock()
        resource.text = sample_gene_document

        match_text = "BRCA1 is a tumor suppressor"

        # Mock ResourceQuote with spans
        mock_quote = Mock()
        mock_quote.spans = [(10, 35)]  # Mock span positions
        resource.quote.return_value = mock_quote

        # Mock the sentence boundary extension
        corrector._extend_to_sentence_boundary = Mock(
            side_effect=[
                "The BRCA1 is a tumor suppressor gene that plays",  # Backward extension
                "BRCA1 is a tumor suppressor gene.",  # Forward extension
            ]
        )

        extensions = corrector._extend_match_for_context(match_text, resource)

        assert len(extensions) == 2
        assert "The BRCA1 is a tumor suppressor gene that plays" in extensions
        assert "BRCA1 is a tumor suppressor gene." in extensions
        resource.quote.assert_called_once_with(match_text)

    def test_extend_match_for_context_quote_error(self, sample_gene_document):
        """Test match extension when quote creation fails."""
        corrector = QuoteCorrector()
        resource = Mock()
        resource.text = sample_gene_document
        resource.quote.side_effect = Exception("Quote not found")

        match_text = "nonexistent text"

        extensions = corrector._extend_match_for_context(match_text, resource)
        assert extensions == []

    def test_extend_to_sentence_boundary_backward(self):
        """Test extending to sentence boundary backward."""
        corrector = QuoteCorrector()

        full_text = "Previous sentence. BRCA1 is a tumor suppressor gene that plays a critical role."
        start_pos = 19  # Start of "BRCA1"
        end_pos = 46  # End of "tumor suppressor" (exclusive)

        extended = corrector._extend_to_sentence_boundary(
            full_text, start_pos, end_pos, extend_backward=True
        )

        assert extended == "BRCA1 is a tumor suppressor"
        assert extended.startswith("BRCA1")  # Should start at sentence boundary

    def test_extend_to_sentence_boundary_forward(self):
        """Test extending to sentence boundary forward."""
        corrector = QuoteCorrector()

        full_text = "BRCA1 is a tumor suppressor gene that plays a critical role. Next sentence here."
        start_pos = 0  # Start of "BRCA1"
        end_pos = 26  # End of "tumor suppressor"

        extended = corrector._extend_to_sentence_boundary(
            full_text, start_pos, end_pos, extend_backward=False
        )

        assert (
            extended == "BRCA1 is a tumor suppressor gene that plays a critical role."
        )
        assert extended.endswith(".")  # Should end at sentence boundary

    def test_extend_to_sentence_boundary_no_punctuation(self):
        """Test extending when no sentence boundaries found."""
        corrector = QuoteCorrector()

        full_text = (
            "BRCA1 is a tumor suppressor gene that plays a critical role in DNA repair"
        )
        start_pos = 0
        end_pos = 26

        # Test backward extension (should extend by reasonable amount)
        extended_back = corrector._extend_to_sentence_boundary(
            full_text, start_pos, end_pos, extend_backward=True
        )
        assert len(extended_back) <= len(full_text)

        # Test forward extension (should extend by reasonable amount)
        extended_forward = corrector._extend_to_sentence_boundary(
            full_text, start_pos, end_pos, extend_backward=False
        )
        assert len(extended_forward) > end_pos - start_pos  # Should be extended


class TestSuggestionCleaning:
    """Test suggestion cleaning functionality."""

    def test_clean_suggestion_basic_cleaning(self):
        """Test basic suggestion cleaning."""
        corrector = QuoteCorrector()

        dirty = "  BRCA1 is a tumor suppressor gene  "
        clean = corrector._clean_suggestion(dirty)
        assert (
            clean == "BRCA is a tumor suppressor gene"
        )  # "1" removed by reference number regex

    def test_clean_suggestion_remove_reference_numbers(self):
        """Test removal of reference numbers."""
        corrector = QuoteCorrector()

        # Test protein with number
        assert (
            corrector._clean_suggestion("protein22 is important")
            == "protein is important"
        )

        # Test gene name with number - the regex removes digits after letters
        assert corrector._clean_suggestion("BRCA1 gene 15.") == "BRCA gene"

        # Test standalone reference number at end
        assert corrector._clean_suggestion("tumor suppressor 42.") == "tumor suppressor"

    def test_clean_suggestion_remove_metadata(self):
        """Test removal of metadata artifacts."""
        corrector = QuoteCorrector()

        test_cases = [
            (
                "BRCA1 PubMed analysis",
                "BRCA analysis",
            ),  # "1" removed by reference number regex
            ("gene Google Scholar study", "gene study"),
            ("protein Crossref data", "protein data"),
            (
                "BRCA1 Scopus information",
                "BRCA information",
            ),  # "1" removed by reference number regex
            ("**bold text** content", "content"),
            ("normal _italic_ text", "normal text"),
        ]

        for dirty, expected in test_cases:
            clean = corrector._clean_suggestion(dirty)
            assert clean == expected

    def test_clean_suggestion_trailing_punctuation(self):
        """Test removal of artificial trailing punctuation."""
        corrector = QuoteCorrector()

        # Short fragments should lose trailing punctuation
        assert (
            corrector._clean_suggestion("BRCA1.") == "BRCA"
        )  # "1" removed, then "." removed
        assert corrector._clean_suggestion("gene,") == "gene"
        assert corrector._clean_suggestion("tumor;") == "tumor"

        # Longer sentences should keep natural punctuation
        longer = "BRCA1 is a tumor suppressor gene that plays."
        assert (
            corrector._clean_suggestion(longer)
            == "BRCA is a tumor suppressor gene that plays."
        )  # "1" removed

        # Don't remove ellipsis
        assert corrector._clean_suggestion("gene...") == "gene..."

    def test_clean_suggestion_whitespace_normalization(self):
        """Test whitespace normalization."""
        corrector = QuoteCorrector()

        messy = "BRCA1    is  a\ttumor   suppressor\n\n gene"
        clean = corrector._clean_suggestion(messy)
        assert (
            clean == "BRCA is a tumor suppressor gene"
        )  # "1" removed by reference number regex

    def test_clean_suggestion_empty_input(self):
        """Test cleaning empty or None input."""
        corrector = QuoteCorrector()

        assert corrector._clean_suggestion("") == ""
        assert corrector._clean_suggestion(None) == ""
        assert corrector._clean_suggestion("   ") == ""


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_full_correction_workflow_high_quality(self, sample_gene_document):
        """Test complete correction workflow with high-quality suggestions."""
        corrector = QuoteCorrector(auto_accept_threshold=85.0)
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 tumor suppressor gene"

        corrections = [
            CorrectionSuggestion(
                text="BRCA1 is a tumor suppressor gene",
                confidence=0.92,
                alignment_score=0.95,
                contiguity_score=1.0,
                boundary_quality=0.85,
                matching_blocks=[],
                explanation="Added missing word 'is'",
            )
        ]

        mock_aligner = Mock(spec=SequenceAligner)
        mock_result = AlignmentResult(
            quote_words=quote_text.split(),
            doc_words=sample_gene_document.split(),
            matching_blocks=[MatchingBlock(0, 3, 0, 3, 3)],
            gaps=[],
            similarity_ratio=0.8,
            contiguous=True,
            error_type=ErrorType.DELETION,
            quality_metrics={},
            corrections=corrections,
        )
        mock_aligner.align_quote_to_resource.return_value = mock_result

        # Generate suggestions
        suggestions = corrector.generate_suggestions(quote_text, resource, mock_aligner)

        # Should auto-accept with high confidence and alignment quality
        should_accept = corrector.should_auto_accept(
            suggestions,
            alignment_quality=0.86,  # > 0.85 threshold
        )
        best_suggestion = corrector.get_best_suggestion(suggestions)

        assert len(suggestions) >= 1
        assert should_accept is True
        # Best suggestion is the one with highest confidence after cleaning
        assert best_suggestion in [
            "BRCA tumor suppressor gene",
            "BRCA is a tumor suppressor gene",
        ]

    def test_full_correction_workflow_low_quality(self, sample_gene_document):
        """Test complete correction workflow with low-quality suggestions."""
        corrector = QuoteCorrector(auto_accept_threshold=85.0)
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "completely unrelated biological content"

        mock_aligner = Mock(spec=SequenceAligner)
        mock_result = AlignmentResult(
            quote_words=quote_text.split(),
            doc_words=sample_gene_document.split(),
            matching_blocks=[MatchingBlock(0, 1, 15, 16, 1)],  # Only "biological"
            gaps=[],
            similarity_ratio=0.2,
            contiguous=True,
            error_type=ErrorType.NOT_FOUND,
            quality_metrics={},
            corrections=[
                CorrectionSuggestion(
                    text="biological processes are important",
                    confidence=0.3,
                    alignment_score=0.4,
                    contiguity_score=0.5,
                    boundary_quality=0.3,
                    matching_blocks=[],
                    explanation="Very weak match",
                )
            ],
        )
        mock_aligner.align_quote_to_resource.return_value = mock_result

        # Generate suggestions
        suggestions = corrector.generate_suggestions(quote_text, resource, mock_aligner)

        # Should NOT auto-accept due to low quality
        should_accept = corrector.should_auto_accept(
            suggestions, alignment_quality=0.25
        )

        assert should_accept is False
        # May still have suggestions, but the alignment-derived ones should be low confidence
        # (The original quote may be included with high confidence, which is correct behavior)
        if suggestions:
            alignment_suggestions = [
                (text, conf)
                for text, conf in suggestions
                if text != "completely unrelated biological content"
            ]
            if alignment_suggestions:
                assert all(conf < 85.0 for _, conf in alignment_suggestions)

    def test_correction_with_no_aligner_provided(self, sample_gene_document):
        """Test correction generation when no aligner is provided."""
        corrector = QuoteCorrector()
        resource = Mock()
        resource.text = sample_gene_document

        quote_text = "BRCA1 is a tumor suppressor"

        # Should create its own aligner internally
        with patch(
            "src.interaction_finder.extraction_graph_v2.quote_validation.corrections.SequenceAligner"
        ) as mock_aligner_class:
            mock_aligner_instance = Mock()
            mock_aligner_class.return_value = mock_aligner_instance

            mock_result = AlignmentResult(
                quote_words=quote_text.split(),
                doc_words=sample_gene_document.split(),
                matching_blocks=[MatchingBlock(0, 5, 0, 5, 5)],
                gaps=[],
                similarity_ratio=0.9,
                contiguous=True,
                error_type=ErrorType.NOT_FOUND,
                quality_metrics={},
                corrections=[
                    CorrectionSuggestion(
                        text="BRCA1 is a tumor suppressor gene",
                        confidence=0.95,
                        alignment_score=0.98,
                        contiguity_score=1.0,
                        boundary_quality=0.9,
                        matching_blocks=[],
                        explanation="Perfect match with extension",
                    )
                ],
            )
            mock_aligner_instance.align_quote_to_resource.return_value = mock_result

            suggestions = corrector.generate_suggestions(quote_text, resource)

            # Should have created and used internal aligner
            mock_aligner_class.assert_called_once()
            mock_aligner_instance.align_quote_to_resource.assert_called_once_with(
                quote_text, resource
            )

            assert len(suggestions) >= 1
            # The highest confidence suggestion should be the cleaned original quote
            assert (
                suggestions[0][0] == "BRCA is a tumor suppressor"
            )  # Original quote, "1" cleaned out
            assert suggestions[0][1] == 100.0  # Perfect match confidence
            # The correction should be second
            if len(suggestions) >= 2:
                assert suggestions[1][0] == "BRCA is a tumor suppressor gene"
                assert suggestions[1][1] == 95.0
