"""
Sequence alignment-based quote validation system.

This module provides the core infrastructure for validating quotes using
word-level sequence alignment with difflib.SequenceMatcher as the foundation.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ...resources import Resource


class ErrorType(Enum):
    """Types of quote validation errors based on alignment patterns."""

    SPLIT_QUOTE = "split_quote"  # Non-contiguous matching blocks
    WORD_SUBSTITUTION = "word_substitution"  # High similarity, specific mismatches
    INSERTION = "insertion"  # Extra words in quote
    DELETION = "deletion"  # Missing words in quote
    PARAPHRASE = "paraphrase"  # Low similarity, some structure preserved
    REORDERING = "reordering"  # Same words, wrong order
    NOT_FOUND = "not_found"  # No significant alignment


@dataclass
class MatchingBlock:
    """Represents a contiguous matching block between quote and document."""

    quote_start: int  # Start position in quote words
    quote_end: int  # End position in quote words
    doc_start: int  # Start position in document words
    doc_end: int  # End position in document words
    length: int  # Number of matching words

    @property
    def quote_span(self) -> Tuple[int, int]:
        """Get quote span as (start, end) tuple."""
        return (self.quote_start, self.quote_end)

    @property
    def doc_span(self) -> Tuple[int, int]:
        """Get document span as (start, end) tuple."""
        return (self.doc_start, self.doc_end)


@dataclass
class Gap:
    """Represents a gap in alignment (insertion or deletion)."""

    position: int  # Position where gap occurs
    gap_type: str  # "insertion" or "deletion"
    words: List[str]  # Words involved in the gap


@dataclass
class CorrectionSuggestion:
    """A suggested correction based on alignment analysis."""

    text: str  # The suggested quote text
    confidence: float  # Confidence score (0.0 to 1.0)
    alignment_score: float  # Quality of sequence alignment
    contiguity_score: float  # How contiguous the matching blocks are
    boundary_quality: float  # How well it aligns with natural boundaries
    matching_blocks: List[MatchingBlock]  # Supporting alignment blocks
    explanation: str  # Human-readable explanation of the correction


@dataclass
class AlignmentResult:
    """Complete analysis of quote alignment with document text."""

    quote_words: List[str]  # Tokenized quote words
    doc_words: List[str]  # Tokenized document words
    similarity_ratio: float  # Overall similarity (0.0 to 1.0)
    matching_blocks: List[MatchingBlock]  # All matching blocks
    gaps: List[Gap]  # Insertions and deletions
    contiguous: bool  # Whether matches are contiguous
    error_type: ErrorType  # Classified error type
    quality_metrics: Dict[str, float]  # Additional quality measures
    corrections: List[CorrectionSuggestion]  # Suggested corrections

    @property
    def total_matching_words(self) -> int:
        """Total number of matching words across all blocks."""
        return sum(block.length for block in self.matching_blocks)

    @property
    def coverage_ratio(self) -> float:
        """Ratio of quote words that have matches."""
        if not self.quote_words:
            return 0.0
        return self.total_matching_words / len(self.quote_words)

    @property
    def is_high_quality(self) -> bool:
        """Whether this alignment represents a high-quality match."""
        return (
            self.similarity_ratio > 0.85
            and self.coverage_ratio > 0.8
            and self.contiguous
        )

    @property
    def is_likely_split(self) -> bool:
        """Whether this alignment indicates a split quote."""
        return (
            len(self.matching_blocks) > 1
            and self.coverage_ratio > 0.7
            and not self.contiguous
        )


class SequenceAligner:
    """Core sequence alignment engine for quote validation."""

    def __init__(
        self,
        min_similarity_threshold: float = 0.1,  # Lower threshold for initial matching
        min_block_length: int = 2,
        split_detection_threshold: float = 0.7,
    ):
        """
        Initialize sequence aligner.

        Args:
            min_similarity_threshold: Minimum similarity to consider a match
            min_block_length: Minimum length of matching blocks to consider
            split_detection_threshold: Coverage threshold for split detection
        """
        self.min_similarity_threshold = min_similarity_threshold
        self.min_block_length = min_block_length
        self.split_detection_threshold = split_detection_threshold

    def align_quote_to_resource(
        self, quote_text: str, resource: "Resource"
    ) -> AlignmentResult:
        """
        Perform comprehensive alignment analysis of quote against resource.

        Args:
            quote_text: The quote text to analyze
            resource: Resource to align against

        Returns:
            Complete alignment analysis result
        """
        # Tokenize both texts
        quote_words = self._tokenize_for_alignment(quote_text)
        doc_words = self._tokenize_for_alignment(resource.text)

        # Find all potential alignment positions
        best_alignments = self._find_best_alignment_positions(quote_words, doc_words)

        if not best_alignments:
            return self._create_no_match_result(quote_words, doc_words)

        # Analyze the best alignment
        best_alignment = best_alignments[0]

        # Create matching blocks from alignment
        matching_blocks = self._extract_matching_blocks(
            best_alignment, quote_words, doc_words
        )

        # Analyze gaps and errors
        gaps = self._analyze_gaps(matching_blocks, quote_words, doc_words)

        # Classify error type
        error_type = self._classify_error_type(
            matching_blocks, gaps, quote_words, doc_words
        )

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            matching_blocks, quote_words, doc_words
        )

        # Generate correction suggestions
        corrections = self._generate_corrections_from_alignment(
            matching_blocks, quote_words, doc_words, resource
        )

        return AlignmentResult(
            quote_words=quote_words,
            doc_words=doc_words,
            similarity_ratio=best_alignment["similarity"],
            matching_blocks=matching_blocks,
            gaps=gaps,
            contiguous=self._is_contiguous(matching_blocks),
            error_type=error_type,
            quality_metrics=quality_metrics,
            corrections=corrections,
        )

    def _tokenize_for_alignment(self, text: str) -> List[str]:
        """Tokenize text into words for alignment analysis."""
        if not text:
            return []

        # Normalize whitespace and convert to lowercase for matching
        normalized = re.sub(r"\s+", " ", text.strip().lower())

        # Split into words, removing empty strings
        words = [word for word in normalized.split() if word]

        return words

    def _find_best_alignment_positions(
        self, quote_words: List[str], doc_words: List[str]
    ) -> List[Dict]:
        """Find all potential alignment positions and rank by quality."""
        if not quote_words or not doc_words:
            return []

        # Use SequenceMatcher to find alignment
        matcher = SequenceMatcher(None, quote_words, doc_words)
        similarity = matcher.ratio()

        if similarity < self.min_similarity_threshold:
            return []

        return [
            {
                "matcher": matcher,
                "similarity": similarity,
                "quote_words": quote_words,
                "doc_words": doc_words,
            }
        ]

    def _extract_matching_blocks(
        self, alignment: Dict, quote_words: List[str], doc_words: List[str]
    ) -> List[MatchingBlock]:
        """Extract matching blocks from SequenceMatcher result."""
        matcher = alignment["matcher"]
        blocks = []

        for block in matcher.get_matching_blocks():
            if block.size >= self.min_block_length:
                blocks.append(
                    MatchingBlock(
                        quote_start=block.a,
                        quote_end=block.a + block.size,
                        doc_start=block.b,
                        doc_end=block.b + block.size,
                        length=block.size,
                    )
                )

        return blocks

    def _analyze_gaps(
        self,
        matching_blocks: List[MatchingBlock],
        quote_words: List[str],
        doc_words: List[str],
    ) -> List[Gap]:
        """Analyze gaps between matching blocks to identify insertions/deletions."""
        gaps = []

        if not matching_blocks:
            return gaps

        # Check for gaps between consecutive matching blocks
        for i in range(len(matching_blocks) - 1):
            current_block = matching_blocks[i]
            next_block = matching_blocks[i + 1]

            # Gap in quote (deletion)
            if next_block.quote_start > current_block.quote_end:
                gap_words = quote_words[
                    current_block.quote_end : next_block.quote_start
                ]
                gaps.append(
                    Gap(
                        position=current_block.quote_end,
                        gap_type="deletion",
                        words=gap_words,
                    )
                )

            # Gap in document (insertion)
            if next_block.doc_start > current_block.doc_end:
                # This represents words that should be in the quote but aren't
                gap_words = doc_words[current_block.doc_end : next_block.doc_start]
                gaps.append(
                    Gap(
                        position=current_block.doc_end,
                        gap_type="insertion",
                        words=gap_words,
                    )
                )

        return gaps

    def _classify_error_type(
        self,
        matching_blocks: List[MatchingBlock],
        gaps: List[Gap],
        quote_words: List[str],
        doc_words: List[str],
    ) -> ErrorType:
        """Classify the type of alignment error based on patterns."""
        if not matching_blocks:
            return ErrorType.NOT_FOUND

        total_matches = sum(block.length for block in matching_blocks)
        coverage = total_matches / len(quote_words) if quote_words else 0.0

        # High coverage but non-contiguous = split quote
        if coverage > self.split_detection_threshold and len(matching_blocks) > 1:
            return ErrorType.SPLIT_QUOTE

        # High coverage, contiguous, with specific gaps = substitution
        if coverage > 0.8 and len(matching_blocks) == 1 and gaps:
            return ErrorType.WORD_SUBSTITUTION

        # Analyze gap types
        if gaps:
            deletion_gaps = [g for g in gaps if g.gap_type == "deletion"]
            insertion_gaps = [g for g in gaps if g.gap_type == "insertion"]

            if deletion_gaps and not insertion_gaps:
                return ErrorType.DELETION
            elif insertion_gaps and not deletion_gaps:
                return ErrorType.INSERTION

        # Low coverage but some structure preserved
        if 0.3 <= coverage < 0.7:
            return ErrorType.PARAPHRASE

        return ErrorType.NOT_FOUND

    def _calculate_quality_metrics(
        self,
        matching_blocks: List[MatchingBlock],
        quote_words: List[str],
        doc_words: List[str],
    ) -> Dict[str, float]:
        """Calculate additional quality metrics for alignment."""
        if not matching_blocks or not quote_words:
            return {
                "contiguity": 0.0,
                "coverage": 0.0,
                "precision": 0.0,
                "order_preservation": 0.0,
            }

        # Contiguity: how close together the matching blocks are
        contiguity = self._calculate_contiguity_score(matching_blocks)

        # Coverage: what fraction of quote words have matches
        total_matches = sum(block.length for block in matching_blocks)
        coverage = total_matches / len(quote_words)

        # Precision: what fraction of matches are actually correct
        precision = coverage  # Simplified for now

        # Order preservation: how well word order is maintained
        order_preservation = self._calculate_order_preservation(matching_blocks)

        return {
            "contiguity": contiguity,
            "coverage": coverage,
            "precision": precision,
            "order_preservation": order_preservation,
        }

    def _calculate_contiguity_score(
        self, matching_blocks: List[MatchingBlock]
    ) -> float:
        """Calculate how contiguous the matching blocks are."""
        if len(matching_blocks) <= 1:
            return 1.0

        # Calculate gaps between blocks
        total_gap = 0
        total_span = 0

        for i in range(len(matching_blocks) - 1):
            current = matching_blocks[i]
            next_block = matching_blocks[i + 1]

            # Gap size in quote positions
            quote_gap = next_block.quote_start - current.quote_end
            total_gap += quote_gap

        # Total span from first to last block
        first_block = matching_blocks[0]
        last_block = matching_blocks[-1]
        total_span = last_block.quote_end - first_block.quote_start

        if total_span == 0:
            return 1.0

        # Contiguity is inverse of gap ratio
        gap_ratio = total_gap / total_span
        return max(0.0, 1.0 - gap_ratio)

    def _calculate_order_preservation(
        self, matching_blocks: List[MatchingBlock]
    ) -> float:
        """Calculate how well word order is preserved across blocks."""
        if len(matching_blocks) <= 1:
            return 1.0

        # Check if blocks are in order
        in_order = True
        for i in range(len(matching_blocks) - 1):
            current = matching_blocks[i]
            next_block = matching_blocks[i + 1]

            # Both quote and doc positions should increase together
            if (
                next_block.quote_start <= current.quote_start
                or next_block.doc_start <= current.doc_start
            ):
                in_order = False
                break

        return 1.0 if in_order else 0.5  # Simplified scoring

    def _is_contiguous(self, matching_blocks: List[MatchingBlock]) -> bool:
        """Check if matching blocks form a contiguous sequence."""
        if len(matching_blocks) <= 1:
            return True

        # Check if blocks are adjacent in both quote and doc
        for i in range(len(matching_blocks) - 1):
            current = matching_blocks[i]
            next_block = matching_blocks[i + 1]

            # Blocks must be adjacent (no gaps)
            if (
                next_block.quote_start != current.quote_end
                or next_block.doc_start != current.doc_end
            ):
                return False

        return True

    def _generate_corrections_from_alignment(
        self,
        matching_blocks: List[MatchingBlock],
        quote_words: List[str],
        doc_words: List[str],
        resource: "Resource",
    ) -> List[CorrectionSuggestion]:
        """Generate correction suggestions based on alignment analysis."""
        corrections = []

        if not matching_blocks:
            return corrections

        # For each matching block, try to extend to natural boundaries
        for block in matching_blocks:
            extended_suggestions = self._extend_block_to_boundaries(
                block, doc_words, resource
            )
            corrections.extend(extended_suggestions)

        # Sort by quality and return top suggestions
        corrections.sort(key=lambda c: c.confidence, reverse=True)
        return corrections[:3]  # Return top 3

    def _extend_block_to_boundaries(
        self, block: MatchingBlock, doc_words: List[str], resource: "Resource"
    ) -> List[CorrectionSuggestion]:
        """Extend a matching block to natural sentence boundaries."""
        suggestions = []

        # Get the original matching text
        block_text = " ".join(doc_words[block.doc_start : block.doc_end])

        # Try extending backward to sentence start
        extended_back = self._extend_to_sentence_start(block, doc_words, resource)
        if extended_back:
            suggestions.append(extended_back)

        # Try extending forward to sentence end
        extended_forward = self._extend_to_sentence_end(block, doc_words, resource)
        if extended_forward:
            suggestions.append(extended_forward)

        # Try extending both directions
        extended_both = self._extend_both_directions(block, doc_words, resource)
        if extended_both:
            suggestions.append(extended_both)

        return suggestions

    def _extend_to_sentence_start(
        self, block: MatchingBlock, doc_words: List[str], resource: "Resource"
    ) -> Optional[CorrectionSuggestion]:
        """Extend matching block backward to sentence start."""
        # Find sentence boundary by looking for punctuation
        start_pos = block.doc_start

        # Look backward for sentence start (after punctuation)
        while start_pos > 0:
            word = doc_words[start_pos - 1]
            if any(punct in word for punct in ".!?"):
                break
            start_pos -= 1

            # Don't extend too far
            if block.doc_start - start_pos > 20:
                break

        if start_pos < block.doc_start:
            extended_text = " ".join(doc_words[start_pos : block.doc_end])

            # Calculate confidence based on extension quality
            extension_length = block.doc_start - start_pos
            confidence = max(0.5, 1.0 - (extension_length / 20))  # Decrease with length

            return CorrectionSuggestion(
                text=extended_text,
                confidence=confidence,
                alignment_score=0.8,  # Simplified
                contiguity_score=1.0,
                boundary_quality=0.9,
                matching_blocks=[block],
                explanation=f"Extended {extension_length} words backward to sentence start",
            )

        return None

    def _extend_to_sentence_end(
        self, block: MatchingBlock, doc_words: List[str], resource: "Resource"
    ) -> Optional[CorrectionSuggestion]:
        """Extend matching block forward to sentence end."""
        # Find sentence boundary by looking for punctuation
        end_pos = block.doc_end

        # Look forward for sentence end
        while end_pos < len(doc_words):
            word = doc_words[end_pos]
            if any(punct in word for punct in ".!?"):
                end_pos += 1  # Include the punctuation
                break
            end_pos += 1

            # Don't extend too far
            if end_pos - block.doc_end > 20:
                break

        if end_pos > block.doc_end:
            extended_text = " ".join(doc_words[block.doc_start : end_pos])

            # Calculate confidence
            extension_length = end_pos - block.doc_end
            confidence = max(0.5, 1.0 - (extension_length / 20))

            return CorrectionSuggestion(
                text=extended_text,
                confidence=confidence,
                alignment_score=0.8,
                contiguity_score=1.0,
                boundary_quality=0.9,
                matching_blocks=[block],
                explanation=f"Extended {extension_length} words forward to sentence end",
            )

        return None

    def _extend_both_directions(
        self, block: MatchingBlock, doc_words: List[str], resource: "Resource"
    ) -> Optional[CorrectionSuggestion]:
        """Extend matching block in both directions to sentence boundaries."""
        # Combine the logic from both extension methods
        start_pos = block.doc_start
        end_pos = block.doc_end

        # Extend backward
        while start_pos > 0:
            word = doc_words[start_pos - 1]
            if any(punct in word for punct in ".!?"):
                break
            start_pos -= 1
            if block.doc_start - start_pos > 15:  # Shorter limit for both directions
                break

        # Extend forward
        while end_pos < len(doc_words):
            word = doc_words[end_pos]
            if any(punct in word for punct in ".!?"):
                end_pos += 1
                break
            end_pos += 1
            if end_pos - block.doc_end > 15:
                break

        if start_pos < block.doc_start or end_pos > block.doc_end:
            extended_text = " ".join(doc_words[start_pos:end_pos])

            # Calculate confidence based on total extension
            total_extension = (block.doc_start - start_pos) + (end_pos - block.doc_end)
            confidence = max(0.6, 1.0 - (total_extension / 30))

            return CorrectionSuggestion(
                text=extended_text,
                confidence=confidence,
                alignment_score=0.85,
                contiguity_score=1.0,
                boundary_quality=0.95,
                matching_blocks=[block],
                explanation=f"Extended to complete sentence boundaries",
            )

        return None

    def _create_no_match_result(
        self, quote_words: List[str], doc_words: List[str]
    ) -> AlignmentResult:
        """Create alignment result for cases with no significant matches."""
        return AlignmentResult(
            quote_words=quote_words,
            doc_words=doc_words,
            similarity_ratio=0.0,
            matching_blocks=[],
            gaps=[],
            contiguous=False,
            error_type=ErrorType.NOT_FOUND,
            quality_metrics={
                "contiguity": 0.0,
                "coverage": 0.0,
                "precision": 0.0,
                "order_preservation": 0.0,
            },
            corrections=[],
        )
