"""
Unit tests for extraction graph V2 agents and split quote detection.

Tests the binary search split quote detection functionality.
"""

import pytest
from interaction_finder.resources import ResourcePool
from interaction_finder.extraction_graph_v2.quote_validation import (
    detect_split_quote,
    find_longest_matching_subquote,
)
from interaction_finder.extraction_graph_v2.quote_validation.utilities import (
    find_longest_matching_prefix,
    find_longest_matching_suffix,
)


class TestSplitQuoteDetection:
    """Test split quote detection with binary search algorithm."""

    @pytest.fixture
    def test_document(self):
        """Create a test document with separated text passages."""
        return """
        BRCA1 is a tumor suppressor gene that plays a critical role in DNA repair.
        
        The protein product of BRCA1 functions in homologous recombination repair.
        
        Many other genes are involved in cancer pathways including p53 and APC.
        
        Mutations in BRCA1 are associated with hereditary breast and ovarian cancer.
        
        The BRCA1 gene was discovered in 1994 through linkage analysis studies.
        """

    @pytest.fixture
    def resource_pool_with_document(self, test_document):
        """Create resource pool with test document."""
        pool = ResourcePool()
        resource = pool.add("http://example.com/test", "Test Document", test_document)
        return pool, resource

    def test_detect_split_quote_success(self, resource_pool_with_document):
        """Test successful detection of a split quote."""
        pool, resource = resource_pool_with_document

        # Create a quote that combines two separate passages
        split_quote = "BRCA1 is a tumor suppressor gene Mutations in BRCA1 are associated with hereditary breast"

        result = detect_split_quote(split_quote, resource)

        assert result is not None
        assert "prefix" in result
        assert "suffix" in result
        assert "prefix_quote" in result
        assert "suffix_quote" in result

        # Check that both parts are found in the document
        # Implementation normalizes quotes to lowercase
        assert "brca1 is a tumor suppressor gene" in result["prefix"]
        assert "mutations in brca1 are associated" in result["suffix"]

        # Check that ResourceQuote objects are returned
        assert hasattr(result["prefix_quote"], "spans")
        assert hasattr(result["suffix_quote"], "spans")
        assert len(result["prefix_quote"].spans) > 0
        assert len(result["suffix_quote"].spans) > 0

    def test_detect_split_quote_no_split_needed(self, resource_pool_with_document):
        """Test that valid quotes are not detected as split."""
        pool, resource = resource_pool_with_document

        # Use a quote that exists as-is in the document
        valid_quote = "BRCA1 is a tumor suppressor gene that plays a critical role"

        result = detect_split_quote(valid_quote, resource)

        # Should not detect this as a split quote since it's contiguous
        assert result is None

    def test_detect_split_quote_too_short(self, resource_pool_with_document):
        """Test that quotes too short for splitting are not processed."""
        pool, resource = resource_pool_with_document

        # Quote with fewer than 6 words
        short_quote = "BRCA1 is important"

        result = detect_split_quote(short_quote, resource)

        assert result is None

    def test_detect_split_quote_prefix_only_matches(self, resource_pool_with_document):
        """Test case where prefix matches but suffix doesn't."""
        pool, resource = resource_pool_with_document

        # Combine real text with fake text
        mixed_quote = (
            "BRCA1 is a tumor suppressor gene this text does not exist anywhere"
        )

        result = detect_split_quote(mixed_quote, resource)

        # Should not detect as split since suffix doesn't match
        assert result is None

    def test_detect_split_quote_binary_search_efficiency(
        self, resource_pool_with_document
    ):
        """Test that binary search finds the longest valid prefix."""
        pool, resource = resource_pool_with_document

        # Create a quote where only part of the first sentence matches with second sentence
        split_quote = "BRCA1 is a tumor suppressor Mutations in BRCA1 are associated with hereditary"

        result = detect_split_quote(split_quote, resource)

        assert result is not None
        # The prefix should be the longest matching part
        assert len(result["prefix"].split()) >= 4  # At least "BRCA1 is a tumor"
        # Implementation normalizes quotes to lowercase
        assert "brca1" in result["prefix"]
        assert "mutations in brca1" in result["suffix"]

    def test_detect_split_quote_edge_case_minimum_words(
        self, resource_pool_with_document
    ):
        """Test edge case with exactly 6 words (minimum for processing)."""
        pool, resource = resource_pool_with_document

        # Create exactly 6-word quote that could be split 3+3
        six_word_quote = "BRCA1 is a Mutations in BRCA1"

        result = detect_split_quote(six_word_quote, resource)

        # Depending on the document content, this might or might not split
        # The test verifies the function handles the minimum case without crashing
        assert result is None or isinstance(result, dict)

    def test_detect_split_quote_no_matches_at_all(self, resource_pool_with_document):
        """Test quote with text that doesn't exist in the document."""
        pool, resource = resource_pool_with_document

        fake_quote = "This text does not exist in the document at all never"

        result = detect_split_quote(fake_quote, resource)

        assert result is None


class TestQuoteRecovery:
    """Test quote recovery functions for paraphrasing detection."""

    @pytest.fixture
    def recovery_document(self):
        """Create a test document for quote recovery testing."""
        return """
        This assay directly measures the product of the LAS enzyme. In agreement with previous results we detected two predominant proteins of 65 and 50 kDa, corresponding to liboic acid-bound PDH-E2 and α-KGDH-E2, respectively.
        
        The analysis showed that BRCA1 mutations are strongly associated with hereditary breast cancer. Multiple studies have confirmed this finding.
        
        Additional research on p53 demonstrates its role as a tumor suppressor gene in various cancer types.
        """

    @pytest.fixture
    def recovery_resource(self, recovery_document):
        """Create resource pool for quote recovery testing."""
        pool = ResourcePool()
        resource = pool.add(
            "http://example.com/recovery", "Recovery Test", recovery_document
        )
        return pool, resource

    def test_find_longest_matching_prefix_success(self, recovery_resource):
        """Test successful prefix matching."""
        pool, resource = recovery_resource

        # Quote that has correct beginning but paraphrased ending
        paraphrased = (
            "In agreement with previous results we detected two major proteins"
        )

        result = find_longest_matching_prefix(paraphrased, resource)

        assert result is not None
        assert "In agreement with previous results we detected two" in result
        # Should find the longest matching prefix
        assert len(result.split()) >= 7

    def test_find_longest_matching_suffix_success(self, recovery_resource):
        """Test successful suffix matching."""
        pool, resource = recovery_resource

        # Quote that has correct ending but paraphrased beginning
        paraphrased = "Based on our findings we detected two predominant proteins of 65 and 50 kDa"

        result = find_longest_matching_suffix(paraphrased, resource)

        assert result is not None
        assert "predominant proteins of 65 and 50 kDa" in result
        assert len(result.split()) >= 7

    def test_find_longest_matching_subquote_prefers_longer(self, recovery_resource):
        """Test that subquote finder returns the longer match."""
        pool, resource = recovery_resource

        # Quote where prefix matches more than suffix
        paraphrased = (
            "In agreement with previous results we detected different proteins"
        )

        result = find_longest_matching_subquote(paraphrased, resource)

        assert result is not None
        # Should prefer the longer prefix match
        # Implementation normalizes quotes to lowercase
        assert result.startswith("in agreement with previous results")

    def test_quote_recovery_no_match(self, recovery_resource):
        """Test quote recovery with no significant matches."""
        pool, resource = recovery_resource

        completely_different = "This text does not exist in the document at all"

        result = find_longest_matching_subquote(completely_different, resource)

        assert result is None

    def test_quote_recovery_insufficient_match(self, recovery_resource):
        """Test that small matches don't trigger recovery."""
        pool, resource = recovery_resource

        # Only first two words match - should not be enough
        minimal_match = (
            "In agreement with completely different text that does not exist"
        )

        result = find_longest_matching_prefix(minimal_match, resource)

        # Should find some match but probably small
        if result:
            assert len(result.split()) <= 3  # Very short match

    def test_paraphrasing_detection_example(self, recovery_resource):
        """Test the specific paraphrasing example from the user."""
        pool, resource = recovery_resource

        # This is the paraphrased quote that should be detected
        paraphrased = "Consistent with previous results we detected two predominant proteins of 65 and 50 kDa, corresponding to liboic acid-bound PDH-E2 and α-KGDH-E2, respectively."

        result = find_longest_matching_subquote(paraphrased, resource)

        assert result is not None
        # Should find substantial matching portion
        assert len(result.split()) > 10  # Significant match
        assert "predominant proteins" in result
        assert "corresponding to" in result
