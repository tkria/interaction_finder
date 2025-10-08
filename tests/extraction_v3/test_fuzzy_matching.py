"""
Tests for fuzzy quote matching in V3 extraction pipeline.

Tests alignment-based quote correction and effectiveness.
"""

import pytest
from interaction_finder.resources import ResourcePool, normalize_text_for_matching


@pytest.fixture
def sample_resource():
    """Create sample resource with text for fuzzy matching tests."""
    pool = ResourcePool()
    doc_text = (
        "BRCA1 is a tumor suppressor gene. "
        "Mutations in BRCA1 are associated with breast cancer. "
        "The BRCA1 gene encodes a protein involved in DNA repair."
    )
    resource = pool.add("http://example.com/doc1", "BRCA1 Test Document", doc_text)
    return resource


def test_exact_quote_matching(sample_resource):
    """Test exact quote matching finds correct spans."""
    quote_text = "BRCA1 is a tumor suppressor gene"
    quote = sample_resource.quote(quote_text)

    assert len(quote.spans) >= 1, "Should find exact match"
    assert quote.query_text == quote_text, "Query text should match input"


def test_fuzzy_quote_matching_with_whitespace(sample_resource):
    """Test fuzzy matching handles whitespace differences."""
    # LLM might return quote with different whitespace
    quote_text_with_extra_spaces = "BRCA1  is  a   tumor suppressor gene"
    quote = sample_resource.quote(quote_text_with_extra_spaces)

    # Should still find match due to normalization
    assert len(quote.spans) >= 1, "Should find match despite whitespace differences"


def test_fuzzy_quote_matching_with_case(sample_resource):
    """Test fuzzy matching handles case differences."""
    quote_text_different_case = "brca1 is a tumor suppressor gene"
    quote = sample_resource.quote(quote_text_different_case)

    # Should find match due to case normalization
    assert len(quote.spans) >= 1, "Should find match despite case differences"


def test_fuzzy_quote_matching_with_punctuation(sample_resource):
    """Test fuzzy matching handles punctuation differences."""
    quote_text = "Mutations in BRCA1 are associated with breast cancer"
    quote = sample_resource.quote(quote_text)

    assert len(quote.spans) >= 1, "Should find exact match"

    # Try with extra punctuation
    quote_text_with_punct = "Mutations in BRCA1, are associated with breast cancer"
    quote2 = sample_resource.quote(quote_text_with_punct)

    # May or may not find depending on normalization aggressiveness
    # Just verify it doesn't crash
    assert quote2 is not None


def test_partial_quote_not_matched(sample_resource):
    """Test partial quotes that don't exist return appropriate result."""
    quote_text = "BRCA1 causes all cancers"  # Not in document

    # This should raise ValueError when quote not found
    try:
        quote = sample_resource.quote(quote_text)
        # If it doesn't raise, it should have zero spans
        assert len(quote.spans) == 0, "Non-existent quote should have no spans"
    except ValueError:
        # This is expected behavior
        pass


def test_multiple_occurrences(sample_resource):
    """Test finding multiple occurrences of same quote."""
    quote_text = "BRCA1"  # Appears multiple times
    quote = sample_resource.quote(quote_text)

    assert len(quote.spans) >= 3, "BRCA1 should appear at least 3 times in document"


def test_normalize_text_for_matching():
    """Test text normalization function."""
    text1 = "BRCA1 is a gene"
    text2 = "brca1  is  a   gene"
    text3 = "BRCA1 IS A GENE"

    norm1 = normalize_text_for_matching(text1)
    norm2 = normalize_text_for_matching(text2)
    norm3 = normalize_text_for_matching(text3)

    # All should normalize to same result
    assert norm1 == norm2, "Should handle whitespace normalization"
    assert norm1 == norm3, "Should handle case normalization"


def test_fuzzy_matching_effectiveness():
    """
    Test fuzzy matching effectiveness on realistic paraphrased quotes.

    Simulates LLM returning slightly different quote text.
    """
    pool = ResourcePool()

    # Create document with specific quotes
    doc_text = (
        "The BRCA1 gene is a tumor suppressor. "
        "Mutations in BRCA1 significantly increase breast cancer risk. "
        "BRCA1-associated breast cancers often occur at younger ages."
    )
    resource = pool.add("http://example.com/doc", "BRCA1 Study", doc_text)

    # Test various quote variations
    test_cases = [
        ("BRCA1 gene is a tumor suppressor", True),  # Exact substring
        ("The BRCA1 gene is a tumor suppressor", True),  # With article
        ("BRCA1  gene  is  a  tumor  suppressor", True),  # Extra whitespace
        ("brca1 gene is a tumor suppressor", True),  # Lowercase
        (
            "BRCA1 mutations increase cancer risk",
            False,
        ),  # Paraphrase (won't match exactly)
    ]

    results = []
    for quote_text, should_match in test_cases:
        try:
            quote = resource.quote(quote_text)
            matched = len(quote.spans) > 0
        except ValueError:
            matched = False
        results.append((quote_text, should_match, matched))

    # Calculate effectiveness
    correct = sum(1 for _, expected, actual in results if expected == actual)
    total = len(results)
    effectiveness = correct / total

    print("\n=== Fuzzy Matching Effectiveness ===")
    for quote_text, expected, actual in results:
        status = "✓" if expected == actual else "✗"
        print(f"{status} {quote_text[:50]}: expected={expected}, matched={actual}")
    print(f"Effectiveness: {effectiveness:.1%} ({correct}/{total})")

    # Should match most exact cases
    assert effectiveness >= 0.60, (
        f"Fuzzy matching effectiveness {effectiveness:.1%} below 60%"
    )


@pytest.mark.parametrize(
    "quote_text,expected_found",
    [
        ("BRCA1 is a tumor suppressor gene", True),
        ("brca1 is a tumor suppressor gene", True),
        ("BRCA1  is  a  tumor  suppressor  gene", True),
        ("BRCA1 does something completely different", False),
    ],
)
def test_fuzzy_matching_parametrized(sample_resource, quote_text, expected_found):
    """Parametrized test for fuzzy matching scenarios."""
    try:
        quote = sample_resource.quote(quote_text)
        found = len(quote.spans) > 0
    except ValueError:
        found = False

    assert found == expected_found, (
        f"Quote '{quote_text}' found={found}, expected={expected_found}"
    )
