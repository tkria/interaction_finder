"""
Tests for NoneExtractor.

NoneExtractor is a null-object pattern extractor that always returns empty
keyword lists, enabling full-content LLM query generation without keyword
extraction. This is used with query_constructor="llm" for the none+llm approach.
"""

import pytest

from interaction_finder.search.reverse.keyword_extractors import (
    NoneExtractor,
    create_extractor,
)


def test_none_extractor_returns_empty_list():
    """Test NoneExtractor returns empty list for any input."""
    extractor = NoneExtractor()
    result = extractor.extract("Some long text with many keywords", top_n=10)
    assert result == []


def test_none_extractor_ignores_text_content():
    """Test NoneExtractor ignores text content."""
    extractor = NoneExtractor()

    # Should return empty regardless of text
    assert extractor.extract("BRCA1 breast cancer mutation", top_n=5) == []
    assert extractor.extract("A" * 10000, top_n=20) == []
    assert extractor.extract("", top_n=5) == []
    assert extractor.extract("   ", top_n=5) == []


def test_none_extractor_ignores_top_n():
    """Test NoneExtractor ignores top_n parameter."""
    extractor = NoneExtractor()
    text = "Gene disease protein pathway interaction biomarker"

    # Should return empty regardless of top_n
    assert extractor.extract(text, top_n=1) == []
    assert extractor.extract(text, top_n=5) == []
    assert extractor.extract(text, top_n=10) == []
    assert extractor.extract(text, top_n=100) == []


def test_none_extractor_name():
    """Test NoneExtractor name property."""
    extractor = NoneExtractor()
    assert extractor.name == "none"


def test_none_extractor_with_unicode():
    """Test NoneExtractor handles Unicode text."""
    extractor = NoneExtractor()
    unicode_text = "α-synuclein β-amyloid γ-secretase 神经退行性疾病"
    result = extractor.extract(unicode_text, top_n=5)
    assert result == []


def test_none_extractor_with_special_characters():
    """Test NoneExtractor handles special characters."""
    extractor = NoneExtractor()
    special_text = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`\n\t\r"
    result = extractor.extract(special_text, top_n=5)
    assert result == []


def test_none_extractor_factory():
    """Test NoneExtractor can be created via factory function."""
    extractor = create_extractor("none")
    assert isinstance(extractor, NoneExtractor)
    assert extractor.name == "none"
    assert extractor.extract("test text", top_n=5) == []


def test_none_extractor_factory_case_insensitive():
    """Test factory accepts case-insensitive 'none' argument."""
    for name in ["none", "None", "NONE", "NoNe"]:
        extractor = create_extractor(name)
        assert isinstance(extractor, NoneExtractor)


def test_none_extractor_with_negative_top_n():
    """Test NoneExtractor handles negative top_n (just returns empty)."""
    extractor = NoneExtractor()
    # NoneExtractor doesn't validate top_n, just returns []
    result = extractor.extract("test text", top_n=-5)
    assert result == []


def test_none_extractor_with_zero_top_n():
    """Test NoneExtractor handles zero top_n (just returns empty)."""
    extractor = NoneExtractor()
    result = extractor.extract("test text", top_n=0)
    assert result == []


def test_none_extractor_multiple_calls():
    """Test NoneExtractor is stateless across multiple calls."""
    extractor = NoneExtractor()

    # Multiple calls should all return empty
    for i in range(10):
        result = extractor.extract(f"Text iteration {i}", top_n=5)
        assert result == []


def test_none_extractor_very_long_text():
    """Test NoneExtractor handles very long text efficiently."""
    extractor = NoneExtractor()
    # 1MB of text
    very_long_text = "A" * 1_000_000

    # Should return instantly (not actually processing the text)
    result = extractor.extract(very_long_text, top_n=10)
    assert result == []


def test_none_extractor_use_case_with_llm():
    """Test NoneExtractor use case: signaling full-content LLM mode."""
    # This is the intended use pattern: none extractor + LLM constructor
    extractor = create_extractor("none")

    # Resource content
    content = """
    FBLN4 mutations cause vascular calcification and arterial stiffness.
    The protein plays a critical role in elastic fiber assembly.
    """

    # Extract keywords (returns empty, signaling "use full content")
    keywords = extractor.extract(content, top_n=10)
    assert keywords == []

    # In real usage, the empty keywords signal the constructor to use
    # full content instead of assembling a query from keywords


def test_none_extractor_docstring_example():
    """Test the example from NoneExtractor docstring."""
    extractor = NoneExtractor()
    result = extractor.extract("Some long paper content...", top_n=5)
    assert result == []


def test_none_extractor_integration_with_query_construction():
    """Test NoneExtractor integration pattern with query construction."""
    extractor = NoneExtractor()

    # Simulate query generation workflow
    resources = [
        "BRCA1 is associated with breast cancer susceptibility",
        "TP53 mutations are common in various cancers",
        "PARP inhibitors show efficacy in BRCA-mutant tumors",
    ]

    all_keywords = []
    for resource in resources:
        keywords = extractor.extract(resource, top_n=5)
        all_keywords.extend(keywords)

    # Should accumulate no keywords
    assert all_keywords == []

    # This signals the constructor: "No keywords available, use full content"
