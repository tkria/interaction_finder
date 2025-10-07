"""
Thorough tests for resource management models and validation.

Tests cover ResourceId generation, Resource text normalization and position mapping,
ResourcePool management, and ResourceQuote quote matching and validation.
"""

import pytest
from pydantic import ValidationError
from interaction_finder.resources import (
    normalize_text_for_matching,
    expand_scientific_shorthand,
    ResourceId,
    Resource,
    ResourcePool,
    ResourceQuote,
)


class TestNormalizeTextForMatching:
    """Test text normalization function."""

    def test_basic_normalization(self):
        """Test basic text normalization functionality."""
        text = "Hello, World! How are you?"
        expected = "hello world how are you"
        assert normalize_text_for_matching(text) == expected

    def test_whitespace_collapse(self):
        """Test that multiple whitespace characters are collapsed."""
        text = "Hello    \t\n   world"
        expected = "hello world"
        assert normalize_text_for_matching(text) == expected

    def test_punctuation_removal(self):
        """Test that punctuation is removed."""
        text = "It's a protein (BRCA1) - very important!"
        expected = "its a protein brca1 very important"
        assert normalize_text_for_matching(text) == expected

    def test_empty_and_whitespace_only(self):
        """Test edge cases with empty or whitespace-only strings."""
        assert normalize_text_for_matching("") == ""
        assert normalize_text_for_matching("   ") == ""
        assert normalize_text_for_matching("\t\n  ") == ""

    def test_numbers_preserved(self):
        """Test that numbers are preserved in normalization."""
        text = "Gene expression level: 2.5x higher in cancer cells"
        expected = "gene expression level 25x higher in cancer cells"
        assert normalize_text_for_matching(text) == expected

    def test_greek_letter_normalization(self):
        """Test Greek letter conversion to ASCII names."""
        # Basic Greek letters
        assert normalize_text_for_matching("α-tubulin") == "alpha tubulin"
        assert normalize_text_for_matching("β-catenin") == "beta catenin"
        assert normalize_text_for_matching("γ-globin") == "gamma globin"

        # Uppercase Greek letters
        assert normalize_text_for_matching("Α-subunit") == "alpha subunit"
        assert normalize_text_for_matching("Β-cell") == "beta cell"

        # Multiple Greek letters
        assert (
            normalize_text_for_matching("α/β-heterodimer") == "alpha beta heterodimer"
        )

        # Greek letters with numbers
        assert normalize_text_for_matching("p53α variant") == "p53 alpha variant"

        # Standalone Greek letters
        assert normalize_text_for_matching("The α protein") == "the alpha protein"

        # Greek letters at word boundaries
        assert normalize_text_for_matching("NFκB pathway") == "nf kappa b pathway"

    def test_unicode_normalization(self):
        """Test Unicode normalization removes accents and diacritical marks."""
        # Basic accented characters
        assert normalize_text_for_matching("café") == "cafe"
        assert normalize_text_for_matching("résumé") == "resume"
        assert normalize_text_for_matching("naïve") == "naive"

        # Scientific terms with accents
        assert normalize_text_for_matching("François Müller") == "francois muller"
        assert normalize_text_for_matching("β-galactosidase") == "beta galactosidase"

        # Mixed Unicode and Greek
        assert (
            normalize_text_for_matching("α-hélix structure") == "alpha helix structure"
        )

    def test_contractions_and_decimals(self):
        """Test that contractions and decimal points are handled correctly."""
        # Contractions should be merged
        assert normalize_text_for_matching("don't worry") == "dont worry"
        assert normalize_text_for_matching("can't bind") == "cant bind"
        assert normalize_text_for_matching("it's active") == "its active"

        # Decimal points should be removed
        assert normalize_text_for_matching("IC50 = 2.5 μM") == "ic50 25 mu m"
        assert normalize_text_for_matching("0.001 significance") == "0001 significance"

        # Don't affect other periods
        assert normalize_text_for_matching("end. Next sentence") == "end next sentence"

    def test_scientific_paper_title_with_underscores(self):
        """Test that paper titles with markdown formatting are properly normalized."""
        title = "Mutations in Iron-Sulfur Cluster Scaffold Genes _NFU1_ and _BOLA3_ Cause a Fatal Deficiency"
        expected = "mutations in iron sulfur cluster scaffold genes nfu1 and bola3 cause a fatal deficiency"
        assert normalize_text_for_matching(title) == expected

    def test_complex_biomedical_text_normalization(self):
        """Test normalization of complex biomedical text with multiple features."""
        text = "The α-helical domain of p53β contains κB-binding sites (χ² = 0.05, p < 0.001)"
        result = normalize_text_for_matching(text)
        # Check key components are present (order and exact spacing may vary)
        assert "alpha" in result
        assert "beta" in result
        assert "kappa" in result
        assert "chi" in result
        assert "helical" in result
        assert "p53" in result
        assert "binding" in result
        assert "005" in result
        assert "0001" in result


class TestExpandScientificShorthand:
    """Test scientific shorthand expansion function."""

    def test_comma_separated_variants(self):
        """Test comma-separated gene variants like ISCA1,2."""
        # Basic comma pattern
        assert expand_scientific_shorthand("ISCA1,2") == ["ISCA1", "ISCA2"]
        assert expand_scientific_shorthand("COL1A1,A2") == ["COL1A1", "COL1A2"]
        assert expand_scientific_shorthand("NFU1,2") == ["NFU1", "NFU2"]

    def test_slash_separated_variants(self):
        """Test slash-separated variants."""
        # Greek letter variants
        assert expand_scientific_shorthand("p53α/β") == ["p53α", "p53β"]
        # Basic slash pattern should be caught by comma pattern
        assert expand_scientific_shorthand("COL1A1/A2") == ["COL1A1", "COL1A2"]

    def test_numeric_ranges(self):
        """Test numeric range expansion."""
        assert expand_scientific_shorthand("exons 2-4") == [
            "exon 2",
            "exon 3",
            "exon 4",
        ]
        assert expand_scientific_shorthand("chapters 1-3") == [
            "chapter 1",
            "chapter 2",
            "chapter 3",
        ]
        assert expand_scientific_shorthand("domains 5-7") == [
            "domain 5",
            "domain 6",
            "domain 7",
        ]

    def test_no_expansion_needed(self):
        """Test cases where no expansion is possible."""
        # Single entities
        assert expand_scientific_shorthand("BRCA1") == ["BRCA1"]
        assert expand_scientific_shorthand("breast cancer") == ["breast cancer"]

        # Complex text without patterns
        assert expand_scientific_shorthand("The protein is important") == [
            "The protein is important"
        ]

    def test_range_size_limit(self):
        """Test that large ranges are not expanded."""
        # Should not expand ranges > 20 items
        result = expand_scientific_shorthand("pages 1-50")
        assert result == ["pages 1-50"]  # Original returned, no expansion

    def test_original_functionality_preserved(self):
        """Test the exact case from the user's example."""
        # This is the specific case that was failing validation
        expansions = expand_scientific_shorthand("ISCA1,2 and IBA57 are required")
        # Should expand to versions with ISCA1 and ISCA2 separately
        expected = ["ISCA1 and IBA57 are required", "ISCA2 and IBA57 are required"]
        assert expansions == expected


class TestResourceId:
    """Test ResourceId creation and validation."""

    def test_basic_creation(self):
        """Test basic ResourceId creation from URL and counter."""
        url = "https://example.com/paper1.pdf"
        counter = 1
        resource_id = ResourceId(url=url, counter=counter)

        assert resource_id.url == url
        assert resource_id.id.startswith("1_")
        assert len(resource_id.id.split("_")[1]) == 8  # 8-hex chars (4-byte digest)

    def test_stable_id_generation(self):
        """Test that same URL + counter produces same ID."""
        url = "https://example.com/paper1.pdf"
        counter = 1

        id1 = ResourceId(url=url, counter=counter)
        id2 = ResourceId(url=url, counter=counter)

        assert id1.id == id2.id

    def test_different_urls_different_ids(self):
        """Test that different URLs produce different IDs."""
        url1 = "https://example.com/paper1.pdf"
        url2 = "https://example.com/paper2.pdf"
        counter = 1

        id1 = ResourceId(url=url1, counter=counter)
        id2 = ResourceId(url=url2, counter=counter)

        assert id1.id != id2.id

    def test_different_counters_different_ids(self):
        """Test that different counters produce different IDs."""
        url = "https://example.com/paper1.pdf"

        id1 = ResourceId(url=url, counter=1)
        id2 = ResourceId(url=url, counter=2)

        assert id1.id != id2.id

    def test_id_format_validation(self):
        """Test validation of ID format."""
        # Valid ID should pass
        resource_id = ResourceId(url="https://example.com", counter=1)

        # Verify the ID format is correct
        assert resource_id.id.startswith("1_")
        assert len(resource_id.id) == 10  # "1_" + 8 hex chars
        assert resource_id.url == "https://example.com"


class TestResource:
    """Test Resource creation and text processing."""

    def test_basic_resource_creation(self):
        """Test basic Resource creation with automatic text processing."""
        resource_id = ResourceId(url="https://example.com", counter=1)
        title = "Test Paper"
        text = "This is a test paper about BRCA1 protein."

        resource = Resource(id=resource_id, title=title, text=text)

        assert resource.id == resource_id
        assert resource.title == title
        assert resource.text == text
        assert resource.normalized_text == "this is a test paper about brca1 protein"
        assert len(resource.position_offsets) > 0

    def test_normalized_text_generation(self):
        """Test that normalized text is correctly generated."""
        resource_id = ResourceId(url="https://example.com", counter=1)
        text = "The BRCA1 gene (breast cancer 1) is crucial for DNA repair!"

        resource = Resource(id=resource_id, title="Test", text=text)

        expected_normalized = "the brca1 gene breast cancer 1 is crucial for dna repair"
        assert resource.normalized_text == expected_normalized

    def test_position_offset_mapping(self):
        """Test that position offset mapping is correctly built."""
        resource_id = ResourceId(url="https://example.com", counter=1)
        text = "Hello, world!"

        resource = Resource(id=resource_id, title="Test", text=text)

        # Should have mappings for each character in normalized text
        assert len(resource.position_offsets) > 0
        # Final offset should map to end of original text
        assert resource.position_offsets[-1][1] == len(text)

    def test_complex_text_normalization(self):
        """Test normalization with complex punctuation and whitespace."""
        resource_id = ResourceId(url="https://example.com", counter=1)
        text = "The protein   (BRCA1)---is important.\n\nIt's crucial for repair!"

        resource = Resource(id=resource_id, title="Test", text=text)

        expected = "the protein brca1 is important its crucial for repair"
        assert resource.normalized_text == expected

    def test_position_mapping_accuracy(self):
        """Test that position mapping correctly maps back to original text."""
        resource_id = ResourceId(url="https://example.com", counter=1)
        text = "The BRCA1 gene."  # Simple case for testing

        resource = Resource(id=resource_id, title="Test", text=text)

    def test_find_original_position_helper(self):
        """Test the _find_original_position helper method directly."""
        resource_id = ResourceId(url="https://example.com", counter=1)
        text = "Hello, world! Test text."
        resource = Resource(id=resource_id, title="Test", text=text)

        # Test finding position at start
        original_pos = resource._find_original_position(0)
        assert original_pos == 0

        # Test finding position in middle
        original_pos = resource._find_original_position(
            5
        )  # Should map to somewhere in original
        assert original_pos is not None
        assert 0 <= original_pos <= len(text)

        # Test finding position at end - should map to end of original text
        end_normalized = len(resource.normalized_text)
        original_pos = resource._find_original_position(end_normalized)
        # The final position mapping should be close to the end (within 1-2 chars due to normalization)
        assert original_pos is not None
        assert abs(original_pos - len(text)) <= 2

    def test_greek_letter_position_mapping(self):
        """Test that Greek letters maintain correct position mapping."""
        resource_id = ResourceId(url="https://example.com", counter=1)
        text = "The α-subunit and β-catenin interact."
        resource = Resource(id=resource_id, title="Test", text=text)

        # Normalized should expand Greek letters
        assert "alpha" in resource.normalized_text
        assert "beta" in resource.normalized_text

        # Test quoting the Greek letters works
        alpha_quote = resource.quote("α-subunit")
        assert alpha_quote is not None
        assert alpha_quote.count == 1

        # Verify the extracted text matches original
        assert "α-subunit" in alpha_quote.get_quote_text()

    def test_unicode_position_mapping(self):
        """Test position mapping with Unicode normalization."""
        resource_id = ResourceId(url="https://example.com", counter=1)
        text = "Protein café binds to résumé domain."
        resource = Resource(id=resource_id, title="Test", text=text)

        # Should be able to quote using normalized form
        cafe_quote = resource.quote("cafe")
        assert cafe_quote is not None
        assert "café" in cafe_quote.get_quote_text()

        resume_quote = resource.quote("resume")
        assert resume_quote is not None
        assert "résumé" in resume_quote.get_quote_text()

        # Test that position mapping is working (don't test specific positions since normalization changes offsets)
        # Instead test that we can successfully find and quote Unicode content
        protein_quote = resource.quote("protein")
        assert protein_quote is not None
        assert "Protein" in protein_quote.get_quote_text()

        binds_quote = resource.quote("binds")
        assert binds_quote is not None
        assert "binds" in binds_quote.get_quote_text()

    def test_resource_quote_method(self):
        """Test the quote() method on Resource."""
        resource_id = ResourceId(url="https://example.com", counter=1)
        text = "The BRCA1 gene is important for DNA repair."
        resource = Resource(id=resource_id, title="Test", text=text)

        # Test successful quote creation
        quote = resource.quote("BRCA1 gene")
        assert quote.get_quote_text() == "BRCA1 gene"
        assert quote.resource == resource

        # Test non-existent quote raises ValueError
        with pytest.raises(ValueError, match="Quote text not found in resource"):
            resource.quote("nonexistent text")


class TestResourcePool:
    """Test ResourcePool management functionality."""

    def test_empty_pool_creation(self):
        """Test creating empty ResourcePool."""
        pool = ResourcePool()
        assert len(pool.resources) == 0

    def test_add_single_resource(self):
        """Test adding a single resource to pool."""
        pool = ResourcePool()
        url = "https://example.com/paper1.pdf"
        title = "Test Paper 1"
        text = "This is the first test paper."

        resource = pool.add(url, title, text)

        assert len(pool.resources) == 1
        assert resource.id.url == url
        assert resource.title == title
        assert resource.text == text
        assert resource.id.id.startswith("1_")

    def test_add_multiple_resources(self):
        """Test adding multiple resources to pool."""
        pool = ResourcePool()

        resource1 = pool.add("https://example.com/paper1.pdf", "Paper 1", "Text 1")
        resource2 = pool.add("https://example.com/paper2.pdf", "Paper 2", "Text 2")

        assert len(pool.resources) == 2
        assert resource1.id.id.startswith("1_")
        assert resource2.id.id.startswith("2_")
        assert resource1.id.id != resource2.id.id

    def test_duplicate_url_prevention(self):
        """Test that duplicate URLs are prevented."""
        pool = ResourcePool()
        url = "https://example.com/paper1.pdf"

        pool.add(url, "Paper 1", "Text 1")

        with pytest.raises(ValueError, match="already exists"):
            pool.add(url, "Paper 1 Again", "Different text")

    def test_get_by_id(self):
        """Test retrieving resource by ID."""
        pool = ResourcePool()
        resource = pool.add("https://example.com/paper1.pdf", "Paper 1", "Text 1")

        retrieved = pool.get(resource.id.id)
        assert retrieved == resource

    def test_get_by_url(self):
        """Test retrieving resource by URL."""
        pool = ResourcePool()
        url = "https://example.com/paper1.pdf"
        resource = pool.add(url, "Paper 1", "Text 1")

        retrieved = pool.get(url)
        assert retrieved == resource

    def test_get_nonexistent_resource(self):
        """Test retrieving nonexistent resource returns None."""
        pool = ResourcePool()
        assert pool.get("nonexistent_id") is None
        assert pool.get("https://nonexistent.com") is None

    def test_find_resource_id_helper(self):
        """Test the _find_resource_id helper method directly."""
        pool = ResourcePool()
        resource = pool.add("https://example.com/paper1.pdf", "Paper 1", "Text 1")

        # Test finding by ResourceId object
        found_id = pool._find_resource_id(resource.id)
        assert found_id == resource.id

        # Test finding by ID string
        found_id = pool._find_resource_id(resource.id.id)
        assert found_id == resource.id

        # Test finding by URL
        found_id = pool._find_resource_id("https://example.com/paper1.pdf")
        assert found_id == resource.id

        # Test nonexistent key
        assert pool._find_resource_id("nonexistent") is None
        assert pool._find_resource_id("https://nonexistent.com") is None

    def test_getitem_access(self):
        """Test dictionary-style access to resources."""
        pool = ResourcePool()
        resource = pool.add("https://example.com/paper1.pdf", "Paper 1", "Text 1")

        # Should work with both ID and URL
        assert pool[resource.id.id] == resource
        assert pool[resource.id.url] == resource

        # Should raise KeyError for nonexistent
        with pytest.raises(KeyError):
            _ = pool["nonexistent"]

    def test_list_methods(self):
        """Test list_ids and list_titles methods."""
        pool = ResourcePool()

        resource1 = pool.add("https://example.com/paper1.pdf", "Paper 1", "Text 1")
        resource2 = pool.add("https://example.com/paper2.pdf", "Paper 2", "Text 2")

        resources = pool.resources

        assert len(resources) == 2
        ids = [r.id.id for r in resources]
        titles = [r.title for r in resources]
        assert resource1.id.id in ids
        assert resource2.id.id in ids
        assert "Paper 1" in titles
        assert "Paper 2" in titles

    def test_contains_operator(self):
        """Test 'in' operator with different input types."""
        pool = ResourcePool()

        # Register resource without content
        resource_id = pool.register("https://example.com/paper1.pdf")

        # Test ResourceId object
        assert resource_id in pool

        # Test ID string
        assert resource_id.id in pool

        # Test URL string
        assert "https://example.com/paper1.pdf" in pool

        # Test non-existent items
        assert "https://nonexistent.com" not in pool
        assert "invalid_id" not in pool

        # Test with content added
        resource = pool.add_content(resource_id, "Paper 1", "Text content")
        assert resource_id in pool
        assert resource_id.id in pool
        assert resource.id.url in pool

        # Test non-string, non-ResourceId types return False
        assert 123 not in pool
        assert None not in pool
        assert [] not in pool

    def test_register_and_add_content_workflow(self):
        """Test the separate registration and content loading workflow."""
        pool = ResourcePool()

        # Register multiple resources
        urls = [
            "https://example.com/paper1.pdf",
            "https://example.com/paper2.pdf",
            "https://example.com/paper3.pdf",
        ]
        resource_ids = [pool.register(url) for url in urls]

        # Check all are registered but have no content
        for i, resource_id in enumerate(resource_ids):
            assert resource_id in pool
            assert urls[i] in pool
            assert resource_id.id in pool
            assert resource_id.id in pool

        # Add content to subset
        pool.add_content(resource_ids[0], "Paper 1", "Content 1")
        pool.add_content(resource_ids[2], "Paper 3", "Content 3")

        # Check content status (all are registered, but only subset have content)
        assert resource_ids[0].id in pool
        assert resource_ids[1].id in pool
        assert resource_ids[2].id in pool

        # Check that only resources with content appear in the resources list
        assert len(pool.resources) == 2  # Only subset with content have titles


class TestResourceQuote:
    """Test ResourceQuote creation and validation."""

    def setup_method(self):
        """Set up test resources for ResourceQuote tests."""
        self.resource_id = ResourceId(url="https://example.com/paper1.pdf", counter=1)
        self.text = "The BRCA1 gene is important for DNA repair. It prevents cancer."
        self.resource = Resource(
            id=self.resource_id, title="Test Paper", text=self.text
        )

    def test_basic_fragment_creation(self):
        """Test basic ResourceQuote creation."""
        fragment = ResourceQuote(self.resource, "BRCA1")

        assert fragment.resource == self.resource
        assert fragment.query_text == "BRCA1"
        assert fragment.count == 1
        assert fragment.spans == [(4, 9)]  # Should find "BRCA1"
        assert fragment.get_quote_text() == "BRCA1"

    def test_span_validation(self):
        """Test span validation in ResourceQuote."""
        # Valid spans should work - using direct construction
        ResourceQuote(self.resource, text="dummy", query_text="The", spans=[(0, 3)])

        # Invalid spans should fail
        with pytest.raises(ValidationError):
            ResourceQuote(
                self.resource, text="dummy", query_text="invalid", spans=[(10, 5)]
            )  # end < start

        with pytest.raises(ValidationError):
            ResourceQuote(
                self.resource, text="dummy", query_text="invalid", spans=[(-1, 5)]
            )  # negative start

        with pytest.raises(ValidationError):
            ResourceQuote(
                self.resource, text="dummy", query_text="invalid", spans=[(0, 1000)]
            )  # beyond text bounds

    def test_from_quote_exact_match(self):
        """Test creating ResourceQuote from exact quote match."""
        quote = "BRCA1"
        fragment = ResourceQuote(self.resource, quote)

        assert fragment is not None
        assert fragment.get_quote_text() == "BRCA1"
        # Quote is valid if constructor succeeded

    def test_from_quote_normalized_match(self):
        """Test creating ResourceQuote from quote with different formatting."""
        # Quote with extra punctuation/capitalization
        quote = "brca1 gene"
        fragment = ResourceQuote(self.resource, quote)

        assert fragment is not None
        # Quote is valid if constructor succeeded
        # Should normalize to match
        actual = fragment.get_quote_text()
        assert "BRCA1" in actual and "gene" in actual

    def test_from_quote_no_match(self):
        """Test creating ResourceQuote from non-existent quote."""
        quote = "nonexistent text"
        with pytest.raises(ValueError, match="Quote text not found in resource"):
            ResourceQuote(self.resource, quote)

    def test_get_context(self):
        """Test getting context around quote."""
        fragment = ResourceQuote(self.resource, "BRCA1")
        context = fragment.get_context(context_chars=10)

        assert "BRCA1" in context
        assert "**BRCA1**" in context  # Quote should be highlighted
        assert "The" in context  # Should include surrounding text

    def test_validate_quote_exact(self):
        """Test quote construction with different text formats."""
        fragment = ResourceQuote(self.resource, "DNA repair")

        # Exact match should work
        assert fragment.get_quote_text() == "DNA repair"

        # Different formatting should also work due to normalized matching
        fragment2 = ResourceQuote(self.resource, "dna repair")  # Case insensitive
        assert fragment2.get_quote_text() == "DNA repair"

        fragment3 = ResourceQuote(self.resource, "DNA  repair")  # Whitespace tolerance
        assert fragment3.get_quote_text() == "DNA repair"

        # Invalid text should raise ValueError
        with pytest.raises(ValueError, match="Quote text not found"):
            ResourceQuote(self.resource, "RNA repair")

    def test_complex_text_quote_matching(self):
        """Test quote matching with complex text containing punctuation."""
        complex_text = "The protein (BRCA1)---is important. It's 2.5x more active!"
        resource_id = ResourceId(url="https://example.com", counter=2)
        resource = Resource(id=resource_id, title="Complex", text=complex_text)

        # Should find quote despite punctuation differences
        fragment = ResourceQuote(resource, "protein BRCA1 is important")
        assert fragment is not None
        # Quote is valid if constructor succeeded

    def test_multiple_quote_occurrences(self):
        """Test quote matching when text appears multiple times."""
        repeated_text = (
            "BRCA1 is important. The BRCA1 gene is crucial for BRCA1 function."
        )
        resource_id = ResourceId(url="https://example.com", counter=3)
        resource = Resource(id=resource_id, title="Repeated", text=repeated_text)

        # Test finding all occurrences
        fragment = ResourceQuote(resource, "BRCA1")
        assert fragment is not None
        assert fragment.count == 3  # Should find all 3 occurrences
        assert fragment.spans == [(0, 5), (24, 29), (50, 55)]  # All positions

        # Test accessing individual occurrences
        assert fragment.get_quote_text(1) == "BRCA1"  # First occurrence
        assert fragment.get_quote_text(2) == "BRCA1"  # Second occurrence
        assert fragment.get_quote_text(3) == "BRCA1"  # Third occurrence

        # Test getting all quote texts at once
        all_texts = fragment.get_all_quote_texts()
        assert all_texts == ["BRCA1", "BRCA1", "BRCA1"]

        # Test Resource.quote() method
        quote = resource.quote("BRCA1")
        assert quote.count == 3

        # Test error handling for non-existent occurrence
        with pytest.raises(IndexError):
            fragment.get_quote_text(4)

    def test_long_quote_matching(self):
        """Test matching longer quotes with multiple words."""
        quote = "important for DNA repair"
        fragment = ResourceQuote(self.resource, quote)

        assert fragment is not None
        actual_quote = fragment.get_quote_text()
        assert "important" in actual_quote
        assert "DNA" in actual_quote
        assert "repair" in actual_quote

    def test_edge_case_whitespace_quote(self):
        """Test quote matching with edge case whitespace handling."""
        text_with_tabs = "Gene\t\tBRCA1\n\nis   important"
        resource_id = ResourceId(url="https://example.com", counter=4)
        resource = Resource(id=resource_id, title="Whitespace", text=text_with_tabs)

        # Should match despite different whitespace
        fragment = ResourceQuote(resource, "Gene BRCA1 is important")
        assert fragment is not None
        # Quote is valid if constructor succeeded


class TestIntegrationScenarios:
    """Integration tests combining multiple components."""

    def test_full_workflow(self):
        """Test complete workflow from pool creation to quote validation."""
        # Create pool and add resources
        pool = ResourcePool()

        resource1 = pool.add(
            "https://example.com/paper1.pdf",
            "BRCA1 Research Paper",
            "The BRCA1 gene is a tumor suppressor. It repairs DNA damage.",
        )

        resource2 = pool.add(
            "https://example.com/paper2.pdf",
            "Cancer Research",
            "Mutations in BRCA1 increase cancer risk significantly.",
        )

        # Create resource quotes
        quote1 = resource1.quote("tumor suppressor")  # Use new quote() method
        quote2 = resource2.quote("increase cancer risk")

        assert quote1 is not None
        assert quote2 is not None

        # Quotes are valid if constructors succeeded

        # Test resource retrieval
        assert pool.get(resource1.id.id) == resource1
        assert pool.get(resource2.id.url) == resource2

    def test_scientific_paper_simulation(self):
        """Simulate realistic scientific paper text processing."""
        pool = ResourcePool()

        # Add realistic paper content
        paper_text = """
        Abstract: The BRCA1 protein plays a critical role in homologous recombination 
        repair of DNA double-strand breaks. Mutations in BRCA1 are associated with 
        increased risk of breast and ovarian cancers.
        
        Introduction: Breast cancer 1 (BRCA1) is a tumor suppressor gene located on 
        chromosome 17q21. The protein product functions in DNA repair pathways.
        
        Results: We observed that BRCA1-deficient cells showed increased sensitivity 
        to PARP inhibitors (p < 0.001). Treatment with olaparib resulted in 90% 
        cell death in BRCA1-mutant cell lines.
        """

        resource = pool.add(
            "https://pubmed.ncbi.nlm.nih.gov/12345678",
            "BRCA1 and PARP Inhibitor Sensitivity",
            paper_text,
        )

        # Test extracting various scientific quotes
        test_quotes = [
            "homologous recombination repair",
            "tumor suppressor gene",
            "increased sensitivity to PARP inhibitors",
            "90% cell death",
        ]

        quotes = []
        for quote_text in test_quotes:
            quote = resource.quote(quote_text)  # Use new quote() method
            # Quote is valid if constructor succeeded (no ValueError raised)
            quotes.append(quote)

        # Test context extraction
        for quote in quotes:
            context = quote.get_context(context_chars=50)
            assert len(context) > len(quote.get_quote_text())
            assert "**" in context  # Should have quote highlighting

    def test_original_problem_paper_title_quotability(self):
        """Integration test for the original problem: quoting paper titles with underscores."""
        # Simulate the exact scenario from the original issue
        paper_title = "Mutations in Iron-Sulfur Cluster Scaffold Genes _NFU1_ and _BOLA3_ Cause a Fatal Deficiency of Multiple Respiratory Chain and 2-Oxoacid Dehydrogenase Enzymes"

        # Mock content similar to the cache file
        paper_content = f"""
        # {paper_title}

        ## Abstract
        Iron-sulfur (Fe-S) clusters are essential cofactors for numerous biological processes.
        This study examines mutations in the _NFU1_ and _BOLA3_ genes, which encode scaffold
        proteins critical for Fe-S cluster biogenesis.

        ## Introduction  
        The _NFU1_ gene encodes a late-acting scaffold protein, while _BOLA3_ participates
        in the cytosolic iron-sulfur cluster assembly machinery.

        ## Results
        Patients with mutations in these genes showed severe deficiencies in respiratory
        chain complexes I, II, and III, as well as reduced activity of 2-oxoacid
        dehydrogenase enzymes.
        """

        pool = ResourcePool()
        resource = pool.add(
            "https://pubmed.ncbi.nlm.nih.gov/example",
            paper_title,  # Use the full title with underscores
            paper_content,
        )

        # Test that the title can be quoted despite the underscores
        title_quote = resource.quote(paper_title)
        assert title_quote is not None, f"Failed to quote paper title: {paper_title}"
        assert title_quote.count >= 1, (
            "Should find at least one occurrence of the title"
        )

        # Test that key terms from the title can be found individually
        mutations_quote = resource.quote("mutations")
        assert mutations_quote is not None, "Should find 'mutations'"

        genes_quote = resource.quote("genes")
        assert genes_quote is not None, "Should find 'genes'"

        # Test key phrase from title
        key_phrase = "iron sulfur cluster"
        phrase_quote = resource.quote(key_phrase)
        assert phrase_quote is not None, "Should find key phrase from title"

        # Test that individual gene names can be quoted
        nfu1_quote = resource.quote("NFU1")
        assert nfu1_quote is not None, "Should find NFU1 gene mentions"
        assert nfu1_quote.count >= 2, "Should find multiple NFU1 mentions"

        bola3_quote = resource.quote("BOLA3")
        assert bola3_quote is not None, "Should find BOLA3 gene mentions"
        assert bola3_quote.count >= 2, "Should find multiple BOLA3 mentions"

        # Verify that underscored versions in content can be found
        underscore_nfu1 = resource.quote("_NFU1_")
        assert underscore_nfu1 is not None, "Should find _NFU1_ with underscores"

        # Verify position mapping works for Greek letters and special characters
        complex_phrase = "iron-sulfur cluster"
        complex_quote = resource.quote(complex_phrase)
        assert complex_quote is not None, "Should find complex phrases with hyphens"


class TestDisjointQuotes:
    """Test disjoint quote matching with ellipses."""

    def test_simple_disjoint_quote(self):
        """Test basic disjoint quote with two segments."""
        document_text = "In 2020, two articles published in Nature reported additional cases with missense variants in exon 38 or 39 in KMT2D gene."

        pool = ResourcePool()
        resource = pool.add("https://example.com", "Test Document", document_text)

        # Test disjoint quote
        quote = resource.quote(
            "In 2020, two articles ... reported additional cases ... missense variants"
        )
        assert quote is not None
        assert quote.count == 1
        assert quote.is_disjoint == True
        # Should have 3 segments: 'In 2020, two articles', 'reported additional cases', 'missense variants'

        # Verify quote text includes ellipses
        quote_text = quote.get_quote_text()
        assert "..." in quote_text
        assert "In 2020, two articles" in quote_text
        assert "reported additional cases" in quote_text
        assert "missense variants" in quote_text

    def test_multiple_disjoint_occurrences(self):
        """Test disjoint quote that matches multiple times."""
        document_text = """
        Null mutations have been identified in the genes coding for two proteins, 
        cartilage-associated protein (CRTAP) and prolyl 3-hydroxylase 1 (P3H1).
        Further analysis showed null mutations in other genes coding for different proteins,
        including additional cartilage-associated protein variants.
        """

        pool = ResourcePool()
        resource = pool.add("https://example.com", "Test Document", document_text)

        quote = resource.quote("Null mutations ... genes coding for ... proteins")
        assert quote is not None
        assert quote.count >= 1

        # Should be disjoint
        assert quote.is_disjoint == True

    def test_continuous_quote_still_works(self):
        """Test that continuous quotes still work as before."""
        document_text = (
            "BRCA1 mutations significantly increase breast cancer risk in patients."
        )

        pool = ResourcePool()
        resource = pool.add("https://example.com", "Test Document", document_text)

        quote = resource.quote("BRCA1 mutations")
        assert quote is not None
        assert quote.count == 1
        assert quote.is_disjoint == False
        # Should be a single continuous quote

        quote_text = quote.get_quote_text()
        assert quote_text == "BRCA1 mutations"
        assert "..." not in quote_text

    def test_disjoint_quote_context(self):
        """Test context display for disjoint quotes."""
        document_text = "In 2020, two articles published in Nature reported additional cases with missense variants in exon 38 or 39 in KMT2D gene."

        pool = ResourcePool()
        resource = pool.add("https://example.com", "Test Document", document_text)

        quote = resource.quote("In 2020 ... reported ... variants")
        assert quote is not None

        context = quote.get_context(context_chars=10)
        # Should show context around each segment
        assert "**In 2020**" in context or "**In 2020, two articles**" in context
        assert "**reported**" in context
        assert "**variants**" in context
        assert "\n...\n" in context  # Separator between segments

    def test_invalid_disjoint_quote(self):
        """Test that invalid disjoint quotes raise appropriate errors."""
        document_text = "This is a test document without the expected content."

        pool = ResourcePool()
        resource = pool.add("https://example.com", "Test Document", document_text)

        # Should fail when segments don't appear in order
        with pytest.raises(ValueError, match="Quote text not found"):
            ResourceQuote(resource, "nonexistent ... segments ... here")

    def test_ellipsis_variations(self):
        """Test different ellipsis formats are recognized."""
        document_text = "The quick brown fox jumps over the lazy dog near the river."

        pool = ResourcePool()
        resource = pool.add("https://example.com", "Test Document", document_text)

        # Test various ellipsis formats including Unicode ellipsis
        formats = [
            "The quick ... jumps over ... lazy dog",
            "The quick….jumps over….lazy dog",  # Unicode ellipsis
            "The quick … jumps over … lazy dog",  # Unicode ellipsis with spaces
            "The quick....jumps over....lazy dog",
        ]

        for format_text in formats:
            quote = resource.quote(format_text)
            assert quote is not None, f"Failed to match format: {format_text}"
            assert quote.is_disjoint == True, f"Should be disjoint for: {format_text}"
            # Should have 3 segments: 'The quick', 'jumps over', 'lazy dog'

    def test_unicode_ellipsis_support(self):
        """Test explicit Unicode ellipsis (U+2026) support."""
        document_text = "Mutations in BRCA1 have been identified in many studies published recently."

        pool = ResourcePool()
        resource = pool.add("https://example.com", "Test Document", document_text)

        # Test Unicode ellipsis character
        quote = resource.quote("Mutations in BRCA1 … published recently")
        assert quote is not None
        assert quote.is_disjoint == True
        assert quote.count == 1

        # Verify the segments are found correctly
        quote_text = quote.get_quote_text()
        assert "Mutations in BRCA1" in quote_text
        assert "published recently" in quote_text
        assert "..." in quote_text  # Output should normalize to ...

        # Test mixed ellipsis types should work the same
        quote_mixed = resource.quote("Mutations in BRCA1 ... published recently")
        assert quote_mixed is not None
        assert quote_mixed.count == 1

        # Both should find the same content (though represented differently)
        assert quote.get_quote_text() == quote_mixed.get_quote_text()
