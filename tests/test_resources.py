"""
Thorough tests for resource management models and validation.

Tests cover ResourceId generation, Resource text normalization and position mapping,
ResourcePool management, and ResourceQuote quote matching and validation.
"""

import pytest
from pydantic import ValidationError
from interaction_finder.resources import (
    normalize_text_for_matching,
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


class TestResourceId:
    """Test ResourceId creation and validation."""

    def test_basic_creation(self):
        """Test basic ResourceId creation from URL and counter."""
        url = "https://example.com/paper1.pdf"
        counter = 1
        resource_id = ResourceId(url=url, counter=counter)

        assert resource_id.url == url
        assert resource_id.id.startswith("1_")
        assert len(resource_id.id.split("_")[1]) == 8  # 8-char hash

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
        assert len(resource_id.id) == 10  # "1_" + 8 character hash
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

        # Test mapping for "brca1" in normalized text
        norm_text = resource.normalized_text  # "the brca1 gene"
        brca1_start = norm_text.find("brca1")

        original_start, original_end = resource.map_normalized_to_original_position(
            brca1_start, 5
        )

        # Should map back to "BRCA1" in original text
        assert text[original_start:original_end] == "BRCA1"

    def test_resource_quote_method(self):
        """Test the quote() method on Resource."""
        resource_id = ResourceId(url="https://example.com", counter=1)
        text = "The BRCA1 gene is important for DNA repair."
        resource = Resource(id=resource_id, title="Test", text=text)

        # Test successful quote creation
        quote = resource.quote("BRCA1 gene")
        assert quote is not None
        assert quote.get_quote_text() == "BRCA1 gene"
        assert quote.resource == resource

        # Test non-existent quote
        no_quote = resource.quote("nonexistent text")
        assert no_quote is None


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
        assert fragment.validate_quote(quote)

    def test_from_quote_normalized_match(self):
        """Test creating ResourceQuote from quote with different formatting."""
        # Quote with extra punctuation/capitalization
        quote = "brca1 gene"
        fragment = ResourceQuote(self.resource, quote)

        assert fragment is not None
        assert fragment.validate_quote(quote)
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
        """Test quote validation with exact match."""
        fragment = ResourceQuote(self.resource, "DNA repair")

        assert fragment.validate_quote("DNA repair")
        assert fragment.validate_quote("dna repair")  # Case insensitive
        assert fragment.validate_quote("DNA  repair")  # Whitespace tolerance
        assert not fragment.validate_quote("RNA repair")  # Different text

    def test_complex_text_quote_matching(self):
        """Test quote matching with complex text containing punctuation."""
        complex_text = "The protein (BRCA1)---is important. It's 2.5x more active!"
        resource_id = ResourceId(url="https://example.com", counter=2)
        resource = Resource(id=resource_id, title="Complex", text=complex_text)

        # Should find quote despite punctuation differences
        fragment = ResourceQuote(resource, "protein BRCA1 is important")
        assert fragment is not None
        assert fragment.validate_quote("protein BRCA1 is important")

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
        assert quote is not None
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
        assert fragment.validate_quote("Gene BRCA1 is important")


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

        # Validate quotes
        assert quote1.validate_quote("tumor suppressor")
        assert quote2.validate_quote("increase cancer risk")

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
            assert quote is not None, f"Could not find quote: {quote_text}"
            assert quote.validate_quote(quote_text)
            quotes.append(quote)

        # Test context extraction
        for quote in quotes:
            context = quote.get_context(context_chars=50)
            assert len(context) > len(quote.get_quote_text())
            assert "**" in context  # Should have quote highlighting
