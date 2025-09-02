"""
Pipeline integration tests for resource tracking.

This module tests the integration points where resource tracking
interfaces with the extraction pipeline.
"""

import pytest
from interaction_finder.extraction_graph.deps import Deps
from interaction_finder.extraction_graph.state import (
    DocumentGroup,
    EntityExtractionState,
)
from interaction_finder.resources import ResourcePool
from interaction_finder.settings import IfetcherConfig


class TestPipelineIntegration:
    """Test resource tracking integration with the pipeline."""

    def test_deps_with_resource_pool(self):
        """Test that Deps can be created with ResourcePool."""
        # Create basic components (mocking the complex PageFetcher setup)
        config = IfetcherConfig()
        resource_pool = ResourcePool()

        # Create mock page fetcher (simplified)
        class MockPageFetcher:
            def __init__(self):
                self.cache = MockURLCache()

        class MockURLCache:
            pass

        page_fetcher = MockPageFetcher()

        # Test creating Deps with ResourcePool
        deps = Deps.from_config(
            config=config,
            page_fetcher=page_fetcher,
            resource_pool=resource_pool,
            verbose=False,
        )

        assert deps.resource_pool is resource_pool
        assert len(deps.resource_pool.resource_map) == 0

    def test_deps_without_resource_pool(self):
        """Test that Deps creates ResourcePool automatically if not provided."""
        config = IfetcherConfig()

        class MockPageFetcher:
            def __init__(self):
                self.cache = MockURLCache()

        class MockURLCache:
            pass

        page_fetcher = MockPageFetcher()

        # Test creating Deps without ResourcePool (should create one automatically)
        deps = Deps.from_config(
            config=config, page_fetcher=page_fetcher, resource_pool=None, verbose=False
        )

        assert deps.resource_pool is not None
        assert isinstance(deps.resource_pool, ResourcePool)
        assert len(deps.resource_pool.resource_map) == 0

    def test_document_group_integration(self):
        """Test DocumentGroup integration with resource tracking."""
        # Create test document group
        doc_group = DocumentGroup(
            urls=["http://example.com/paper1", "http://example.com/paper2"],
            chunks={
                "http://example.com/paper1": [
                    {"text": "BRCA1 is a tumor suppressor gene."},
                    {"text": "Mutations in BRCA1 cause breast cancer."},
                ],
                "http://example.com/paper2": [
                    {"text": "p53 and BRCA1 interact in DNA repair pathways."},
                ],
            },
            metadata={
                "http://example.com/paper1": {"title": "BRCA1 Function"},
                "http://example.com/paper2": {"title": "p53-BRCA1 Interactions"},
            },
        )

        # Test ResourcePool population
        resource_pool = ResourcePool()
        resources = doc_group.populate_resource_pool(resource_pool)

        assert len(resources) == 2
        assert "http://example.com/paper1" in resources
        assert "http://example.com/paper2" in resources

        # Verify resource content combines chunks correctly
        paper1_resource = resources["http://example.com/paper1"]
        assert "BRCA1 is a tumor suppressor gene" in paper1_resource.text
        assert "Mutations in BRCA1 cause breast cancer" in paper1_resource.text
        assert paper1_resource.title == "BRCA1 Function"

        # Test ResourceQuote creation
        brca1_quote = paper1_resource.quote("BRCA1")
        assert brca1_quote is not None
        assert brca1_quote.count == 2  # Should find 2 occurrences in paper1

        # Test cross-document quotes
        all_resources = list(resources.values())
        for resource in all_resources:
            brca1_quotes = resource.quote("BRCA1")
            if brca1_quotes:
                # Each document should have BRCA1 mentions
                assert brca1_quotes.count >= 1

    def test_entity_extraction_state_integration(self):
        """Test that EntityExtractionState works with resource tracking."""
        # Create extraction state
        state = EntityExtractionState()

        # Add document group
        doc_group = DocumentGroup(
            urls=["http://example.com/study"],
            chunks={
                "http://example.com/study": [
                    {
                        "text": "This study examines BRCA1 mutations in breast cancer patients."
                    },
                    {
                        "text": "Results show that BRCA1 deficiency increases cancer risk by 80%."
                    },
                ]
            },
            metadata={"http://example.com/study": {"title": "BRCA1 Cancer Study"}},
        )

        state.document_groups = [doc_group]
        state.current_group_index = 0

        # Test getting current group
        current_group = state.get_current_group()
        assert current_group is doc_group

        # Test resource population
        resource_pool = ResourcePool()
        resources = current_group.populate_resource_pool(resource_pool)

        assert len(resources) == 1
        study_resource = resources["http://example.com/study"]

        # Verify combined text from multiple chunks
        assert "This study examines BRCA1 mutations" in study_resource.text
        assert "Results show that BRCA1 deficiency" in study_resource.text

        # Test quote extraction across chunks
        brca1_quote = study_resource.quote("BRCA1")
        assert brca1_quote is not None
        assert brca1_quote.count == 2  # One in each chunk

    def test_resource_id_generation(self):
        """Test that ResourceIds are generated consistently."""
        resource_pool = ResourcePool()

        # Test that same URL cannot be registered twice (should raise error)
        id1 = resource_pool.register("http://example.com/paper")
        with pytest.raises(ValueError, match="already exists"):
            resource_pool.register("http://example.com/paper")

        # Test different URLs get different ResourceIds
        id2 = resource_pool.register("http://example.com/different")
        assert id2.id != id1.id
        assert id2.url != id1.url

        # Test counter increments for different URLs
        assert "1_" in id1.id  # First resource
        assert "2_" in id2.id  # Second resource

        # Test that we can retrieve the same ResourceId by URL lookup
        assert id1.url in resource_pool  # URL lookup should work
        assert id2.url in resource_pool

    def test_resource_quote_basic_functionality(self):
        """Test that ResourceQuotes work with basic functionality."""
        # Set up test resources
        resource_pool = ResourcePool()
        resource_id = resource_pool.register("http://example.com/functionality_test")
        resource = resource_pool.add_content(
            resource_id,
            "Functionality Test Paper",
            "The BRCA1 gene is essential for DNA repair. When BRCA1 is defective, cells cannot properly repair DNA breaks, leading to genomic instability and increased cancer risk.",
        )

        # Create ResourceQuote
        brca1_quote = resource.quote("BRCA1")
        assert brca1_quote is not None

        # Test basic functionality
        assert brca1_quote.count == 2  # Two BRCA1 mentions
        assert brca1_quote.validate_quote("BRCA1")  # Basic validation

        # Test quote text retrieval
        first_quote = brca1_quote.get_quote_text(1)
        assert first_quote == "BRCA1"

        all_quotes = brca1_quote.get_all_quote_texts()
        assert len(all_quotes) == 2
        assert all(quote == "BRCA1" for quote in all_quotes)

    def test_prompt_resource_integration(self):
        """Test that resource information can be included in prompts."""
        # This tests the pattern that will be used in actual agent prompts
        resource_pool = ResourcePool()
        doc_group = DocumentGroup(
            urls=["http://example.com/prompt_test"],
            chunks={
                "http://example.com/prompt_test": [
                    {"text": "BRCA1 is a tumor suppressor that prevents breast cancer."}
                ]
            },
            metadata={"http://example.com/prompt_test": {"title": "Prompt Test Paper"}},
        )

        # Populate ResourcePool
        resources = doc_group.populate_resource_pool(resource_pool)
        resource = resources["http://example.com/prompt_test"]

        # Test prompt construction (similar to what nodes will do)
        url = "http://example.com/prompt_test"
        document_text = "BRCA1 is a tumor suppressor that prevents breast cancer."

        resource_info = f"""
RESOURCE TRACKING INFORMATION:
- Resource ID: {resource.id.id}
- Resource URL: {resource.id.url}
- Available for creating ResourceQuotes to support entity extractions

When extracting entities, you can reference this Resource ID in your response.
"""

        prompt = f"""Extract gene, disease entities from this document:

Source URL: {url}
{resource_info}
Content:
{document_text}

Identify all entities found in this document. If Resource ID is available, consider providing ResourceQuotes for supporting evidence."""

        # Verify prompt contains resource information
        assert resource.id.id in prompt
        assert resource.id.url in prompt
        assert "ResourceQuotes" in prompt
        assert "RESOURCE TRACKING INFORMATION" in prompt

        # This prompt could now be passed to an LLM agent that understands ResourceQuotes
        assert len(prompt) > 200  # Should be substantial prompt
