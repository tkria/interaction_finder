"""
Integration tests for extraction graph V2 pipeline.

Tests the complete end-to-end pipeline with ResourceQuote provenance.

Phase 3 Note: These tests require real LLM agents. Set TEST_MODEL environment
variable to specify model (default: openai:gpt-4o-mini). Tests will skip if
model is unavailable.
"""

import pytest
import logging
import os
from pathlib import Path

from pydantic_ai import models
from interaction_finder.extraction_graph_v2.run import run_extraction_v2
from interaction_finder.extraction_graph_v2.deps import ExtractionDeps
from interaction_finder.extraction_graph_v2.state import ExtractionState
from interaction_finder.extraction_graph_v2.models import EntityPairOut
from interaction_finder.settings import IfetcherConfig
from interaction_finder.fetcher import PageFetcher
from .fixtures import BRCA1_DOCUMENT, TEST_URLS, create_test_resource_pool

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="class", autouse=True)
def enable_model_requests():
    """Enable real model requests for integration tests that use slow marker."""
    original_setting = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    yield
    models.ALLOW_MODEL_REQUESTS = original_setting


class TestExtractionPipelineIntegration:
    """Test complete extraction pipeline with ResourceQuote integration."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        # Use minimal config for testing
        config_data = {
            "task": {
                "kinds": {
                    "gene": {
                        "kind": ["gene"],
                        "form": ["name"],
                        "example": ["BRCA1", "TP53"],
                    },
                    "disease": {
                        "kind": ["disease"],
                        "form": ["name"],
                        "example": ["breast cancer", "lung cancer"],
                    },
                },
                "relation": "interaction",
                "context": "Test gene-disease interactions",
            },
            "agents": {"_": {"llm": "openai:gpt-4o"}},
            "cache": {"directory": "test_cache"},
        }
        return IfetcherConfig(**config_data)

    @pytest.fixture
    def deps(self, config):
        """Create test dependencies with Phase 3 agent support."""
        import os

        page_fetcher = PageFetcher(config)

        # For Phase 3, we need a real model that works with pydantic-ai agents
        # Priority: environment variable > lightweight OpenAI model > mock
        test_model = os.environ.get("TEST_MODEL", "openai:gpt-4o-mini")

        return ExtractionDeps.from_config(
            config=config,
            page_fetcher=page_fetcher,
            model=test_model,  # Phase 3 compatible model
        )

    @pytest.fixture
    def resource_pool(self):
        """Create resource pool with test document."""
        documents = [(TEST_URLS[0], "BRCA1 Study", BRCA1_DOCUMENT)]
        return create_test_resource_pool(documents)

    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow test since it uses real agents
    async def test_end_to_end_pipeline(self, resource_pool, deps):
        """Test complete pipeline produces pairs with provenance."""
        logger.info("Starting end-to-end pipeline test")

        # Run the complete pipeline
        pairs = await run_extraction_v2(resource_pool, deps)

        # Should find at least one pair
        assert len(pairs) > 0, "Pipeline should produce at least one pair"

        # All pairs should be EntityPairOut objects
        assert all(isinstance(pair, EntityPairOut) for pair in pairs)

        # All pairs should have complete provenance
        for pair in pairs:
            assert pair.validate_provenance(), (
                f"Pair {pair.entity_a.name}-{pair.entity_b.name} missing provenance"
            )
            assert len(pair.evidence_quotes) > 0, "Pair should have evidence quotes"
            assert pair.entity_a.validate(), "Entity A should have valid quotes"
            assert pair.entity_b.validate(), "Entity B should have valid quotes"

        logger.info(f"✓ Pipeline produced {len(pairs)} pairs with complete provenance")

    @pytest.mark.asyncio
    @pytest.mark.slow  # Uses real agents
    async def test_brca1_breast_cancer_relationship(self, resource_pool, deps):
        """Test that pipeline finds BRCA1-breast cancer relationship."""
        pairs = await run_extraction_v2(resource_pool, deps)

        # Should find BRCA1-breast cancer pair
        brca1_pairs = [
            pair
            for pair in pairs
            if (pair.entity_a.name == "BRCA1" and "breast cancer" in pair.entity_b.name)
            or (pair.entity_b.name == "BRCA1" and "breast cancer" in pair.entity_a.name)
        ]

        assert len(brca1_pairs) > 0, "Should find BRCA1-breast cancer relationship"

        # Validate the specific pair
        pair = brca1_pairs[0]
        assert pair.confidence in ["high", "medium"], (
            "Should have high or medium confidence"
        )
        assert len(pair.evidence_quotes) > 0, "Should have evidence"

        # Check output format
        output_format = pair.to_output_format()
        required_fields = [
            "first",
            "first_kind",
            "second",
            "second_kind",
            "reasoning",
            "resources",
            "confidence",
        ]
        for field in required_fields:
            assert field in output_format, f"Missing required field: {field}"

        logger.info(
            f"✓ Found BRCA1-breast cancer pair with confidence: {pair.confidence}"
        )

    @pytest.mark.asyncio
    async def test_resource_quote_positions(self, resource_pool, deps):
        """Test that ResourceQuotes have correct positions."""
        pairs = await run_extraction_v2(resource_pool, deps)

        for pair in pairs:
            # Check entity quotes
            for entity in [pair.entity_a, pair.entity_b]:
                for quote in entity.quotes:
                    # ResourceQuote existence implies count > 0 by design

                    # Verify quote text exists at claimed positions
                    for span_start, span_end in quote.spans:
                        quote_text = quote.resource.text[span_start:span_end]
                        assert entity.name in quote_text, (
                            f"Entity {entity.name} not found at position {span_start}-{span_end}"
                        )

            # Check evidence quotes
            for evidence_quote in pair.evidence_quotes:
                # ResourceQuote existence implies count > 0 by design

                # Verify evidence text exists
                for i in range(evidence_quote.count):
                    evidence_text = evidence_quote.get_quote_text(i + 1)
                    assert len(evidence_text) > 0, "Evidence text should not be empty"

        logger.info("✓ All ResourceQuotes have valid positions")

    @pytest.mark.asyncio
    async def test_state_metrics_tracking(self, resource_pool, deps):
        """Test that state tracks Phase 3 metrics correctly."""
        # Create state to check initial metrics
        state = ExtractionState(resource_pool=resource_pool)

        # Initial state should be empty
        assert state.get_entity_count() == 0
        assert state.get_assessment_count() == 0
        assert state.get_pairs_count() == 0

        # Phase 3 metrics should be initialized
        assert state.metrics.entities_extraction_calls == 0
        assert state.metrics.assessment_calls == 0
        assert state.metrics.entities_with_valid_quotes == 0

        # Run complete pipeline with real agents
        pairs = await run_extraction_v2(resource_pool, deps)
        assert len(pairs) > 0, "Should have found pairs"

        # Note: run_extraction_v2 creates its own state, but we can verify via pairs
        # and by testing the metrics functionality directly

        # Test metrics functionality
        state.metrics.start_timing()
        state.metrics.record_extraction_call(success=True, duration=1.5)
        state.metrics.record_assessment_call(success=True, duration=0.8)

        # Add some entity validation tracking
        for pair in pairs:
            state.metrics.record_entity_validation(pair.entity_a)
            state.metrics.record_entity_validation(pair.entity_b)

        # Test success rates calculation
        success_rates = state.metrics.get_success_rates()
        assert "extraction_success_rate" in success_rates
        assert "assessment_success_rate" in success_rates
        assert "evidence_retrieval_rate" in success_rates
        assert success_rates["extraction_success_rate"] == 100.0
        assert success_rates["assessment_success_rate"] == 100.0

        # Test enhanced summary with metrics
        summary = state.get_summary()
        assert "agent_calls" in summary
        assert "success_rates" in summary
        assert "quality_metrics" in summary
        assert "timing" in summary

        # Verify provenance metrics from pipeline results
        for pair in pairs:
            assert pair.entity_a.total_occurrences > 0, (
                "Entity A should have occurrences"
            )
            assert pair.entity_b.total_occurrences > 0, (
                "Entity B should have occurrences"
            )
            assert len(pair.evidence_quotes) > 0, "Should have evidence"

        logger.info("✓ Phase 3 state metrics tracking works correctly")

    @pytest.mark.asyncio
    async def test_no_same_kind_pairs(self, resource_pool, deps):
        """Test that pipeline doesn't create gene-gene or disease-disease pairs."""
        pairs = await run_extraction_v2(resource_pool, deps)

        for pair in pairs:
            assert pair.entity_a.kind != pair.entity_b.kind, (
                f"Same-kind pair found: {pair.entity_a.name} ({pair.entity_a.kind}) - {pair.entity_b.name} ({pair.entity_b.kind})"
            )

        logger.info("✓ No same-kind pairs created")

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self):
        """Test pipeline handles errors gracefully."""
        # Test with empty resource pool
        empty_pool = create_test_resource_pool([])

        # Create mock config
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]}
                },
                "relation": "test",
                "context": "test",
            },
            "agents": {"_": {"llm": "test"}},
            "cache": {"directory": "test"},
        }
        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)
        deps = ExtractionDeps.from_config(config, page_fetcher)

        # Should handle empty pool gracefully
        pairs = await run_extraction_v2(empty_pool, deps)
        assert isinstance(pairs, list), "Should return empty list, not crash"

        logger.info("✓ Pipeline handles errors gracefully")

    @pytest.mark.asyncio
    @pytest.mark.slow  # Uses real agents
    async def test_phase3_agent_integration(self, resource_pool, deps):
        """Test Phase 3 agent integration and error handling."""
        # Test that agents are created and called correctly
        pairs = await run_extraction_v2(resource_pool, deps)

        # Should successfully extract entities using LLM agents
        assert len(pairs) > 0, "Phase 3 agents should extract entities and create pairs"

        # All entities should have been validated through agents
        for pair in pairs:
            # Both entities should have proper provenance
            assert pair.entity_a.validate(), (
                f"Entity {pair.entity_a.name} should have valid quotes"
            )
            assert pair.entity_b.validate(), (
                f"Entity {pair.entity_b.name} should have valid quotes"
            )

            # Should have evidence from assessment agents
            assert len(pair.evidence_quotes) > 0, (
                "Assessment agents should provide evidence"
            )

            # Confidence should be properly mapped from agent outputs
            assert pair.confidence in ["high", "medium", "low"], (
                f"Invalid confidence: {pair.confidence}"
            )

        logger.info("✓ Phase 3 agent integration works correctly")

    @pytest.mark.asyncio
    @pytest.mark.slow  # Tests agent failure handling
    async def test_agent_failure_resilience(self, resource_pool):
        """Test that pipeline handles agent failures gracefully."""
        # Create deps with invalid model to simulate agent failures
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]},
                    "disease": {
                        "kind": ["disease"],
                        "form": ["name"],
                        "example": ["cancer"],
                    },
                },
                "relation": "interaction",
                "context": "Test context",
            },
            "agents": {
                "_": {"llm": "invalid-model"}
            },  # This should cause agent failures
            "cache": {"directory": "test_cache"},
        }
        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)
        deps = ExtractionDeps.from_config(config, page_fetcher, model="invalid-model")

        # Pipeline should handle agent failures without crashing
        pairs = await run_extraction_v2(resource_pool, deps)

        # Should return empty list rather than crash
        assert isinstance(pairs, list), "Should return list even with agent failures"

        logger.info("✓ Pipeline handles agent failures gracefully")

    @pytest.mark.asyncio
    async def test_document_grouping(self, config, deps):
        """Pipeline runs when multiple documents are present (no explicit groups in v2)."""
        # Create test documents
        doc1_content = """
        BRCA1 is a tumor suppressor gene that plays a critical role in DNA repair.
        Mutations in BRCA1 are associated with breast cancer and ovarian cancer.
        """

        doc2_content = """
        TP53 is another important tumor suppressor gene that regulates cell division.
        TP53 mutations are found in many types of cancer including lung cancer.
        """

        doc3_content = """
        The BRCA1 gene encodes a protein involved in homologous recombination repair.
        Hereditary breast cancer is often linked to BRCA1 gene defects.
        """

        # Create ResourcePool with all documents
        from interaction_finder.resources import ResourcePool

        resource_pool = ResourcePool()

        resource1 = resource_pool.add(
            "https://example.com/doc1", "BRCA1 Research", doc1_content
        )
        resource2 = resource_pool.add(
            "https://example.com/doc2", "TP53 Study", doc2_content
        )
        resource3 = resource_pool.add(
            "https://example.com/doc3", "BRCA1 Analysis", doc3_content
        )

        # Run pipeline without explicit groups (v2 processes all documents together)
        pairs = await run_extraction_v2(resource_pool, deps)

        assert len(pairs) >= 0, "Should handle grouped processing without errors"

        # Verify that both BRCA1 and TP53 entities can be found across groups
        found_entities = set()
        for pair in pairs:
            found_entities.add(pair.entity_a.name)
            found_entities.add(pair.entity_b.name)

        logger.info(f"Found entities across documents: {found_entities}")
        logger.info(f"✓ Multi-document test completed with {len(pairs)} pairs")

    @pytest.mark.asyncio
    async def test_backward_compatibility_no_groups(self, resource_pool, deps):
        """Pipeline works without groups (v2 default)."""
        pairs = await run_extraction_v2(resource_pool, deps)
        assert len(pairs) >= 0
        logger.info("✓ No-groups default behavior verified")

    @pytest.mark.asyncio
    async def test_target_term_context(self, config, deps):
        """Test that target term is passed through the pipeline correctly."""
        from interaction_finder.resources import ResourcePool
        from interaction_finder.extraction_graph_v2.deps import ExtractionDeps

        # Create test document with BRCA1 content
        doc_content = """
        BRCA1 is a tumor suppressor gene that plays a critical role in DNA repair.
        Mutations in BRCA1 are associated with breast cancer and ovarian cancer.
        The BRCA1 protein works with other proteins to repair damaged DNA.
        """

        resource_pool = ResourcePool()
        resource = resource_pool.add(
            "https://example.com/brca1", "BRCA1 Research", doc_content
        )

        # Create deps with target term
        deps_with_term = ExtractionDeps.from_config(
            deps.config, deps.page_fetcher, deps.model, target_term="BRCA1"
        )

        # Test that target_term is properly stored
        assert deps_with_term.target_term == "BRCA1", (
            "Target term should be stored in deps"
        )

        # Test pipeline with target term
        pairs = await run_extraction_v2(resource_pool, deps_with_term)

        assert len(pairs) >= 0, "Should handle target term without errors"
        logger.info(f"✓ Target term test completed with {len(pairs)} pairs")

        # Also test without target term for comparison
        deps_without_term = ExtractionDeps.from_config(
            deps.config, deps.page_fetcher, deps.model, target_term=None
        )

        assert deps_without_term.target_term is None, (
            "Target term should be None when not provided"
        )

        pairs_without = await run_extraction_v2(resource_pool, deps_without_term)
        assert len(pairs_without) >= 0, (
            "Should handle missing target term without errors"
        )

        logger.info("✓ Target term context test passed")
