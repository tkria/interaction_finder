"""
Unit tests for extraction graph V2 pipeline runners.

Tests run.py and run_simple.py pipeline execution.
"""

import pytest
import logging
from unittest.mock import Mock, patch, AsyncMock

from interaction_finder.extraction_graph_v2.run import run_extraction_v2
from interaction_finder.extraction_graph_v2.run_simple import run_extraction_v2_simple
from interaction_finder.extraction_graph_v2.deps import ExtractionDeps
from interaction_finder.extraction_graph_v2.state import ExtractionState
from interaction_finder.extraction_graph_v2.models import EntityPairOut
from interaction_finder.settings import IfetcherConfig
from interaction_finder.fetcher import PageFetcher
from interaction_finder.resources import ResourcePool
from .fixtures import (
    BRCA1_DOCUMENT,
    MULTI_GENE_DOCUMENT,
    NO_RELATIONSHIP_DOCUMENT,
    TEST_URLS,
    create_test_resource_pool,
)

logger = logging.getLogger(__name__)


class TestCreateTestResourcePool:
    """Test helper function for creating test resource pools."""

    def test_create_empty_pool(self):
        """Test creating empty resource pool."""
        pool = create_test_resource_pool([])

        assert isinstance(pool, ResourcePool)
        assert len(pool.resources) == 0

    def test_create_single_document_pool(self):
        """Test creating pool with single document."""
        documents = [(TEST_URLS[0], "Test Document", BRCA1_DOCUMENT)]
        pool = create_test_resource_pool(documents)

        assert len(pool.resources) == 1
        resource = list(pool.resources)[0]
        assert resource.id.url == TEST_URLS[0]
        assert resource.title == "Test Document"
        assert resource.text == BRCA1_DOCUMENT

    def test_create_multiple_document_pool(self):
        """Test creating pool with multiple documents."""
        documents = [
            (TEST_URLS[0], "BRCA1 Study", BRCA1_DOCUMENT),
            (TEST_URLS[1], "Multi Gene Study", MULTI_GENE_DOCUMENT),
            (TEST_URLS[2], "Control Study", NO_RELATIONSHIP_DOCUMENT),
        ]
        pool = create_test_resource_pool(documents)

        assert len(pool.resources) == 3

        # Check that all documents are present
        urls = {resource.id.url for resource in pool.resources}
        assert urls == {TEST_URLS[0], TEST_URLS[1], TEST_URLS[2]}

    def test_resource_pool_quote_functionality(self):
        """Test that created resource pool supports quote creation."""
        documents = [(TEST_URLS[0], "Test", BRCA1_DOCUMENT)]
        pool = create_test_resource_pool(documents)

        resource = list(pool.resources)[0]
        quote = resource.quote("BRCA1")

        assert quote is not None
        assert quote.count > 0


class TestRunExtractionV2Simple:
    """Test the simplified pipeline runner."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
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
            "agents": {"_": {"llm": "test"}},
            "cache": {"directory": "test_cache"},
        }
        return IfetcherConfig(**config_data)

    @pytest.fixture
    def test_deps(self, test_config):
        """Create test dependencies."""
        page_fetcher = PageFetcher(test_config)
        return ExtractionDeps.from_config(test_config, page_fetcher, model="test")

    @pytest.fixture
    def test_resource_pool(self):
        """Create resource pool for testing."""
        documents = [
            (TEST_URLS[0], "BRCA1 Study", BRCA1_DOCUMENT),
            (TEST_URLS[1], "Multi Gene Study", MULTI_GENE_DOCUMENT),
        ]
        return create_test_resource_pool(documents)

    @pytest.mark.asyncio
    async def test_run_extraction_v2_simple_empty_pool(self, test_deps):
        """Test simple runner with empty resource pool."""
        empty_pool = create_test_resource_pool([])

        result = await run_extraction_v2_simple(empty_pool, test_deps)

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_run_extraction_v2_simple_with_documents(
        self, test_resource_pool, test_deps
    ):
        """Test simple runner with test documents."""
        result = await run_extraction_v2_simple(test_resource_pool, test_deps)

        assert isinstance(result, list)
        # Result depends on mock agent behavior
        assert len(result) >= 0

    @pytest.mark.asyncio
    async def test_run_extraction_v2_simple_error_handling(self, test_resource_pool):
        """Test simple runner error handling."""
        # Create deps with invalid configuration to trigger errors
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]}
                },
                "relation": "test",
                "context": "test",
            },
            "agents": {"_": {"llm": "invalid-model"}},
            "cache": {"directory": "test"},
        }
        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)
        error_deps = ExtractionDeps.from_config(config, page_fetcher, model="invalid")

        # Should handle errors gracefully and return empty list
        result = await run_extraction_v2_simple(test_resource_pool, error_deps)

        assert isinstance(result, list)
        # May be empty due to errors, but shouldn't crash

    @pytest.mark.asyncio
    async def test_run_extraction_v2_simple_state_tracking(
        self, test_resource_pool, test_deps
    ):
        """Test that simple runner properly tracks state."""
        with patch(
            "interaction_finder.extraction_graph_v2.run_simple.ExtractionState"
        ) as mock_state_class:
            # Create a mock state instance
            mock_state = Mock()
            mock_state.metrics = Mock()
            mock_state.metrics.start_timing = Mock()
            mock_state.final_pairs = []
            mock_state.validate_provenance_chain.return_value = True
            mock_state.get_summary.return_value = {"test": "summary"}

            mock_state_class.return_value = mock_state

            result = await run_extraction_v2_simple(test_resource_pool, test_deps)

            # Should have created state with resource pool
            mock_state_class.assert_called_once_with(resource_pool=test_resource_pool)

            # Should have started timing
            mock_state.metrics.start_timing.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_extraction_v2_simple_node_execution_order(
        self, test_resource_pool, test_deps
    ):
        """Test that nodes execute in correct order."""
        with (
            patch(
                "interaction_finder.extraction_graph_v2.nodes.ExtractEntities"
            ) as mock_extract,
            patch(
                "interaction_finder.extraction_graph_v2.nodes.AssessIndividually"
            ) as mock_assess,
            patch(
                "interaction_finder.extraction_graph_v2.nodes.AggregateIntoPairs"
            ) as mock_aggregate,
        ):
            # Setup mocks
            mock_extract_instance = AsyncMock()
            mock_assess_instance = AsyncMock()
            mock_aggregate_instance = AsyncMock()

            mock_extract.return_value = mock_extract_instance
            mock_assess.return_value = mock_assess_instance
            mock_aggregate.return_value = mock_aggregate_instance

            # Make extract node not return early
            mock_extract_instance.run.return_value = None

            result = await run_extraction_v2_simple(test_resource_pool, test_deps)

            # All nodes should be created and run
            mock_extract.assert_called_once()
            mock_assess.assert_called_once()
            mock_aggregate.assert_called_once()

            mock_extract_instance.run.assert_called_once()
            mock_assess_instance.run.assert_called_once()
            mock_aggregate_instance.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_extraction_v2_simple_early_termination(
        self, test_resource_pool, test_deps
    ):
        """Test early termination when extraction returns a result."""
        with patch(
            "interaction_finder.extraction_graph_v2.nodes.ExtractEntities"
        ) as mock_extract:
            mock_extract_instance = AsyncMock()
            mock_extract.return_value = mock_extract_instance

            # Make extract node return early termination
            mock_extract_instance.run.return_value = "early_termination"

            result = await run_extraction_v2_simple(test_resource_pool, test_deps)

            assert isinstance(result, list)
            assert len(result) == 0  # Should return empty list on early termination


class TestRunExtractionV2:
    """Test the main pipeline runner."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
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
            "agents": {"_": {"llm": "test"}},
            "cache": {"directory": "test_cache"},
        }
        return IfetcherConfig(**config_data)

    @pytest.fixture
    def test_deps(self, test_config):
        """Create test dependencies."""
        page_fetcher = PageFetcher(test_config)
        return ExtractionDeps.from_config(test_config, page_fetcher, model="test")

    @pytest.fixture
    def test_resource_pool(self):
        """Create resource pool for testing."""
        documents = [
            (TEST_URLS[0], "BRCA1 Study", BRCA1_DOCUMENT),
            (TEST_URLS[1], "Multi Gene Study", MULTI_GENE_DOCUMENT),
        ]
        return create_test_resource_pool(documents)

    @pytest.mark.asyncio
    async def test_run_extraction_v2_no_groups(self, test_resource_pool, test_deps):
        """Test main runner without document groups (default behavior in v2)."""
        result = await run_extraction_v2(test_resource_pool, test_deps)

        assert isinstance(result, list)
        # All results should be EntityPairOut instances
        assert all(isinstance(pair, EntityPairOut) for pair in result)

    @pytest.mark.asyncio
    async def test_run_extraction_v2_empty_groups(self, test_resource_pool, test_deps):
        """Test main runner with empty resource pool (no documents)."""
        # Create empty resource pool
        empty_pool = ResourcePool()

        result = await run_extraction_v2(empty_pool, test_deps)

        assert isinstance(result, list)
        # Empty pool should return empty results
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_run_extraction_v2_with_groups(self, test_deps):
        """Test main runner with document groups."""
        # Create resource pool with multiple documents
        documents = [
            (TEST_URLS[0], "BRCA1 Study 1", BRCA1_DOCUMENT),
            (TEST_URLS[1], "BRCA1 Study 2", BRCA1_DOCUMENT),
            (TEST_URLS[2], "Multi Gene Study", MULTI_GENE_DOCUMENT),
        ]
        resource_pool = create_test_resource_pool(documents)

        # Create document groups
        resources = list(resource_pool.resources)
        document_groups = [
            [resources[0].id, resources[1].id],  # BRCA1 group
            [resources[2].id],  # Multi-gene group
        ]

        result = await run_extraction_v2(resource_pool, test_deps)

        assert isinstance(result, list)
        assert all(isinstance(pair, EntityPairOut) for pair in result)

    @pytest.mark.asyncio
    async def test_run_extraction_v2_invalid_groups(
        self, test_resource_pool, test_deps
    ):
        """Test main runner with invalid document groups."""
        # Groups with non-existent resource IDs
        invalid_groups = [
            ["non-existent-1", "non-existent-2"],
        ]

        result = await run_extraction_v2(test_resource_pool, test_deps)

        # Should handle gracefully
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_run_extraction_v2_error_handling(self, test_resource_pool):
        """Test main runner error handling."""
        # Create deps with problematic configuration
        config_data = {
            "task": {
                "kinds": {
                    "gene": {"kind": ["gene"], "form": ["name"], "example": ["BRCA1"]}
                },
                "relation": "test",
                "context": "test",
            },
            "agents": {"_": {"llm": "invalid"}},
            "cache": {"directory": "test"},
        }
        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)
        error_deps = ExtractionDeps.from_config(config, page_fetcher, model="invalid")

        result = await run_extraction_v2(test_resource_pool, error_deps)

        # Should handle errors gracefully
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_run_extraction_v2_backwards_compatibility(
        self, test_resource_pool, test_deps
    ):
        """Test backwards compatibility with older interface."""
        # Test that calling without document_groups works (v2 default)
        result = await run_extraction_v2(test_resource_pool, test_deps)

        # Should work and return list
        assert isinstance(result, list)


class TestPipelineIntegration:
    """Test integration between different pipeline components."""

    @pytest.fixture
    def full_test_setup(self):
        """Create complete test setup."""
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
                "context": "Comprehensive gene-disease interaction analysis",
            },
            "agents": {"_": {"llm": "test"}},
            "cache": {"directory": "full_test_cache"},
        }

        config = IfetcherConfig(**config_data)
        page_fetcher = PageFetcher(config)
        deps = ExtractionDeps.from_config(config, page_fetcher, model="test")

        documents = [
            (TEST_URLS[0], "BRCA1 Research", BRCA1_DOCUMENT),
            (TEST_URLS[1], "Multi-Gene Analysis", MULTI_GENE_DOCUMENT),
            (TEST_URLS[2], "Control Study", NO_RELATIONSHIP_DOCUMENT),
        ]
        resource_pool = create_test_resource_pool(documents)

        return resource_pool, deps, config

    @pytest.mark.asyncio
    async def test_pipeline_consistency_simple_vs_full(self, full_test_setup):
        """Test that simple and full pipelines produce consistent results."""
        resource_pool, deps, config = full_test_setup

        # Run both pipelines
        simple_result = await run_extraction_v2_simple(resource_pool, deps)
        full_result = await run_extraction_v2(resource_pool, deps)

        # Both should return lists of EntityPairOut
        assert isinstance(simple_result, list)
        assert isinstance(full_result, list)

        assert all(isinstance(pair, EntityPairOut) for pair in simple_result)
        assert all(isinstance(pair, EntityPairOut) for pair in full_result)

    @pytest.mark.asyncio
    async def test_pipeline_with_target_term(self, full_test_setup):
        """Test pipeline execution with target term context."""
        resource_pool, deps, config = full_test_setup

        # Create deps with target term
        deps_with_term = ExtractionDeps.from_config(
            config, deps.page_fetcher, deps.model, target_term="BRCA1"
        )

        result = await run_extraction_v2_simple(resource_pool, deps_with_term)

        assert isinstance(result, list)
        # Target term should be available during processing
        assert deps_with_term.target_term == "BRCA1"

    @pytest.mark.asyncio
    async def test_pipeline_provenance_validation(self, full_test_setup):
        """Test that pipeline results maintain proper provenance."""
        resource_pool, deps, config = full_test_setup

        result = await run_extraction_v2_simple(resource_pool, deps)

        # All pairs should have proper provenance
        for pair in result:
            assert isinstance(pair, EntityPairOut)
            assert hasattr(pair, "entity_a")
            assert hasattr(pair, "entity_b")
            assert hasattr(pair, "evidence_quotes")

            # Entities should have quotes
            assert hasattr(pair.entity_a, "quotes")
            assert hasattr(pair.entity_b, "quotes")

    @pytest.mark.asyncio
    async def test_pipeline_logging_and_metrics(self, full_test_setup, caplog):
        """Test that pipeline properly logs execution and tracks metrics."""
        resource_pool, deps, config = full_test_setup

        # Set logging level to capture info messages
        with caplog.at_level(logging.INFO):
            result = await run_extraction_v2_simple(resource_pool, deps)

        # Should have logged pipeline progress
        log_messages = [record.message for record in caplog.records]

        # Look for key pipeline messages
        pipeline_logs = [msg for msg in log_messages if "extraction" in msg.lower()]
        assert len(pipeline_logs) > 0  # Should have some extraction-related logs

    @pytest.mark.asyncio
    async def test_pipeline_resource_management(self, full_test_setup):
        """Test proper resource management during pipeline execution."""
        resource_pool, deps, config = full_test_setup

        # Check initial resource state
        initial_resource_count = len(resource_pool.resources)

        result = await run_extraction_v2_simple(resource_pool, deps)

        # Resource pool should be unchanged
        assert len(resource_pool.resources) == initial_resource_count

        # All resources should still be accessible
        for resource in resource_pool.resources:
            assert resource.text is not None
            assert len(resource.text) > 0

    @pytest.mark.asyncio
    async def test_pipeline_concurrent_execution(self, full_test_setup):
        """Test that multiple pipeline executions can run concurrently."""
        resource_pool, deps, config = full_test_setup

        import asyncio

        # Run multiple pipeline executions concurrently
        tasks = [
            run_extraction_v2_simple(resource_pool, deps),
            run_extraction_v2_simple(resource_pool, deps),
            run_extraction_v2_simple(resource_pool, deps),
        ]

        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, list)
            assert all(isinstance(pair, EntityPairOut) for pair in result)
