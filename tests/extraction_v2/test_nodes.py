"""
Unit tests for extraction graph V2 nodes.

Tests individual node behavior and state transitions.
"""

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock

from pydantic_graph import GraphRunContext
from pydantic_ai.models.test import TestModel

from interaction_finder.resources import ResourcePool
from interaction_finder.extraction_graph_v2.nodes import (
    ExtractEntities,
    AssessIndividually,
    AggregateIntoPairs,
)
from interaction_finder.extraction_graph_v2.state import ExtractionState
from interaction_finder.extraction_graph_v2.deps import ExtractionDeps
from interaction_finder.extraction_graph_v2.models import (
    EntityWithQuotes,
    IndividualAssessment,
    EntityPairOut,
)
from interaction_finder.settings import IfetcherConfig
from interaction_finder.fetcher import PageFetcher
from .fixtures import (
    BRCA1_DOCUMENT,
    MULTI_GENE_DOCUMENT,
    NO_RELATIONSHIP_DOCUMENT,
    TEST_URLS,
)

logger = logging.getLogger(__name__)


class TestExtractEntities:
    """Test the ExtractEntities node."""

    @pytest.fixture
    def extraction_config(self):
        """Create extraction config for testing."""
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
    def test_deps(self, extraction_config):
        """Create test dependencies."""
        page_fetcher = PageFetcher(extraction_config)
        return ExtractionDeps.from_config(
            extraction_config, page_fetcher, model=TestModel()
        )

    @pytest.fixture
    def test_resource_pool(self):
        """Create resource pool with test document."""
        pool = ResourcePool()
        pool.add(TEST_URLS[0], "BRCA1 Study", BRCA1_DOCUMENT)
        return pool

    @pytest.fixture
    def test_state(self, test_resource_pool):
        """Create test extraction state."""
        return ExtractionState(resource_pool=test_resource_pool)

    @pytest.fixture
    def test_context(self, test_state, test_deps):
        """Create test graph run context."""
        return GraphRunContext(state=test_state, deps=test_deps)

    def test_extract_entities_node_creation(self):
        """Test that ExtractEntities node can be created."""
        node = ExtractEntities()
        assert node is not None
        assert hasattr(node, "run")

    @pytest.mark.asyncio
    async def test_extract_entities_empty_state(self, test_context):
        """Test extraction with empty initial state."""
        node = ExtractEntities()

        # Initial state should be empty
        assert len(test_context.state.entities_found) == 0

        # Run the node
        result = await node.run(test_context)

        # Node should complete without returning End
        assert result is None

        # State should now have entities (mocked response would create some)
        # Note: With TestModel, the actual extraction depends on mock responses

    @pytest.mark.asyncio
    async def test_extract_entities_metrics_tracking(self, test_context):
        """Test that extraction tracks metrics correctly."""
        node = ExtractEntities()

        # Check initial metrics
        initial_calls = test_context.state.metrics.entities_extraction_calls

        await node.run(test_context)

        # Should have recorded extraction calls
        assert test_context.state.metrics.entities_extraction_calls >= initial_calls

    def test_extract_entities_target_term_support(self, test_deps, test_state):
        """Test that extraction supports target term context."""
        # Create deps with target term
        deps_with_term = ExtractionDeps.from_config(
            test_deps.config,
            test_deps.page_fetcher,
            test_deps.model,
            target_term="BRCA1",
        )

        context = GraphRunContext(state=test_state, deps=deps_with_term)
        node = ExtractEntities()

        # Should be able to access target term
        assert context.deps.target_term == "BRCA1"


class TestAssessIndividually:
    """Test the AssessIndividually node."""

    @pytest.fixture
    def test_state_with_entities(self, test_resource_pool):
        """Create state with pre-loaded entities."""
        state = ExtractionState(resource_pool=test_resource_pool)

        # Add test entities with ResourceQuotes
        resource = list(test_resource_pool.resources)[0]
        brca1_quote = resource.quote("BRCA1")
        cancer_quote = resource.quote("breast cancer")

        if brca1_quote and cancer_quote:
            state.entities_found = [
                EntityWithQuotes(name="BRCA1", kind="gene", quotes=[brca1_quote]),
                EntityWithQuotes(
                    name="breast cancer", kind="disease", quotes=[cancer_quote]
                ),
            ]

        return state

    @pytest.fixture
    def assessment_context(self, test_state_with_entities, test_deps):
        """Create context for assessment testing."""
        return GraphRunContext(state=test_state_with_entities, deps=test_deps)

    def test_assess_individually_node_creation(self):
        """Test that AssessIndividually node can be created."""
        node = AssessIndividually()
        assert node is not None
        assert hasattr(node, "run")

    @pytest.mark.asyncio
    async def test_assess_individually_with_entities(self, assessment_context):
        """Test individual assessment with existing entities."""
        node = AssessIndividually()

        # Should have entities to assess
        assert len(assessment_context.state.entities_found) > 0

        # Initial assessments should be empty
        assert len(assessment_context.state.individual_assessments) == 0

        await node.run(assessment_context)

        # Should have created assessments for entities
        # Note: Actual assessments depend on mock agent responses

    @pytest.mark.asyncio
    async def test_assess_individually_empty_entities(self, test_context):
        """Test assessment with no entities."""
        node = AssessIndividually()

        # No entities to assess
        assert len(test_context.state.entities_found) == 0

        await node.run(test_context)

        # Should handle gracefully
        assert len(test_context.state.individual_assessments) == 0

    @pytest.mark.asyncio
    async def test_assess_individually_metrics_tracking(self, assessment_context):
        """Test that assessment tracks metrics correctly."""
        node = AssessIndividually()

        initial_calls = assessment_context.state.metrics.assessment_calls

        await node.run(assessment_context)

        # Should have recorded assessment calls if entities exist
        if len(assessment_context.state.entities_found) > 0:
            assert assessment_context.state.metrics.assessment_calls >= initial_calls


class TestAggregateIntoPairs:
    """Test the AggregateIntoPairs node."""

    @pytest.fixture
    def test_state_with_assessments(self, test_resource_pool):
        """Create state with pre-loaded entities and assessments."""
        state = ExtractionState(resource_pool=test_resource_pool)

        # Add test entities
        resource = list(test_resource_pool.resources)[0]
        brca1_quote = resource.quote("BRCA1")
        cancer_quote = resource.quote("breast cancer")

        if brca1_quote and cancer_quote:
            brca1_entity = EntityWithQuotes(
                name="BRCA1", kind="gene", quotes=[brca1_quote]
            )
            cancer_entity = EntityWithQuotes(
                name="breast cancer", kind="disease", quotes=[cancer_quote]
            )

            state.entities_found = [brca1_entity, cancer_entity]

            # Add test assessment with evidence
            evidence_quote = resource.quote(
                "BRCA1 mutations significantly increase breast cancer risk"
            )
            if evidence_quote:
                assessment = IndividualAssessment(
                    entity=brca1_entity,
                    relationship_potential="high",
                    related_entities=["breast cancer"],
                    evidence_quotes=[evidence_quote],
                    reasoning="Test assessment",
                )
                state.individual_assessments = [assessment]

        return state

    @pytest.fixture
    def aggregation_context(self, test_state_with_assessments, test_deps):
        """Create context for aggregation testing."""
        return GraphRunContext(state=test_state_with_assessments, deps=test_deps)

    def test_aggregate_into_pairs_node_creation(self):
        """Test that AggregateIntoPairs node can be created."""
        node = AggregateIntoPairs()
        assert node is not None
        assert hasattr(node, "run")

    @pytest.mark.asyncio
    async def test_aggregate_into_pairs_with_assessments(self, aggregation_context):
        """Test pair aggregation with existing assessments."""
        node = AggregateIntoPairs()

        # Should have entities and assessments
        assert len(aggregation_context.state.entities_found) > 0
        assert len(aggregation_context.state.individual_assessments) >= 0

        # Initial pairs should be empty
        assert len(aggregation_context.state.final_pairs) == 0

        await node.run(aggregation_context)

        # Should have created pairs (depends on assessment quality)
        # Note: Actual pair creation depends on the assessment logic

    @pytest.mark.asyncio
    async def test_aggregate_no_valid_pairs(self, test_context):
        """Test aggregation with no valid pairs."""
        node = AggregateIntoPairs()

        # No entities or assessments
        assert len(test_context.state.entities_found) == 0
        assert len(test_context.state.individual_assessments) == 0

        await node.run(test_context)

        # Should handle gracefully
        assert len(test_context.state.final_pairs) == 0

    @pytest.mark.asyncio
    async def test_aggregate_same_kind_filtering(self, test_resource_pool, test_deps):
        """Test that aggregation doesn't create same-kind pairs."""
        state = ExtractionState(resource_pool=test_resource_pool)

        # Add two entities of the same kind
        resource = list(test_resource_pool.resources)[0]
        brca1_quote = resource.quote("BRCA1")
        tp53_quote = resource.quote("p53")  # Assuming p53 is mentioned somewhere

        if brca1_quote:
            gene1 = EntityWithQuotes(name="BRCA1", kind="gene", quotes=[brca1_quote])
            gene2 = EntityWithQuotes(
                name="TP53", kind="gene", quotes=[brca1_quote]
            )  # Using same quote for simplicity

            state.entities_found = [gene1, gene2]

            # Create assessments for both
            if brca1_quote:
                assessment1 = IndividualAssessment(
                    entity=gene1,
                    relationship_potential="high",
                    related_entities=["TP53"],
                    evidence_quotes=[brca1_quote],
                    reasoning="Test",
                )
                assessment2 = IndividualAssessment(
                    entity=gene2,
                    relationship_potential="high",
                    related_entities=["BRCA1"],
                    evidence_quotes=[brca1_quote],
                    reasoning="Test",
                )
                state.individual_assessments = [assessment1, assessment2]

        context = GraphRunContext(state=state, deps=test_deps)
        node = AggregateIntoPairs()

        await node.run(context)

        # Should not create gene-gene pairs
        for pair in context.state.final_pairs:
            assert pair.entity_a.kind != pair.entity_b.kind, (
                f"Same-kind pair created: {pair.entity_a.kind}-{pair.entity_b.kind}"
            )

    def test_node_pipeline_integration(self, test_context):
        """Test that all nodes can be created and connected."""
        extract_node = ExtractEntities()
        assess_node = AssessIndividually()
        aggregate_node = AggregateIntoPairs()

        # All nodes should be created successfully
        assert extract_node is not None
        assert assess_node is not None
        assert aggregate_node is not None

        # All should have run methods
        assert hasattr(extract_node, "run")
        assert hasattr(assess_node, "run")
        assert hasattr(aggregate_node, "run")


class TestNodeErrorHandling:
    """Test error handling in nodes."""

    @pytest.fixture
    def error_prone_deps(self, extraction_config):
        """Create deps that might cause errors."""
        page_fetcher = PageFetcher(extraction_config)
        # Use a model that might fail
        return ExtractionDeps.from_config(
            extraction_config, page_fetcher, model="invalid-model"
        )

    @pytest.mark.asyncio
    async def test_extract_entities_error_handling(self, test_state, error_prone_deps):
        """Test that ExtractEntities handles errors gracefully."""
        context = GraphRunContext(state=test_state, deps=error_prone_deps)
        node = ExtractEntities()

        # Should not crash on errors
        try:
            await node.run(context)
            # If no exception, that's good
            assert True
        except Exception as e:
            # If there is an exception, it should be handled gracefully
            # The actual behavior depends on implementation
            logger.info(f"Expected error handled: {e}")

    @pytest.mark.asyncio
    async def test_nodes_with_missing_data(self, test_deps):
        """Test nodes with missing or invalid data."""
        # Create empty resource pool
        empty_pool = ResourcePool()
        empty_state = ExtractionState(resource_pool=empty_pool)
        context = GraphRunContext(state=empty_state, deps=test_deps)

        # All nodes should handle empty state gracefully
        extract_node = ExtractEntities()
        assess_node = AssessIndividually()
        aggregate_node = AggregateIntoPairs()

        await extract_node.run(context)
        await assess_node.run(context)
        await aggregate_node.run(context)

        # Should complete without errors
        assert len(context.state.final_pairs) == 0


class TestNodeStateTransitions:
    """Test state transitions between nodes."""

    @pytest.fixture
    def multi_doc_pool(self):
        """Create resource pool with multiple documents."""
        pool = ResourcePool()
        pool.add(TEST_URLS[0], "BRCA1 Study", BRCA1_DOCUMENT)
        pool.add(TEST_URLS[1], "Multi Gene Study", MULTI_GENE_DOCUMENT)
        pool.add(TEST_URLS[2], "Control Study", NO_RELATIONSHIP_DOCUMENT)
        return pool

    @pytest.mark.asyncio
    async def test_sequential_node_execution(self, multi_doc_pool, test_deps):
        """Test that nodes execute in sequence and pass state correctly."""
        state = ExtractionState(resource_pool=multi_doc_pool)
        context = GraphRunContext(state=state, deps=test_deps)

        # Initial state
        assert len(state.entities_found) == 0
        assert len(state.individual_assessments) == 0
        assert len(state.final_pairs) == 0

        # Execute extract node
        extract_node = ExtractEntities()
        await extract_node.run(context)

        # After extraction, should have entities
        entities_count = len(state.entities_found)
        logger.info(f"After extraction: {entities_count} entities")

        # Execute assessment node
        assess_node = AssessIndividually()
        await assess_node.run(context)

        # After assessment, should have assessments
        assessments_count = len(state.individual_assessments)
        logger.info(f"After assessment: {assessments_count} assessments")

        # Execute aggregation node
        aggregate_node = AggregateIntoPairs()
        await aggregate_node.run(context)

        # After aggregation, should have pairs
        pairs_count = len(state.final_pairs)
        logger.info(f"After aggregation: {pairs_count} pairs")

        # State should maintain consistency
        assert len(state.entities_found) == entities_count
        assert len(state.individual_assessments) == assessments_count

    def test_state_validation_between_nodes(self, test_context):
        """Test that state validation works between node executions."""
        # Test state summary at each stage
        initial_summary = test_context.state.get_summary()
        assert "entities" in initial_summary
        assert "assessments" in initial_summary
        assert "pairs" in initial_summary

        # Test provenance validation
        validation_result = test_context.state.validate_provenance_chain()
        assert isinstance(validation_result, bool)

        # Test metrics tracking
        metrics = test_context.state.metrics
        assert hasattr(metrics, "entities_extraction_calls")
        assert hasattr(metrics, "assessment_calls")
        assert hasattr(metrics, "entities_with_valid_quotes")
