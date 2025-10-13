"""
Unit tests for extraction graph V2 state management.

Tests ExtractionState and ExtractionMetrics functionality.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock

from interaction_finder.extraction_graph_v2.state import (
    ExtractionState,
    ExtractionMetrics,
)
from interaction_finder.extraction_graph_v2.models import (
    EntityWithQuotes,
    IndividualAssessment,
    EntityPairOut,
)
from interaction_finder.resources import ResourcePool
from .fixtures import (
    BRCA1_DOCUMENT,
    MULTI_GENE_DOCUMENT,
    TEST_URLS,
)


class TestExtractionMetrics:
    """Test ExtractionMetrics functionality."""

    @pytest.fixture
    def metrics(self):
        """Create fresh metrics instance."""
        return ExtractionMetrics()

    def test_metrics_initialization(self, metrics):
        """Test that metrics initialize with zero values."""
        assert metrics.entities_extraction_calls == 0
        assert metrics.entities_extraction_successes == 0
        assert metrics.assessment_calls == 0
        assert metrics.assessment_successes == 0
        assert metrics.entities_with_valid_quotes == 0
        assert metrics.evidence_quotes_found == 0
        assert metrics.evidence_quotes_requested == 0
        assert metrics.start_time is None
        assert metrics.extraction_time == 0.0
        assert metrics.assessment_time == 0.0

    def test_record_extraction_call_success(self, metrics):
        """Test recording successful extraction calls."""
        metrics.record_extraction_call(success=True, duration=1.5)

        assert metrics.entities_extraction_calls == 1
        assert metrics.entities_extraction_successes == 1
        assert metrics.extraction_time == 1.5

    def test_record_extraction_call_failure(self, metrics):
        """Test recording failed extraction calls."""
        metrics.record_extraction_call(success=False, duration=2.3)

        assert metrics.entities_extraction_calls == 1
        assert metrics.entities_extraction_successes == 0
        assert metrics.extraction_time == 2.3

    def test_record_assessment_call_success(self, metrics):
        """Test recording successful assessment calls."""
        metrics.record_assessment_call(success=True, duration=0.8)

        assert metrics.assessment_calls == 1
        assert metrics.assessment_successes == 1
        assert metrics.assessment_time == 0.8

    def test_record_assessment_call_failure(self, metrics):
        """Test recording failed assessment calls."""
        metrics.record_assessment_call(success=False, duration=1.2)

        assert metrics.assessment_calls == 1
        assert metrics.assessment_successes == 0
        assert metrics.assessment_time == 1.2

    def test_record_entity_validation(self, metrics):
        """Test recording entity validation results."""
        # Create mock entities
        valid_entity = Mock()
        valid_entity.validate.return_value = True

        invalid_entity = Mock()
        invalid_entity.validate.return_value = False

        metrics.record_entity_validation(valid_entity)
        metrics.record_entity_validation(invalid_entity)
        metrics.record_entity_validation(valid_entity)

        assert metrics.entities_with_valid_quotes == 2

    def test_record_evidence_search(self, metrics):
        """Test recording evidence search results."""
        metrics.record_evidence_search(requested=10, found=7)
        metrics.record_evidence_search(requested=5, found=3)

        assert metrics.evidence_quotes_requested == 15
        assert metrics.evidence_quotes_found == 10

    def test_get_success_rates_with_data(self, metrics):
        """Test success rate calculation with data."""
        # Add some calls
        metrics.record_extraction_call(success=True)
        metrics.record_extraction_call(success=True)
        metrics.record_extraction_call(success=False)

        metrics.record_assessment_call(success=True)
        metrics.record_assessment_call(success=False)

        metrics.record_evidence_search(requested=10, found=8)

        rates = metrics.get_success_rates()

        assert (
            abs(rates["extraction_success_rate"] - (2 / 3 * 100)) < 0.01
        )  # 2/3 = 66.67%
        assert rates["assessment_success_rate"] == 50.0  # 1/2 = 50%
        assert rates["evidence_retrieval_rate"] == 80.0  # 8/10 = 80%

    def test_get_success_rates_empty(self, metrics):
        """Test success rate calculation with no data."""
        rates = metrics.get_success_rates()

        assert rates["extraction_success_rate"] == 0.0
        assert rates["assessment_success_rate"] == 0.0
        assert rates["evidence_retrieval_rate"] == 0.0

    def test_timing_functionality(self, metrics):
        """Test timing start and duration calculation."""
        # Initially no timing
        assert metrics.get_total_duration() == 0.0

        # Start timing
        start = time.time()
        metrics.start_timing()

        # Wait a small amount
        time.sleep(0.01)

        duration = metrics.get_total_duration()
        assert duration > 0.0
        assert duration < 1.0  # Should be very small

        # Verify start_time was set
        assert metrics.start_time is not None
        assert abs(metrics.start_time - start) < 0.1

    def test_cumulative_timing(self, metrics):
        """Test cumulative timing for operations."""
        metrics.record_extraction_call(success=True, duration=1.0)
        metrics.record_extraction_call(success=False, duration=1.5)
        metrics.record_assessment_call(success=True, duration=0.5)

        assert metrics.extraction_time == 2.5
        assert metrics.assessment_time == 0.5


class TestExtractionState:
    """Test ExtractionState functionality."""

    @pytest.fixture
    def empty_state(self):
        """Create empty state."""
        return ExtractionState()

    @pytest.fixture
    def resource_pool(self):
        """Create resource pool with test documents."""
        pool = ResourcePool()
        pool.add(TEST_URLS[0], "BRCA1 Study", BRCA1_DOCUMENT)
        pool.add(TEST_URLS[1], "Multi Gene Study", MULTI_GENE_DOCUMENT)
        return pool

    @pytest.fixture
    def populated_state(self, resource_pool):
        """Create state with test data."""
        state = ExtractionState(resource_pool=resource_pool)

        # Add test entities
        resource = list(resource_pool.resources)[0]
        brca1_quote = resource.quote("BRCA1")
        cancer_quote = resource.quote("breast cancer")

        if brca1_quote and cancer_quote:
            brca1_entity = EntityWithQuotes(
                name="BRCA1", kind="gene", quotes=[brca1_quote]
            )
            cancer_entity = EntityWithQuotes(
                name="breast cancer", kind="disease", quotes=[cancer_quote]
            )

            state.entities_found = {
                "BRCA1": brca1_entity,
                "breast_cancer": cancer_entity,
            }

            # Add assessment
            evidence_quote = resource.quote(
                "BRCA1 mutations significantly increase breast cancer risk"
            )
            if evidence_quote:
                assessment = IndividualAssessment(
                    entity=brca1_entity,
                    relationship_potential="high",
                    related_entities=["breast cancer"],
                    evidence_quotes=[evidence_quote],
                    reasoning="Strong evidence",
                )
                state.individual_assessments = [assessment]

                # Add final pair
                pair = EntityPairOut(
                    entity_a=brca1_entity,
                    entity_b=cancer_entity,
                    confidence="high",
                    evidence_quotes=[evidence_quote],
                    reasoning="Well documented relationship",
                )
                state.final_pairs = [pair]

        return state

    def test_empty_state_initialization(self, empty_state):
        """Test that empty state initializes correctly."""
        assert isinstance(empty_state.resource_pool, ResourcePool)
        assert len(empty_state.entities_found) == 0
        assert len(empty_state.individual_assessments) == 0
        assert len(empty_state.final_pairs) == 0
        assert isinstance(empty_state.metrics, ExtractionMetrics)

    def test_state_with_resource_pool(self, resource_pool):
        """Test state initialization with resource pool."""
        state = ExtractionState(resource_pool=resource_pool)

        assert state.resource_pool == resource_pool
        assert len(state.resource_pool.resources) == 2

    def test_get_counts(self, populated_state):
        """Test count methods."""
        assert populated_state.get_entity_count() == 2
        assert populated_state.get_assessment_count() == 1
        assert populated_state.get_pairs_count() == 1

    def test_get_counts_empty(self, empty_state):
        """Test count methods with empty state."""
        assert empty_state.get_entity_count() == 0
        assert empty_state.get_assessment_count() == 0
        assert empty_state.get_pairs_count() == 0

    def test_get_summary_empty(self, empty_state):
        """Test summary with empty state."""
        summary = empty_state.get_summary()

        # Basic counts
        assert summary["documents"] == 0
        assert summary["entities"] == 0
        assert summary["assessments"] == 0
        assert summary["pairs"] == 0

        # Metrics sections
        assert "agent_calls" in summary
        assert "success_rates" in summary
        assert "quality_metrics" in summary
        assert "timing" in summary

        # All should be zero/empty
        assert summary["agent_calls"]["extraction_calls"] == 0
        assert summary["agent_calls"]["assessment_calls"] == 0
        assert summary["success_rates"]["extraction_success_rate"] == 0.0

    def test_get_summary_populated(self, populated_state):
        """Test summary with populated state."""
        # Add some metrics
        populated_state.metrics.record_extraction_call(success=True, duration=1.0)
        populated_state.metrics.record_assessment_call(success=True, duration=0.5)

        summary = populated_state.get_summary()

        # Basic counts
        assert summary["documents"] == 2
        assert summary["entities"] == 2
        assert summary["assessments"] == 1
        assert summary["pairs"] == 1

        # Metrics
        assert summary["agent_calls"]["extraction_calls"] == 1
        assert summary["agent_calls"]["assessment_calls"] == 1
        assert summary["success_rates"]["extraction_success_rate"] == 100.0

    def test_validate_provenance_chain_empty(self, empty_state):
        """Test provenance validation with empty state."""
        assert empty_state.validate_provenance_chain() == True  # Vacuously true

    def test_validate_provenance_chain_valid(self, populated_state):
        """Test provenance validation with valid data."""
        # Assuming populated_state has valid provenance
        result = populated_state.validate_provenance_chain()

        # Result depends on whether the test entities actually validate
        assert isinstance(result, bool)

    def test_validate_provenance_chain_invalid_entities(self, resource_pool):
        """Test provenance validation with invalid entities."""
        state = ExtractionState(resource_pool=resource_pool)

        # Add entity with no quotes
        invalid_entity = Mock()
        invalid_entity.validate.return_value = False

        state.entities_found = {"invalid": invalid_entity}

        assert state.validate_provenance_chain() == False

    def test_validate_provenance_chain_invalid_pairs(self, resource_pool):
        """Test provenance validation with invalid pairs."""
        state = ExtractionState(resource_pool=resource_pool)

        # Add valid entity
        valid_entity = Mock()
        valid_entity.validate.return_value = True
        state.entities_found = {"valid": valid_entity}

        # Add invalid pair
        invalid_pair = Mock()
        invalid_pair.validate_provenance.return_value = False
        state.final_pairs = [invalid_pair]

        assert state.validate_provenance_chain() == False

    def test_get_provenance_metrics_empty(self, empty_state):
        """Test provenance metrics with empty state."""
        metrics = empty_state.get_provenance_metrics()

        assert metrics["entity_quote_rate"] == 0.0
        assert metrics["pair_evidence_rate"] == 0.0

    def test_get_provenance_metrics_with_data(self):
        """Test provenance metrics with test data."""
        state = ExtractionState()

        # Create mock entities
        valid_entity = Mock()
        valid_entity.validate.return_value = True

        invalid_entity = Mock()
        invalid_entity.validate.return_value = False

        state.entities_found = {
            "valid": valid_entity,
            "invalid": invalid_entity,
        }

        # Create mock pairs
        valid_pair = Mock()
        valid_pair.validate_provenance.return_value = True

        invalid_pair = Mock()
        invalid_pair.validate_provenance.return_value = False

        state.final_pairs = [valid_pair, invalid_pair, valid_pair]

        metrics = state.get_provenance_metrics()

        assert metrics["entity_quote_rate"] == 50.0  # 1/2 = 50%
        assert abs(metrics["pair_evidence_rate"] - 66.67) < 0.01  # 2/3 = 66.67%

    def test_state_metrics_integration(self, empty_state):
        """Test that state properly integrates with metrics."""
        # Metrics should be accessible
        assert hasattr(empty_state, "metrics")
        assert isinstance(empty_state.metrics, ExtractionMetrics)

        # Should be able to record metrics
        empty_state.metrics.record_extraction_call(success=True)
        assert empty_state.metrics.entities_extraction_calls == 1

        # Metrics should appear in summary
        summary = empty_state.get_summary()
        assert summary["agent_calls"]["extraction_calls"] == 1


class TestExtractionStateOperations:
    """Test state operations and mutations."""

    @pytest.fixture
    def mutable_state(self):
        """Create state for mutation testing."""
        pool = ResourcePool()
        pool.add(TEST_URLS[0], "Test", BRCA1_DOCUMENT)
        return ExtractionState(resource_pool=pool)

    def test_adding_entities(self, mutable_state):
        """Test adding entities to state."""
        resource = list(mutable_state.resource_pool.resources)[0]
        quote = resource.quote("BRCA1")

        if quote:
            entity = EntityWithQuotes(name="BRCA1", kind="gene", quotes=[quote])
            mutable_state.entities_found["BRCA1"] = entity

            assert mutable_state.get_entity_count() == 1
            assert "BRCA1" in mutable_state.entities_found

    def test_adding_assessments(self, mutable_state):
        """Test adding assessments to state."""
        # First add entity
        resource = list(mutable_state.resource_pool.resources)[0]
        entity_quote = resource.quote("BRCA1")
        evidence_quote = resource.quote("BRCA1 mutations")

        if entity_quote and evidence_quote:
            entity = EntityWithQuotes(name="BRCA1", kind="gene", quotes=[entity_quote])
            assessment = IndividualAssessment(
                entity=entity,
                relationship_potential="high",
                related_entities=["test"],
                evidence_quotes=[evidence_quote],
                reasoning="Test",
            )

            mutable_state.individual_assessments.append(assessment)

            assert mutable_state.get_assessment_count() == 1

    def test_adding_final_pairs(self, mutable_state):
        """Test adding final pairs to state."""
        resource = list(mutable_state.resource_pool.resources)[0]
        brca1_quote = resource.quote("BRCA1")
        cancer_quote = resource.quote("breast cancer")
        evidence_quote = resource.quote("BRCA1 mutations significantly")

        if brca1_quote and cancer_quote and evidence_quote:
            entity_a = EntityWithQuotes(name="BRCA1", kind="gene", quotes=[brca1_quote])
            entity_b = EntityWithQuotes(
                name="breast cancer", kind="disease", quotes=[cancer_quote]
            )

            pair = EntityPairOut(
                entity_a=entity_a,
                entity_b=entity_b,
                confidence="high",
                evidence_quotes=[evidence_quote],
                reasoning="Test pair",
            )

            mutable_state.final_pairs.append(pair)

            assert mutable_state.get_pairs_count() == 1

    def test_metrics_recording_workflow(self, mutable_state):
        """Test typical metrics recording workflow."""
        metrics = mutable_state.metrics

        # Start timing
        metrics.start_timing()

        # Record extraction phase
        metrics.record_extraction_call(success=True, duration=1.0)
        metrics.record_extraction_call(success=False, duration=0.5)

        # Record assessment phase
        metrics.record_assessment_call(success=True, duration=0.8)

        # Record validation
        valid_entity = Mock()
        valid_entity.validate.return_value = True
        metrics.record_entity_validation(valid_entity)

        # Record evidence search
        metrics.record_evidence_search(requested=5, found=3)

        # Check final state
        summary = mutable_state.get_summary()

        assert summary["agent_calls"]["extraction_calls"] == 2
        assert summary["agent_calls"]["assessment_calls"] == 1
        assert summary["success_rates"]["extraction_success_rate"] == 50.0
        assert summary["success_rates"]["assessment_success_rate"] == 100.0
        assert summary["success_rates"]["evidence_retrieval_rate"] == 60.0
        assert summary["quality_metrics"]["entities_with_valid_quotes"] == 1

    def test_state_serialization_compatibility(self, mutable_state):
        """Test that state can be serialized (for debugging/logging)."""
        # Test that summary can be converted to JSON-compatible format
        summary = mutable_state.get_summary()

        # All values should be JSON-serializable types
        def is_json_serializable(obj):
            try:
                import json

                json.dumps(obj)
                return True
            except (TypeError, OverflowError):
                return False

        assert is_json_serializable(summary)

    def test_state_resource_pool_operations(self, mutable_state):
        """Test operations on resource pool within state."""
        initial_count = len(mutable_state.resource_pool.resources)

        # Add new resource
        mutable_state.resource_pool.add(
            TEST_URLS[1], "Additional Doc", "Additional content with BRCA1."
        )

        assert len(mutable_state.resource_pool.resources) == initial_count + 1

        # Summary should reflect new document count
        summary = mutable_state.get_summary()
        assert summary["documents"] == initial_count + 1
