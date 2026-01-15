"""Tests for relationship label consolidation and polarity classification.

Tests the consolidation of relationship labels (e.g., "linked_to" → "associated_with")
and the application of polarity mappings.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from interaction_finder.extraction.stages.consolidate_relationships import (
    _collect_unique_relationships,
    _apply_consolidations,
    _store_polarity_mappings,
    consolidate_relationships,
)
from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    EvidenceQuality,
    PairAssessment,
    RelationshipConsolidation,
)
from interaction_finder.extraction.state import State
from interaction_finder.resources import ResourceId


def make_evidence(level: int = 5) -> EvidenceQuality:
    return EvidenceQuality(
        directness="explicit",
        source_type="primary",
        specificity="mechanistic",
        language="definitive",
        overall=level,
    )


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    return {kind: {kind} for kind in kinds}


def make_assessment(
    resource: ResourceId, relationship: str = "associated_with"
) -> PairAssessment:
    entity1 = EntityMention(
        kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="test"
    )
    entity2 = EntityMention(
        kind="disease", name="Cancer", aliases=[], quotes=[], reasoning="test"
    )
    return PairAssessment(
        topic_relevance=3,
        resource_id=resource,
        entity1=EntityRef(canonical="BRCA1", mentions=[entity1]),
        entity2=EntityRef(canonical="Cancer", mentions=[entity2]),
        relationship=relationship,
        quotes=[],
        evidence=make_evidence(7),
        reasoning="Test",
    )


class TestCollectUniqueRelationships:
    """Test collecting unique relationship labels from assessments."""

    def test_single_document_single_relationship(self):
        """Collects single relationship from one document."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://example.com/doc1", counter=0)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "associated_with")
        ]
        unique = _collect_unique_relationships(state)
        assert unique == {"associated_with"}

    def test_multiple_documents_multiple_relationships(self):
        """Collects unique relationships across multiple documents."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)
        state.pair_assessments_by_resource[resource1] = [
            make_assessment(resource1, "associated_with"),
            make_assessment(resource1, "regulates"),
        ]
        state.pair_assessments_by_resource[resource2] = [
            make_assessment(resource2, "inhibits"),
            make_assessment(resource2, "associated_with"),  # Duplicate
        ]
        unique = _collect_unique_relationships(state)
        assert unique == {"associated_with", "regulates", "inhibits"}

    def test_empty_assessments(self):
        """Returns empty set when no assessments."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        unique = _collect_unique_relationships(state)
        assert unique == set()

    def test_deduplicates_within_document(self):
        """Same relationship in one document counted once."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://example.com/doc1", counter=0)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "activates"),
            make_assessment(resource, "activates"),
            make_assessment(resource, "activates"),
        ]
        unique = _collect_unique_relationships(state)
        assert unique == {"activates"}


class TestApplyConsolidations:
    """Test applying relationship consolidations to assessments."""

    def test_consolidates_single_relationship(self):
        """Single relationship label is consolidated."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://example.com/doc1", counter=0)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "linked_to")
        ]
        consolidations = [
            RelationshipConsolidation(
                original="linked_to",
                consolidated="associated_with",
                polarity="positive",
                reasoning="linked_to is a synonym for associated_with in this context",
            )
        ]
        _apply_consolidations(consolidations, state)
        assert (
            state.pair_assessments_by_resource[resource][0].relationship
            == "associated_with"
        )

    def test_no_change_when_same_label(self):
        """No change when original equals consolidated."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://example.com/doc1", counter=0)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "associated_with")
        ]
        consolidations = [
            RelationshipConsolidation(
                original="associated_with",
                consolidated="associated_with",
                polarity="positive",
                reasoning="associated_with is already the canonical form for this relationship",
            )
        ]
        _apply_consolidations(consolidations, state)
        assert (
            state.pair_assessments_by_resource[resource][0].relationship
            == "associated_with"
        )

    def test_multiple_consolidations_applied(self):
        """Multiple different consolidations all applied."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://example.com/doc1", counter=0)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "linked_to"),
            make_assessment(resource, "related_to"),
        ]
        consolidations = [
            RelationshipConsolidation(
                original="linked_to",
                consolidated="associated_with",
                polarity="positive",
                reasoning="linked_to is a synonym for associated_with in this context",
            ),
            RelationshipConsolidation(
                original="related_to",
                consolidated="associated_with",
                polarity="positive",
                reasoning="related_to is a synonym for associated_with in this context",
            ),
        ]
        _apply_consolidations(consolidations, state)
        assert (
            state.pair_assessments_by_resource[resource][0].relationship
            == "associated_with"
        )
        assert (
            state.pair_assessments_by_resource[resource][1].relationship
            == "associated_with"
        )

    def test_tracks_merge_count(self):
        """relationships_merged counter is incremented."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://example.com/doc1", counter=0)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "linked_to"),
            make_assessment(resource, "linked_to"),
        ]
        assert state.relationships_merged == 0
        consolidations = [
            RelationshipConsolidation(
                original="linked_to",
                consolidated="associated_with",
                polarity="positive",
                reasoning="linked_to is a synonym for associated_with in this context",
            )
        ]
        _apply_consolidations(consolidations, state)
        assert state.relationships_merged == 2  # Both assessments updated


class TestStorePolarityMappings:
    """Test storing polarity mappings from consolidations."""

    def test_stores_original_polarity(self):
        """Original relationship gets polarity mapping."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        consolidations = [
            RelationshipConsolidation(
                original="activates",
                consolidated="activates",
                polarity="positive",
                reasoning="activates indicates a positive regulatory relationship",
            )
        ]
        _store_polarity_mappings(consolidations, state)
        assert state.relationship_polarities["activates"] == "positive"

    def test_stores_consolidated_polarity(self):
        """Consolidated relationship also gets polarity mapping."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        consolidations = [
            RelationshipConsolidation(
                original="linked_to",
                consolidated="associated_with",
                polarity="neutral",
                reasoning="linked_to is a generic association without clear directionality",
            )
        ]
        _store_polarity_mappings(consolidations, state)
        assert state.relationship_polarities["linked_to"] == "neutral"
        assert state.relationship_polarities["associated_with"] == "neutral"

    def test_multiple_polarities(self):
        """Multiple consolidations with different polarities stored."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        consolidations = [
            RelationshipConsolidation(
                original="activates",
                consolidated="activates",
                polarity="positive",
                reasoning="activates indicates a positive regulatory relationship",
            ),
            RelationshipConsolidation(
                original="inhibits",
                consolidated="inhibits",
                polarity="negative",
                reasoning="inhibits indicates a negative regulatory relationship",
            ),
            RelationshipConsolidation(
                original="co-occurs",
                consolidated="associated_with",
                polarity="neutral",
                reasoning="co-occurs is a generic association without clear directionality",
            ),
        ]
        _store_polarity_mappings(consolidations, state)
        assert state.relationship_polarities["activates"] == "positive"
        assert state.relationship_polarities["inhibits"] == "negative"
        assert state.relationship_polarities["co-occurs"] == "neutral"
        assert state.relationship_polarities["associated_with"] == "neutral"


class TestLLMErrorHandling:
    """Test that LLM errors are handled gracefully."""

    @pytest.fixture
    def mock_deps(self):
        """Create mock dependencies for async tests."""
        deps = MagicMock()
        deps.config = MagicMock()
        deps.config.stage.extraction.filter_irrelevant_relationships = False
        deps.logger = MagicMock()
        deps.agent_semaphore = AsyncMock()
        deps.agent_semaphore.__aenter__ = AsyncMock(return_value=None)
        deps.agent_semaphore.__aexit__ = AsyncMock(return_value=None)
        deps.checkpoint_path = None
        deps.input_checkpoint = None
        deps.progress = None
        return deps

    @pytest.mark.asyncio
    async def test_llm_error_returns_true_and_logs(self, mock_deps):
        """LLM error doesn't crash pipeline, logs error, continues with neutral polarity."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://example.com/doc1", counter=0)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "associated_with")
        ]
        # Mock agent that raises an error
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=ValueError("LLM API error"))
        mock_agent._name = "mock_agent"
        with patch(
            "interaction_finder.extraction.stages.consolidate_relationships.get_relationship_consolidation_agent",
            return_value=mock_agent,
        ):
            result = await consolidate_relationships(state, mock_deps)
        # Should return True (continue pipeline) even on error
        assert result is True
        # Error should be logged
        mock_deps.logger.error.assert_called_once()
        error_msg = mock_deps.logger.error.call_args[0][0]
        assert "ValueError" in error_msg
        # Relationships should get neutral polarity fallback
        assert state.relationship_polarities["associated_with"] == "neutral"
        # No consolidations applied
        assert state.relationships_merged == 0

    @pytest.mark.asyncio
    async def test_timeout_error_handled(self, mock_deps):
        """TimeoutError is caught and handled gracefully."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://example.com/doc1", counter=0)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "regulates")
        ]
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=TimeoutError("Request timed out"))
        mock_agent._name = "mock_agent"
        with patch(
            "interaction_finder.extraction.stages.consolidate_relationships.get_relationship_consolidation_agent",
            return_value=mock_agent,
        ):
            result = await consolidate_relationships(state, mock_deps)
        assert result is True
        mock_deps.logger.error.assert_called_once()
        # Fallback polarity assigned
        assert state.relationship_polarities["regulates"] == "neutral"

    @pytest.mark.asyncio
    async def test_connection_error_handled(self, mock_deps):
        """ConnectionError is caught and handled gracefully."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        resource = ResourceId(url="https://example.com/doc1", counter=0)
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "inhibits")
        ]
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=ConnectionError("Network unavailable"))
        mock_agent._name = "mock_agent"
        with patch(
            "interaction_finder.extraction.stages.consolidate_relationships.get_relationship_consolidation_agent",
            return_value=mock_agent,
        ):
            result = await consolidate_relationships(state, mock_deps)
        assert result is True
        # Fallback polarity assigned
        assert state.relationship_polarities["inhibits"] == "neutral"


class TestCanonicalRelationshipFiltering:
    """Test that only canonical relationships are exposed to downstream stages."""

    def test_relationship_mappings_excludes_unconsolidated_from_known(self):
        """Unconsolidated labels should be filtered from known_relationships for sweep."""
        from interaction_finder.extraction.utils import normalize_for_comparison

        state = State(
            topic="test",
            target_entity_types=["gene", "cell_type"],
            permitted_pairs=build_permitted_pairs(["gene", "cell_type"]),
        )
        # Simulate consolidation: marker_for -> marks, marker_of -> marks
        state.relationship_polarities = {
            "marker_for": "neutral",  # unconsolidated (original)
            "marker_of": "neutral",  # unconsolidated (original)
            "marks": "neutral",  # canonical
            "associated_with": "neutral",  # canonical (unchanged)
            "expressed_in": "neutral",  # canonical
            "expressed_by": "neutral",  # unconsolidated (original)
        }
        state.relationship_mappings = {
            normalize_for_comparison("marker_for"): "marks",
            normalize_for_comparison("marker_of"): "marks",
            normalize_for_comparison("expressed_by"): "expressed_in",
        }
        # Filter to only canonical labels (as done in sweep stage)
        unconsolidated_normalized = set(state.relationship_mappings.keys())
        known_relationships = sorted(
            label
            for label in state.relationship_polarities.keys()
            if normalize_for_comparison(label) not in unconsolidated_normalized
        )
        # Should only include canonical labels
        assert "marks" in known_relationships
        assert "associated_with" in known_relationships
        assert "expressed_in" in known_relationships
        # Should exclude unconsolidated labels
        assert "marker_for" not in known_relationships
        assert "marker_of" not in known_relationships
        assert "expressed_by" not in known_relationships

    def test_empty_relationship_mappings_keeps_all(self):
        """When no consolidation happened, all labels should be available."""
        from interaction_finder.extraction.utils import normalize_for_comparison

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        state.relationship_polarities = {
            "activates": "positive",
            "inhibits": "negative",
            "associated_with": "neutral",
        }
        state.relationship_mappings = {}  # No consolidation
        unconsolidated_normalized = set(state.relationship_mappings.keys())
        known_relationships = sorted(
            label
            for label in state.relationship_polarities.keys()
            if normalize_for_comparison(label) not in unconsolidated_normalized
        )
        # All labels should be included when no consolidation
        assert known_relationships == ["activates", "associated_with", "inhibits"]
