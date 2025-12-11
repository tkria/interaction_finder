"""Tests for relationship consolidation and polarity filtering.

Tests cover:
1. _collect_unique_relationships - deduplication across documents
2. _apply_consolidations - relationship label updates
3. _store_polarity_mappings - polarity storage
4. _filter_irrelevant_assessments - filtering logic
5. _ensure_polarities_for_all_relationships - fallback polarity assignment
6. End-to-end relationship consolidation integration
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    EvidenceQuality,
    PairAssessment,
    RelationshipConsolidation,
    RelationshipConsolidations,
)
from interaction_finder.extraction.stages.consolidate_relationships import (
    _apply_consolidations,
    _collect_unique_relationships,
    _ensure_polarities_for_all_relationships,
    _filter_irrelevant_assessments,
    _store_polarity_mappings,
    consolidate_relationships,
)
from interaction_finder.extraction.state import State
from interaction_finder.resources import ResourceId


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    """Build permitted pairs dict for testing."""
    return {kind: {kind} for kind in kinds}


def make_evidence(level: int = 5) -> EvidenceQuality:
    """Create a default EvidenceQuality for testing."""
    return EvidenceQuality(
        directness="explicit",
        source_type="primary",
        specificity="mechanistic",
        language="definitive",
        overall=level,
    )


def make_assessment(
    resource: ResourceId,
    relationship: str = "associated_with",
    entity1_name: str = "BRCA1",
    entity2_name: str = "Cancer",
) -> PairAssessment:
    """Create a test PairAssessment."""
    return PairAssessment(
        topic_relevance=3,
        resource_id=resource,
        entity1=EntityRef(
            canonical=entity1_name,
            mentions=[
                EntityMention(
                    kind="gene",
                    name=entity1_name,
                    aliases=[],
                    quotes=[],
                    reasoning="test",
                )
            ],
        ),
        entity2=EntityRef(
            canonical=entity2_name,
            mentions=[
                EntityMention(
                    kind="disease",
                    name=entity2_name,
                    aliases=[],
                    quotes=[],
                    reasoning="test",
                )
            ],
        ),
        relationship=relationship,
        quotes=[],
        evidence=make_evidence(),
        reasoning="Test assessment",
    )


class TestCollectUniqueRelationships:
    """Tests for _collect_unique_relationships function."""

    def test_collects_from_single_document(self):
        """Collects relationships from a single document."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        state.pair_assessments_by_resource = {
            resource: [
                make_assessment(resource, "associated_with"),
                make_assessment(resource, "causes"),
            ]
        }
        relationships = _collect_unique_relationships(state)
        assert relationships == {"associated_with", "causes"}

    def test_deduplicates_across_documents(self):
        """Same relationship in multiple documents is counted once."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        state.pair_assessments_by_resource = {
            resource1: [make_assessment(resource1, "associated_with")],
            resource2: [
                make_assessment(resource2, "associated_with"),
                make_assessment(resource2, "linked_to"),
            ],
        }
        relationships = _collect_unique_relationships(state)
        assert relationships == {"associated_with", "linked_to"}

    def test_empty_assessments_returns_empty(self):
        """Returns empty set when no assessments exist."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        state.pair_assessments_by_resource = {}
        relationships = _collect_unique_relationships(state)
        assert relationships == set()


class TestApplyConsolidations:
    """Tests for _apply_consolidations function."""

    def test_applies_consolidation_to_assessments(self):
        """Consolidation changes relationship labels in assessments."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        assessment = make_assessment(resource, "linked_to")
        state.pair_assessments_by_resource = {resource: [assessment]}
        consolidations = [
            RelationshipConsolidation(
                original="linked_to",
                consolidated="associated_with",
                polarity="positive",
                reasoning="Synonym consolidation - linked_to and associated_with are semantically equivalent",
            )
        ]
        _apply_consolidations(consolidations, state)
        assert assessment.relationship == "associated_with"
        assert state.relationships_merged == 1

    def test_no_change_when_same(self):
        """No update when original equals consolidated."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        assessment = make_assessment(resource, "associated_with")
        state.pair_assessments_by_resource = {resource: [assessment]}
        consolidations = [
            RelationshipConsolidation(
                original="associated_with",
                consolidated="associated_with",
                polarity="positive",
                reasoning="No change needed - already in canonical form for this context",
            )
        ]
        _apply_consolidations(consolidations, state)
        assert assessment.relationship == "associated_with"
        assert state.relationships_merged == 0

    def test_multiple_consolidations(self):
        """Multiple consolidations are applied."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        assessment1 = make_assessment(resource, "linked_to")
        assessment2 = make_assessment(resource, "related_to")
        state.pair_assessments_by_resource = {resource: [assessment1, assessment2]}
        consolidations = [
            RelationshipConsolidation(
                original="linked_to",
                consolidated="associated_with",
                polarity="positive",
                reasoning="Synonym consolidation - both terms indicate association",
            ),
            RelationshipConsolidation(
                original="related_to",
                consolidated="associated_with",
                polarity="positive",
                reasoning="Synonym consolidation - both terms indicate association",
            ),
        ]
        _apply_consolidations(consolidations, state)
        assert assessment1.relationship == "associated_with"
        assert assessment2.relationship == "associated_with"
        assert state.relationships_merged == 2


class TestStorePolarityMappings:
    """Tests for _store_polarity_mappings function."""

    def test_stores_both_original_and_consolidated(self):
        """Both original and consolidated forms get polarity mappings."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        consolidations = [
            RelationshipConsolidation(
                original="linked_to",
                consolidated="associated_with",
                polarity="positive",
                reasoning="Synonym consolidation for test purposes of polarity mapping",
            )
        ]
        _store_polarity_mappings(consolidations, state)
        assert state.relationship_polarities["linked_to"] == "positive"
        assert state.relationship_polarities["associated_with"] == "positive"

    def test_stores_multiple_polarities(self):
        """Multiple consolidations store correct polarities."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        consolidations = [
            RelationshipConsolidation(
                original="inhibits",
                consolidated="inhibits",
                polarity="negative",
                reasoning="Negative regulation relationship indicating suppression or reduction",
            ),
            RelationshipConsolidation(
                original="colocalizes_with",
                consolidated="spatial_proximity",
                polarity="irrelevant",
                reasoning="Spatial proximity is not causally relevant to the research question",
            ),
        ]
        _store_polarity_mappings(consolidations, state)
        assert state.relationship_polarities["inhibits"] == "negative"
        assert state.relationship_polarities["colocalizes_with"] == "irrelevant"
        assert state.relationship_polarities["spatial_proximity"] == "irrelevant"


class TestEnsurePolaritiesForAllRelationships:
    """Tests for _ensure_polarities_for_all_relationships function."""

    def test_assigns_neutral_to_missing(self):
        """Missing relationships get neutral polarity."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        state.relationship_polarities = {"known": "positive"}
        relationships = {"known", "unknown1", "unknown2"}
        _ensure_polarities_for_all_relationships(relationships, state)
        assert state.relationship_polarities["known"] == "positive"
        assert state.relationship_polarities["unknown1"] == "neutral"
        assert state.relationship_polarities["unknown2"] == "neutral"

    def test_no_change_when_all_present(self):
        """No changes when all relationships have polarities."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        state.relationship_polarities = {"a": "positive", "b": "negative"}
        _ensure_polarities_for_all_relationships({"a", "b"}, state)
        assert state.relationship_polarities == {"a": "positive", "b": "negative"}


class TestFilterIrrelevantAssessments:
    """Tests for _filter_irrelevant_assessments function."""

    def test_mixed_polarities_not_filtered(self):
        """Pair with some relevant relationships is kept."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        # Same pair, two different relationships
        assessment1 = make_assessment(resource, "associated_with")
        assessment2 = make_assessment(resource, "colocalizes_with")
        state.pair_assessments_by_resource = {resource: [assessment1, assessment2]}
        state.relationship_polarities = {
            "associated_with": "positive",  # relevant
            "colocalizes_with": "irrelevant",
        }
        deps = MagicMock()
        deps.logger = MagicMock()
        deps.config.tools.extraction.filter_irrelevant_relationships = True
        _filter_irrelevant_assessments(state, deps)
        # Pair not filtered because one relationship is relevant
        assert len(state.pair_assessments_by_resource[resource]) == 2

    def test_all_irrelevant_filtered(self):
        """Pair where ALL relationships are irrelevant is filtered."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        assessment1 = make_assessment(resource, "spatial_proximity")
        assessment2 = make_assessment(resource, "colocalizes_with")
        state.pair_assessments_by_resource = {resource: [assessment1, assessment2]}
        state.relationship_polarities = {
            "spatial_proximity": "irrelevant",
            "colocalizes_with": "irrelevant",
        }
        deps = MagicMock()
        deps.logger = MagicMock()
        deps.config.tools.extraction.filter_irrelevant_relationships = True
        _filter_irrelevant_assessments(state, deps)
        # Pair filtered - assessments removed
        assert len(state.pair_assessments_by_resource[resource]) == 0
        # Pre-rejected judgment created
        assert len(state.pair_judgments) == 1

    def test_empty_assessments_handled(self):
        """Empty assessment list doesn't cause errors."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        state.pair_assessments_by_resource = {}
        state.relationship_polarities = {}
        deps = MagicMock()
        deps.logger = MagicMock()
        deps.config.tools.extraction.filter_irrelevant_relationships = True
        _filter_irrelevant_assessments(state, deps)
        # No errors, no judgments created
        assert len(state.pair_judgments) == 0

    def test_missing_polarity_defaults_to_not_irrelevant(self):
        """Relationship with no polarity mapping is not treated as irrelevant."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        assessment = make_assessment(resource, "unknown_relationship")
        state.pair_assessments_by_resource = {resource: [assessment]}
        state.relationship_polarities = {}  # No polarity for unknown_relationship
        deps = MagicMock()
        deps.logger = MagicMock()
        deps.config.tools.extraction.filter_irrelevant_relationships = True
        _filter_irrelevant_assessments(state, deps)
        # Not filtered because missing polarity != "irrelevant"
        assert len(state.pair_assessments_by_resource[resource]) == 1


class TestPolarityFiltering:
    """Integration tests for polarity filtering behavior."""

    def test_only_all_irrelevant_pairs_filtered(self):
        """Only pairs where ALL assessments are irrelevant get filtered."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        # Pair 1: BRCA1-Cancer with mixed polarities (keep)
        pair1_relevant = make_assessment(resource, "associated_with", "BRCA1", "Cancer")
        pair1_irrelevant = make_assessment(
            resource, "colocalizes_with", "BRCA1", "Cancer"
        )
        # Pair 2: TP53-Tumor with all irrelevant (filter)
        pair2_irrelevant1 = make_assessment(
            resource, "spatial_proximity", "TP53", "Tumor"
        )
        pair2_irrelevant2 = make_assessment(
            resource, "colocalizes_with", "TP53", "Tumor"
        )
        state.pair_assessments_by_resource = {
            resource: [
                pair1_relevant,
                pair1_irrelevant,
                pair2_irrelevant1,
                pair2_irrelevant2,
            ]
        }
        state.relationship_polarities = {
            "associated_with": "positive",
            "colocalizes_with": "irrelevant",
            "spatial_proximity": "irrelevant",
        }
        deps = MagicMock()
        deps.logger = MagicMock()
        deps.config.tools.extraction.filter_irrelevant_relationships = True
        _filter_irrelevant_assessments(state, deps)
        # Pair 1 kept (2 assessments), Pair 2 filtered (0 assessments)
        remaining = state.pair_assessments_by_resource[resource]
        assert len(remaining) == 2
        assert all(a.entity1.canonical == "BRCA1" for a in remaining)
        # One judgment created for filtered pair
        assert len(state.pair_judgments) == 1

    def test_partial_irrelevant_keeps_pair(self):
        """Pair with some relevant assessments is fully kept."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        # 3 assessments for same pair: 2 irrelevant, 1 relevant
        assessment1 = make_assessment(resource, "colocalizes_with")
        assessment2 = make_assessment(resource, "spatial_proximity")
        assessment3 = make_assessment(resource, "causes")
        state.pair_assessments_by_resource = {
            resource: [assessment1, assessment2, assessment3]
        }
        state.relationship_polarities = {
            "colocalizes_with": "irrelevant",
            "spatial_proximity": "irrelevant",
            "causes": "positive",
        }
        deps = MagicMock()
        deps.logger = MagicMock()
        deps.config.tools.extraction.filter_irrelevant_relationships = True
        _filter_irrelevant_assessments(state, deps)
        # All 3 assessments kept because pair has at least one relevant
        assert len(state.pair_assessments_by_resource[resource]) == 3
        # No pre-rejected judgments
        assert len(state.pair_judgments) == 0


class TestEndToEndRelationshipConsolidation:
    """Integration tests for the full relationship consolidation flow."""

    @pytest.fixture
    def mock_deps(self):
        """Create mock dependencies for async tests."""
        deps = MagicMock()
        deps.config = MagicMock()
        deps.config.tools.extraction.filter_irrelevant_relationships = True
        deps.logger = MagicMock()
        deps.agent_semaphore = AsyncMock()
        deps.agent_semaphore.__aenter__ = AsyncMock(return_value=None)
        deps.agent_semaphore.__aexit__ = AsyncMock(return_value=None)
        deps.checkpoint_path = None
        deps.input_checkpoint = None
        deps.progress = None
        return deps

    @pytest.mark.asyncio
    async def test_full_consolidation_flow(self, mock_deps):
        """End-to-end test: collect → consolidate → apply → store polarities → filter."""
        state = State(
            topic="PAH genetic risk factors",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource1 = ResourceId(url="https://doc1.com", counter=0)
        resource2 = ResourceId(url="https://doc2.com", counter=1)
        # Doc 1: Two relationships - one to consolidate, one irrelevant
        state.pair_assessments_by_resource[resource1] = [
            make_assessment(resource1, "linked_to", "BRCA1", "Cancer"),
            make_assessment(resource1, "spatial_colocalization", "BRCA1", "Cancer"),
        ]
        # Doc 2: Same pair with risk relationship
        state.pair_assessments_by_resource[resource2] = [
            make_assessment(resource2, "increases_risk_of", "BRCA1", "Cancer"),
        ]
        # Mock LLM response with consolidations and polarities
        mock_consolidations = RelationshipConsolidations(
            consolidations=[
                RelationshipConsolidation(
                    original="linked_to",
                    consolidated="associated_with",
                    polarity="positive",
                    reasoning="linked_to is synonymous with associated_with in this context",
                ),
                RelationshipConsolidation(
                    original="increases_risk_of",
                    consolidated="increases_risk_of",
                    polarity="positive",
                    reasoning="This relationship indicates a genetic risk factor association",
                ),
                RelationshipConsolidation(
                    original="spatial_colocalization",
                    consolidated="spatial_colocalization",
                    polarity="irrelevant",
                    reasoning="Spatial colocalization is too mechanistic for risk factor study",
                ),
            ]
        )
        mock_result = MagicMock()
        mock_result.output = mock_consolidations
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "mock_agent"
        with patch(
            "interaction_finder.extraction.stages.consolidate_relationships.get_relationship_consolidation_agent",
            return_value=mock_agent,
        ):
            result = await consolidate_relationships(state, mock_deps)
        # Pipeline should continue
        assert result is True
        # Check consolidation was applied
        doc1_assessments = state.pair_assessments_by_resource[resource1]
        assert doc1_assessments[0].relationship == "associated_with"  # Consolidated
        assert (
            doc1_assessments[1].relationship == "spatial_colocalization"
        )  # Unchanged label
        # Check polarity mappings stored
        assert state.relationship_polarities["linked_to"] == "positive"
        assert state.relationship_polarities["associated_with"] == "positive"
        assert state.relationship_polarities["increases_risk_of"] == "positive"
        assert state.relationship_polarities["spatial_colocalization"] == "irrelevant"
        # Check relationships_merged count (only linked_to changed)
        assert state.relationships_merged == 1
        # The pair has mixed polarities (positive + irrelevant), so should NOT be filtered
        assert len(state.pair_judgments) == 0  # No pre-rejected judgments

    @pytest.mark.asyncio
    async def test_consolidation_with_all_irrelevant_filtered(self, mock_deps):
        """Test that pairs with ALL irrelevant relationships get filtered."""
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        resource = ResourceId(url="https://doc1.com", counter=0)
        # Single pair with only irrelevant relationships
        state.pair_assessments_by_resource[resource] = [
            make_assessment(resource, "spatial_proximity", "TP53", "Tumor"),
            make_assessment(resource, "colocalizes_with", "TP53", "Tumor"),
        ]
        mock_consolidations = RelationshipConsolidations(
            consolidations=[
                RelationshipConsolidation(
                    original="spatial_proximity",
                    consolidated="spatial_proximity",
                    polarity="irrelevant",
                    reasoning="Spatial proximity is not causally relevant to the research question",
                ),
                RelationshipConsolidation(
                    original="colocalizes_with",
                    consolidated="colocalizes_with",
                    polarity="irrelevant",
                    reasoning="Colocalization is not causally relevant to the research question",
                ),
            ]
        )
        mock_result = MagicMock()
        mock_result.output = mock_consolidations
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "mock_agent"
        with patch(
            "interaction_finder.extraction.stages.consolidate_relationships.get_relationship_consolidation_agent",
            return_value=mock_agent,
        ):
            result = await consolidate_relationships(state, mock_deps)
        assert result is True
        # Pair should be filtered (all relationships irrelevant)
        assert len(state.pair_judgments) == 1
        # The assessments should be removed
        assert len(state.pair_assessments_by_resource[resource]) == 0
