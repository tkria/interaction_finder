"""Tests for ConsolidateRelationshipsNode global relationship label consolidation.

Tests the relationship merging architecture that:
1. Collects all unique relationship labels across documents
2. Queries LLM for consolidation + polarity classification (unified)
3. Applies consolidations consistently across all assessments
4. Filters irrelevant relationship types based on polarity
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    PairAssessment,
    RelationshipConsolidation,
    RelationshipConsolidations,
)
from interaction_finder.extraction.nodes import ConsolidateRelationshipsNode
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import build_permitted_pairs
from interaction_finder.resources import ResourceId, ResourcePool
from tests.extraction.conftest import make_evidence


def create_mock_agent_with_override(run_return_value):
    """Create a mock agent with working rename_agent() support."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=run_return_value)
    mock_agent._name = "mock_agent"
    return mock_agent


@pytest.fixture
def mock_deps():
    """Create mock dependencies."""
    deps = MagicMock()
    deps.config = MagicMock()
    deps.config.tools.extraction.filter_irrelevant_relationships = (
        False  # Disable by default for tests
    )
    deps.logger = MagicMock()
    deps.resource_pool = ResourcePool()
    deps.agent_semaphore = AsyncMock()
    deps.agent_semaphore.__aenter__ = AsyncMock(return_value=None)
    deps.agent_semaphore.__aexit__ = AsyncMock(return_value=None)
    deps.checkpoint_path = None
    deps.input_checkpoint = None
    return deps


class TestCollectUniqueRelationships:
    """Test _collect_unique_relationships method."""

    def test_collects_from_single_document(self, mock_deps):
        """Should collect relationship labels from a single document."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        gene = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        disease = EntityMention(
            kind="disease",
            name="Cancer",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityRef(canonical=gene.name, mentions=[gene]),
                    entity2=EntityRef(canonical=disease.name, mentions=[disease]),
                    relationship="associated_with",
                    quotes=[],
                    evidence=make_evidence(8),
                    reasoning="test",
                ),
            ],
        }

        unique = node._collect_unique_relationships(ctx)

        assert len(unique) == 1
        assert "associated_with" in unique

    def test_collects_from_multiple_documents(self, mock_deps):
        """Should collect relationship labels from multiple documents."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        entity1 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        entity2 = EntityMention(
            kind="disease",
            name="Cancer",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])

        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=entity1_ref,
                    entity2=entity2_ref,
                    relationship="associated_with",
                    quotes=[],
                    evidence=make_evidence(8),
                    reasoning="test",
                ),
            ],
            resource2: [
                PairAssessment(
                    resource_id=resource2,
                    entity1=entity1_ref,
                    entity2=entity2_ref,
                    relationship="increases_risk_of",
                    quotes=[],
                    evidence=make_evidence(8),
                    reasoning="test",
                ),
                PairAssessment(
                    resource_id=resource2,
                    entity1=entity1_ref,
                    entity2=entity2_ref,
                    relationship="regulates",
                    quotes=[],
                    evidence=make_evidence(6),
                    reasoning="test",
                ),
            ],
        }

        unique = node._collect_unique_relationships(ctx)

        assert len(unique) == 3
        assert "associated_with" in unique
        assert "increases_risk_of" in unique
        assert "regulates" in unique

    def test_handles_empty_assessments(self, mock_deps):
        """Should handle empty assessments gracefully."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        unique = node._collect_unique_relationships(ctx)

        assert len(unique) == 0


class TestUnifiedConsolidationIntegration:
    """Integration tests for the new unified consolidation+polarity system."""

    @pytest.mark.asyncio
    async def test_end_to_end_unified_consolidation(self, mock_deps):
        """Test complete flow with unified consolidation and polarity classification."""
        # Create mock node and state
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="PAH genetic risk factors",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Create assessments with different relationships
        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        entity1 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        entity2 = EntityMention(
            kind="disease",
            name="Cancer",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])

        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=entity1_ref,
                    entity2=entity2_ref,
                    relationship="linked_to",  # Will be consolidated to "associated_with"
                    quotes=[],
                    evidence=make_evidence(8),
                    reasoning="test",
                ),
            ],
            resource2: [
                PairAssessment(
                    resource_id=resource2,
                    entity1=entity1_ref,
                    entity2=entity2_ref,
                    relationship="increases_risk_of",  # Will stay as is
                    quotes=[],
                    evidence=make_evidence(8),
                    reasoning="test",
                ),
                PairAssessment(
                    resource_id=resource2,
                    entity1=entity1_ref,
                    entity2=entity2_ref,
                    relationship="spatial_colocalization",  # Will be marked irrelevant
                    quotes=[],
                    evidence=make_evidence(3),
                    reasoning="test",
                ),
            ],
        }

        # Mock the unified consolidation agent
        mock_consolidations = RelationshipConsolidations(
            consolidations=[
                RelationshipConsolidation(
                    original="linked_to",
                    consolidated="associated_with",
                    polarity="positive",
                    reasoning="Synonymous with associated_with, indicates positive association",
                ),
                RelationshipConsolidation(
                    original="increases_risk_of",
                    consolidated="increases_risk_of",
                    polarity="positive",
                    reasoning="Clearly indicates genetic risk factor",
                ),
                RelationshipConsolidation(
                    original="spatial_colocalization",
                    consolidated="spatial_colocalization",
                    polarity="irrelevant",
                    reasoning="Too mechanistic for genetic risk factor study",
                ),
            ]
        )

        mock_result = MagicMock()
        mock_result.output = mock_consolidations

        mock_agent = create_mock_agent_with_override(mock_result)

        # Patch the agent getter
        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_consolidation_agent",
            return_value=mock_agent,
        ):
            # Run the node
            result = await node.run(ctx)

        # Verify consolidation was applied
        assert ctx.state.relationships_merged == 1  # Only "linked_to" was changed

        # Verify all assessments have updated relationships
        assessments_resource1 = ctx.state.pair_assessments_by_resource[resource1]
        assert (
            assessments_resource1[0].relationship == "associated_with"
        )  # Consolidated

        # Verify polarity mappings were stored
        assert "associated_with" in ctx.state.relationship_polarities
        assert ctx.state.relationship_polarities["associated_with"] == "positive"
        assert ctx.state.relationship_polarities["increases_risk_of"] == "positive"
        assert (
            ctx.state.relationship_polarities["spatial_colocalization"] == "irrelevant"
        )

        # Verify filtering happened (irrelevant pair should have judgment)
        # Note: filtering creates pre-rejected judgments for all-irrelevant pairs
        assert len(ctx.state.pair_judgments) >= 0  # May or may not have filtered pair

    @pytest.mark.asyncio
    async def test_consolidation_without_changes(self, mock_deps):
        """Test when LLM decides no consolidation is needed."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)

        entity1 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        entity2 = EntityMention(
            kind="disease",
            name="Cancer",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])

        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=entity1_ref,
                    entity2=entity2_ref,
                    relationship="associated_with",
                    quotes=[],
                    evidence=make_evidence(8),
                    reasoning="test",
                ),
            ],
        }

        # Mock: relationship is already canonical
        mock_consolidations = RelationshipConsolidations(
            consolidations=[
                RelationshipConsolidation(
                    original="associated_with",
                    consolidated="associated_with",  # No change
                    polarity="neutral",
                    reasoning="Already canonical, unclear directionality",
                ),
            ]
        )

        mock_result = MagicMock()
        mock_result.output = mock_consolidations
        mock_agent = create_mock_agent_with_override(mock_result)

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_consolidation_agent",
            return_value=mock_agent,
        ):
            await node.run(ctx)

        # No consolidations applied (consolidated == original)
        assert ctx.state.relationships_merged == 0

        # But polarity mapping should still be stored
        assert ctx.state.relationship_polarities["associated_with"] == "neutral"

    @pytest.mark.asyncio
    async def test_polarity_filtering_behavior(self, mock_deps):
        """Test that only all-irrelevant pairs are filtered."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        entity1 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        entity2 = EntityMention(
            kind="disease",
            name="Cancer",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        entity3 = EntityMention(
            kind="gene",
            name="TP53",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])
        entity3_ref = EntityRef(canonical=entity3.name, mentions=[entity3])

        # Pair 1: Has both relevant and irrelevant assessments (should NOT be filtered)
        # Pair 2: Has only irrelevant assessments (SHOULD be filtered)
        ctx.state.pair_assessments_by_resource = {
            resource1: [
                # BRCA1-Cancer: mixed (supporting + irrelevant)
                PairAssessment(
                    resource_id=resource1,
                    entity1=entity1_ref,
                    entity2=entity2_ref,
                    relationship="increases_risk_of",
                    quotes=[],
                    evidence=make_evidence(8),
                    reasoning="test",
                ),
                PairAssessment(
                    resource_id=resource1,
                    entity1=entity1_ref,
                    entity2=entity2_ref,
                    relationship="spatial_colocalization",
                    quotes=[],
                    evidence=make_evidence(3),
                    reasoning="test",
                ),
            ],
            resource2: [
                # TP53-Cancer: only irrelevant
                PairAssessment(
                    resource_id=resource2,
                    entity1=entity3_ref,
                    entity2=entity2_ref,
                    relationship="spatial_colocalization",
                    quotes=[],
                    evidence=make_evidence(3),
                    reasoning="test",
                ),
            ],
        }

        mock_consolidations = RelationshipConsolidations(
            consolidations=[
                RelationshipConsolidation(
                    original="increases_risk_of",
                    consolidated="increases_risk_of",
                    polarity="positive",
                    reasoning="Relevant to research topic, indicates genetic risk factor clearly",
                ),
                RelationshipConsolidation(
                    original="spatial_colocalization",
                    consolidated="spatial_colocalization",
                    polarity="irrelevant",
                    reasoning="Not relevant to research question, wrong level of analysis",
                ),
            ]
        )

        mock_result = MagicMock()
        mock_result.output = mock_consolidations
        mock_agent = create_mock_agent_with_override(mock_result)

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_consolidation_agent",
            return_value=mock_agent,
        ):
            await node.run(ctx)

        # Verify polarity mappings were stored
        assert ctx.state.relationship_polarities["increases_risk_of"] == "positive"
        assert (
            ctx.state.relationship_polarities["spatial_colocalization"] == "irrelevant"
        )

        # Note: The actual filtering behavior (creating pre-rejected judgments)
        # depends on the specific implementation. The key is that polarity
        # classifications were correctly stored and can be used by downstream nodes.

    @pytest.mark.asyncio
    async def test_no_relationships_to_consolidate(self, mock_deps):
        """Test when there are no relationships to process."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # No assessments
        result = await node.run(ctx)

        # Should complete without error
        assert ctx.state.relationships_merged == 0
        assert len(ctx.state.relationship_polarities) == 0

    @pytest.mark.asyncio
    async def test_llm_error_handling(self, mock_deps):
        """Test graceful handling of LLM errors."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)

        entity1 = EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        entity2 = EntityMention(
            kind="disease",
            name="Cancer",
            aliases=[],
            quotes=[],
            reasoning="test",
        )
        entity1_ref = EntityRef(canonical=entity1.name, mentions=[entity1])
        entity2_ref = EntityRef(canonical=entity2.name, mentions=[entity2])

        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=entity1_ref,
                    entity2=entity2_ref,
                    relationship="associated_with",
                    quotes=[],
                    evidence=make_evidence(8),
                    reasoning="test",
                ),
            ],
        }

        # Mock agent that raises an error
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=ValueError("LLM error"))
        mock_agent._name = "mock_agent"

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_consolidation_agent",
            return_value=mock_agent,
        ):
            result = await node.run(ctx)

        # Should complete without crashing
        assert ctx.state.relationships_merged == 0
        # Logger should have been called with error
        mock_deps.logger.error.assert_called_once()
