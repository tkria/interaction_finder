"""Tests for ConsolidateRelationshipsNode global relationship label consolidation.

Tests the relationship merging architecture that:
1. Collects all unique relationship labels across documents
2. Queries LLM for mapping decisions (merge/rename)
3. Resolves transitive mappings
4. Applies mappings consistently across all assessments
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.usage import RunUsage
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.models import (
    EntityMention,
    PairAssessment,
    RelationshipMapping,
    RelationshipMappings,
)
from interaction_finder.extraction.nodes import ConsolidateRelationshipsNode
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import build_permitted_pairs
from interaction_finder.resources import ResourceId, ResourcePool


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
        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="disease",
                        name="Cancer",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="associated_with",
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                ),
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="disease",
                        name="Cancer",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="regulates",
                    quotes=[],
                    confidence="medium",
                    reasoning="test",
                ),
            ]
        }

        unique = node._collect_unique_relationships(ctx)

        assert len(unique) == 2
        assert "associated_with" in unique
        assert "regulates" in unique

    def test_collects_from_multiple_documents(self, mock_deps):
        """Should collect relationship labels across multiple documents."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="linked_to",
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ],
            resource2: [
                PairAssessment(
                    resource_id=resource2,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="associated_with",
                    quotes=[],
                    confidence="medium",
                    reasoning="test",
                )
            ],
        }

        unique = node._collect_unique_relationships(ctx)

        assert len(unique) == 2
        assert "linked_to" in unique
        assert "associated_with" in unique

    def test_handles_empty_assessments(self, mock_deps):
        """Should return empty set when no assessments."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        unique = node._collect_unique_relationships(ctx)

        assert len(unique) == 0


class TestResolveTransitiveMappings:
    """Test _resolve_transitive_mappings method."""

    def test_resolves_simple_chain(self):
        """Should resolve A→B→C to A→C and B→C."""
        node = ConsolidateRelationshipsNode()
        mappings = {"linked_to": "related_to", "related_to": "associated_with"}

        resolved = node._resolve_transitive_mappings(mappings)

        # Both should map to final target
        assert resolved["linked_to"] == "associated_with"
        assert resolved["related_to"] == "associated_with"

    def test_handles_no_chains(self):
        """Should pass through mappings without chains."""
        node = ConsolidateRelationshipsNode()
        mappings = {"linked_to": "associated_with", "regulates": "modulates"}

        resolved = node._resolve_transitive_mappings(mappings)

        assert resolved == mappings

    def test_handles_cycle_detection(self):
        """Should break cycles and not get stuck in infinite loop."""
        node = ConsolidateRelationshipsNode()
        # Create a cycle: A→B→C→A
        mappings = {
            "linked_to": "related_to",
            "related_to": "associated_with",
            "associated_with": "linked_to",
        }

        # Should not raise, should handle gracefully
        resolved = node._resolve_transitive_mappings(mappings)

        # Verify it completed without getting stuck
        assert len(resolved) == 3

    def test_handles_empty_mappings(self):
        """Should handle empty mappings dict."""
        node = ConsolidateRelationshipsNode()
        mappings = {}

        resolved = node._resolve_transitive_mappings(mappings)

        assert resolved == {}


class TestApplyRelationshipMappings:
    """Test _apply_relationship_mappings method."""

    def test_updates_matching_relationships(self, mock_deps):
        """Should update relationships that match mappings."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="linked_to",
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ]
        }

        # linked to → associated with (normalized forms)
        mappings = {"linked to": "associated with"}

        node._apply_relationship_mappings(mappings, ctx)

        assessment = ctx.state.pair_assessments_by_resource[resource1][0]
        assert assessment.relationship == "associated with"
        assert ctx.state.relationships_merged == 1

    def test_preserves_unmatched_relationships(self, mock_deps):
        """Should not modify relationships without mappings."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="regulates",
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ]
        }

        # Mapping for different relationship
        mappings = {"linked to": "associated with"}

        node._apply_relationship_mappings(mappings, ctx)

        assessment = ctx.state.pair_assessments_by_resource[resource1][0]
        assert assessment.relationship == "regulates"  # Unchanged
        assert ctx.state.relationships_merged == 0

    def test_handles_multiple_documents(self, mock_deps):
        """Should update relationships across all documents."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="linked_to",
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ],
            resource2: [
                PairAssessment(
                    resource_id=resource2,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="related_to",
                    quotes=[],
                    confidence="medium",
                    reasoning="test",
                )
            ],
        }

        # Both map to same target (normalized forms)
        mappings = {"linked to": "associated with", "related to": "associated with"}

        node._apply_relationship_mappings(mappings, ctx)

        assert (
            ctx.state.pair_assessments_by_resource[resource1][0].relationship
            == "associated with"
        )
        assert (
            ctx.state.pair_assessments_by_resource[resource2][0].relationship
            == "associated with"
        )
        assert ctx.state.relationships_merged == 2

    def test_handles_empty_mappings(self, mock_deps):
        """Should handle empty mappings gracefully."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="associated_with",
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ]
        }

        mappings = {}

        node._apply_relationship_mappings(mappings, ctx)

        # Should not change anything
        assert (
            ctx.state.pair_assessments_by_resource[resource1][0].relationship
            == "associated_with"
        )
        assert ctx.state.relationships_merged == 0


class TestGetRelationshipMappings:
    """Test _get_relationship_mappings method (integration with LLM)."""

    @pytest.mark.asyncio
    async def test_calls_agent_and_parses_output(self, mock_deps):
        """Should call LLM agent and parse mappings."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="PAH genetics",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Mock agent response
        mock_output = RelationshipMappings(
            mappings=[
                RelationshipMapping(
                    old="linked_to",
                    new="associated_with",
                    reasoning="Synonyms in this context",
                ),
                RelationshipMapping(
                    old="upregulates",
                    new="activates",
                    reasoning="Functional equivalence",
                ),
            ]
        )
        mock_result = MagicMock()
        mock_result.output = mock_output
        mock_agent = create_mock_agent_with_override(mock_result)

        relationships = {"linked_to", "upregulates", "associated_with", "activates"}

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_mapping_agent",
            return_value=mock_agent,
        ):
            mappings = await node._get_relationship_mappings(relationships, ctx)

        # Should normalize and return mappings (normalized forms)
        assert "linked to" in mappings
        assert mappings["linked to"] == "associated with"
        assert "upregulates" in mappings
        assert mappings["upregulates"] == "activates"

    @pytest.mark.asyncio
    async def test_skips_self_mappings(self, mock_deps):
        """Should skip mappings where old == new."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Mock agent response with self-mapping
        mock_output = RelationshipMappings(
            mappings=[
                RelationshipMapping(
                    old="associated_with",
                    new="associated_with",  # Self-mapping
                    reasoning="Already canonical form, no change needed",
                ),
                RelationshipMapping(
                    old="linked_to",
                    new="associated_with",
                    reasoning="Synonym consolidation for consistency",
                ),
            ]
        )
        mock_result = MagicMock()
        mock_result.output = mock_output
        mock_agent = create_mock_agent_with_override(mock_result)

        relationships = {"linked_to", "associated_with"}

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_mapping_agent",
            return_value=mock_agent,
        ):
            mappings = await node._get_relationship_mappings(relationships, ctx)

        # Should skip self-mapping (normalized forms)
        assert "associated with" not in mappings
        assert "linked to" in mappings

    @pytest.mark.asyncio
    async def test_handles_agent_error(self, mock_deps):
        """Should handle agent errors gracefully."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Mock agent that raises error
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=TimeoutError("LLM timeout"))
        mock_agent._name = "mock_agent"

        relationships = {"linked_to", "associated_with"}

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_mapping_agent",
            return_value=mock_agent,
        ):
            mappings = await node._get_relationship_mappings(relationships, ctx)

        # Should return empty dict on error
        assert mappings == {}
        # Should log error
        mock_deps.logger.error.assert_called_once()


class TestFullNodeFlow:
    """Integration test for complete node execution."""

    @pytest.mark.asyncio
    async def test_end_to_end_merging(self, mock_deps):
        """Should execute full merging workflow."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="PAH genetics",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Set up assessments
        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="disease",
                        name="Cancer",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="linked_to",
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ],
            resource2: [
                PairAssessment(
                    resource_id=resource2,
                    entity1=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="disease",
                        name="Cancer",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="related_to",
                    quotes=[],
                    confidence="medium",
                    reasoning="test",
                )
            ],
        }

        # Mock agent response
        mock_output = RelationshipMappings(
            mappings=[
                RelationshipMapping(
                    old="linked_to",
                    new="associated_with",
                    reasoning="Synonym consolidation",
                ),
                RelationshipMapping(
                    old="related_to",
                    new="associated_with",
                    reasoning="Synonym consolidation",
                ),
            ]
        )
        mock_result = MagicMock()
        mock_result.output = mock_output
        mock_agent = create_mock_agent_with_override(mock_result)

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_mapping_agent",
            return_value=mock_agent,
        ):
            next_node = await node.run(ctx)

        # Verify results (normalized forms)
        assert ctx.state.relationships_merged == 2
        assert (
            ctx.state.pair_assessments_by_resource[resource1][0].relationship
            == "associated with"
        )
        assert (
            ctx.state.pair_assessments_by_resource[resource2][0].relationship
            == "associated with"
        )
        # Should store mappings
        assert len(ctx.state.relationship_mappings) == 2

    @pytest.mark.asyncio
    async def test_handles_no_relationships(self, mock_deps):
        """Should handle case with no assessments."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        next_node = await node.run(ctx)

        # Should complete without error
        assert ctx.state.relationships_merged == 0

    @pytest.mark.asyncio
    async def test_handles_no_mappings_suggested(self, mock_deps):
        """Should handle case where LLM suggests no mappings."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="associated_with",
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ]
        }

        # Mock agent response with no mappings
        mock_output = RelationshipMappings(mappings=[])
        mock_result = MagicMock()
        mock_result.output = mock_output
        mock_agent = create_mock_agent_with_override(mock_result)

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_mapping_agent",
            return_value=mock_agent,
        ):
            next_node = await node.run(ctx)

        # Should complete without changes
        assert ctx.state.relationships_merged == 0
        assert (
            ctx.state.pair_assessments_by_resource[resource1][0].relationship
            == "associated_with"
        )


class TestFilterIrrelevantRelationships:
    """Test _filter_irrelevant_relationships and _get_relevance_decisions methods."""

    @pytest.mark.asyncio
    async def test_filters_irrelevant_relationship_type(self, mock_deps):
        """Should filter pairs with irrelevant relationship types."""
        from interaction_finder.extraction.models import (
            RelationshipRelevanceDecision,
            RelationshipRelevanceDecisions,
        )

        node = ConsolidateRelationshipsNode()
        state = State(
            topic="PAH genetic associations",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="disease",
                        name="PAH",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="binds to",  # Irrelevant for genetic associations
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ]
        }

        # Mock agent response - marks "binds to" as irrelevant
        mock_output = RelationshipRelevanceDecisions(
            decisions=[
                RelationshipRelevanceDecision(
                    relationship="binds to",
                    is_relevant=False,
                    reasoning="Molecular binding is too mechanistic for a genetic association study",
                )
            ]
        )
        mock_result = MagicMock()
        mock_result.output = mock_output
        mock_agent = create_mock_agent_with_override(mock_result)

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_relevance_agent",
            return_value=mock_agent,
        ):
            await node._filter_irrelevant_relationships(ctx)

        # Should create pre-rejected judgment
        assert len(ctx.state.pair_judgments) == 1
        judgment = list(ctx.state.pair_judgments.values())[0]
        assert judgment.accepted is False
        assert judgment.confidence == "high"
        assert "filtered as irrelevant" in judgment.reasoning.lower()
        assert "mechanistic" in judgment.reasoning.lower()

    @pytest.mark.asyncio
    async def test_preserves_relevant_relationships(self, mock_deps):
        """Should not filter pairs with relevant relationship types."""
        from interaction_finder.extraction.models import (
            RelationshipRelevanceDecision,
            RelationshipRelevanceDecisions,
        )

        node = ConsolidateRelationshipsNode()
        state = State(
            topic="PAH genetic associations",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="disease",
                        name="PAH",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="associated with",  # Relevant
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ]
        }

        # Mock agent response - marks as relevant
        mock_output = RelationshipRelevanceDecisions(
            decisions=[
                RelationshipRelevanceDecision(
                    relationship="associated with",
                    is_relevant=True,
                    reasoning="Genetic associations are exactly what this research seeks",
                )
            ]
        )
        mock_result = MagicMock()
        mock_result.output = mock_output
        mock_agent = create_mock_agent_with_override(mock_result)

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_relevance_agent",
            return_value=mock_agent,
        ):
            await node._filter_irrelevant_relationships(ctx)

        # Should NOT create any pre-rejected judgments
        assert len(ctx.state.pair_judgments) == 0

    @pytest.mark.asyncio
    async def test_handles_multiple_relationship_types(self, mock_deps):
        """Should filter only irrelevant types, preserve relevant ones."""
        from interaction_finder.extraction.models import (
            RelationshipRelevanceDecision,
            RelationshipRelevanceDecisions,
        )

        node = ConsolidateRelationshipsNode()
        state = State(
            topic="PAH genetics",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="disease",
                        name="PAH",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="associated with",  # Relevant
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ],
            resource2: [
                PairAssessment(
                    resource_id=resource2,
                    entity1=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="disease",
                        name="PAH",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="phosphorylates",  # Irrelevant
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ],
        }

        # Mock agent response
        mock_output = RelationshipRelevanceDecisions(
            decisions=[
                RelationshipRelevanceDecision(
                    relationship="associated with",
                    is_relevant=True,
                    reasoning="Core relationship for genetic association research",
                ),
                RelationshipRelevanceDecision(
                    relationship="phosphorylates",
                    is_relevant=False,
                    reasoning="Molecular mechanism too detailed for genetic association study",
                ),
            ]
        )
        mock_result = MagicMock()
        mock_result.output = mock_output
        mock_agent = create_mock_agent_with_override(mock_result)

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_relevance_agent",
            return_value=mock_agent,
        ):
            await node._filter_irrelevant_relationships(ctx)

        # Should create judgment only for irrelevant pair
        assert len(ctx.state.pair_judgments) == 1
        judgment = list(ctx.state.pair_judgments.values())[0]
        assert judgment.relationship == "phosphorylates"
        assert judgment.accepted is False

    @pytest.mark.asyncio
    async def test_handles_agent_error_gracefully(self, mock_deps):
        """Should handle LLM errors without crashing."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.pair_assessments_by_resource = {
            resource1: [
                PairAssessment(
                    resource_id=resource1,
                    entity1=EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    entity2=EntityMention(
                        kind="gene",
                        name="TP53",
                        aliases=[],
                        quotes=[],
                        reasoning="test",
                    ),
                    relationship="regulates",
                    quotes=[],
                    confidence="high",
                    reasoning="test",
                )
            ]
        }

        # Mock agent that raises error
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=TimeoutError("LLM timeout"))
        mock_agent._name = "mock_agent"

        with patch(
            "interaction_finder.extraction.consolidate_relationships.get_relationship_relevance_agent",
            return_value=mock_agent,
        ):
            await node._filter_irrelevant_relationships(ctx)

        # Should not create any judgments on error
        assert len(ctx.state.pair_judgments) == 0
        # Should log error
        mock_deps.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_empty_relationships(self, mock_deps):
        """Should handle case with no assessments gracefully."""
        node = ConsolidateRelationshipsNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # No assessments
        await node._filter_irrelevant_relationships(ctx)

        # Should complete without error or judgments
        assert len(ctx.state.pair_judgments) == 0
