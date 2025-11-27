"""Tests for canonical name variant tracking during merging.

Validates that merge decisions are recorded using normalized keys but applied
using the actual canonical names present in each document.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.usage import RunUsage
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.models import (
    EntityConsolidationDecision,
    EntityConsolidationDecisions,
    EntityMention,
)
from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
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
    deps.config.tools.extraction.merge_batch_size = 50
    deps.config.tools.extraction.max_rename_iterations = 3
    deps.logger = MagicMock()
    deps.resource_pool = ResourcePool()
    return deps


class TestCanonicalNameTracking:
    """Test that canonical name variants are tracked and used correctly."""

    def test_tracks_all_canonical_variants(self, mock_deps):
        """Should track all canonical variants for each normalized entity."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["disease"],
            permitted_pairs=build_permitted_pairs(["disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Three documents with different capitalizations
        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)
        resource3 = ResourceId(url="https://example.com/doc3", counter=2)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "Pulmonary Arterial Hypertension": EntityMention(
                    kind="disease",
                    name="Pulmonary Arterial Hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="doc1",
                ),
            },
            resource2: {
                "pulmonary arterial hypertension": EntityMention(
                    kind="disease",
                    name="pulmonary arterial hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="doc2",
                ),
            },
            resource3: {
                "Pulmonary arterial hypertension": EntityMention(
                    kind="disease",
                    name="Pulmonary arterial hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="doc3",
                ),
            },
        }

        # Run collection phase (no merges expected, just tracking)
        node._collect_unique_entities(ctx)

        # Should track all three canonical variants
        norm_key = ("pulmonary arterial hypertension", "disease")
        assert norm_key in ctx.state.canonical_name_variants
        variants = ctx.state.canonical_name_variants[norm_key]
        assert len(variants) == 3
        assert "Pulmonary Arterial Hypertension" in variants
        assert "pulmonary arterial hypertension" in variants
        assert "Pulmonary arterial hypertension" in variants


class TestCrossDocumentCanonicalMerging:
    """Test that merges work correctly with different canonical forms."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_merges_using_per_document_canonical_names(self, mock_deps):
        """Should use the canonical name present in each specific document.

        Note: This is an integration test that requires proper LLM mocking.
        Currently skipped in non-integration test runs.
        """
        node = ConsolidateEntitiesNode()
        state = State(
            topic="PAH genetics",
            target_entity_types=["disease"],
            permitted_pairs=build_permitted_pairs(["disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Cached decision: merge idiopathic PAH into PAH (normalized forms)
        # Cache stores target canonical name (the parent to merge into)
        ctx.state.merge_decision_cache[
            (
                "pulmonary arterial hypertension",
                "idiopathic pulmonary arterial hypertension",
                "disease",
            )
        ] = "Pulmonary Arterial Hypertension"

        # Three documents with different capitalizations
        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)
        resource3 = ResourceId(url="https://example.com/doc3", counter=2)

        ctx.state.validated_entities_by_resource = {
            # Doc1: Capital PAH + Capital Idiopathic PAH
            resource1: {
                "Pulmonary Arterial Hypertension": EntityMention(
                    kind="disease",
                    name="Pulmonary Arterial Hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="doc1",
                ),
                "Idiopathic Pulmonary Arterial Hypertension": EntityMention(
                    kind="disease",
                    name="Idiopathic Pulmonary Arterial Hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="doc1",
                ),
            },
            # Doc2: lowercase pah + lowercase idiopathic pah
            resource2: {
                "pulmonary arterial hypertension": EntityMention(
                    kind="disease",
                    name="pulmonary arterial hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="doc2",
                ),
                "idiopathic pulmonary arterial hypertension": EntityMention(
                    kind="disease",
                    name="idiopathic pulmonary arterial hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="doc2",
                ),
            },
            # Doc3: Mixed case PAH + Mixed case Idiopathic PAH
            resource3: {
                "Pulmonary arterial hypertension": EntityMention(
                    kind="disease",
                    name="Pulmonary arterial hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="doc3",
                ),
                "Idiopathic pulmonary arterial hypertension": EntityMention(
                    kind="disease",
                    name="Idiopathic pulmonary arterial hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="doc3",
                ),
            },
        }

        # Apply global merging
        await node.run(ctx)

        # All three documents should have merged idiopathic into main PAH
        # but using their respective canonical forms
        for resource_id in [resource1, resource2, resource3]:
            entities = ctx.state.validated_entities_by_resource[resource_id]

            # Should only have one entity left (idiopathic merged away)
            assert len(entities) == 1

            # The remaining entity should be the parent (shorter name)
            entity_name = list(entities.keys())[0]
            entity = entities[entity_name]

            # Check that it's the parent form (normalized: "pulmonary arterial hypertension")
            from interaction_finder.extraction.utils import normalize_for_comparison

            assert (
                normalize_for_comparison(entity_name)
                == "pulmonary arterial hypertension"
            )

            # The child should be in aliases
            assert len(entity.aliases) >= 1
            child_aliases = [
                alias
                for alias in entity.aliases
                if normalize_for_comparison(alias)
                == "idiopathic pulmonary arterial hypertension"
            ]
            assert len(child_aliases) >= 1

            # Check reasoning shows merge
            assert "MERGED" in entity.reasoning

        # Should have merged 3 entities total (one per document)
        assert ctx.state.entities_merged == 3

    @pytest.mark.asyncio
    async def test_handles_abbreviation_expansion_in_merges(self, mock_deps):
        """Should match cache via parenthetical expansion.

        When entity names have parentheticals like "PAH (Pulmonary arterial hypertension)",
        the extract_all_forms function expands them to extract both base and abbreviation.
        This means "PAH (XYZ)" matches cache entries for "PAH" or "XYZ" individually.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        node = ConsolidateEntitiesNode()
        state = State(
            topic="PAH genetics",
            target_entity_types=["disease"],
            permitted_pairs=build_permitted_pairs(["disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)
        # Cache decision for the BASE forms (without abbreviation suffixes)
        # extract_all_forms will expand "Name (Abbrev)" to get "Name", matching cache
        # Cache stores target canonical name (the parent to merge into)
        ctx.state.merge_decision_cache[
            (
                "pulmonary arterial hypertension",
                "idiopathic pulmonary arterial hypertension",
                "disease",
            )
        ] = "Pulmonary arterial hypertension (PAH)"
        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        # Document has names WITH (PAH) suffix
        ctx.state.validated_entities_by_resource = {
            resource1: {
                "Pulmonary arterial hypertension (PAH)": EntityMention(
                    kind="disease",
                    name="Pulmonary arterial hypertension (PAH)",
                    aliases=[],
                    quotes=[],
                    reasoning="doc1",
                ),
                "Idiopathic pulmonary arterial hypertension (IPAH)": EntityMention(
                    kind="disease",
                    name="Idiopathic pulmonary arterial hypertension (IPAH)",
                    aliases=[],
                    quotes=[],
                    reasoning="doc1",
                ),
            },
        }
        # Mock LLM for any additional pairs (pah/ipah forms create extra pairs)
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(decisions=[])
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "test_agent"
        with patch(
            "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
            return_value=mock_agent,
        ):
            # Apply global merging - should hit the cache via expanded forms
            await node.run(ctx)
        entities = ctx.state.validated_entities_by_resource[resource1]
        # The merge SHOULD happen because extract_all_forms expands:
        # "Pulmonary arterial hypertension (PAH)" -> "Pulmonary arterial hypertension"
        # "Idiopathic pulmonary arterial hypertension (IPAH)" -> "Idiopathic pulmonary arterial hypertension"
        # And "pulmonary arterial hypertension" IS a substring of "idiopathic pulmonary arterial hypertension"
        assert len(entities) == 1  # Merge happens via expanded forms
        assert ctx.state.merge_cache_hits >= 1  # Cache was used


class TestMergeWithMissingParent:
    """Test behavior when parent entity is not in document."""

    @pytest.mark.asyncio
    async def test_no_merge_when_parent_missing(self, mock_deps):
        """Should not merge if parent doesn't exist in document."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["disease"],
            permitted_pairs=build_permitted_pairs(["disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Cached decision says to merge
        ctx.state.merge_decision_cache[
            (
                "pulmonary arterial hypertension",
                "idiopathic pulmonary arterial hypertension",
                "disease",
            )
        ] = True

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)

        # Document only has child, not parent
        ctx.state.validated_entities_by_resource = {
            resource1: {
                "Idiopathic Pulmonary Arterial Hypertension": EntityMention(
                    kind="disease",
                    name="Idiopathic Pulmonary Arterial Hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="doc1",
                ),
            },
        }

        # Apply global merging
        await node.run(ctx)

        # Should NOT merge (parent not present)
        entities = ctx.state.validated_entities_by_resource[resource1]
        assert len(entities) == 1
        assert "Idiopathic Pulmonary Arterial Hypertension" in entities
        assert ctx.state.entities_merged == 0
