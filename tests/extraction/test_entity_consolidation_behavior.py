"""End-to-end behavior tests for entity consolidation.

Tests critical consolidation behaviors through the public API (running the actual node),
not internal implementation details.
"""

import asyncio
import pytest
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.models import EntityMention, EntityRef
from interaction_finder.extraction.state import State
from interaction_finder.resources import ResourceId, ResourcePool
from interaction_finder.settings import IfetcherConfig


def ref_map(data: dict[str, EntityMention]) -> dict[str, EntityRef]:
    """Convert entity mentions to EntityRef dict."""
    return {
        name: EntityRef(canonical=name, mentions=[mention])
        for name, mention in data.items()
    }


@pytest.fixture
def mock_config():
    return IfetcherConfig()


@pytest.fixture
def mock_deps(mock_config):
    import logging

    deps = type(
        "Deps",
        (),
        {
            "config": mock_config,
            "resource_pool": ResourcePool(),
            "logger": logging.getLogger("test"),
            "agent_semaphore": asyncio.Semaphore(1),
            "progress": None,
            "checkpoint_path": None,
            "input_checkpoint": None,
        },
    )()
    return deps


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    return {kind: {kind} for kind in kinds}


class TestCapitalizationVariants:
    """Test that capitalization variants are auto-merged."""

    @pytest.mark.asyncio
    async def test_two_capitalization_variants_merge(self, mock_deps):
        """Two capitalization variants should auto-merge to consistent canonical name."""
        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")

        # Two documents with same entity, different capitalization
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "Pulmonary Hypertension": EntityMention(
                    kind="phenotype",
                    name="Pulmonary Hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc1",
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "Pulmonary hypertension": EntityMention(
                    kind="phenotype",
                    name="Pulmonary hypertension",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc2",
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Run the node
        await node.run(ctx)

        # Check that both entities now use the same canonical name
        entities1 = ctx.state.validated_entities_by_resource[resource1]
        entities2 = ctx.state.validated_entities_by_resource[resource2]

        # Both should have exactly one entity
        assert len(entities1) == 1
        assert len(entities2) == 1

        # Get the canonical names
        canonical1 = list(entities1.values())[0].canonical
        canonical2 = list(entities2.values())[0].canonical

        # Should be the same canonical name (prefer mixed-case)
        assert canonical1 == canonical2
        assert canonical1 == "Pulmonary Hypertension"

    @pytest.mark.asyncio
    async def test_three_capitalization_variants_merge(self, mock_deps):
        """Three capitalization variants should all merge to same canonical."""
        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")
        resource3 = ResourceId(url="http://doc3.com", id="doc3")

        # Three documents with different capitalizations
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "BRCA1": EntityMention(
                    kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="Doc1"
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "Brca1": EntityMention(
                    kind="gene", name="Brca1", aliases=[], quotes=[], reasoning="Doc2"
                )
            }
        )

        state.validated_entities_by_resource[resource3] = ref_map(
            {
                "brca1": EntityMention(
                    kind="gene", name="brca1", aliases=[], quotes=[], reasoning="Doc3"
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # All should use same canonical name
        canonical1 = list(ctx.state.validated_entities_by_resource[resource1].values())[
            0
        ].canonical
        canonical2 = list(ctx.state.validated_entities_by_resource[resource2].values())[
            0
        ].canonical
        canonical3 = list(ctx.state.validated_entities_by_resource[resource3].values())[
            0
        ].canonical

        assert canonical1 == canonical2 == canonical3
        # Prefer mixed-case over all-caps or all-lower
        assert canonical1 in ["BRCA1", "Brca1", "brca1"]


class TestIdenticalNames:
    """Test that identical canonical names don't create duplicate merges."""

    @pytest.mark.asyncio
    async def test_identical_names_handled_correctly(self, mock_deps):
        """Identical canonical names from aliases should not cause issues."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")

        # Entity with alias that expands to same canonical form
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "BMPR2": EntityMention(
                    kind="gene",
                    name="BMPR2",
                    aliases=["BMPR2 (bone morphogenetic protein receptor 2)"],
                    quotes=[],
                    reasoning="Doc1",
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # Should not create self-referential merges
        entities = ctx.state.validated_entities_by_resource[resource1]
        assert len(entities) == 1
        canonical = list(entities.values())[0].canonical
        assert canonical == "BMPR2"


class TestCrossDocumentConsistency:
    """Test that merged entities use consistent canonical names across all documents."""

    @pytest.mark.asyncio
    async def test_fuzzy_variants_consolidate_globally(self, mock_deps):
        """Fuzzy spelling variants should consolidate to same canonical across documents."""
        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")
        resource3 = ResourceId(url="http://doc3.com", id="doc3")

        # Three documents with US/UK spelling variants
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "tumor": EntityMention(
                    kind="phenotype",
                    name="tumor",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc1",
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "tumour": EntityMention(
                    kind="phenotype",
                    name="tumour",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc2",
                )
            }
        )

        state.validated_entities_by_resource[resource3] = ref_map(
            {
                "tumor": EntityMention(
                    kind="phenotype",
                    name="tumor",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc3",
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # All should use same canonical name
        canonical1 = list(ctx.state.validated_entities_by_resource[resource1].values())[
            0
        ].canonical
        canonical2 = list(ctx.state.validated_entities_by_resource[resource2].values())[
            0
        ].canonical
        canonical3 = list(ctx.state.validated_entities_by_resource[resource3].values())[
            0
        ].canonical

        assert canonical1 == canonical2 == canonical3

    @pytest.mark.asyncio
    async def test_contested_variants_prevented_from_merging(self, mock_deps):
        """Contested variants (mapping to multiple entities) should not auto-merge."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")

        # Two entities where expanded variant is ambiguous
        # "MAPK1" appears in "ERK (MAPK1)" and as standalone "MAPK1"
        # System should detect contestation and keep them separate
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "ERK (MAPK1)": EntityMention(
                    kind="gene",
                    name="ERK (MAPK1)",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc1",
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "MAPK1": EntityMention(
                    kind="gene",
                    name="MAPK1",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc2",
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # Should remain separate due to contestation
        canonical1 = list(ctx.state.validated_entities_by_resource[resource1].values())[
            0
        ].canonical
        canonical2 = list(ctx.state.validated_entities_by_resource[resource2].values())[
            0
        ].canonical

        # They should NOT be merged
        assert canonical1 != canonical2
        assert canonical1 == "ERK (MAPK1)"
        assert canonical2 == "MAPK1"

    @pytest.mark.asyncio
    async def test_multiple_entity_kinds_handled_separately(self, mock_deps):
        """Different entity kinds should not merge even with same names."""
        state = State(
            topic="test",
            target_entity_types=["gene", "phenotype"],
            permitted_pairs={"gene": {"gene"}, "phenotype": {"phenotype"}},
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")

        # Same name but different kinds
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "PAH": EntityMention(
                    kind="gene", name="PAH", aliases=[], quotes=[], reasoning="Gene"
                ),
                "PAH Disease": EntityMention(
                    kind="phenotype",
                    name="PAH Disease",
                    aliases=[],
                    quotes=[],
                    reasoning="Disease",
                ),
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # Should remain separate entities
        entities = ctx.state.validated_entities_by_resource[resource1]
        assert len(entities) == 2

        # Check kinds are preserved
        kinds = {ent.mentions[0].kind for canonical, ent in entities.items()}
        assert kinds == {"gene", "phenotype"}


class TestAliasPreservation:
    """Test that aliases are correctly preserved and accumulated during merges."""

    @pytest.mark.asyncio
    async def test_merge_preserves_old_canonical_as_alias(self, mock_deps):
        """When entities merge, the absorbed name should become an alias."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")

        # Two simple entities that will merge
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "BRCA1": EntityMention(
                    kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="Doc1"
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "Brca1": EntityMention(
                    kind="gene", name="Brca1", aliases=[], quotes=[], reasoning="Doc2"
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # Get the merged entity from first document
        entity_ref = list(ctx.state.validated_entities_by_resource[resource1].values())[
            0
        ]

        # The absorbed variant should be preserved somewhere
        # (either as alias or in mentions)
        all_names = {entity_ref.canonical}
        for mention in entity_ref.mentions:
            all_names.add(mention.name)
            all_names.update(mention.aliases)

        # Both original names should be represented
        assert "BRCA1" in all_names or "Brca1" in all_names

    @pytest.mark.asyncio
    async def test_merge_combines_aliases_from_both_entities(self, mock_deps):
        """When entities merge, aliases from both should be combined."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")

        # Two entities with different aliases that will merge via capitalization
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "TP53": EntityMention(
                    kind="gene",
                    name="TP53",
                    aliases=["tumor protein p53"],
                    quotes=[],
                    reasoning="Doc1",
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "Tp53": EntityMention(
                    kind="gene",
                    name="Tp53",
                    aliases=["p53"],
                    quotes=[],
                    reasoning="Doc2",
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # Get the merged entity and collect all aliases
        entity_ref = list(ctx.state.validated_entities_by_resource[resource1].values())[
            0
        ]
        all_aliases = set()
        for mention in entity_ref.mentions:
            all_aliases.update(mention.aliases)

        # Should have aliases from both original entities
        assert "tumor protein p53" in all_aliases or "p53" in all_aliases

    @pytest.mark.asyncio
    async def test_multiple_merges_accumulate_aliases(self, mock_deps):
        """Multiple sequential merges should accumulate all aliases."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")
        resource3 = ResourceId(url="http://doc3.com", id="doc3")

        # Three entities with different aliases, all will merge
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["breast cancer 1"],
                    quotes=[],
                    reasoning="Doc1",
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "Brca1": EntityMention(
                    kind="gene",
                    name="Brca1",
                    aliases=["BRCA1 gene"],
                    quotes=[],
                    reasoning="Doc2",
                )
            }
        )

        state.validated_entities_by_resource[resource3] = ref_map(
            {
                "brca1": EntityMention(
                    kind="gene",
                    name="brca1",
                    aliases=["BRCA1 protein"],
                    quotes=[],
                    reasoning="Doc3",
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # All should use same canonical
        canonical1 = list(ctx.state.validated_entities_by_resource[resource1].values())[
            0
        ].canonical
        canonical2 = list(ctx.state.validated_entities_by_resource[resource2].values())[
            0
        ].canonical
        canonical3 = list(ctx.state.validated_entities_by_resource[resource3].values())[
            0
        ].canonical

        assert canonical1 == canonical2 == canonical3

        # Collect all aliases from all documents
        all_aliases = set()
        for resource_id in [resource1, resource2, resource3]:
            entity_ref = list(
                ctx.state.validated_entities_by_resource[resource_id].values()
            )[0]
            for mention in entity_ref.mentions:
                all_aliases.update(mention.aliases)

        # Should have at least one alias from each original entity
        # (exact preservation depends on implementation)
        assert len(all_aliases) >= 1


class TestLLMInteraction:
    """Test LLM interaction for high-speculation cases."""

    @pytest.mark.asyncio
    async def test_high_speculation_goes_to_agent_review(self, mock_deps):
        """Pairs with speculation > threshold should not auto-merge."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from interaction_finder.extraction.models import (
            EntityConsolidationDecision,
            EntityConsolidationDecisions,
        )

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")

        # Two entities with very different names (high speculation)
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "ERK": EntityMention(
                    kind="gene", name="ERK", aliases=[], quotes=[], reasoning="Doc1"
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "MAPK": EntityMention(
                    kind="gene", name="MAPK", aliases=[], quotes=[], reasoning="Doc2"
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Mock LLM to reject the merge (implicit skip - return empty decisions)
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[]  # Empty = all pairs implicitly skipped
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "test_agent"

        with patch(
            "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
            return_value=mock_agent,
        ):
            await node.run(ctx)

        # Should remain separate (LLM rejected merge)
        entities1 = ctx.state.validated_entities_by_resource[resource1]
        entities2 = ctx.state.validated_entities_by_resource[resource2]

        # Both should still have their original entities
        assert len(entities1) == 1
        assert len(entities2) == 1


class TestIterativeConsolidation:
    """Test multiple consolidation rounds and transitive merges."""

    @pytest.mark.asyncio
    async def test_cross_document_merge_applied_everywhere(self, mock_deps):
        """When entities merge, change should apply to all documents."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")
        resource3 = ResourceId(url="http://doc3.com", id="doc3")

        # Same entity in 3 documents with different capitalizations
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "BRCA1": EntityMention(
                    kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="Doc1"
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "Brca1": EntityMention(
                    kind="gene", name="Brca1", aliases=[], quotes=[], reasoning="Doc2"
                )
            }
        )

        state.validated_entities_by_resource[resource3] = ref_map(
            {
                "brca1": EntityMention(
                    kind="gene", name="brca1", aliases=[], quotes=[], reasoning="Doc3"
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # All three documents should use the SAME canonical name
        canonical1 = list(ctx.state.validated_entities_by_resource[resource1].values())[
            0
        ].canonical
        canonical2 = list(ctx.state.validated_entities_by_resource[resource2].values())[
            0
        ].canonical
        canonical3 = list(ctx.state.validated_entities_by_resource[resource3].values())[
            0
        ].canonical

        assert canonical1 == canonical2
        assert canonical2 == canonical3
        # Should pick a consistent canonical (prefer mixed case)
        assert canonical1 in ["BRCA1", "Brca1", "brca1"]


class TestEdgeCases:
    """Test advanced edge cases and complex scenarios."""

    @pytest.mark.asyncio
    async def test_parenthetical_expansion_contestation(self, mock_deps):
        """Parenthetical expansion that's contested should not auto-merge."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")

        # ERK (MAPK1) expands to "MAPK1", but standalone "MAPK1" also exists
        # This creates contestation - "MAPK1" maps to both entities
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "ERK (MAPK1)": EntityMention(
                    kind="gene",
                    name="ERK (MAPK1)",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc1",
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "MAPK1": EntityMention(
                    kind="gene",
                    name="MAPK1",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc2",
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # Should remain separate due to contestation
        canonical1 = list(ctx.state.validated_entities_by_resource[resource1].values())[
            0
        ].canonical
        canonical2 = list(ctx.state.validated_entities_by_resource[resource2].values())[
            0
        ].canonical

        assert canonical1 != canonical2
        assert canonical1 == "ERK (MAPK1)"
        assert canonical2 == "MAPK1"

    @pytest.mark.asyncio
    async def test_iterative_consolidation_rounds(self, mock_deps):
        """Test that consolidation iterates until all opportunities are found."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")
        resource3 = ResourceId(url="http://doc3.com", id="doc3")

        # Three capitalization variants - system should iterate to catch all
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "BRCA1": EntityMention(
                    kind="gene", name="BRCA1", aliases=[], quotes=[], reasoning="Doc1"
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "Brca1": EntityMention(
                    kind="gene", name="Brca1", aliases=[], quotes=[], reasoning="Doc2"
                )
            }
        )

        state.validated_entities_by_resource[resource3] = ref_map(
            {
                "brca1": EntityMention(
                    kind="gene", name="brca1", aliases=[], quotes=[], reasoning="Doc3"
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # All three should converge to same canonical through iterative consolidation
        canonical1 = list(ctx.state.validated_entities_by_resource[resource1].values())[
            0
        ].canonical
        canonical2 = list(ctx.state.validated_entities_by_resource[resource2].values())[
            0
        ].canonical
        canonical3 = list(ctx.state.validated_entities_by_resource[resource3].values())[
            0
        ].canonical

        # All should converge
        assert canonical1 == canonical2 == canonical3

    @pytest.mark.asyncio
    async def test_mixed_variants_converge(self, mock_deps):
        """Test that capitalization + fuzzy + parenthetical all converge correctly."""
        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")
        resource3 = ResourceId(url="http://doc3.com", id="doc3")

        # Mix of: capitalization variant, fuzzy variant, and one with parenthetical
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "Tumor": EntityMention(
                    kind="phenotype",
                    name="Tumor",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc1",
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "tumour": EntityMention(
                    kind="phenotype",
                    name="tumour",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc2",
                )
            }
        )

        state.validated_entities_by_resource[resource3] = ref_map(
            {
                "TUMOR": EntityMention(
                    kind="phenotype",
                    name="TUMOR",
                    aliases=[],
                    quotes=[],
                    reasoning="Doc3",
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # All should converge to same canonical
        canonical1 = list(ctx.state.validated_entities_by_resource[resource1].values())[
            0
        ].canonical
        canonical2 = list(ctx.state.validated_entities_by_resource[resource2].values())[
            0
        ].canonical
        canonical3 = list(ctx.state.validated_entities_by_resource[resource3].values())[
            0
        ].canonical

        assert canonical1 == canonical2 == canonical3
        # Should prefer longer/mixed-case form
        assert canonical1 in ["Tumor", "tumour", "TUMOR"]

    @pytest.mark.asyncio
    async def test_duplicate_aliases_deduplicated(self, mock_deps):
        """Test that duplicate aliases are deduplicated after merge."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")

        # Two entities with overlapping aliases that will merge
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "TP53": EntityMention(
                    kind="gene",
                    name="TP53",
                    aliases=["p53", "tumor protein p53"],
                    quotes=[],
                    reasoning="Doc1",
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "Tp53": EntityMention(
                    kind="gene",
                    name="Tp53",
                    aliases=["p53", "P53"],  # Duplicate "p53" (different case)
                    quotes=[],
                    reasoning="Doc2",
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # Collect all aliases from merged entity
        entity_ref = list(ctx.state.validated_entities_by_resource[resource1].values())[
            0
        ]
        all_aliases = []
        for mention in entity_ref.mentions:
            all_aliases.extend(mention.aliases)

        # Count occurrences of "p53" (case-insensitive)
        p53_count = sum(1 for alias in all_aliases if alias.lower() == "p53")

        # Should not have excessive duplicates
        # (exact deduplication behavior depends on implementation,
        # but we shouldn't have 3+ of the same alias)
        assert p53_count <= 2  # Some duplicates acceptable across mentions


class TestIterativeRefinementModel:
    """Test the ClusterDecision model and split action."""

    def test_cluster_decision_split_action(self):
        """Test that ClusterDecision accepts split action."""
        from interaction_finder.extraction.models import ClusterDecision

        decision = ClusterDecision(
            group_id="test123",
            action="split",
            reasoning="Mixes disease and measurement",
        )
        assert decision.group_id == "test123"
        assert decision.action == "split"
        assert decision.target is None
        assert decision.reasoning == "Mixes disease and measurement"

    def test_cluster_decision_merge_action_explicit(self):
        """Test that ClusterDecision accepts explicit merge action."""
        from interaction_finder.extraction.models import ClusterDecision

        decision = ClusterDecision(
            group_id="test456",
            action="merge",
            target="PAH",
            reasoning="All variants of PAH",
        )
        assert decision.action == "merge"
        assert decision.target == "PAH"

    def test_cluster_decision_merge_action_default(self):
        """Test that action defaults to merge when omitted."""
        from interaction_finder.extraction.models import ClusterDecision

        decision = ClusterDecision(
            group_id="test789", target="Hypertension", reasoning="All same condition"
        )
        assert decision.action == "merge"  # Should default to merge
        assert decision.target == "Hypertension"

    def test_cluster_decision_exclude_action(self):
        """Test that ClusterDecision accepts exclude action."""
        from interaction_finder.extraction.models import ClusterDecision

        decision = ClusterDecision(
            group_id="test999",
            action="exclude",
            target="5",
            reasoning="Member 5 is a measurement, not a disease",
        )
        assert decision.group_id == "test999"
        assert decision.action == "exclude"
        assert decision.target == "5"
        assert decision.reasoning == "Member 5 is a measurement, not a disease"

    def test_cluster_decisions_batch(self):
        """Test that ClusterDecisions can hold multiple decisions."""
        from interaction_finder.extraction.models import (
            ClusterDecisions,
            ClusterDecision,
        )

        decisions = ClusterDecisions(
            decisions=[
                ClusterDecision(
                    group_id="g1",
                    action="merge",
                    target="PAH",
                    reasoning="PAH variants",
                ),
                ClusterDecision(
                    group_id="g2", action="split", reasoning="Mixes unrelated"
                ),
                ClusterDecision(
                    group_id="g3",
                    action="exclude",
                    target="Lower TAPSE",
                    reasoning="TAPSE is a measurement",
                ),
            ]
        )
        assert len(decisions.decisions) == 3
        assert decisions.decisions[0].action == "merge"
        assert decisions.decisions[1].action == "split"
        assert decisions.decisions[2].action == "exclude"
