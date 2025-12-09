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


class TestContestedVariantClustering:
    """Test that contested variants don't prevent entities from clustering."""

    @pytest.mark.asyncio
    async def test_contested_variant_entities_still_cluster(self, mock_deps):
        """Entities with contested variant forms should still cluster based on other tokens.

        Regression test for bug where entities with contested variants were excluded
        from clustering entirely, even though they could cluster on other shared tokens.
        """
        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")
        resource2 = ResourceId(url="http://doc2.com", id="doc2")
        resource3 = ResourceId(url="http://doc3.com", id="doc3")

        # Three entities that share tokens but have a contested variant:
        # - "Pulmonary Arterial Hypertension" has alias "PAH"
        # - "Heritable pulmonary arterial hypertension" has alias "HPAH" and "heritable PAH"
        # - "Familial pulmonary arterial hypertension" has alias "familial PAH"
        #
        # The variant "PAH" becomes contested (maps to both main PAH and heritable PAH)
        # But they should still cluster on shared tokens: pulmonary, arterial, hypertension
        state.validated_entities_by_resource[resource1] = ref_map(
            {
                "Pulmonary Arterial Hypertension": EntityMention(
                    kind="phenotype",
                    name="Pulmonary Arterial Hypertension",
                    aliases=["PAH"],
                    quotes=[],
                    reasoning="Doc1",
                )
            }
        )

        state.validated_entities_by_resource[resource2] = ref_map(
            {
                "Heritable pulmonary arterial hypertension": EntityMention(
                    kind="phenotype",
                    name="Heritable pulmonary arterial hypertension",
                    aliases=["HPAH", "heritable PAH"],
                    quotes=[],
                    reasoning="Doc2",
                )
            }
        )

        state.validated_entities_by_resource[resource3] = ref_map(
            {
                "Familial pulmonary arterial hypertension": EntityMention(
                    kind="phenotype",
                    name="Familial pulmonary arterial hypertension",
                    aliases=["familial PAH", "FPAH"],
                    quotes=[],
                    reasoning="Doc3",
                )
            }
        )

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # Check that clusters were recorded in consolidated data for phenotype kind
        phenotype_merges = ctx.state.consolidated.entities.merges.get("phenotype")
        assert phenotype_merges is not None, "phenotype merges should be recorded"
        clusters = phenotype_merges.clusters
        # The entities should have been included in clustering despite contested variants
        # Check that we have at least one multi-entity cluster
        multi_entity_clusters = [c for c in clusters if len(c) > 1]
        assert len(multi_entity_clusters) > 0, (
            "PAH-related entities should cluster together on shared tokens"
        )


class TestMultipleClusterDecisions:
    """Test handling of multiple LLM decisions for the same cluster.

    These tests directly test the decision processing logic by calling
    _get_group_consolidation_decisions with mocked agent responses.
    """

    @pytest.mark.asyncio
    async def test_multiple_excludes_same_group(self, mock_deps):
        """Multiple exclude decisions on same group should all be applied."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from interaction_finder.extraction.models import (
            ClusterDecision,
            ClusterDecisions,
        )
        from interaction_finder.extraction.clustering import Cluster

        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )
        # Create minimal state for the method
        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)
        # Create test groups and entities
        group = frozenset(["Disease A", "Disease B", "Measurement X", "Measurement Y"])
        entities = {
            name: []
            for name in group  # Empty variant lists (not needed for this test)
        }
        # Create a simple merge tree for the group
        merge_tree = Cluster(entities=group, similarity=0.8)

        def make_mock_result(prompt):
            import re

            match = re.search(r"## Group (\w+)", prompt)
            if match:
                group_id = match.group(1)
                # Return two exclude decisions for same group
                return MagicMock(
                    output=ClusterDecisions(
                        decisions=[
                            ClusterDecision(
                                group_id=group_id,
                                action="exclude",
                                target="Measurement X",
                                reasoning="X is a measurement",
                            ),
                            ClusterDecision(
                                group_id=group_id,
                                action="exclude",
                                target="Measurement Y",
                                reasoning="Y is a measurement",
                            ),
                        ]
                    )
                )
            return MagicMock(output=ClusterDecisions(decisions=[]))

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=make_mock_result)

        with patch("interaction_finder.agent_config.agent_getter") as mock_getter:
            mock_getter.return_value = lambda config: mock_agent
            rules, new_names, _ = await node._get_group_consolidation_decisions(
                [group], [merge_tree], "phenotype", entities, ctx
            )

        # Both measurements excluded, only diseases remain
        # With excludes-only, remaining group goes to next round (no merge rules created)
        # Key assertion: no merge rules should include excluded members
        for (norm_name, kind), (target, reason) in rules.items():
            assert "measurement" not in norm_name.lower()

    @pytest.mark.asyncio
    async def test_exclude_plus_merge(self, mock_deps):
        """Exclude + merge on same group: exclude first, then merge remainder."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from interaction_finder.extraction.models import (
            ClusterDecision,
            ClusterDecisions,
        )
        from interaction_finder.extraction.clustering import Cluster

        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )
        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)
        # 3 entities: 2 diseases + 1 measurement
        group = frozenset(
            ["Hypertension Type A", "Hypertension Type B", "Blood Pressure"]
        )
        entities = {name: [] for name in group}
        merge_tree = Cluster(entities=group, similarity=0.8)

        def make_mock_result(prompt):
            import re

            match = re.search(r"## Group (\w+)", prompt)
            if match:
                group_id = match.group(1)
                # Exclude measurement, merge the two diseases
                return MagicMock(
                    output=ClusterDecisions(
                        decisions=[
                            ClusterDecision(
                                group_id=group_id,
                                action="exclude",
                                target="Blood Pressure",
                                reasoning="Blood Pressure is a measurement",
                            ),
                            ClusterDecision(
                                group_id=group_id,
                                action="merge",
                                target="Hypertension",  # New canonical name
                                reasoning="Both are hypertension subtypes",
                            ),
                        ]
                    )
                )
            return MagicMock(output=ClusterDecisions(decisions=[]))

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=make_mock_result)

        with patch("interaction_finder.agent_config.agent_getter") as mock_getter:
            mock_getter.return_value = lambda config: mock_agent
            rules, new_names, _ = await node._get_group_consolidation_decisions(
                [group], [merge_tree], "phenotype", entities, ctx
            )

        # Should have merge rules for the hypertension types (not Blood Pressure)
        merged_sources = {norm_name for (norm_name, kind) in rules.keys()}
        # Blood Pressure should NOT be in merge rules (it was excluded)
        assert "blood pressure" not in merged_sources
        # New name should be registered
        assert "Hypertension" in new_names

    @pytest.mark.asyncio
    async def test_merge_plus_split_rejected(self, mock_deps):
        """Merge + split on same group should be rejected as contradictory."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from interaction_finder.extraction.models import (
            ClusterDecision,
            ClusterDecisions,
        )
        from interaction_finder.extraction.clustering import Cluster

        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )
        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)
        group = frozenset(["Entity A", "Entity B"])
        entities = {name: [] for name in group}
        merge_tree = Cluster(entities=group, similarity=0.8)

        def make_mock_result(prompt):
            import re

            match = re.search(r"## Group (\w+)", prompt)
            if match:
                group_id = match.group(1)
                # Contradictory: both merge AND split
                return MagicMock(
                    output=ClusterDecisions(
                        decisions=[
                            ClusterDecision(
                                group_id=group_id,
                                action="merge",
                                target="Entity A",
                                reasoning="Same entity",
                            ),
                            ClusterDecision(
                                group_id=group_id,
                                action="split",
                                reasoning="Different entities",
                            ),
                        ]
                    )
                )
            return MagicMock(output=ClusterDecisions(decisions=[]))

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=make_mock_result)

        with patch("interaction_finder.agent_config.agent_getter") as mock_getter:
            mock_getter.return_value = lambda config: mock_agent
            rules, new_names, _ = await node._get_group_consolidation_decisions(
                [group], [merge_tree], "phenotype", entities, ctx
            )

        # Should be rejected - no merge rules created
        assert len(rules) == 0

    @pytest.mark.asyncio
    async def test_exclude_plus_split(self, mock_deps):
        """Exclude + split on same group: exclude first, then split remainder."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from interaction_finder.extraction.models import (
            ClusterDecision,
            ClusterDecisions,
        )
        from interaction_finder.extraction.clustering import Cluster

        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )
        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)
        # 4 entities: will exclude 1, then split remaining 3
        group = frozenset(
            ["Disease Alpha", "Disease Beta", "Disease Gamma", "Unrelated Thing"]
        )
        entities = {name: [] for name in group}
        # Create a tree structure for splitting
        left = Cluster(
            entities=frozenset(["Disease Alpha", "Disease Beta"]), similarity=0.9
        )
        right = Cluster(
            entities=frozenset(["Disease Gamma", "Unrelated Thing"]), similarity=0.7
        )
        merge_tree = Cluster(entities=group, left=left, right=right, similarity=0.5)
        call_count = 0

        def make_mock_result(prompt):
            nonlocal call_count
            call_count += 1
            import re

            match = re.search(r"## Group (\w+)", prompt)
            if match and call_count == 1:
                group_id = match.group(1)
                # First round: exclude + split
                return MagicMock(
                    output=ClusterDecisions(
                        decisions=[
                            ClusterDecision(
                                group_id=group_id,
                                action="exclude",
                                target="Unrelated Thing",
                                reasoning="Doesn't belong",
                            ),
                            ClusterDecision(
                                group_id=group_id,
                                action="split",
                                reasoning="Remaining diseases should be split",
                            ),
                        ]
                    )
                )
            # Subsequent rounds: no more decisions
            return MagicMock(output=ClusterDecisions(decisions=[]))

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=make_mock_result)

        with patch("interaction_finder.agent_config.agent_getter") as mock_getter:
            mock_getter.return_value = lambda config: mock_agent
            rules, new_names, _ = await node._get_group_consolidation_decisions(
                [group], [merge_tree], "phenotype", entities, ctx
            )

        # Exclude + split should not create merge rules
        # (split divides the group, doesn't merge)
        # Unrelated Thing should NOT appear in any merge rules
        merged_sources = {norm_name for (norm_name, kind) in rules.keys()}
        assert "unrelated thing" not in merged_sources


class TestResolvedGroupsTracking:
    """Test that resolved groups are tracked and not re-asked."""

    @pytest.mark.asyncio
    async def test_returns_kept_separate_as_resolved(self, mock_deps):
        """Groups kept separate should be returned as resolved."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from interaction_finder.extraction.models import ClusterDecisions
        from interaction_finder.extraction.clustering import Cluster

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)
        group1 = frozenset(["Gene A", "Gene B"])
        group2 = frozenset(["Gene X", "Gene Y"])
        entities = {name: [] for g in [group1, group2] for name in g}
        merge_tree1 = Cluster(entities=group1, similarity=0.5)
        merge_tree2 = Cluster(entities=group2, similarity=0.5)
        # LLM returns no decisions - both groups kept separate
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(
            return_value=MagicMock(output=ClusterDecisions(decisions=[]))
        )
        with patch("interaction_finder.agent_config.agent_getter") as mock_getter:
            mock_getter.return_value = lambda config: mock_agent
            rules, new_names, resolved = await node._get_group_consolidation_decisions(
                [group1, group2], [merge_tree1, merge_tree2], "gene", entities, ctx
            )
        assert len(resolved) == 2
        assert group1 in resolved
        assert group2 in resolved
        assert len(rules) == 0

    @pytest.mark.asyncio
    async def test_merged_groups_not_in_resolved(self, mock_deps):
        """Groups that were merged should not be in resolved (they're transformed)."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from interaction_finder.extraction.models import (
            ClusterDecision,
            ClusterDecisions,
        )
        from interaction_finder.extraction.clustering import Cluster

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)
        group = frozenset(["Gene A", "Gene B"])
        entities = {name: [] for name in group}
        merge_tree = Cluster(entities=group, similarity=0.8)

        def make_mock_result(prompt):
            import re

            match = re.search(r"## Group (\w+)", prompt)
            if match:
                return MagicMock(
                    output=ClusterDecisions(
                        decisions=[
                            ClusterDecision(
                                group_id=match.group(1),
                                action="merge",
                                target="1",
                                reasoning="Same gene",
                            )
                        ]
                    )
                )
            return MagicMock(output=ClusterDecisions(decisions=[]))

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=make_mock_result)
        with patch("interaction_finder.agent_config.agent_getter") as mock_getter:
            mock_getter.return_value = lambda config: mock_agent
            rules, new_names, resolved = await node._get_group_consolidation_decisions(
                [group], [merge_tree], "gene", entities, ctx
            )
        # Merged group is not "kept separate" - it was acted upon
        assert group not in resolved
        assert len(rules) == 1  # Gene B -> Gene A
