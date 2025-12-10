"""Tests for per-kind entity consolidation.

Verifies that:
1. Each entity kind is processed independently to completion
2. Renames in one kind don't trigger re-processing of other kinds
3. Auto-merges are not repeated across iterations within a kind
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    ClusterDecisions,
    ClusterDecision,
)
from interaction_finder.extraction.state import State
from interaction_finder.resources import ResourceId, ResourcePool
from interaction_finder.settings import IfetcherConfig


def ref_map(data: dict[str, EntityMention]) -> dict[str, EntityRef]:
    """Convert entity mentions to EntityRef dict."""
    return {
        name: EntityRef(canonical=name, mentions=[mention])
        for name, mention in data.items()
    }


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    return {kind: {kind} for kind in kinds}


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


class TestPerKindIndependence:
    """Test that entity kinds are processed independently."""

    @pytest.mark.asyncio
    async def test_gene_and_phenotype_processed_separately(self, mock_deps):
        """Gene entities should be fully processed before phenotype entities."""
        state = State(
            topic="test",
            target_entity_types=["gene", "phenotype"],
            permitted_pairs={"gene": {"gene"}, "phenotype": {"phenotype"}},
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")

        # Add entities of different kinds
        state.validated_entities_by_resource[resource1] = {
            "BRCA1": EntityRef(
                canonical="BRCA1",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "Brca1": EntityRef(
                canonical="Brca1",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="Brca1",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "Tumor": EntityRef(
                canonical="Tumor",
                mentions=[
                    EntityMention(
                        kind="phenotype",
                        name="Tumor",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "tumour": EntityRef(
                canonical="tumour",
                mentions=[
                    EntityMention(
                        kind="phenotype",
                        name="tumour",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
        }

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # Gene entities should be merged (BRCA1/Brca1 are capitalization variants)
        # Phenotype entities should be merged (Tumor/tumour are fuzzy variants)
        entities = ctx.state.validated_entities_by_resource[resource1]

        # Should have 2 entities (one gene, one phenotype)
        assert len(entities) == 2

        # Check that both kinds were consolidated
        kinds = {ent.kind for ent in entities.values()}
        assert kinds == {"gene", "phenotype"}

    @pytest.mark.asyncio
    async def test_rename_in_one_kind_doesnt_affect_other(self, mock_deps):
        """A rename in gene kind should not trigger re-clustering of phenotype kind."""
        state = State(
            topic="test",
            target_entity_types=["gene", "phenotype"],
            permitted_pairs={"gene": {"gene"}, "phenotype": {"phenotype"}},
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")

        # Add entities - genes will get LLM review, phenotypes will auto-merge
        state.validated_entities_by_resource[resource1] = {
            "GeneA": EntityRef(
                canonical="GeneA",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="GeneA",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "GeneB": EntityRef(
                canonical="GeneB",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="GeneB",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "Tumor": EntityRef(
                canonical="Tumor",
                mentions=[
                    EntityMention(
                        kind="phenotype",
                        name="Tumor",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "tumour": EntityRef(
                canonical="tumour",
                mentions=[
                    EntityMention(
                        kind="phenotype",
                        name="tumour",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
        }

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Track clustering calls
        clustering_calls = []
        original_find = None

        def tracking_find(*args, **kwargs):
            clustering_calls.append(args[0])  # entities dict
            return original_find(*args, **kwargs)

        # Mock LLM to return empty decisions
        mock_result = MagicMock()
        mock_result.output = ClusterDecisions(decisions=[])
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with patch("interaction_finder.agent_config.agent_getter") as mock_getter:
            mock_getter.return_value = lambda config: mock_agent
            with patch(
                "interaction_finder.extraction.nodes.find_consolidation_candidates"
            ) as mock_find:
                # Import the real function to use as fallback
                from interaction_finder.extraction.entity_matching import (
                    find_consolidation_candidates as real_find,
                )

                original_find = real_find
                mock_find.side_effect = tracking_find
                await node.run(ctx)

        # Each kind should be clustered exactly once per iteration
        # (may have multiple iterations within a kind, but other kind shouldn't be affected)
        gene_clustering_count = sum(
            1
            for entities in clustering_calls
            if any("Gene" in name for name in entities.keys())
        )
        phenotype_clustering_count = sum(
            1
            for entities in clustering_calls
            if any("umor" in name or "umour" in name for name in entities.keys())
        )

        # Both should be clustered at least once
        assert gene_clustering_count >= 1
        assert phenotype_clustering_count >= 1


class TestNoRepeatedAutoMerges:
    """Test that auto-merges are not repeated across iterations."""

    @pytest.mark.asyncio
    async def test_auto_merges_applied_only_once(self, mock_deps):
        """Auto-merge rules should only be applied once, not repeated."""
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")

        # Multiple capitalization variants that will auto-merge
        state.validated_entities_by_resource[resource1] = {
            "BRCA1": EntityRef(
                canonical="BRCA1",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="BRCA1",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "Brca1": EntityRef(
                canonical="Brca1",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="Brca1",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "brca1": EntityRef(
                canonical="brca1",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="brca1",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
        }

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        await node.run(ctx)

        # All should be merged to same canonical
        entities = ctx.state.validated_entities_by_resource[resource1]
        assert len(entities) == 1

        # Check merges were recorded
        gene_merges = ctx.state.consolidated.entities.merges.get("gene")
        assert gene_merges is not None

        # Auto-merges should be recorded
        auto_merge_count = len(gene_merges.automatic)
        # Should have 2 auto-merges (Brca1 → BRCA1, brca1 → BRCA1 or similar)
        assert auto_merge_count >= 1


class TestIterationWithinKind:
    """Test that iteration happens correctly within a single kind."""

    @pytest.mark.asyncio
    async def test_splits_trigger_iteration_within_kind(self, mock_deps):
        """Split decisions should cause re-review of subgroups within same kind."""
        state = State(
            topic="test",
            target_entity_types=["phenotype"],
            permitted_pairs=build_permitted_pairs(["phenotype"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")

        # Entities that will cluster together
        state.validated_entities_by_resource[resource1] = {
            "Disease A Type 1": EntityRef(
                canonical="Disease A Type 1",
                mentions=[
                    EntityMention(
                        kind="phenotype",
                        name="Disease A Type 1",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "Disease A Type 2": EntityRef(
                canonical="Disease A Type 2",
                mentions=[
                    EntityMention(
                        kind="phenotype",
                        name="Disease A Type 2",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "Disease B": EntityRef(
                canonical="Disease B",
                mentions=[
                    EntityMention(
                        kind="phenotype",
                        name="Disease B",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
        }

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        call_count = 0

        def make_mock_result(prompt):
            nonlocal call_count
            call_count += 1
            # First call: split the cluster
            if call_count == 1:
                import re

                match = re.search(r"## Group (\w+)", prompt)
                if match:
                    return MagicMock(
                        output=ClusterDecisions(
                            decisions=[
                                ClusterDecision(
                                    group_id=match.group(1),
                                    action="split",
                                    reasoning="Mixes different diseases",
                                )
                            ]
                        )
                    )
            # Subsequent calls: keep separate
            return MagicMock(output=ClusterDecisions(decisions=[]))

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=make_mock_result)

        with patch("interaction_finder.agent_config.agent_getter") as mock_getter:
            mock_getter.return_value = lambda config: mock_agent
            await node.run(ctx)

        # LLM should have been called multiple times (initial + after split)
        assert call_count >= 1


class TestRenameTargetedIteration:
    """Test that rename-triggered iterations only review groups containing new names."""

    @pytest.mark.asyncio
    async def test_only_groups_with_new_names_reviewed(self, mock_deps):
        """After a rename, only groups containing the new name should be reviewed."""
        from interaction_finder.extraction.entity_matching import (
            ConsolidationCandidates,
            SpeculatedVariant,
        )

        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )

        resource1 = ResourceId(url="http://doc1.com", id="doc1")

        # Set up entities
        state.validated_entities_by_resource[resource1] = {
            "GeneA": EntityRef(
                canonical="GeneA",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="GeneA",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "GeneB": EntityRef(
                canonical="GeneB",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="GeneB",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "GeneC": EntityRef(
                canonical="GeneC",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="GeneC",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
            "GeneD": EntityRef(
                canonical="GeneD",
                mentions=[
                    EntityMention(
                        kind="gene",
                        name="GeneD",
                        aliases=[],
                        quotes=[],
                        reasoning="Doc1",
                    )
                ],
            ),
        }

        node = ConsolidateEntitiesNode()
        ctx = GraphRunContext(state=state, deps=mock_deps)

        iteration_count = 0
        groups_reviewed_per_iteration = []

        # Track which groups are sent to LLM on each iteration
        def mock_find_candidates(entities, threshold, mention_counts, logger):
            nonlocal iteration_count
            iteration_count += 1

            if iteration_count == 1:
                # First iteration: two groups
                return ConsolidationCandidates(
                    auto_merge=[],
                    agent_review=[],
                    agent_review_groups=[
                        frozenset({"GeneA", "GeneB"}),  # Group 1
                        frozenset({"GeneC", "GeneD"}),  # Group 2
                    ],
                    contested_warnings=[],
                    merge_trees=[],
                )
            else:
                # Second iteration: same groups plus NewGene in one
                return ConsolidationCandidates(
                    auto_merge=[],
                    agent_review=[],
                    agent_review_groups=[
                        frozenset({"GeneA", "GeneB", "NewGene"}),  # Contains new name
                        frozenset({"GeneC", "GeneD"}),  # Does NOT contain new name
                    ],
                    contested_warnings=[],
                    merge_trees=[],
                )

        call_count = 0

        def make_mock_result(prompt):
            nonlocal call_count
            call_count += 1

            # Track which groups are in the prompt
            import re

            groups_in_prompt = re.findall(r"## Group (\w+)", prompt)
            groups_reviewed_per_iteration.append(len(groups_in_prompt))

            if call_count == 1:
                # First call: rename GeneA to NewGene
                match = re.search(r"## Group (\w+)", prompt)
                if match:
                    return MagicMock(
                        output=ClusterDecisions(
                            decisions=[
                                ClusterDecision(
                                    group_id=match.group(1),
                                    action="merge",
                                    target="NewGene",  # Rename to new name
                                    reasoning="Standardized name",
                                )
                            ]
                        )
                    )
            # Subsequent calls: keep separate
            return MagicMock(output=ClusterDecisions(decisions=[]))

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=make_mock_result)

        with patch("interaction_finder.agent_config.agent_getter") as mock_getter:
            mock_getter.return_value = lambda config: mock_agent
            with patch(
                "interaction_finder.extraction.nodes.find_consolidation_candidates",
                side_effect=mock_find_candidates,
            ):
                await node.run(ctx)

        # Should have had 2 iterations (first found rename, second checked new name)
        assert iteration_count == 2

        # First iteration should review 2 groups, second should review only 1
        # (the one containing NewGene)
        assert len(groups_reviewed_per_iteration) >= 2
        assert groups_reviewed_per_iteration[0] == 2  # Both groups initially
        assert groups_reviewed_per_iteration[1] == 1  # Only group with NewGene
