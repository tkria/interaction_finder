"""Tests for rename decision caching behavior.

All merge/rename decisions are cached uniformly as the target canonical name
(or False for skip). This ensures:
1. Cross-document consistency - same decision applied everywhere
2. Efficiency - no redundant LLM calls for the same decision
3. New merge opportunities are checked when target is a new name
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.models import (
    EntityMention,
    EntityConsolidationDecision,
    EntityConsolidationDecisions,
)
from interaction_finder.extraction.state import State
from interaction_finder.resources import ResourceId, ResourcePool
from interaction_finder.settings import IfetcherConfig


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
        },
    )()
    return deps


def build_permitted_pairs(kinds: list[str]) -> dict[str, set[str]]:
    return {kind: {kind} for kind in kinds}


class TestRenameCaching:
    """Test that rename decisions are cached correctly."""

    @pytest.mark.asyncio
    async def test_rename_decision_cached_as_target(self, mock_deps):
        """Rename decisions should be cached as target canonical name."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)
        # Setup: LLM returns rename decision
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[
                EntityConsolidationDecision(
                    pair_id=1,
                    pair_token="ABCD",
                    action="rename",
                    target="TGF-β",
                    reasoning="Standard abbreviation",
                )
            ]
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "test_agent"
        # Prepare batch data - target "TGF-β" doesn't exist yet
        batch = [("tgf", "transforming growth factor beta")]
        canonical_lookup = {
            ("tgf", "gene"): "TGF",
            (
                "transforming growth factor beta",
                "gene",
            ): "Transforming growth factor beta",
        }
        kind_entities: dict[str, set[str]] = {
            "tgf": {"TGF"},
            "transforming growth factor beta": {"Transforming growth factor beta"},
        }
        with (
            patch(
                "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
                return_value=mock_agent,
            ),
            patch(
                "interaction_finder.extraction.nodes._generate_token",
                return_value="ABCD",
            ),
        ):
            rules, new_names = await node._process_consolidation_batch(
                batch, "gene", canonical_lookup, kind_entities, 1, ctx
            )
        # Verify cache entry is the target canonical name
        cache_key = ("tgf", "transforming growth factor beta", "gene")
        assert cache_key in ctx.state.merge_decision_cache
        cached_value = ctx.state.merge_decision_cache[cache_key]
        assert cached_value == "TGF-β"
        # Verify new_names returned (target doesn't exist in kind_entities)
        assert ("TGF-β", "gene") in new_names

    @pytest.mark.asyncio
    async def test_cached_decision_applied_without_llm(self, mock_deps):
        """Cached decisions should be applied without LLM call."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)
        # Pre-populate cache with target canonical name
        ctx.state.merge_decision_cache[
            ("tgf", "transforming growth factor beta", "gene")
        ] = "TGF-β"
        # Setup pairs and entities
        candidate_pairs = {"gene": [("tgf", "transforming growth factor beta")]}
        canonical_lookup = {
            ("tgf", "gene"): "TGF",
            (
                "transforming growth factor beta",
                "gene",
            ): "Transforming growth factor beta",
        }
        unique_entities = {
            "gene": {
                "tgf": {"TGF"},
                "transforming growth factor beta": {"Transforming growth factor beta"},
            }
        }
        # Mock LLM (should NOT be called)
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        with patch(
            "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
            return_value=mock_agent,
        ):
            rules, new_names = await node._get_consolidation_decisions(
                candidate_pairs, canonical_lookup, unique_entities, ctx
            )
        # LLM should not have been called
        assert not mock_agent.run.called
        # Rule should be created from cache - value is (target, reasoning) tuple
        assert ("transforming growth factor beta", "gene") in rules
        target, reasoning = rules[("transforming growth factor beta", "gene")]
        assert target == "TGF-β"
        # New name should be returned (target not in unique_entities)
        assert ("TGF-β", "gene") in new_names
        # Cache hit should be recorded
        assert ctx.state.merge_cache_hits == 1


class TestRenameMergeOpportunities:
    """Test that renames create new merge opportunities."""

    def test_add_new_names_to_entities(self, mock_deps):
        """New canonical names from renames should be added to unique_entities."""
        node = ConsolidateEntitiesNode()
        unique_entities = {
            "gene": {
                "tgf": {"TGF"},
                "tgf receptor": {"TGF receptor"},
            }
        }
        # Rename created "TGF-β" which normalizes to "tgf beta"
        new_names = [("TGF-β", "gene")]
        node._add_new_names_to_entities(new_names, unique_entities)
        # New normalized form should be added
        assert "tgf beta" in unique_entities["gene"]
        assert "TGF-β" in unique_entities["gene"]["tgf beta"]

    def test_new_name_creates_merge_opportunity(self, mock_deps):
        """Rename to existing entity name should create auto-merge opportunity."""
        node = ConsolidateEntitiesNode()
        # Setup: "TGF" already exists, rename creates another "TGF" canonical
        unique_entities = {
            "gene": {
                "tgf": {"TGF", "tgf"},  # Two canonicals for same normalized form
            }
        }
        # Find merge candidates should create auto-merge rule
        auto_rules, _, _ = node._find_merge_candidates(unique_entities)
        # One of the canonicals should be merged into the other
        assert len(auto_rules) == 1
        # Target should be the best canonical (uppercase) - value is (target, reasoning) tuple
        target, reasoning = list(auto_rules.values())[0]
        assert target == "TGF"


class TestCachedDecisionsAcrossRuns:
    """Test that cached decisions behave correctly across multiple processing runs.

    These tests simulate realistic scenarios where:
    1. First run: LLM makes decisions (merge/skip/rename)
    2. Second run: Same pairs encountered, should use cached decisions
    3. Interactions: Renames create new entities that may merge with existing ones
    """

    @pytest.mark.asyncio
    async def test_merge_decision_persists_across_runs(self, mock_deps):
        """Merge decision from first run should be reused in second run."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Setup: Two entities that are substring-related
        unique_entities = {
            "gene": {
                "brca": {"BRCA"},
                "brca1": {"BRCA1"},
            }
        }
        candidate_pairs = {"gene": [("brca", "brca1")]}
        canonical_lookup = {
            ("brca", "gene"): "BRCA",
            ("brca1", "gene"): "BRCA1",
        }

        # RUN 1: LLM decides to merge
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[
                EntityConsolidationDecision(
                    pair_id=1,
                    pair_token="ABCD",
                    action="merge",
                    reasoning="BRCA1 is a variant of BRCA",
                )
            ]
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "test_agent"

        with (
            patch(
                "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
                return_value=mock_agent,
            ),
            patch(
                "interaction_finder.extraction.nodes._generate_token",
                return_value="ABCD",
            ),
        ):
            rules1, new_names1 = await node._get_consolidation_decisions(
                candidate_pairs, canonical_lookup, unique_entities, ctx
            )

        # Verify first run called LLM and created rule - value is (target, reasoning) tuple
        assert mock_agent.run.called
        assert ("brca1", "gene") in rules1
        target1, reasoning1 = rules1[("brca1", "gene")]
        assert target1 == "BRCA"
        assert ctx.state.merge_decision_cache[("brca", "brca1", "gene")] == "BRCA"

        # RUN 2: Same pair encountered again (reset mock)
        mock_agent.run.reset_mock()

        with patch(
            "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
            return_value=mock_agent,
        ):
            rules2, new_names2 = await node._get_consolidation_decisions(
                candidate_pairs, canonical_lookup, unique_entities, ctx
            )

        # Should NOT call LLM - use cache
        assert not mock_agent.run.called
        # Should still produce the same rule - value is (target, reasoning) tuple
        assert ("brca1", "gene") in rules2
        target2, reasoning2 = rules2[("brca1", "gene")]
        assert target2 == "BRCA"

    @pytest.mark.asyncio
    async def test_skip_decision_persists_across_runs(self, mock_deps):
        """Skip decision from first run should prevent merge in second run."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        unique_entities = {
            "gene": {
                "tp": {"TP"},
                "tp53": {"TP53"},
            }
        }
        candidate_pairs = {"gene": [("tp", "tp53")]}
        canonical_lookup = {
            ("tp", "gene"): "TP",
            ("tp53", "gene"): "TP53",
        }

        # RUN 1: LLM decides to skip (not merge)
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[
                EntityConsolidationDecision(
                    pair_id=1,
                    pair_token="ABCD",
                    action="skip",
                    reasoning="TP and TP53 are different proteins",
                )
            ]
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "test_agent"

        with (
            patch(
                "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
                return_value=mock_agent,
            ),
            patch(
                "interaction_finder.extraction.nodes._generate_token",
                return_value="ABCD",
            ),
        ):
            rules1, _ = await node._get_consolidation_decisions(
                candidate_pairs, canonical_lookup, unique_entities, ctx
            )

        # Verify first run: no rule created, but cached as False
        assert mock_agent.run.called
        assert len(rules1) == 0
        assert ctx.state.merge_decision_cache[("tp", "tp53", "gene")] is False

        # RUN 2: Same pair - should use cache and not merge
        mock_agent.run.reset_mock()

        with patch(
            "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
            return_value=mock_agent,
        ):
            rules2, _ = await node._get_consolidation_decisions(
                candidate_pairs, canonical_lookup, unique_entities, ctx
            )

        # Should NOT call LLM
        assert not mock_agent.run.called
        # Should still produce no rules
        assert len(rules2) == 0

    @pytest.mark.asyncio
    async def test_rename_then_merge_with_existing(self, mock_deps):
        """Rename creates entity that matches existing - should merge on next iteration."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Setup: "TGF-beta protein" exists, and so does "TGF-β"
        # Rename will change "TGF-beta protein" to "TGF-β", creating a collision
        unique_entities = {
            "gene": {
                "tgf beta protein": {"TGF-beta protein"},
                "tgf beta": {"TGF-β"},  # Already exists!
            }
        }
        candidate_pairs = {"gene": [("tgf beta", "tgf beta protein")]}
        canonical_lookup = {
            ("tgf beta", "gene"): "TGF-β",
            ("tgf beta protein", "gene"): "TGF-beta protein",
        }

        # RUN 1: LLM decides to rename "TGF-beta protein" to "TGF-β"
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[
                EntityConsolidationDecision(
                    pair_id=1,
                    pair_token="ABCD",
                    action="rename",
                    target="TGF-β",
                    reasoning="Use standard symbol for consistency",
                )
            ]
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "test_agent"

        with (
            patch(
                "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
                return_value=mock_agent,
            ),
            patch(
                "interaction_finder.extraction.nodes._generate_token",
                return_value="ABCD",
            ),
        ):
            rules1, new_names1 = await node._get_consolidation_decisions(
                candidate_pairs, canonical_lookup, unique_entities, ctx
            )

        # Verify rename rule created - value is (target, reasoning) tuple
        assert ("tgf beta protein", "gene") in rules1
        target, reasoning = rules1[("tgf beta protein", "gene")]
        assert target == "TGF-β"

        # Cache should have the target
        assert (
            ctx.state.merge_decision_cache[("tgf beta", "tgf beta protein", "gene")]
            == "TGF-β"
        )

        # new_names should NOT include TGF-β since it already exists in unique_entities
        assert ("TGF-β", "gene") not in new_names1

    @pytest.mark.asyncio
    async def test_rename_to_new_name_triggers_recheck(self, mock_deps):
        """Rename to brand new name should be flagged for merge opportunity check."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Setup: Only verbose name exists, rename will create new short name
        unique_entities = {
            "gene": {
                "transforming growth factor beta": {"Transforming growth factor beta"},
            }
        }
        # Simulate a pair where LLM will rename to "TGF-β"
        candidate_pairs = {"gene": [("tgf", "transforming growth factor beta")]}
        canonical_lookup = {
            ("tgf", "gene"): "TGF",  # Would be the parent in substring match
            (
                "transforming growth factor beta",
                "gene",
            ): "Transforming growth factor beta",
        }
        # Add "tgf" to unique_entities for the lookup to work
        unique_entities["gene"]["tgf"] = {"TGF"}

        # LLM decides to rename to brand new name "TGF-β"
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[
                EntityConsolidationDecision(
                    pair_id=1,
                    pair_token="ABCD",
                    action="rename",
                    target="TGF-β",
                    reasoning="Standard abbreviation",
                )
            ]
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "test_agent"

        with (
            patch(
                "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
                return_value=mock_agent,
            ),
            patch(
                "interaction_finder.extraction.nodes._generate_token",
                return_value="ABCD",
            ),
        ):
            rules, new_names = await node._get_consolidation_decisions(
                candidate_pairs, canonical_lookup, unique_entities, ctx
            )

        # "TGF-β" normalizes to "tgf beta" which doesn't exist in unique_entities
        # So it should be flagged as a new name
        assert ("TGF-β", "gene") in new_names

        # When we add this new name and re-run _find_merge_candidates,
        # it might find new merge opportunities
        node._add_new_names_to_entities(new_names, unique_entities)
        assert "tgf beta" in unique_entities["gene"]
        assert "TGF-β" in unique_entities["gene"]["tgf beta"]

    @pytest.mark.asyncio
    async def test_full_interaction_sequence(self, mock_deps):
        """Test a realistic sequence: skip → merge → rename across multiple pairs.

        Scenario:
        - Pair 1 (TP, TP53): Skip - these are different proteins
        - Pair 2 (BRCA, BRCA1): Merge - BRCA1 is a variant
        - Pair 3 (TGF-beta protein, TGF): Rename to "TGF-β"

        Then verify all decisions are cached and reused correctly.
        """
        node = ConsolidateEntitiesNode()
        state = State(
            topic="cancer genetics",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        unique_entities = {
            "gene": {
                "tp": {"TP"},
                "tp53": {"TP53"},
                "brca": {"BRCA"},
                "brca1": {"BRCA1"},
                "tgf": {"TGF"},
                "tgf beta protein": {"TGF-beta protein"},
            }
        }
        candidate_pairs = {
            "gene": [
                ("tp", "tp53"),
                ("brca", "brca1"),
                ("tgf", "tgf beta protein"),
            ]
        }
        canonical_lookup = {
            ("tp", "gene"): "TP",
            ("tp53", "gene"): "TP53",
            ("brca", "gene"): "BRCA",
            ("brca1", "gene"): "BRCA1",
            ("tgf", "gene"): "TGF",
            ("tgf beta protein", "gene"): "TGF-beta protein",
        }

        # First run: LLM makes all three decisions
        call_count = 0

        def make_decisions(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            mock_result.output = EntityConsolidationDecisions(
                decisions=[
                    EntityConsolidationDecision(
                        pair_id=1,
                        pair_token="TOK1",
                        action="skip",
                        reasoning="TP and TP53 are different proteins with distinct functions",
                    ),
                    EntityConsolidationDecision(
                        pair_id=2,
                        pair_token="TOK2",
                        action="merge",
                        reasoning="BRCA1 is a well-known variant of BRCA gene family",
                    ),
                    EntityConsolidationDecision(
                        pair_id=3,
                        pair_token="TOK3",
                        action="rename",
                        target="TGF-β",
                        reasoning="Use standard symbol for transforming growth factor beta",
                    ),
                ]
            )
            return mock_result

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=make_decisions)
        mock_agent._name = "test_agent"

        token_sequence = iter(["TOK1", "TOK2", "TOK3"])

        with (
            patch(
                "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
                return_value=mock_agent,
            ),
            patch(
                "interaction_finder.extraction.nodes._generate_token",
                side_effect=lambda: next(token_sequence),
            ),
        ):
            rules1, new_names1 = await node._get_consolidation_decisions(
                candidate_pairs, canonical_lookup, unique_entities, ctx
            )

        # Verify first run results - values are (target, reasoning) tuples
        assert call_count == 1  # LLM called once

        # Skip: no rule for TP/TP53
        assert ("tp53", "gene") not in rules1

        # Merge: BRCA1 → BRCA
        assert ("brca1", "gene") in rules1
        merge_target, merge_reasoning = rules1[("brca1", "gene")]
        assert merge_target == "BRCA"

        # Rename: TGF-beta protein → TGF-β
        assert ("tgf beta protein", "gene") in rules1
        rename_target, rename_reasoning = rules1[("tgf beta protein", "gene")]
        assert rename_target == "TGF-β"

        # New name flagged (TGF-β doesn't exist yet)
        assert ("TGF-β", "gene") in new_names1

        # Verify cache state
        assert ctx.state.merge_decision_cache[("tp", "tp53", "gene")] is False
        assert ctx.state.merge_decision_cache[("brca", "brca1", "gene")] == "BRCA"
        assert (
            ctx.state.merge_decision_cache[("tgf", "tgf beta protein", "gene")]
            == "TGF-β"
        )

        # SECOND RUN: Same pairs - should all use cache
        mock_agent.run.reset_mock()
        call_count = 0

        with patch(
            "interaction_finder.extraction.nodes.get_entity_consolidation_agent",
            return_value=mock_agent,
        ):
            rules2, new_names2 = await node._get_consolidation_decisions(
                candidate_pairs, canonical_lookup, unique_entities, ctx
            )

        # LLM should NOT be called
        assert not mock_agent.run.called

        # Same rules should be produced from cache - values are (target, reasoning) tuples
        assert ("tp53", "gene") not in rules2  # Still skip
        merge_target2, merge_reasoning2 = rules2[("brca1", "gene")]
        assert merge_target2 == "BRCA"  # Still merge
        rename_target2, rename_reasoning2 = rules2[("tgf beta protein", "gene")]
        assert rename_target2 == "TGF-β"  # Still rename

        # Cache hits should be recorded
        assert ctx.state.merge_cache_hits == 3
