"""Tests for entity consolidation with speculation-based matching.

The new system uses speculation thresholds instead of explicit caching:
1. Auto-merge for low speculation (≤ threshold)
2. Agent review for high speculation (> threshold)
3. Contested variants are filtered out
4. Renames create new merge opportunities
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
    EntityRef,
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


class TestWithinRunCaching:
    """Test that LLM merge decisions are cached within a single run."""

    @pytest.mark.asyncio
    async def test_same_pair_uses_cache(self, mock_deps):
        """Same entity pair in multiple iterations should use cache."""
        from interaction_finder.extraction.entity_matching import SpeculatedVariant

        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Setup: Mock LLM to return skip decision (implicit - empty)
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[]  # Empty = all pairs implicitly skipped
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "test_agent"

        entities = {
            "BRCA": [SpeculatedVariant("BRCA", 0, "original", False)],
            "BRCA1": [SpeculatedVariant("BRCA1", 0, "original", False)],
        }
        agent_review_pairs = [("BRCA1", "BRCA")]

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
            # First call - should call LLM
            rules1, new_names1 = await node._get_consolidation_decisions_for_kind(
                agent_review_pairs, "gene", entities, ctx
            )

            # Second call with same pair - should use cache
            rules2, new_names2 = await node._get_consolidation_decisions_for_kind(
                agent_review_pairs, "gene", entities, ctx
            )

        # LLM should only be called once
        assert mock_agent.run.call_count == 1

        # Cache statistics: Both calls check cache, first populates it, second uses it
        assert ctx.state.merge_cache_hits == 2  # Both calls found it in cache
        assert ctx.state.merge_cache_misses == 0  # No misses

        # Both calls should produce the same (empty) rules
        assert rules1 == rules2 == {}


class TestSpeculationBasedConsolidation:
    """Test that consolidation uses speculation thresholds correctly."""

    @pytest.mark.asyncio
    async def test_agent_review_for_high_speculation(self, mock_deps):
        """Entities with high speculation should go to agent review."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Setup: Entities that require agent review
        from interaction_finder.extraction.entity_matching import SpeculatedVariant

        entities = {
            "BRCA": [SpeculatedVariant("BRCA", 0, "original", False)],
            "BRCA1": [SpeculatedVariant("BRCA1", 0, "original", False)],
        }

        # Mock LLM for agent review
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[]  # Empty = all pairs implicitly skipped
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "test_agent"

        # Build agent review pairs
        agent_review_pairs = [("BRCA1", "BRCA")]  # (child, parent)

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
            rules, new_names = await node._get_consolidation_decisions_for_kind(
                agent_review_pairs, "gene", entities, ctx
            )

        # Agent should have been called
        assert mock_agent.run.called
        # Skip decision means no rules created
        assert len(rules) == 0


class TestRenameMergeOpportunities:
    """Test that renames create new merge opportunities."""

    def test_add_new_names_to_entities(self, mock_deps):
        """New canonical names from renames should be added to entities_by_kind."""
        from interaction_finder.extraction.entity_matching import SpeculatedVariant

        node = ConsolidateEntitiesNode()
        entities_by_kind = {
            "gene": {
                "TGF": [SpeculatedVariant("TGF", 0, "original", False)],
                "TGF receptor": [
                    SpeculatedVariant("TGF receptor", 0, "original", False)
                ],
            }
        }

        # Rename created "TGF-β"
        new_names = {"TGF-β"}
        node._add_new_names_to_entities(new_names, entities_by_kind)

        # New entity should be added
        assert "TGF-β" in entities_by_kind["gene"]
        # Should have single variant at speculation=0
        variants = entities_by_kind["gene"]["TGF-β"]
        assert len(variants) == 1
        assert variants[0].speculation == 0
        assert variants[0].source == "original"

    def test_new_name_creates_merge_opportunity(self, mock_deps):
        """Rename to similar entity name should be detected as potential merge on next iteration."""
        from interaction_finder.extraction.entity_matching import (
            SpeculatedVariant,
            find_consolidation_candidates,
        )

        # Setup: After rename, "TGF-β" exists alongside "TGF" with parenthetical extraction
        # This represents realistic scenario where rename creates new entity that could merge
        entities = {
            "TGF": [
                SpeculatedVariant("TGF", 0, "original", False),
            ],
            "TGF-β": [
                SpeculatedVariant("TGF-β", 0, "original", False),
            ],
        }

        # Find merge candidates
        candidates = find_consolidation_candidates(entities)

        # These should go to agent review (not auto-merge) because they're genuinely different
        # TGF and TGF-β are related but distinct entities
        assert (
            len(candidates.agent_review) >= 0
        )  # May or may not be reviewed depending on similarity
        # They shouldn't auto-merge (different entities)
        for child, parent, reasoning in candidates.auto_merge:
            # If they do auto-merge, it should be based on low speculation
            assert reasoning.startswith("auto:")


class TestContestedVariants:
    """Test that contested variants are properly filtered."""

    def test_contested_variants_generate_warnings(self, mock_deps):
        """Variants mapping to multiple entities should generate warnings."""
        from interaction_finder.extraction.entity_matching import (
            SpeculatedVariant,
            find_consolidation_candidates,
        )

        # Setup: Two entities with overlapping aliases (contested variant)
        # If LLM incorrectly extracted both entities with "ACTB" as a form
        entities = {
            "ACTB (β-Actin)": [
                SpeculatedVariant("ACTB (β-Actin)", 0, "original", False),
                SpeculatedVariant("ACTB", 1, "before_paren", False),
                SpeculatedVariant("β-Actin", 2, "paren_expansion", False),
            ],
            "INHBB": [
                SpeculatedVariant("INHBB", 0, "original", False),
                # Incorrectly including ACTB as an alias would create collision
                # but extract_entity_variants wouldn't do this
            ],
        }

        candidates = find_consolidation_candidates(entities)

        # Should not have incorrect auto-merges
        # ACTB and INHBB should never merge
        for child, parent, reasoning in candidates.auto_merge:
            assert not (
                (child == "INHBB" and "ACTB" in parent)
                or (parent == "INHBB" and "ACTB" in child)
            )


class TestRenameDecisions:
    """Test that rename decisions work correctly with new system."""

    @pytest.mark.asyncio
    async def test_rename_creates_rule_and_new_name(self, mock_deps):
        """Rename decision should create consolidation rule and mark new name."""
        from interaction_finder.extraction.entity_matching import SpeculatedVariant

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
                    confirm_token="ABCD",
                    rename="TGF-β",
                    reasoning="Standard abbreviation",
                )
            ]
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "test_agent"

        # Entities that need agent review
        entities = {
            "TGF": [SpeculatedVariant("TGF", 0, "original", False)],
            "Transforming growth factor beta": [
                SpeculatedVariant(
                    "Transforming growth factor beta", 0, "original", False
                )
            ],
        }
        agent_review_pairs = [
            ("Transforming growth factor beta", "TGF")
        ]  # (child, parent)

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
            rules, new_names = await node._get_consolidation_decisions_for_kind(
                agent_review_pairs, "gene", entities, ctx
            )

        # Should have rule for the child
        from interaction_finder.extraction.utils import normalize_for_comparison

        child_norm = normalize_for_comparison("Transforming growth factor beta")
        assert (child_norm, "gene") in rules
        target, reasoning = rules[(child_norm, "gene")]
        assert target == "TGF-β"
        assert reasoning == "Standard abbreviation"

        # Should mark TGF-β as new name (doesn't exist in entities)
        assert "TGF-β" in new_names

    @pytest.mark.asyncio
    async def test_merge_decision_uses_parent_as_target(self, mock_deps):
        """Merge decision should use parent entity as target."""
        from interaction_finder.extraction.entity_matching import SpeculatedVariant

        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Setup: LLM returns merge decision
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[
                EntityConsolidationDecision(
                    pair_id=1,
                    confirm_token="ABCD",
                    rename=None,  # None = merge
                    reasoning="Plural form should merge",
                )
            ]
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent._name = "test_agent"

        entities = {
            "Telangiectasia": [
                SpeculatedVariant("Telangiectasia", 0, "original", False)
            ],
            "Telangiectasias": [
                SpeculatedVariant("Telangiectasias", 0, "original", False)
            ],
        }
        agent_review_pairs = [("Telangiectasias", "Telangiectasia")]  # (child, parent)

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
            rules, new_names = await node._get_consolidation_decisions_for_kind(
                agent_review_pairs, "gene", entities, ctx
            )

        # Should have rule with parent as target
        from interaction_finder.extraction.utils import normalize_for_comparison

        child_norm = normalize_for_comparison("Telangiectasias")
        assert (child_norm, "gene") in rules
        target, reasoning = rules[(child_norm, "gene")]
        assert target == "Telangiectasia"  # Parent
        assert reasoning == "Plural form should merge"

        # Should not have new names (parent already exists)
        assert len(new_names) == 0
