"""Test that identical canonical names are not sent to LLM for merge decisions.

This tests the fix for the bug where alias expansion causes the same canonical
entity name to appear under multiple normalized forms, leading to prompts like:
  Parent: 'BMPR2' (type: gene)
  Child: 'BMPR2' (type: gene)
"""

import pytest
from unittest.mock import AsyncMock, patch

from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.models import EntityMention
from interaction_finder.extraction.state import State
from interaction_finder.extraction.deps import Deps
from interaction_finder.resources import ResourcePool, ResourceId
from interaction_finder.settings import IfetcherConfig
from pydantic_graph import GraphRunContext


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return IfetcherConfig()


@pytest.fixture
def mock_deps(mock_config):
    """Create mock dependencies."""
    import asyncio
    import logging

    deps = Deps(
        config=mock_config,
        resource_pool=ResourcePool(),
        logger=logging.getLogger("test"),
        agent_semaphore=asyncio.Semaphore(1),
        progress=None,
    )
    return deps


def test_identical_canonical_names_auto_merged(mock_deps):
    """Test that pairs with identical canonical names are auto-merged without LLM call."""

    # Setup: Entity "BMPR2" with alias "BMPR2 gene" in one document
    # This creates:
    #   "bmpr2" -> {"BMPR2"}
    #   "bmpr2 gene" -> {"BMPR2"}  (from alias expansion)
    # When substring matching finds ("bmpr2", "bmpr2 gene"), both lookups return "BMPR2"

    state = State(
        topic="pulmonary arterial hypertension",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    # Create entities in state
    resource_id1 = ResourceId(url="http://doc1.com", id="doc1")
    state.validated_entities_by_resource[resource_id1] = {
        "BMPR2": EntityMention(
            kind="gene",
            name="BMPR2",
            aliases=["BMPR2 gene"],  # This alias causes the bug
            quotes=[],
            reasoning="Test entity",
        )
    }

    resource_id2 = ResourceId(url="http://doc2.com", id="doc2")
    state.validated_entities_by_resource[resource_id2] = {
        "BMPR2": EntityMention(
            kind="gene",
            name="BMPR2",
            aliases=[],
            quotes=[],
            reasoning="Test entity in second doc",
        )
    }

    # Create node and run collection
    node = ConsolidateEntitiesNode()
    unique_entities = node._collect_unique_entities(
        GraphRunContext(state=state, deps=mock_deps)
    )

    # Verify collection phase creates the problematic structure
    assert "gene" in unique_entities
    assert "bmpr2" in unique_entities["gene"]
    assert "bmpr2 gene" in unique_entities["gene"]
    assert unique_entities["gene"]["bmpr2"] == {"BMPR2"}
    assert unique_entities["gene"]["bmpr2 gene"] == {"BMPR2"}  # Same canonical name!

    # Find merge candidates
    exact_rules, substring_pairs, _ = node._find_merge_candidates(unique_entities)

    # Should find substring relationship
    assert "gene" in substring_pairs
    assert ("bmpr2", "bmpr2 gene") in substring_pairs["gene"]

    # Now test that _get_consolidation_decisions doesn't send these to LLM
    with patch(
        "interaction_finder.extraction.nodes.get_entity_consolidation_agent"
    ) as mock_agent:
        # Setup mock agent that should NOT be called
        mock_result = AsyncMock()
        mock_result.output.decisions = []
        mock_agent.return_value.run = AsyncMock(return_value=mock_result)

        # Run merge decisions
        import asyncio
        from tests.extraction.test_consolidate_entities_node import (
            build_canonical_lookup_from_unique_entities,
        )

        canonical_lookup = build_canonical_lookup_from_unique_entities(unique_entities)
        ctx = GraphRunContext(state=state, deps=mock_deps)
        merge_rules, new_names = asyncio.run(
            node._get_consolidation_decisions(
                substring_pairs, canonical_lookup, unique_entities, ctx
            )
        )

        # Agent should NOT have been called (batch was empty after filtering)
        mock_agent.return_value.run.assert_not_called()

        # Should have auto-created merge rule using canonical names
        # Since both "bmpr2" and "bmpr2 gene" normalize to "BMPR2", no merge rule is needed
        # (they're already the same entity)
        assert len(merge_rules) == 0

        # Should have cached the decision with target canonical name
        cache_key = ("bmpr2", "bmpr2 gene", "gene")
        assert cache_key in state.merge_decision_cache
        assert state.merge_decision_cache[cache_key] == "BMPR2"


def test_different_canonical_names_sent_to_llm(mock_deps):
    """Test that pairs with different canonical names ARE sent to LLM."""

    state = State(
        topic="pulmonary arterial hypertension",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    # Create entities with genuinely different canonical names
    resource_id1 = ResourceId(url="http://doc1.com", id="doc1")
    state.validated_entities_by_resource[resource_id1] = {
        "BMPR2": EntityMention(
            kind="gene",
            name="BMPR2",
            aliases=[],
            quotes=[],
            reasoning="Test entity",
        )
    }

    resource_id2 = ResourceId(url="http://doc2.com", id="doc2")
    state.validated_entities_by_resource[resource_id2] = {
        "BMPR2 gene": EntityMention(  # Different canonical name
            kind="gene",
            name="BMPR2 gene",
            aliases=[],
            quotes=[],
            reasoning="Test entity in second doc",
        )
    }

    # Create node and run collection
    node = ConsolidateEntitiesNode()
    unique_entities = node._collect_unique_entities(
        GraphRunContext(state=state, deps=mock_deps)
    )

    # Verify different canonical names
    assert unique_entities["gene"]["bmpr2"] == {"BMPR2"}
    assert unique_entities["gene"]["bmpr2 gene"] == {"BMPR2 gene"}  # Different!

    # Find merge candidates
    exact_rules, substring_pairs, _ = node._find_merge_candidates(unique_entities)

    # Should find substring relationship
    assert ("bmpr2", "bmpr2 gene") in substring_pairs["gene"]

    # Now test that _get_consolidation_decisions DOES send these to LLM
    with patch(
        "interaction_finder.extraction.nodes.get_entity_consolidation_agent"
    ) as mock_agent:
        # Setup mock agent that SHOULD be called
        mock_result = AsyncMock()
        mock_result.output.decisions = []
        mock_agent.return_value.run = AsyncMock(return_value=mock_result)

        # Run merge decisions
        import asyncio
        from tests.extraction.test_consolidate_entities_node import (
            build_canonical_lookup_from_unique_entities,
        )

        canonical_lookup = build_canonical_lookup_from_unique_entities(unique_entities)
        ctx = GraphRunContext(state=state, deps=mock_deps)
        merge_rules, new_names = asyncio.run(
            node._get_consolidation_decisions(
                substring_pairs, canonical_lookup, unique_entities, ctx
            )
        )

        # Agent SHOULD have been called
        mock_agent.return_value.run.assert_called_once()

        # Check the prompt contains the correct canonical names (new format: 'child' → 'parent')
        call_args = mock_agent.return_value.run.call_args
        prompt = call_args[0][0]
        assert "'BMPR2 gene' → 'BMPR2'" in prompt
        # Should NOT have identical names in the pair
        assert "'BMPR2' → 'BMPR2'" not in prompt


def test_mixed_identical_and_different_pairs(mock_deps):
    """Test batch with both identical and different canonical name pairs."""

    state = State(
        topic="pulmonary arterial hypertension",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    # Create three entities:
    # 1. BMPR2 with alias -> creates identical pair (bmpr2, bmpr2 gene)
    # 2. BMP9 standalone -> creates different pair (bmp9, bmpr2)
    resource_id1 = ResourceId(url="http://doc1.com", id="doc1")
    state.validated_entities_by_resource[resource_id1] = {
        "BMPR2": EntityMention(
            kind="gene",
            name="BMPR2",
            aliases=["BMPR2 gene"],  # Identical canonical case
            quotes=[],
            reasoning="Test",
        ),
        "BMP9": EntityMention(
            kind="gene",
            name="BMP9",
            aliases=[],
            quotes=[],
            reasoning="Test",
        ),
    }

    node = ConsolidateEntitiesNode()
    unique_entities = node._collect_unique_entities(
        GraphRunContext(state=state, deps=mock_deps)
    )

    # Manually create substring pairs that would be found
    substring_pairs = {
        "gene": [
            ("bmpr2", "bmpr2 gene"),  # Identical canonical names
            ("bmp9", "bmpr2"),  # Different canonical names (substring match)
        ]
    }

    with patch(
        "interaction_finder.extraction.nodes.get_entity_consolidation_agent"
    ) as mock_agent:
        mock_result = AsyncMock()
        mock_result.output.decisions = []
        mock_agent.return_value.run = AsyncMock(return_value=mock_result)

        import asyncio
        from tests.extraction.test_consolidate_entities_node import (
            build_canonical_lookup_from_unique_entities,
        )

        canonical_lookup = build_canonical_lookup_from_unique_entities(unique_entities)
        ctx = GraphRunContext(state=state, deps=mock_deps)
        merge_rules, new_names = asyncio.run(
            node._get_consolidation_decisions(
                substring_pairs, canonical_lookup, unique_entities, ctx
            )
        )

        # Agent SHOULD be called (for the BMP9/BMPR2 pair)
        mock_agent.return_value.run.assert_called_once()

        # Check prompt only contains the different-canonical-name pair
        call_args = mock_agent.return_value.run.call_args
        prompt = call_args[0][0]
        assert "BMP9" in prompt and "BMPR2" in prompt
        # Should NOT contain the identical pair
        assert prompt.count("BMPR2") == 1  # Only appears once (in the BMP9/BMPR2 pair)

        # Should have auto-merged the identical pair (no merge rule since same canonical)
        # But should have cached the decision with target canonical name
        cache_key = ("bmpr2", "bmpr2 gene", "gene")
        assert cache_key in state.merge_decision_cache
        assert state.merge_decision_cache[cache_key] == "BMPR2"
