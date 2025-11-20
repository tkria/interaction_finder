"""Test that canonical_name_variants doesn't accumulate across multiple runs.

This addresses Issue 4: canonical_name_variants should be cleared at the start
of each consolidation to prevent stale data accumulation.
"""

import asyncio
import pytest
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.models import EntityMention
from interaction_finder.extraction.state import State
from interaction_finder.resources import ResourceId, ResourcePool
from interaction_finder.settings import IfetcherConfig


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return IfetcherConfig()


@pytest.fixture
def mock_deps(mock_config):
    """Create mock dependencies."""
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


def test_canonical_name_variants_cleared_between_runs(mock_deps):
    """Test that canonical_name_variants is cleared on each consolidation run.

    Scenario:
    - First run: collect entities from doc1
    - Add new entity to doc2
    - Second run: collect entities from both docs
    - Verify: canonical_name_variants reflects only current run, no stale data
    """
    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")
    resource2 = ResourceId(url="http://doc2.com", id="doc2")

    # First run: Only doc1
    state.validated_entities_by_resource[resource1] = {
        "BRCA1": EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=[],
            quotes=[],
            reasoning="First run",
        )
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    # Run collection first time
    node._collect_unique_entities(ctx)

    # Check what's in canonical_name_variants
    assert ("brca1", "gene") in ctx.state.canonical_name_variants
    first_run_keys = set(ctx.state.canonical_name_variants.keys())
    assert len(first_run_keys) == 1

    # Add second document with different entity
    state.validated_entities_by_resource[resource2] = {
        "TP53": EntityMention(
            kind="gene",
            name="TP53",
            aliases=[],
            quotes=[],
            reasoning="Second run",
        )
    }

    # Run collection second time
    node._collect_unique_entities(ctx)

    # Check canonical_name_variants was cleared and repopulated
    second_run_keys = set(ctx.state.canonical_name_variants.keys())

    # Should have both entities now
    assert ("brca1", "gene") in ctx.state.canonical_name_variants
    assert ("tp53", "gene") in ctx.state.canonical_name_variants
    assert len(second_run_keys) == 2

    # Most importantly: no stale duplicates
    # Each key should have exactly one canonical name
    for key, variants in ctx.state.canonical_name_variants.items():
        assert isinstance(variants, set)
        assert len(variants) >= 1, f"Key {key} should have at least 1 variant"


def test_canonical_name_variants_no_stale_data_after_entity_removal(mock_deps):
    """Test that removing entities clears their canonical_name_variants entries.

    Scenario:
    - First run: BRCA1 and TP53
    - Remove TP53
    - Second run: Only BRCA1
    - Verify: TP53's entries are gone from canonical_name_variants
    """
    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")

    # First run: Two entities
    state.validated_entities_by_resource[resource1] = {
        "BRCA1": EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=[],
            quotes=[],
            reasoning="First",
        ),
        "TP53": EntityMention(
            kind="gene",
            name="TP53",
            aliases=[],
            quotes=[],
            reasoning="First",
        ),
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    node._collect_unique_entities(ctx)

    # Both should be present
    assert ("brca1", "gene") in ctx.state.canonical_name_variants
    assert ("tp53", "gene") in ctx.state.canonical_name_variants

    # Remove TP53
    del state.validated_entities_by_resource[resource1]["TP53"]

    # Second run
    node._collect_unique_entities(ctx)

    # Only BRCA1 should remain
    assert ("brca1", "gene") in ctx.state.canonical_name_variants
    assert ("tp53", "gene") not in ctx.state.canonical_name_variants, (
        "TP53 should be cleared after removal"
    )

    # Should have exactly 1 key
    assert len(ctx.state.canonical_name_variants) == 1
