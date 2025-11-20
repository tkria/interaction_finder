"""Test that all instances of merged entities use the same canonical name.

When entities with capitalization or spelling variants are merged, ALL instances
across ALL documents should be renamed to use the same global canonical name.
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


def test_capitalization_variants_use_same_canonical_name(mock_deps):
    """Test that capitalization variants all use the same canonical name after merge."""

    state = State(
        topic="test",
        target_entity_types=["phenotype"],
        permitted_pairs={"phenotype": {"phenotype"}},
    )

    # Create three documents with different capitalizations
    resource1 = ResourceId(url="http://doc1.com", id="doc1")
    resource2 = ResourceId(url="http://doc2.com", id="doc2")
    resource3 = ResourceId(url="http://doc3.com", id="doc3")

    state.validated_entities_by_resource[resource1] = {
        "Pulmonary arterial hypertension": EntityMention(
            kind="phenotype",
            name="Pulmonary arterial hypertension",
            aliases=[],
            quotes=[],
            reasoning="Test1",
        )
    }

    state.validated_entities_by_resource[resource2] = {
        "Pulmonary Arterial Hypertension": EntityMention(
            kind="phenotype",
            name="Pulmonary Arterial Hypertension",
            aliases=[],
            quotes=[],
            reasoning="Test2",
        )
    }

    state.validated_entities_by_resource[resource3] = {
        "pulmonary arterial hypertension": EntityMention(
            kind="phenotype",
            name="pulmonary arterial hypertension",
            aliases=[],
            quotes=[],
            reasoning="Test3",
        )
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    # Collect and apply merges
    unique_entities = node._collect_unique_entities(ctx)
    exact_rules, _, _ = node._find_merge_candidates(unique_entities)
    resolved_rules = node._resolve_transitive_merges(exact_rules)
    node._apply_merge_rules_globally(resolved_rules, ctx)

    # Check that ALL documents now have the SAME canonical name
    canonical_names = set()
    for entities in state.validated_entities_by_resource.values():
        assert len(entities) == 1, (
            "Should have exactly one entity per document after merge"
        )
        canonical_name = list(entities.keys())[0]
        canonical_names.add(canonical_name)

    # All should use the same canonical name (most complex capitalization)
    assert len(canonical_names) == 1, (
        f"All documents should use the same canonical name, got: {canonical_names}"
    )

    # Should pick the most complex capitalization (mixed case)
    global_canonical = canonical_names.pop()
    assert global_canonical == "Pulmonary Arterial Hypertension", (
        f"Expected 'Pulmonary Arterial Hypertension', got '{global_canonical}'"
    )

    # Check that renamed entities have their old names in aliases
    doc1_entity = list(state.validated_entities_by_resource[resource1].values())[0]
    doc2_entity = list(state.validated_entities_by_resource[resource2].values())[0]
    doc3_entity = list(state.validated_entities_by_resource[resource3].values())[0]

    # Doc1 was "Pulmonary arterial hypertension", renamed to "Pulmonary Arterial Hypertension"
    assert "Pulmonary arterial hypertension" in doc1_entity.aliases
    # Doc2 was already "Pulmonary Arterial Hypertension", no rename needed
    assert len(doc2_entity.aliases) == 0
    # Doc3 was "pulmonary arterial hypertension", renamed to "Pulmonary Arterial Hypertension"
    assert "pulmonary arterial hypertension" in doc3_entity.aliases


def test_spelling_variants_use_same_canonical_name(mock_deps):
    """Test that UK/US spelling variants all use the same canonical name after merge."""

    state = State(
        topic="test",
        target_entity_types=["phenotype"],
        permitted_pairs={"phenotype": {"phenotype"}},
    )

    # Create two documents with UK/US spelling variants
    resource1 = ResourceId(url="http://doc1.com", id="doc1")
    resource2 = ResourceId(url="http://doc2.com", id="doc2")

    state.validated_entities_by_resource[resource1] = {
        "Hereditary haemorrhagic telangiectasia": EntityMention(
            kind="phenotype",
            name="Hereditary haemorrhagic telangiectasia",
            aliases=[],
            quotes=[],
            reasoning="Test1",
        )
    }

    state.validated_entities_by_resource[resource2] = {
        "Hereditary Hemorrhagic Telangiectasia": EntityMention(
            kind="phenotype",
            name="Hereditary Hemorrhagic Telangiectasia",
            aliases=[],
            quotes=[],
            reasoning="Test2",
        )
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    # Collect and apply merges (fuzzy matching should catch these)
    unique_entities = node._collect_unique_entities(ctx)
    exact_rules, substring_pairs, _ = node._find_merge_candidates(unique_entities)
    # Fuzzy auto-merge rules are in exact_rules (Tier 2 fuzzy matching)
    resolved_rules = node._resolve_transitive_merges(exact_rules)
    node._apply_merge_rules_globally(resolved_rules, ctx)

    # Check that ALL documents now have the SAME canonical name
    canonical_names = set()
    for entities in state.validated_entities_by_resource.values():
        assert len(entities) == 1, (
            "Should have exactly one entity per document after merge"
        )
        canonical_name = list(entities.keys())[0]
        canonical_names.add(canonical_name)

    # All should use the same canonical name
    assert len(canonical_names) == 1, (
        f"All documents should use the same canonical name, got: {canonical_names}"
    )


def test_mixed_variants_across_many_documents(mock_deps):
    """Test global canonical naming with many documents and mixed variants."""

    state = State(
        topic="test",
        target_entity_types=["phenotype"],
        permitted_pairs={"phenotype": {"phenotype"}},
    )

    # Create 10 documents with various capitalizations
    variants = [
        "pulmonary arterial hypertension",
        "Pulmonary arterial hypertension",
        "Pulmonary Arterial Hypertension",
        "PULMONARY ARTERIAL HYPERTENSION",
        "pulmonary arterial hypertension",  # duplicate
        "Pulmonary Arterial Hypertension",  # duplicate
        "Pulmonary arterial Hypertension",  # different pattern
        "pulmonary Arterial hypertension",  # different pattern
        "Pulmonary arterial hypertension",  # duplicate
        "Pulmonary Arterial Hypertension",  # duplicate
    ]

    for i, variant in enumerate(variants):
        resource = ResourceId(url=f"http://doc{i}.com", id=f"doc{i}")
        state.validated_entities_by_resource[resource] = {
            variant: EntityMention(
                kind="phenotype",
                name=variant,
                aliases=[],
                quotes=[],
                reasoning=f"Test{i}",
            )
        }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    # Collect and apply merges
    unique_entities = node._collect_unique_entities(ctx)
    exact_rules, _, _ = node._find_merge_candidates(unique_entities)
    resolved_rules = node._resolve_transitive_merges(exact_rules)
    node._apply_merge_rules_globally(resolved_rules, ctx)

    # Check that ALL 10 documents now have the SAME canonical name
    canonical_names = set()
    for entities in state.validated_entities_by_resource.values():
        assert len(entities) == 1, (
            "Should have exactly one entity per document after merge"
        )
        canonical_name = list(entities.keys())[0]
        canonical_names.add(canonical_name)

    # All should use the same canonical name
    assert len(canonical_names) == 1, (
        f"All 10 documents should use the same canonical name, got: {canonical_names}"
    )

    # Should pick the most complex capitalization (mixed case)
    global_canonical = canonical_names.pop()
    # The heuristic picks based on lowercase * uppercase counts
    # "Pulmonary Arterial Hypertension" has 3*3=9, which is highest
    assert global_canonical == "Pulmonary Arterial Hypertension"
