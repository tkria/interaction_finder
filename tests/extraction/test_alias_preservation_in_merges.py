"""Test that aliases are correctly preserved and merged during entity consolidation.

This validates that when entities are merged or renamed:
1. The absorbed entity's canonical name is added to aliases
2. The absorbed entity's existing aliases are merged into the parent
3. Renamed entities preserve their old names and existing aliases
4. Cross-document merges preserve all variant names
"""

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
    import asyncio
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


def test_merge_combines_aliases_from_both_entities(mock_deps):
    """Test that merging two entities combines aliases from both."""

    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")

    # Parent has its own aliases, child has its own aliases
    state.validated_entities_by_resource[resource1] = {
        "BRCA": EntityMention(
            kind="gene",
            name="BRCA",
            aliases=["breast cancer gene", "BRCA-related"],
            quotes=[],
            reasoning="parent",
        ),
        "BRCA1": EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["BRCA1 gene", "breast cancer 1"],
            quotes=[],
            reasoning="child",
        ),
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    # Create merge rule: BRCA1 → BRCA (using canonical names)
    merge_rules = {("BRCA1", "gene"): "BRCA"}

    # Note: canonical_name_variants no longer needed with new implementation

    node._apply_merge_rules_globally(merge_rules, ctx)

    # Check that parent now has:
    # - Its own original aliases
    # - Child's canonical name
    # - Child's aliases
    entities = state.validated_entities_by_resource[resource1]
    assert "BRCA" in entities
    assert "BRCA1" not in entities

    merged_entity = entities["BRCA"]
    expected_aliases = {
        "breast cancer gene",  # Parent's original alias
        "BRCA-related",  # Parent's original alias
        "BRCA1",  # Child's canonical name
        "BRCA1 gene",  # Child's alias
        "breast cancer 1",  # Child's alias
    }
    assert set(merged_entity.aliases) == expected_aliases


def test_rename_preserves_existing_aliases(mock_deps):
    """Test that renaming an entity preserves its existing aliases."""

    state = State(
        topic="test",
        target_entity_types=["phenotype"],
        permitted_pairs={"phenotype": {"phenotype"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")

    # Entity with existing aliases that will be renamed
    state.validated_entities_by_resource[resource1] = {
        "pulmonary arterial hypertension": EntityMention(
            kind="phenotype",
            name="pulmonary arterial hypertension",
            aliases=["PAH", "pulmonary hypertension"],
            quotes=[],
            reasoning="test",
        )
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    # Create rule that renames to title case (cross-document merge scenario)
    # This simulates: another document has "Pulmonary Arterial Hypertension"
    # and the merge rule says this entity should adopt that canonical name
    merge_rules = {
        (
            "pulmonary arterial hypertension",
            "phenotype",
        ): "Pulmonary Arterial Hypertension"
    }

    node._apply_merge_rules_globally(merge_rules, ctx)

    # Entity should be renamed but keep all its aliases
    entities = state.validated_entities_by_resource[resource1]
    assert "Pulmonary Arterial Hypertension" in entities
    assert "pulmonary arterial hypertension" not in entities

    renamed_entity = entities["Pulmonary Arterial Hypertension"]
    expected_aliases = {
        "PAH",  # Original alias
        "pulmonary hypertension",  # Original alias
        "pulmonary arterial hypertension",  # Old canonical name
    }
    assert set(renamed_entity.aliases) == expected_aliases


def test_cross_document_fuzzy_merge_preserves_all_variants(mock_deps):
    """Test that fuzzy merges across documents preserve all name variants."""

    state = State(
        topic="test",
        target_entity_types=["phenotype"],
        permitted_pairs={"phenotype": {"phenotype"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")
    resource2 = ResourceId(url="http://doc2.com", id="doc2")

    # UK spelling with aliases
    state.validated_entities_by_resource[resource1] = {
        "Hereditary haemorrhagic telangiectasia": EntityMention(
            kind="phenotype",
            name="Hereditary haemorrhagic telangiectasia",
            aliases=["HHT", "Osler-Weber-Rendu syndrome"],
            quotes=[],
            reasoning="UK doc",
        )
    }

    # US spelling with different aliases
    state.validated_entities_by_resource[resource2] = {
        "Hereditary Hemorrhagic Telangiectasia": EntityMention(
            kind="phenotype",
            name="Hereditary Hemorrhagic Telangiectasia",
            aliases=["HHT", "Osler disease"],
            quotes=[],
            reasoning="US doc",
        )
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    # Collect entities (this populates canonical_name_variants)
    unique_entities = node._collect_unique_entities(ctx)

    # Find fuzzy match rules (should auto-merge UK→US spelling)
    exact_rules, _ = node._find_merge_candidates(unique_entities)
    resolved_rules = node._resolve_transitive_merges(exact_rules)

    node._apply_merge_rules_globally(resolved_rules, ctx)

    # Both documents should now use US spelling
    for entities in state.validated_entities_by_resource.values():
        assert "Hereditary Hemorrhagic Telangiectasia" in entities
        assert "Hereditary haemorrhagic telangiectasia" not in entities

    # Doc1 entity should have old UK name in aliases plus original aliases
    doc1_entity = state.validated_entities_by_resource[resource1][
        "Hereditary Hemorrhagic Telangiectasia"
    ]
    assert "Hereditary haemorrhagic telangiectasia" in doc1_entity.aliases
    assert "HHT" in doc1_entity.aliases
    assert "Osler-Weber-Rendu syndrome" in doc1_entity.aliases

    # Doc2 entity should keep its original aliases (no rename occurred)
    doc2_entity = state.validated_entities_by_resource[resource2][
        "Hereditary Hemorrhagic Telangiectasia"
    ]
    assert "HHT" in doc2_entity.aliases
    assert "Osler disease" in doc2_entity.aliases


def test_multiple_merges_accumulate_aliases(mock_deps):
    """Test that multiple sequential merges accumulate all aliases."""

    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")

    # Three entities that will merge: BRCA2 → BRCA1 → BRCA
    state.validated_entities_by_resource[resource1] = {
        "BRCA": EntityMention(
            kind="gene",
            name="BRCA",
            aliases=["breast cancer associated"],
            quotes=[],
            reasoning="parent",
        ),
        "BRCA1": EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["BRCA1 gene"],
            quotes=[],
            reasoning="child1",
        ),
        "BRCA2": EntityMention(
            kind="gene",
            name="BRCA2",
            aliases=["BRCA2 gene"],
            quotes=[],
            reasoning="child2",
        ),
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    # Create transitive merge rules: BRCA2→BRCA1, BRCA1→BRCA (using canonical names)
    merge_rules = {
        ("BRCA2", "gene"): "BRCA1",
        ("BRCA1", "gene"): "BRCA",
    }

    # Note: canonical_name_variants no longer needed with new implementation

    node._apply_merge_rules_globally(merge_rules, ctx)

    # Should have only BRCA left with all aliases
    entities = state.validated_entities_by_resource[resource1]
    assert "BRCA" in entities
    assert "BRCA1" not in entities
    assert "BRCA2" not in entities

    final_entity = entities["BRCA"]
    expected_aliases = {
        "breast cancer associated",  # BRCA's original alias
        "BRCA1",  # BRCA1's canonical name
        "BRCA1 gene",  # BRCA1's alias
        "BRCA2",  # BRCA2's canonical name
        "BRCA2 gene",  # BRCA2's alias
    }
    assert set(final_entity.aliases) == expected_aliases


def test_no_duplicate_aliases(mock_deps):
    """Test that duplicate aliases are not added when merging."""

    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")

    # Both entities have overlapping aliases
    state.validated_entities_by_resource[resource1] = {
        "BRCA": EntityMention(
            kind="gene",
            name="BRCA",
            aliases=[
                "breast cancer gene",
                "BRCA-related",
                "BRCA1",
            ],  # Already has BRCA1!
            quotes=[],
            reasoning="parent",
        ),
        "BRCA1": EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=["breast cancer gene", "BRCA1 specific"],  # Overlapping alias
            quotes=[],
            reasoning="child",
        ),
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    merge_rules = {("brca1", "gene"): "brca"}

    ctx.state.canonical_name_variants[("brca", "gene")] = {"BRCA"}
    ctx.state.canonical_name_variants[("brca1", "gene")] = {"BRCA1"}

    node._apply_merge_rules_globally(merge_rules, ctx)

    entities = state.validated_entities_by_resource[resource1]
    merged_entity = entities["BRCA"]

    # Check that "breast cancer gene" and "BRCA1" don't appear multiple times
    alias_counts = {}
    for alias in merged_entity.aliases:
        alias_counts[alias] = alias_counts.get(alias, 0) + 1

    for alias, count in alias_counts.items():
        assert count == 1, f"Alias '{alias}' appears {count} times (should be 1)"
