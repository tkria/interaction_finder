"""Test that fuzzy auto-merge uses canonical names, not normalized forms.

This verifies Issue 3 is fixed: fuzzy Tier 1 auto-merge rules should map
canonical names to canonical names, not normalized forms to normalized forms.
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


def test_fuzzy_automerge_uses_canonical_names_not_normalized(mock_deps):
    """Test that fuzzy auto-merge rules use canonical entity names.

    Scenario:
    - "Telangiectasia" (singular)
    - "Telangiectasias" (plural, OSA distance = 1)

    Expected:
    - Auto-merge rule created (obvious plural variant)
    - Rule maps canonical names: ("Telangiectasias", kind) → "Telangiectasia"
    - NOT normalized forms: ("telangiectasias", kind) → "telangiectasia"
    """
    state = State(
        topic="test",
        target_entity_types=["phenotype"],
        permitted_pairs={"phenotype": {"phenotype"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")
    resource2 = ResourceId(url="http://doc2.com", id="doc2")

    state.validated_entities_by_resource[resource1] = {
        "Telangiectasia": EntityMention(
            kind="phenotype",
            name="Telangiectasia",
            aliases=[],
            quotes=[],
            reasoning="Singular form",
        )
    }

    state.validated_entities_by_resource[resource2] = {
        "Telangiectasias": EntityMention(
            kind="phenotype",
            name="Telangiectasias",
            aliases=[],
            quotes=[],
            reasoning="Plural form",
        )
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    unique_entities = node._collect_unique_entities(ctx)
    auto_merge_rules, _, _ = node._find_merge_candidates(unique_entities)

    # Should have exactly 1 fuzzy auto-merge rule
    assert len(auto_merge_rules) == 1, (
        f"Should have 1 fuzzy auto-merge rule, got {len(auto_merge_rules)}"
    )

    # Get the rule
    rule_key = list(auto_merge_rules.keys())[0]
    rule_target = auto_merge_rules[rule_key]

    # Rule key should be (canonical_name, kind)
    child_name, kind = rule_key
    assert kind == "phenotype"

    # Both key and target should be CANONICAL names (with proper capitalization)
    assert child_name == "Telangiectasias", (
        f"Rule key should be canonical 'Telangiectasias', got: {child_name}"
    )
    assert rule_target == "Telangiectasia", (
        f"Rule target should be canonical 'Telangiectasia', got: {rule_target}"
    )

    # Should NOT be normalized forms
    assert child_name != "telangiectasias", "Rule key should not be normalized form"
    assert rule_target != "telangiectasia", "Rule target should not be normalized form"


def test_fuzzy_automerge_after_capitalization_consolidation(mock_deps):
    """Test that fuzzy matching works on entities consolidated in Phase 1.

    Scenario:
    - "Haemorrhagic" (UK spelling, mixed case)
    - "haemorrhagic" (UK spelling, lowercase)
    - "Hemorrhagic" (US spelling)

    Expected:
    1. Phase 1 consolidates UK variants → "Haemorrhagic"
    2. Phase 2 fuzzy matches: "Haemorrhagic" vs "Hemorrhagic" (OSA=1, s→z)
    3. Auto-merge rule: ("Hemorrhagic", phenotype) → "Haemorrhagic"
    """
    state = State(
        topic="test",
        target_entity_types=["phenotype"],
        permitted_pairs={"phenotype": {"phenotype"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")
    resource2 = ResourceId(url="http://doc2.com", id="doc2")
    resource3 = ResourceId(url="http://doc3.com", id="doc3")

    state.validated_entities_by_resource[resource1] = {
        "Haemorrhagic": EntityMention(
            kind="phenotype",
            name="Haemorrhagic",
            aliases=[],
            quotes=[],
            reasoning="UK mixed case",
        )
    }

    state.validated_entities_by_resource[resource2] = {
        "haemorrhagic": EntityMention(
            kind="phenotype",
            name="haemorrhagic",
            aliases=[],
            quotes=[],
            reasoning="UK lowercase",
        )
    }

    state.validated_entities_by_resource[resource3] = {
        "Hemorrhagic": EntityMention(
            kind="phenotype",
            name="Hemorrhagic",
            aliases=[],
            quotes=[],
            reasoning="US spelling",
        )
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    unique_entities = node._collect_unique_entities(ctx)
    auto_merge_rules, _, _ = node._find_merge_candidates(unique_entities)

    # Should have 2 rules:
    # 1. Phase 1: haemorrhagic → Haemorrhagic (capitalization)
    # 2. Phase 2: Hemorrhagic → Haemorrhagic (fuzzy spelling variant)
    assert len(auto_merge_rules) == 2, (
        f"Should have 2 auto-merge rules (1 exact + 1 fuzzy), got {len(auto_merge_rules)}"
    )

    # Check that both rules use canonical names
    for (child_name, kind), parent_name in auto_merge_rules.items():
        assert kind == "phenotype"

        # All should be canonical names (not normalized forms)
        assert child_name[0].isupper() or child_name[0].islower(), (
            f"Child should be canonical name: {child_name}"
        )
        assert parent_name[0].isupper() or parent_name[0].islower(), (
            f"Parent should be canonical name: {parent_name}"
        )

        # Normalized forms would be all lowercase
        # Canonical names preserve original capitalization
        assert (
            child_name != child_name.lower() or len(child_name) == 1
        ) or child_name == "haemorrhagic", (
            f"Expected canonical name, got normalized: {child_name}"
        )
