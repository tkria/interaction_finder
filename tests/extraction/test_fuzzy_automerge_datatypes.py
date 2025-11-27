"""Test that fuzzy auto-merge uses normalized keys and canonical targets.

Rules are keyed by normalized form for cross-document consistency.
Targets are canonical names to preserve proper casing in output.
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


def test_fuzzy_automerge_uses_normalized_keys_and_canonical_targets(mock_deps):
    """Test that fuzzy auto-merge uses normalized keys and canonical targets.

    Scenario:
    - "Telangiectasia" (singular)
    - "Telangiectasias" (plural, OSA distance = 1)

    Expected:
    - Auto-merge rule created (obvious plural variant)
    - Rule key is normalized: ("telangiectasias", kind)
    - Rule target is canonical: "Telangiectasia"
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

    # Rule key should be (normalized_form, kind) for cross-document consistency
    child_norm, kind = rule_key
    assert kind == "phenotype"
    assert child_norm == "telangiectasias", (
        f"Rule key should be normalized 'telangiectasias', got: {child_norm}"
    )

    # Target should be CANONICAL name (with proper capitalization)
    assert rule_target == "Telangiectasia", (
        f"Rule target should be canonical 'Telangiectasia', got: {rule_target}"
    )


def test_fuzzy_automerge_after_capitalization_consolidation(mock_deps):
    """Test that Phase 2 preserves Phase 1 winners as parents.

    Scenario:
    - "Haemorrhagic" (UK spelling, mixed case)
    - "haemorrhagic" (UK spelling, lowercase)
    - "Hemorrhagic" (US spelling)

    Expected:
    1. Phase 1 consolidates UK variants: ("haemorrhagic", phenotype) → "Haemorrhagic"
    2. Phase 2 fuzzy matches: "haemorrhagic" is established (had Phase 1 rule),
       so US spelling merges INTO UK: ("hemorrhagic", phenotype) → "Haemorrhagic"

    This ensures that corpus-established spellings (with multiple cap variants)
    take precedence over single-variant spellings in fuzzy auto-merge.
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

    # With normalized keys:
    # - "Haemorrhagic" and "haemorrhagic" both normalize to "haemorrhagic" (12 chars)
    # - "Hemorrhagic" normalizes to "hemorrhagic" (11 chars)
    #
    # Phase 1: UK variants consolidated to "Haemorrhagic"
    #   Rule: ("haemorrhagic", phenotype) → "Haemorrhagic"
    #
    # Phase 2: Fuzzy match haemorrhagic (12) vs hemorrhagic (11)
    #   "haemorrhagic" had Phase 1 rule (established), so it becomes parent
    #   Rule: ("hemorrhagic", phenotype) → "Haemorrhagic"
    #
    # Final: 2 rules, both mapping to UK canonical "Haemorrhagic"

    assert len(auto_merge_rules) == 2, (
        f"Should have 2 rules (Phase 1 + Phase 2), got {len(auto_merge_rules)}"
    )

    # Phase 1 rule: UK lowercase → UK mixed case
    assert ("haemorrhagic", "phenotype") in auto_merge_rules
    assert auto_merge_rules[("haemorrhagic", "phenotype")] == "Haemorrhagic"

    # Phase 2 rule: US spelling → UK canonical (established form wins)
    assert ("hemorrhagic", "phenotype") in auto_merge_rules
    assert auto_merge_rules[("hemorrhagic", "phenotype")] == "Haemorrhagic"
