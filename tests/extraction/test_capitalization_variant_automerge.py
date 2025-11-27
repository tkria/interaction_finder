"""Test that capitalization variants are auto-merged without LLM calls.

This test file specifically targets the bug where entities differing only in
capitalization (e.g., "Pulmonary Hypertension" vs "Pulmonary hypertension")
are incorrectly sent to the LLM for merge decisions instead of being
automatically merged.
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


def test_two_capitalization_variants_automerge_without_llm(mock_deps):
    """Test that two capitalization variants are auto-merged without LLM.

    Bug symptom: "Pulmonary Hypertension" vs "Pulmonary hypertension" sent to LLM.
    Expected: Auto-merged because they normalize to the same form.
    """
    state = State(
        topic="test",
        target_entity_types=["phenotype"],
        permitted_pairs={"phenotype": {"phenotype"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")
    resource2 = ResourceId(url="http://doc2.com", id="doc2")

    # Two documents with same entity, different capitalization
    state.validated_entities_by_resource[resource1] = {
        "Pulmonary Hypertension": EntityMention(
            kind="phenotype",
            name="Pulmonary Hypertension",
            aliases=[],
            quotes=[],
            reasoning="Doc1",
        )
    }

    state.validated_entities_by_resource[resource2] = {
        "Pulmonary hypertension": EntityMention(
            kind="phenotype",
            name="Pulmonary hypertension",
            aliases=[],
            quotes=[],
            reasoning="Doc2",
        )
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    # Collect unique entities
    unique_entities = node._collect_unique_entities(ctx)

    # Both should map to the same normalized form
    assert "phenotype" in unique_entities
    assert "pulmonary hypertension" in unique_entities["phenotype"]
    canonical_set = unique_entities["phenotype"]["pulmonary hypertension"]
    assert len(canonical_set) == 2, "Should have 2 capitalization variants"
    assert canonical_set == {"Pulmonary Hypertension", "Pulmonary hypertension"}

    # Find merge candidates
    exact_rules, substring_pairs, _ = node._find_merge_candidates(unique_entities)

    # BUG REVEALED: exact_rules should contain a rule to merge the variants
    # but currently it creates a useless self-referential rule
    # After fix: should have exactly 1 rule merging child into parent
    assert len(exact_rules) == 1, (
        f"Should have exactly 1 exact match rule, got {len(exact_rules)}: {exact_rules}"
    )

    # Rules are keyed by normalized form for cross-document consistency
    rule_keys = list(exact_rules.keys())
    child_key = rule_keys[0]
    parent_canonical = exact_rules[child_key]

    # Check that the key is (normalized_form, kind)
    assert child_key[1] == "phenotype", "Second element should be kind"
    child_norm = child_key[0]
    assert child_norm == "pulmonary hypertension", (
        f"Key should be normalized form, got: {child_norm}"
    )
    # Target should be the best canonical name (most complex capitalization)
    assert parent_canonical == "Pulmonary Hypertension", (
        f"Target should be best canonical name, got: {parent_canonical}"
    )

    # BUG: Currently substring_pairs contains this pair because exact match failed!
    # After fix: Should be empty - no pairs should go to LLM
    assert (
        "phenotype" not in substring_pairs
        or len(substring_pairs.get("phenotype", [])) == 0
    ), (
        f"Capitalization variants should not be in substring pairs (LLM candidates), "
        f"got: {substring_pairs}"
    )


def test_three_capitalization_variants_all_automerge(mock_deps):
    """Test that three or more capitalization variants all merge correctly.

    With normalized rule keys, all 3 variants share the same normalized form,
    so there's only 1 rule that applies to all of them.
    """
    state = State(
        topic="test",
        target_entity_types=["phenotype"],
        permitted_pairs={"phenotype": {"phenotype"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")
    resource2 = ResourceId(url="http://doc2.com", id="doc2")
    resource3 = ResourceId(url="http://doc3.com", id="doc3")

    # Three documents with same entity, different capitalizations
    state.validated_entities_by_resource[resource1] = {
        "HEREDITARY HEMORRHAGIC TELANGIECTASIA": EntityMention(
            kind="phenotype",
            name="HEREDITARY HEMORRHAGIC TELANGIECTASIA",
            aliases=[],
            quotes=[],
            reasoning="Doc1",
        )
    }

    state.validated_entities_by_resource[resource2] = {
        "Hereditary Hemorrhagic Telangiectasia": EntityMention(
            kind="phenotype",
            name="Hereditary Hemorrhagic Telangiectasia",
            aliases=[],
            quotes=[],
            reasoning="Doc2",
        )
    }

    state.validated_entities_by_resource[resource3] = {
        "hereditary hemorrhagic telangiectasia": EntityMention(
            kind="phenotype",
            name="hereditary hemorrhagic telangiectasia",
            aliases=[],
            quotes=[],
            reasoning="Doc3",
        )
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    unique_entities = node._collect_unique_entities(ctx)
    exact_rules, substring_pairs, _ = node._find_merge_candidates(unique_entities)

    # With normalized keys, all 3 variants share the same normalized form
    # So there's only 1 rule that applies to all of them
    assert len(exact_rules) == 1, (
        f"Should have 1 exact match rule (normalized key), got {len(exact_rules)}"
    )

    # The key should be the normalized form
    rule_keys = list(exact_rules.keys())
    child_key = rule_keys[0]
    assert child_key == ("hereditary hemorrhagic telangiectasia", "phenotype"), (
        f"Key should be normalized form, got: {child_key}"
    )

    # Target should be the best canonical (title case)
    assert exact_rules[child_key] == "Hereditary Hemorrhagic Telangiectasia", (
        f"Target should be best canonical, got: {exact_rules[child_key]}"
    )

    # No variants should go to LLM
    assert "phenotype" not in substring_pairs, (
        f"Capitalization variants should not be in substring pairs: {substring_pairs}"
    )


def test_capitalization_variants_with_real_substring(mock_deps):
    """Test capitalization variants when there's also a REAL substring relationship.

    Use entities that actually have a substring relationship:
    - "BRCA" (all caps)
    - "brca" (lowercase)
    - "BRCA1" (different entity, true substring)

    Expected behavior:
    1. First two auto-merge (capitalization variants of "BRCA")
    2. After merge, only ONE "brca" canonical name exists
    3. LLM sees: "BRCA" vs "BRCA1" (one pair, not two)
    4. NOT: multiple "BRCA" capitalization variants vs "BRCA1"
    """
    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")
    resource2 = ResourceId(url="http://doc2.com", id="doc2")
    resource3 = ResourceId(url="http://doc3.com", id="doc3")

    state.validated_entities_by_resource[resource1] = {
        "BRCA": EntityMention(
            kind="gene",
            name="BRCA",
            aliases=[],
            quotes=[],
            reasoning="Doc1",
        )
    }

    state.validated_entities_by_resource[resource2] = {
        "brca": EntityMention(
            kind="gene",
            name="brca",
            aliases=[],
            quotes=[],
            reasoning="Doc2",
        )
    }

    state.validated_entities_by_resource[resource3] = {
        "BRCA1": EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=[],
            quotes=[],
            reasoning="Doc3",
        )
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    unique_entities = node._collect_unique_entities(ctx)
    exact_rules, substring_pairs, _ = node._find_merge_candidates(unique_entities)

    # Should have exactly 1 exact match rule (2 BRCA variants merge)
    assert len(exact_rules) == 1, (
        f"Should have 1 exact match rule for BRCA variants, got {len(exact_rules)}"
    )

    # Should have exactly 1 substring pair for LLM: BRCA vs BRCA1
    assert "gene" in substring_pairs, "Should have substring pairs for LLM"
    assert len(substring_pairs["gene"]) == 1, (
        f"Should have exactly 1 substring pair (BRCA vs BRCA1), got: {substring_pairs}"
    )

    # The substring pair should be between normalized forms
    brca_norm = "brca"
    brca1_norm = "brca1"
    pair = substring_pairs["gene"][0]
    assert pair == (brca_norm, brca1_norm), (
        f"Expected ({brca_norm}, {brca1_norm}), got {pair}"
    )


def test_exact_match_rules_use_canonical_names_not_normalized(mock_deps):
    """Test that exact match rules map canonical names, not normalized forms.

    Bug: Currently rules are {(norm_form, kind): norm_form}
    Expected: Rules should be {(child_canonical, kind): parent_canonical}
    """
    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")
    resource2 = ResourceId(url="http://doc2.com", id="doc2")

    state.validated_entities_by_resource[resource1] = {
        "BRCA1": EntityMention(
            kind="gene",
            name="BRCA1",
            aliases=[],
            quotes=[],
            reasoning="Doc1",
        )
    }

    state.validated_entities_by_resource[resource2] = {
        "brca1": EntityMention(
            kind="gene",
            name="brca1",
            aliases=[],
            quotes=[],
            reasoning="Doc2",
        )
    }

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    unique_entities = node._collect_unique_entities(ctx)
    exact_rules, _, _ = node._find_merge_candidates(unique_entities)

    # Get the rule
    assert len(exact_rules) == 1, f"Should have 1 rule, got {len(exact_rules)}"

    rule_key = list(exact_rules.keys())[0]
    rule_target = exact_rules[rule_key]

    # Key should be (canonical_name, kind)
    child_name, kind = rule_key
    assert kind == "gene"

    # Child name should be a real canonical name (either "BRCA1" or "brca1")
    assert child_name in {"BRCA1", "brca1"}, (
        f"Rule key should be a canonical name, got: {child_name}"
    )

    # Target should also be a canonical name, not normalized form
    assert rule_target in {"BRCA1", "brca1"}, (
        f"Rule target should be a canonical name, got: {rule_target}"
    )

    # They should be different (parent != child)
    assert child_name != rule_target, (
        f"Rule should not be self-referential: {child_name} -> {rule_target}"
    )
