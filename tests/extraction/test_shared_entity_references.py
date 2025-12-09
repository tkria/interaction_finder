"""Test that EntityMention shared references work correctly during consolidation.

This addresses Issue 5: EntityMention objects are shared between
validated_entities_by_resource and pair_assessments_by_resource.
When entities are renamed/merged, both structures should reflect the changes.

This is intentional mutation-based architecture for efficiency.
"""

import asyncio
import pytest
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    PairAssessment,
)
from interaction_finder.extraction.state import State
from interaction_finder.resources import ResourceId, ResourcePool
from interaction_finder.settings import IfetcherConfig
from tests.extraction.conftest import make_evidence


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


def test_pair_assessments_see_entity_renames_via_shared_references(mock_deps):
    """Test that PairAssessments automatically see entity renames.

    Scenario:
    - Create entities: BRCA (parent), brca (child)
    - Create PairAssessment referencing the child entity object
    - Merge child → parent (renames child entity)
    - Verify: PairAssessment.entity1 now has the updated name

    This works because PairAssessment holds references to the same EntityMention
    objects that are in validated_entities_by_resource.
    """
    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")

    # Create entities
    brca_entity = EntityMention(
        kind="gene",
        name="BRCA",
        aliases=[],
        quotes=[],
        reasoning="Parent",
    )

    brca_lower = EntityMention(
        kind="gene",
        name="brca",
        aliases=[],
        quotes=[],
        reasoning="Child",
    )

    state.validated_entities_by_resource[resource1] = {
        "BRCA": EntityRef(canonical="BRCA", mentions=[brca_entity]),
        "brca": EntityRef(canonical="brca", mentions=[brca_lower]),
    }

    # Create PairAssessment that references the child entity via EntityRef
    assessment = PairAssessment(
        resource_id=resource1,
        entity1=EntityRef(canonical=brca_lower.name, mentions=[brca_lower]),
        entity2=EntityRef(canonical=brca_entity.name, mentions=[brca_entity]),
        relationship="interacts_with",
        quotes=[],
        evidence=make_evidence(8),
        reasoning="Test",
    )

    state.pair_assessments_by_resource[resource1] = [assessment]

    # Merge child → parent - value is (target, reasoning) tuple
    merge_rules = {("brca", "gene"): ("BRCA", "test")}

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    node._apply_merge_rules_globally(merge_rules, ctx)
    node._update_pair_entity_references(merge_rules, ctx)

    # Verify: canonical updated, mentions preserved
    assert assessment.entity1.canonical == "BRCA"
    assert "brca" in assessment.entity1.aliases()


def test_pair_assessments_see_entity_merges_via_shared_references(mock_deps):
    """Test that PairAssessments see merged entity data within same document.

    Scenario:
    - Doc1: BRCA (parent) with quote1, brca (child) with quote2
    - Create assessment in doc1 referencing parent
    - Merge child → parent (combines quotes)
    - Verify: Assessment.entity1 has both quotes

    This demonstrates that when entities in the SAME document are merged,
    the parent entity accumulates quotes from children, and any assessments
    referencing the parent see the updated data.

    Note: Cross-document merges work differently (rename without merge).
    """
    state = State(
        topic="test",
        target_entity_types=["gene"],
        permitted_pairs={"gene": {"gene"}},
    )

    resource1 = ResourceId(url="http://doc1.com", id="doc1")

    # Same doc: Parent and child entity
    # Use model_construct to bypass validation (test doesn't need real ResourceQuote objects)
    brca_parent = EntityMention.model_construct(
        kind="gene",
        name="BRCA",
        aliases=[],
        quotes=["Quote from parent"],
        reasoning="Parent",
    )

    brca_child = EntityMention.model_construct(
        kind="gene",
        name="brca",
        aliases=[],
        quotes=["Quote from child"],
        reasoning="Child",
    )

    state.validated_entities_by_resource[resource1] = {
        "BRCA": EntityRef(canonical="BRCA", mentions=[brca_parent]),
        "brca": EntityRef(canonical="brca", mentions=[brca_child]),
    }

    # Create assessment referencing parent via EntityRef
    assessment = PairAssessment(
        resource_id=resource1,
        entity1=EntityRef(canonical=brca_parent.name, mentions=[brca_parent]),
        entity2=EntityRef(canonical=brca_parent.name, mentions=[brca_parent]),
        relationship="self_reference",
        quotes=[],
        evidence=make_evidence(8),
        reasoning="Test",
    )

    state.pair_assessments_by_resource[resource1] = [assessment]

    # Merge child → parent within same document - value is (target, reasoning) tuple
    merge_rules = {("brca", "gene"): ("BRCA", "test")}

    node = ConsolidateEntitiesNode()
    ctx = GraphRunContext(state=state, deps=mock_deps)

    node._apply_merge_rules_globally(merge_rules, ctx)
    node._update_pair_entity_references(merge_rules, ctx)

    # Verify: Parent entity now has quotes from both entities
    assert "Quote from parent" in brca_parent.quotes
    assert "Quote from child" in brca_child.quotes

    # Assessment sees combined mentions via EntityRef aliases/quotes aggregation
    assert "brca" in assessment.entity1.aliases()
    assert len(assessment.entity1.quotes()) == 2

    # Child should be removed from entities dict (only canonical remains)
    assert "brca" not in state.validated_entities_by_resource[resource1]
    assert "BRCA" in state.validated_entities_by_resource[resource1]


def test_entity_mutation_contract_documented(mock_deps):
    """Document the mutation contract: EntityMention objects are mutable.

    This test serves as documentation of the intended behavior:
    1. EntityMention objects are shared between dicts
    2. Mutations to entity.name, entity.quotes, entity.aliases are visible everywhere
    3. Dict keys must be updated separately when entity.name changes
    4. This is intentional for efficiency and simplicity

    If this test fails, the mutation contract has changed.
    """
    # Create an entity
    entity = EntityMention.model_construct(
        kind="gene",
        name="BRCA1",
        aliases=["Breast Cancer 1"],
        quotes=["Original quote"],
        reasoning="Test",
    )

    # Store in two dicts (simulating validated_entities and pair assessment)
    dict1 = {"BRCA1": entity}
    dict2 = {"ref": entity}

    # Mutate the entity
    entity.name = "BRCA1-modified"
    entity.aliases.append("New alias")
    entity.quotes.append("New quote")

    # Both dicts see the changes (same object reference)
    assert dict1["BRCA1"].name == "BRCA1-modified"
    assert dict2["ref"].name == "BRCA1-modified"
    assert "New alias" in dict1["BRCA1"].aliases
    assert "New quote" in dict2["ref"].quotes

    # HOWEVER: Dict keys are NOT automatically updated
    # The key is still "BRCA1" even though entity.name changed
    assert "BRCA1" in dict1
    assert "BRCA1-modified" not in dict1

    # To update the key, must explicitly move the entry
    dict1["BRCA1-modified"] = dict1.pop("BRCA1")
    assert "BRCA1-modified" in dict1
    assert "BRCA1" not in dict1
