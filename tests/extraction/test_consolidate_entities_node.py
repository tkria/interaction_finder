"""Tests for ConsolidateEntitiesNode global cross-document merging.

Tests the new global merging architecture that:
1. Collects all unique normalized entities across documents
2. Finds substring relationships globally
3. Queries LLM once per unique normalized pair (with caching)
4. Applies merge rules consistently across all documents
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.usage import RunUsage
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.models import (
    EntityMention,
    EntityConsolidationDecision,
    EntityConsolidationDecisions,
)
from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import build_permitted_pairs
from interaction_finder.resources import ResourceId, ResourcePool


def build_canonical_lookup_from_unique_entities(unique_entities):
    """Helper to build canonical_lookup from unique_entities dict."""
    canonical_lookup = {}
    for kind, norm_dict in unique_entities.items():
        for norm, canonicals in norm_dict.items():
            canonical_lookup[(norm, kind)] = next(iter(canonicals))
    return canonical_lookup


def create_mock_agent_with_override(run_return_value):
    """Create a mock agent with working rename_agent() support."""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=run_return_value)
    mock_agent._name = "mock_agent"
    return mock_agent


@pytest.fixture
def mock_deps():
    """Create mock dependencies."""
    deps = MagicMock()
    deps.config = MagicMock()
    deps.config.tools.extraction.merge_batch_size = 50
    deps.config.tools.extraction.max_rename_iterations = 3
    deps.logger = MagicMock()
    deps.resource_pool = ResourcePool()
    return deps


class TestCollectUniqueEntities:
    """Test _collect_unique_entities method."""

    def test_collects_from_single_document(self, mock_deps):
        """Should collect entities from a single document."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Add entities to state
        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="test",
                ),
                "TP53": EntityMention(
                    kind="gene",
                    name="TP53",
                    aliases=["TP53"],
                    quotes=[],
                    reasoning="test",
                ),
            }
        }

        unique = node._collect_unique_entities(ctx)

        assert "gene" in unique
        assert "brca1" in unique["gene"]
        assert "tp53" in unique["gene"]
        assert "BRCA1" in unique["gene"]["brca1"]
        assert "TP53" in unique["gene"]["tp53"]

    def test_collects_from_multiple_documents(self, mock_deps):
        """Should collect entities across multiple documents."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="test",
                )
            },
            resource2: {
                "brca1": EntityMention(
                    kind="gene",
                    name="brca1",
                    aliases=["brca1"],
                    quotes=[],
                    reasoning="test",
                )
            },
        }

        unique = node._collect_unique_entities(ctx)

        # Both variants should be under same normalized key
        assert "gene" in unique
        assert "brca1" in unique["gene"]
        assert len(unique["gene"]["brca1"]) == 2
        assert "BRCA1" in unique["gene"]["brca1"]
        assert "brca1" in unique["gene"]["brca1"]

    def test_tracks_canonical_variants_globally(self, mock_deps):
        """Should track all canonical name variants in state."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="test",
                )
            },
            resource2: {
                "brca1": EntityMention(
                    kind="gene",
                    name="brca1",
                    aliases=["brca1"],
                    quotes=[],
                    reasoning="test",
                ),
                "Brca1": EntityMention(
                    kind="gene",
                    name="Brca1",
                    aliases=["Brca1"],
                    quotes=[],
                    reasoning="test",
                ),
            },
        }

        node._collect_unique_entities(ctx)

        # Check canonical_name_variants tracking
        norm_key = ("brca1", "gene")
        assert norm_key in ctx.state.canonical_name_variants
        assert ctx.state.canonical_name_variants[norm_key] == {
            "BRCA1",
            "brca1",
            "Brca1",
        }

    def test_handles_multiple_entity_kinds(self, mock_deps):
        """Should separate entities by kind."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="test",
                ),
                "breast cancer": EntityMention(
                    kind="disease",
                    name="breast cancer",
                    aliases=["breast cancer"],
                    quotes=[],
                    reasoning="test",
                ),
            }
        }

        unique = node._collect_unique_entities(ctx)

        assert "gene" in unique
        assert "disease" in unique
        assert "brca1" in unique["gene"]
        assert "breast cancer" in unique["disease"]


class TestFindGlobalSubstringPairs:
    """Test _find_merge_candidates method."""

    def test_finds_simple_substring(self):
        """Should find substring relationships."""
        node = ConsolidateEntitiesNode()

        unique_entities = {"gene": {"brca": {"BRCA"}, "brca1": {"BRCA1"}}}

        exact_matches, pairs, _ = node._find_merge_candidates(unique_entities)

        assert "gene" in pairs
        assert len(pairs["gene"]) == 1
        assert ("brca", "brca1") in pairs["gene"]

    def test_finds_multiple_substrings(self):
        """Should find multiple substring relationships."""
        node = ConsolidateEntitiesNode()

        unique_entities = {
            "gene": {
                "brca": {"BRCA"},
                "brca1": {"BRCA1"},
                "tp": {"TP"},
                "tp53": {"TP53"},
            }
        }

        exact_matches, pairs, _ = node._find_merge_candidates(unique_entities)

        assert "gene" in pairs
        assert len(pairs["gene"]) == 2
        pair_set = set(pairs["gene"])
        assert ("brca", "brca1") in pair_set
        assert ("tp", "tp53") in pair_set

    def test_no_substrings_found(self):
        """Should return empty dict when no substrings exist."""
        node = ConsolidateEntitiesNode()

        unique_entities = {"gene": {"brca1": {"BRCA1"}, "tp53": {"TP53"}}}

        exact_matches, pairs, _ = node._find_merge_candidates(unique_entities)

        assert pairs == {}

    def test_handles_exact_normalized_match(self):
        """Should create exact match rules for identical normalized names."""
        node = ConsolidateEntitiesNode()

        unique_entities = {"gene": {"brca1": {"BRCA1", "brca1", "Brca1"}}}

        exact_matches, pairs, _ = node._find_merge_candidates(unique_entities)

        # Should have exact match rules but no substring pairs
        assert pairs == {}
        assert len(exact_matches) > 0  # Multiple variants should create merge rules

    def test_bidirectional_substring_detection(self):
        """Should detect substrings in both directions."""
        node = ConsolidateEntitiesNode()

        unique_entities = {
            "gene": {
                "short": {"short"},
                "short name": {"short name"},
                "name": {"name"},
            }
        }

        exact_matches, pairs, _ = node._find_merge_candidates(unique_entities)

        assert "gene" in pairs
        pair_set = set(pairs["gene"])
        # "short" is in "short name"
        assert ("short", "short name") in pair_set
        # "name" is in "short name"
        assert ("name", "short name") in pair_set


class TestGetGlobalMergeDecisions:
    """Test _get_consolidation_decisions method."""

    @pytest.mark.asyncio
    async def test_cache_miss_queries_llm(self, mock_deps):
        """First encounter should query LLM."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        substring_pairs = {"gene": [("brca", "brca1")]}
        unique_entities = {"gene": {"brca": {"BRCA"}, "brca1": {"BRCA1"}}}
        canonical_lookup = build_canonical_lookup_from_unique_entities(unique_entities)

        # Mock LLM response (pair_id=1 corresponds to BRCA/BRCA1 pair)
        # Token "test" won't match the real token, but ID-based lookup will be used
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[
                EntityConsolidationDecision(
                    pair_id=1,
                    pair_token="test",
                    action="merge",
                    reasoning="BRCA1 is specific gene, BRCA is shorthand",
                )
            ]
        )
        mock_agent = create_mock_agent_with_override(mock_result)

        import interaction_finder.extraction.nodes as nodes_module

        original_getter = nodes_module.get_entity_consolidation_agent
        nodes_module.get_entity_consolidation_agent = lambda config: mock_agent

        try:
            merge_rules, new_names = await node._get_consolidation_decisions(
                substring_pairs, canonical_lookup, unique_entities, ctx
            )

            # Should have called LLM
            assert mock_agent.run.called
            assert len(merge_rules) == 1
            # Merge rules keyed by normalized form, target is canonical name
            assert ("brca1", "gene") in merge_rules
            assert merge_rules[("brca1", "gene")] == "BRCA"

            # Cache should be populated with target canonical name
            assert ctx.state.merge_decision_cache[("brca", "brca1", "gene")] == "BRCA"
            assert ctx.state.merge_cache_misses == 1
            assert ctx.state.merge_cache_hits == 0

        finally:
            nodes_module.get_entity_consolidation_agent = original_getter

    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm(self, mock_deps):
        """Second encounter should use cache."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Pre-populate cache with target canonical name
        ctx.state.merge_decision_cache[("brca", "brca1", "gene")] = "BRCA"

        substring_pairs = {"gene": [("brca", "brca1")]}
        unique_entities = {"gene": {"brca": {"BRCA"}, "brca1": {"BRCA1"}}}
        canonical_lookup = build_canonical_lookup_from_unique_entities(unique_entities)

        # Mock LLM (should not be called)
        mock_agent = create_mock_agent_with_override(None)

        import interaction_finder.extraction.nodes as nodes_module

        original_getter = nodes_module.get_entity_consolidation_agent
        nodes_module.get_entity_consolidation_agent = lambda config: mock_agent

        try:
            merge_rules, new_names = await node._get_consolidation_decisions(
                substring_pairs, canonical_lookup, unique_entities, ctx
            )

            # Should NOT have called LLM
            assert not mock_agent.run.called
            # Should return merge rule from cache (keyed by normalized form)
            assert len(merge_rules) == 1
            assert ("brca1", "gene") in merge_rules
            assert merge_rules[("brca1", "gene")] == "BRCA"
            # Metrics
            assert ctx.state.merge_cache_hits == 1
            assert ctx.state.merge_cache_misses == 0

        finally:
            nodes_module.get_entity_consolidation_agent = original_getter

    @pytest.mark.asyncio
    async def test_respects_reject_decisions(self, mock_deps):
        """Should not create merge rule when cache says no merge."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Cache says NOT to merge
        ctx.state.merge_decision_cache[("tp", "tp53", "gene")] = False

        substring_pairs = {"gene": [("tp", "tp53")]}
        unique_entities = {"gene": {"tp": {"TP"}, "tp53": {"TP53"}}}
        canonical_lookup = build_canonical_lookup_from_unique_entities(unique_entities)

        merge_rules, new_names = await node._get_consolidation_decisions(
            substring_pairs, canonical_lookup, unique_entities, ctx
        )

        # No merge rules created
        assert len(merge_rules) == 0
        assert ctx.state.merge_cache_hits == 1

    @pytest.mark.asyncio
    async def test_mixed_cached_and_uncached(self, mock_deps):
        """Should handle mix of cached and uncached pairs efficiently."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Pre-populate cache for one pair with target canonical name
        ctx.state.merge_decision_cache[("brca", "brca1", "gene")] = "BRCA"

        substring_pairs = {"gene": [("brca", "brca1"), ("tp", "tp53")]}
        unique_entities = {
            "gene": {
                "brca": {"BRCA"},
                "brca1": {"BRCA1"},
                "tp": {"TP"},
                "tp53": {"TP53"},
            }
        }
        canonical_lookup = build_canonical_lookup_from_unique_entities(unique_entities)

        # Mock LLM for uncached pair only (pair_id=1 is tp/tp53)
        # Token "test" won't match the real token, but ID-based lookup will be used
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[
                EntityConsolidationDecision(
                    pair_id=1,
                    pair_token="test",
                    action="skip",
                    reasoning="TP and TP53 are different proteins with distinct functions",
                )
            ]
        )
        mock_agent = create_mock_agent_with_override(mock_result)

        import interaction_finder.extraction.nodes as nodes_module

        original_getter = nodes_module.get_entity_consolidation_agent
        nodes_module.get_entity_consolidation_agent = lambda config: mock_agent

        try:
            merge_rules, new_names = await node._get_consolidation_decisions(
                substring_pairs, canonical_lookup, unique_entities, ctx
            )

            # Should call LLM only for uncached pair
            assert mock_agent.run.called
            # Should have merge rule only for cached pair (keyed by normalized form)
            assert len(merge_rules) == 1
            assert ("brca1", "gene") in merge_rules
            assert merge_rules[("brca1", "gene")] == "BRCA"
            # Metrics
            assert ctx.state.merge_cache_hits == 1
            assert ctx.state.merge_cache_misses == 1

        finally:
            nodes_module.get_entity_consolidation_agent = original_getter

    @pytest.mark.asyncio
    async def test_handles_llm_errors_gracefully(self, mock_deps):
        """Should continue if LLM call fails."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        substring_pairs = {"gene": [("brca", "brca1")]}
        unique_entities = {"gene": {"brca": {"BRCA"}, "brca1": {"BRCA1"}}}
        canonical_lookup = build_canonical_lookup_from_unique_entities(unique_entities)

        # Mock LLM to raise error
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=TimeoutError("LLM timeout"))
        mock_agent._name = "mock_agent"

        import interaction_finder.extraction.nodes as nodes_module

        original_getter = nodes_module.get_entity_consolidation_agent
        nodes_module.get_entity_consolidation_agent = lambda config: mock_agent

        try:
            merge_rules, new_names = await node._get_consolidation_decisions(
                substring_pairs, canonical_lookup, unique_entities, ctx
            )

            # Should return empty (no merge rules)
            assert len(merge_rules) == 0

            # Logger should have been called
            assert mock_deps.logger.error.called

        finally:
            nodes_module.get_entity_consolidation_agent = original_getter


class TestResolveTransitiveMerges:
    """Test _resolve_transitive_merges method."""

    def test_resolves_simple_chain(self):
        """Should resolve A→B→C to A→C, B→C."""
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("a", "gene"): "b",
            ("b", "gene"): "c",
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # Both should point to final parent 'c'
        assert resolved[("a", "gene")] == "c"
        assert resolved[("b", "gene")] == "c"

    def test_resolves_long_chain(self):
        """Should resolve A→B→C→D to A→D, B→D, C→D."""
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("a", "gene"): "b",
            ("b", "gene"): "c",
            ("c", "gene"): "d",
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # All should point to final parent 'd'
        assert resolved[("a", "gene")] == "d"
        assert resolved[("b", "gene")] == "d"
        assert resolved[("c", "gene")] == "d"

    def test_handles_multiple_independent_chains(self):
        """Should handle multiple independent merge chains."""
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("a", "gene"): "b",
            ("b", "gene"): "c",
            ("x", "gene"): "y",
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # First chain
        assert resolved[("a", "gene")] == "c"
        assert resolved[("b", "gene")] == "c"

        # Second chain (no transitivity)
        assert resolved[("x", "gene")] == "y"

    def test_handles_no_chains(self):
        """Should pass through rules with no chains."""
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("a", "gene"): "b",
            ("c", "gene"): "d",
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # No changes - no chains to resolve
        assert resolved[("a", "gene")] == "b"
        assert resolved[("c", "gene")] == "d"

    def test_handles_empty_rules(self):
        """Should handle empty merge rules."""
        node = ConsolidateEntitiesNode()

        merge_rules = {}
        resolved = node._resolve_transitive_merges(merge_rules)

        assert resolved == {}

    def test_handles_circular_reference(self):
        """Should detect and stop at circular references."""
        node = ConsolidateEntitiesNode()

        # Create artificial cycle: a→b, b→c, c→a
        merge_rules = {
            ("a", "gene"): "b",
            ("b", "gene"): "c",
            ("c", "gene"): "a",
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # Should detect cycle and stop - exact behavior depends on traversal order
        # Just verify it doesn't crash and produces some result
        assert len(resolved) == 3

    def test_respects_entity_kinds(self):
        """Should handle different entity kinds independently."""
        node = ConsolidateEntitiesNode()

        merge_rules = {
            ("a", "gene"): "b",
            ("b", "gene"): "c",
            ("a", "disease"): "b",  # Different kind, no chain
        }

        resolved = node._resolve_transitive_merges(merge_rules)

        # Gene chain resolved
        assert resolved[("a", "gene")] == "c"
        assert resolved[("b", "gene")] == "c"

        # Disease - no chain
        assert resolved[("a", "disease")] == "b"


class TestApplyMergeRulesGlobally:
    """Test _apply_merge_rules_globally method."""

    def test_applies_merge_to_single_document(self, mock_deps):
        """Should merge entities within a document."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BRCA": EntityMention(
                    kind="gene",
                    name="BRCA",
                    aliases=["BRCA"],
                    quotes=[],
                    reasoning="parent",
                ),
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="child",
                ),
            }
        }

        # Rules keyed by normalized form, target is canonical name
        merge_rules = {("brca1", "gene"): "BRCA"}

        node._apply_merge_rules_globally(merge_rules, ctx)

        # Child should be merged into parent
        entities = ctx.state.validated_entities_by_resource[resource1]
        assert "BRCA" in entities
        assert "BRCA1" not in entities
        assert "BRCA1" in entities["BRCA"].aliases
        assert "MERGED(BRCA1)" in entities["BRCA"].reasoning
        assert ctx.state.entities_merged == 1

    def test_applies_merge_across_multiple_documents(self, mock_deps):
        """Should apply same rule consistently across documents."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BRCA": EntityMention(
                    kind="gene",
                    name="BRCA",
                    aliases=["BRCA"],
                    quotes=[],
                    reasoning="parent1",
                ),
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="child1",
                ),
            },
            resource2: {
                "brca": EntityMention(
                    kind="gene",
                    name="brca",
                    aliases=["brca"],
                    quotes=[],
                    reasoning="parent2",
                ),
                "brca1": EntityMention(
                    kind="gene",
                    name="brca1",
                    aliases=["brca1"],
                    quotes=[],
                    reasoning="child2",
                ),
            },
        }

        # Rules keyed by normalized form - matches both BRCA1 and brca1
        # First matching entity's canonical name is used as target
        merge_rules = {("brca1", "gene"): "BRCA"}

        node._apply_merge_rules_globally(merge_rules, ctx)

        # Both documents should have merges applied
        entities1 = ctx.state.validated_entities_by_resource[resource1]
        entities2 = ctx.state.validated_entities_by_resource[resource2]

        assert "BRCA" in entities1
        assert "BRCA1" not in entities1

        assert "brca" in entities2
        assert "brca1" not in entities2

        assert ctx.state.entities_merged == 2

    def test_only_merges_when_both_present(self, mock_deps):
        """Should only merge when both parent and child exist in document."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BRCA": EntityMention(
                    kind="gene",
                    name="BRCA",
                    aliases=["BRCA"],
                    quotes=[],
                    reasoning="parent",
                ),
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="child",
                ),
            },
            resource2: {
                # Only has child, no parent
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="child",
                )
            },
        }

        # Rules keyed by normalized form, target is canonical name
        merge_rules = {("brca1", "gene"): "BRCA"}

        node._apply_merge_rules_globally(merge_rules, ctx)

        # Doc1 should merge (both parent and child present)
        assert "BRCA" in ctx.state.validated_entities_by_resource[resource1]
        assert "BRCA1" not in ctx.state.validated_entities_by_resource[resource1]

        # Doc2: child gets renamed to parent (cross-document merge behavior)
        # Since parent doesn't exist in doc2, child is renamed to parent's canonical name
        assert "BRCA" in ctx.state.validated_entities_by_resource[resource2]
        assert "BRCA1" not in ctx.state.validated_entities_by_resource[resource2]
        # Old name should be in aliases
        assert (
            "BRCA1"
            in ctx.state.validated_entities_by_resource[resource2]["BRCA"].aliases
        )

        assert ctx.state.entities_merged == 1  # Only doc1 has actual merge

    def test_handles_empty_merge_rules(self, mock_deps):
        """Should handle empty merge rules gracefully."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="test",
                )
            }
        }

        merge_rules = {}

        node._apply_merge_rules_globally(merge_rules, ctx)

        # Nothing should change
        assert "BRCA1" in ctx.state.validated_entities_by_resource[resource1]
        assert ctx.state.entities_merged == 0

    def test_handles_multiple_entity_kinds(self, mock_deps):
        """Should apply merges for different entity kinds independently."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BRCA": EntityMention(
                    kind="gene",
                    name="BRCA",
                    aliases=["BRCA"],
                    quotes=[],
                    reasoning="gene_parent",
                ),
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="gene_child",
                ),
                "PAH": EntityMention(
                    kind="disease",
                    name="PAH",
                    aliases=["PAH"],
                    quotes=[],
                    reasoning="disease_parent",
                ),
                "Pulmonary Arterial Hypertension": EntityMention(
                    kind="disease",
                    name="Pulmonary Arterial Hypertension",
                    aliases=["PAH"],
                    quotes=[],
                    reasoning="disease_child",
                ),
            }
        }

        # Rules keyed by normalized form, target is canonical name
        merge_rules = {
            ("brca1", "gene"): "BRCA",
            ("pulmonary arterial hypertension", "disease"): "PAH",
        }

        node._apply_merge_rules_globally(merge_rules, ctx)

        entities = ctx.state.validated_entities_by_resource[resource1]

        # Both merges should happen
        assert "BRCA" in entities
        assert "BRCA1" not in entities
        assert "PAH" in entities
        assert "Pulmonary Arterial Hypertension" not in entities

        assert ctx.state.entities_merged == 2

    def test_resolves_transitive_merge_chains(self, mock_deps):
        """Should resolve transitive merge chains (A→B→C becomes A→C).

        When we have a chain like:
        - "Associated pulmonary arterial hypertension" → "pulmonary arterial hypertension"
        - "pulmonary arterial hypertension" → "PAH"

        The transitive resolution should make:
        - "Associated pulmonary arterial hypertension" → "PAH"
        - "pulmonary arterial hypertension" → "PAH"

        So everything merges directly to the final parent.
        """
        node = ConsolidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["disease"],
            permitted_pairs=build_permitted_pairs(["disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.validated_entities_by_resource = {
            resource1: {
                "PAH": EntityMention(
                    kind="disease",
                    name="PAH",
                    aliases=["PAH"],
                    quotes=[],
                    reasoning="short form",
                ),
                "pulmonary arterial hypertension": EntityMention(
                    kind="disease",
                    name="pulmonary arterial hypertension",
                    aliases=["pulmonary arterial hypertension"],
                    quotes=[],
                    reasoning="full form",
                ),
                "Associated pulmonary arterial hypertension": EntityMention(
                    kind="disease",
                    name="Associated pulmonary arterial hypertension",
                    aliases=["Associated pulmonary arterial hypertension"],
                    quotes=[],
                    reasoning="specific variant",
                ),
            }
        }

        # Create merge chain: A→B→C - keys are normalized, targets are canonical
        merge_rules = {
            (
                "associated pulmonary arterial hypertension",
                "disease",
            ): "pulmonary arterial hypertension",
            ("pulmonary arterial hypertension", "disease"): "PAH",
        }
        # Rules must be resolved transitively before applying
        resolved_rules = node._resolve_transitive_merges(merge_rules)
        node._apply_merge_rules_globally(resolved_rules, ctx)

        entities = ctx.state.validated_entities_by_resource[resource1]

        # Both should merge directly into PAH (the final parent)
        assert "PAH" in entities
        assert "pulmonary arterial hypertension" not in entities
        assert "Associated pulmonary arterial hypertension" not in entities

        # Both entities should be in PAH's aliases
        assert "pulmonary arterial hypertension" in entities["PAH"].aliases
        assert "Associated pulmonary arterial hypertension" in entities["PAH"].aliases

        assert ctx.state.entities_merged == 2


class TestIntegration:
    """Integration tests for complete ConsolidateEntitiesNode flow."""

    @pytest.mark.asyncio
    async def test_full_merge_flow(self, mock_deps):
        """Test complete flow from collection to application."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="breast cancer genetics",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BRCA": EntityMention(
                    kind="gene",
                    name="BRCA",
                    aliases=["BRCA"],
                    quotes=[],
                    reasoning="parent",
                ),
                "BRCA1": EntityMention(
                    kind="gene",
                    name="BRCA1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="child",
                ),
            },
            resource2: {
                "brca": EntityMention(
                    kind="gene",
                    name="brca",
                    aliases=["brca"],
                    quotes=[],
                    reasoning="parent",
                ),
                "brca1": EntityMention(
                    kind="gene",
                    name="brca1",
                    aliases=["brca1"],
                    quotes=[],
                    reasoning="child",
                ),
            },
        }

        # Mock LLM to approve merge (pair_id=1 for brca/brca1)
        # Token "test" won't match the real token, but ID-based lookup will be used
        mock_result = MagicMock()
        mock_result.output = EntityConsolidationDecisions(
            decisions=[
                EntityConsolidationDecision(
                    pair_id=1,
                    pair_token="test",
                    action="merge",
                    reasoning="BRCA1 is specific gene",
                )
            ]
        )
        mock_agent = create_mock_agent_with_override(mock_result)

        import interaction_finder.extraction.nodes as nodes_module

        original_getter = nodes_module.get_entity_consolidation_agent
        nodes_module.get_entity_consolidation_agent = lambda config: mock_agent

        try:
            result = await node.run(ctx)

            # Should return next node (ConsolidateRelationshipsNode)
            from interaction_finder.extraction.nodes import ConsolidateRelationshipsNode

            assert isinstance(result, ConsolidateRelationshipsNode)

            # Phase 1 merges capitalization variants, Phase 3 merges substring pairs
            # Expected final state: Only "BRCA" remains (best capitalization)
            # - Phase 1: "brca" -> "BRCA", "brca1" -> "BRCA1" (2 merges)
            # - Phase 3: "BRCA1" -> "BRCA" (2 more merges, one per document)

            # Both documents should have only "BRCA"
            assert "BRCA" in ctx.state.validated_entities_by_resource[resource1]
            assert "BRCA1" not in ctx.state.validated_entities_by_resource[resource1]

            # Resource2 also has only "BRCA" (lowercase variants merged)
            assert "BRCA" in ctx.state.validated_entities_by_resource[resource2]
            assert "brca" not in ctx.state.validated_entities_by_resource[resource2]
            assert "brca1" not in ctx.state.validated_entities_by_resource[resource2]

            # Total merges: 2 from Phase 1 (cross-document renames) + 2 from Phase 3
            # But entities_merged only counts actual merges (when both exist in same doc)
            # Phase 1 does renames (not counted), Phase 3 does 2 merges (1 per doc)
            assert ctx.state.entities_merged == 2

        finally:
            nodes_module.get_entity_consolidation_agent = original_getter
