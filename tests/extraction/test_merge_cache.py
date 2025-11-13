"""Tests for merge decision caching and cross-document normalization.

NOTE: These tests are for the OLD per-document merge architecture.
The new architecture (MergeEntitiesNode) handles merging globally.
See test_canonical_name_variants.py for tests of the new architecture.

These tests are currently SKIPPED and kept for reference.
They should be either updated or removed in the future.

Validates that:
1. Merge decisions are cached and reused across documents
2. Cache keys use normalized entity names
3. Post-hoc normalization applies cached decisions consistently
4. Cache metrics are tracked correctly
"""

import pytest

pytest.skip(
    "Old per-document merge tests - new architecture uses global merging",
    allow_module_level=True,
)

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.usage import RunUsage
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.models import (
    EntityInfo,
    EntityMention,
    EntityMergeDecision,
    EntityMergeDecisions,
)
from interaction_finder.extraction.nodes import ValidateEntitiesNode
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import build_permitted_pairs
from interaction_finder.resources import (
    Resource,
    ResourceId,
    ResourcePool,
    ResourceQuote,
)


def create_mock_agent_with_override(run_return_value):
    """Create a mock agent with working rename_agent() support.

    Parameters:
        run_return_value: The value to return from agent.run()

    Returns:
        Mock agent with ._name attribute and .run() method
    """
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=run_return_value)
    mock_agent._name = "mock_agent"  # Add _name attribute for rename_agent
    return mock_agent


@pytest.fixture
def mock_deps():
    """Create mock dependencies."""
    deps = MagicMock()
    deps.config = MagicMock()
    deps.config.tools.extraction.merge_batch_size = 50
    deps.logger = MagicMock()
    deps.resource_pool = ResourcePool()
    return deps


@pytest.fixture
def basic_resource():
    """Create a basic test resource."""
    return Resource(
        id=ResourceId(url="https://example.com/doc1"),
        title="Test Document",
        text="This is a test document with some entities.",
        chunks=[(0, 20), (20, 50)],
        metadata={},
    )


@pytest.fixture
def entity_with_quotes(basic_resource):
    """Create an entity mention with valid quotes."""
    return EntityMention(
        kind="gene",
        name="brca1",
        aliases=["BRCA1"],
        quotes=[
            ResourceQuote(
                text="BRCA1 gene",
                resource=basic_resource,
                chunk_indices=[0],
                start_pos=0,
                end_pos=10,
            )
        ],
        reasoning="Test entity",
    )


class TestMergeCaching:
    """Test merge decision caching behavior."""

    @pytest.mark.asyncio
    async def test_cache_miss_queries_llm(self, mock_deps):
        """First encounter with entity pair should query LLM."""
        node = ValidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Create entities with substring relationship
        entities = {
            "brca1": EntityMention(
                kind="gene",
                name="brca1",
                aliases=["BRCA1"],
                quotes=[],
                reasoning="test",
            ),
            "brca": EntityMention(
                kind="gene", name="brca", aliases=["BRCA"], quotes=[], reasoning="test"
            ),
        }

        # Mock LLM response
        mock_result = MagicMock()
        mock_result.output = EntityMergeDecisions(
            decisions=[
                EntityMergeDecision(
                    parent_entity="brca",
                    child_entity="brca1",
                    should_merge=True,
                    reasoning="BRCA1 is a specific gene, BRCA is shorthand",
                )
            ]
        )
        mock_agent = create_mock_agent_with_override(mock_result)

        # Patch the agent getter
        import interaction_finder.extraction.nodes as nodes_module

        original_getter = nodes_module.get_entity_merge_agent
        nodes_module.get_entity_merge_agent = lambda config: mock_agent

        try:
            decisions = await node._get_merge_decisions(
                [("brca", "brca1")], entities, ctx
            )

            # Should have made LLM call
            assert mock_agent.run.called
            assert len(decisions) == 1
            assert decisions[0].should_merge is True

            # Cache should be populated
            cache_key = ("brca", "brca1", "gene")
            assert cache_key in ctx.state.merge_decision_cache
            assert ctx.state.merge_decision_cache[cache_key] is True

            # Metrics should reflect cache miss
            assert ctx.state.merge_cache_misses == 1
            assert ctx.state.merge_cache_hits == 0

        finally:
            nodes_module.get_entity_merge_agent = original_getter

    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm(self, mock_deps):
        """Second encounter with same pair should use cache."""
        node = ValidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Pre-populate cache
        cache_key = ("brca", "brca1", "gene")
        ctx.state.merge_decision_cache[cache_key] = True

        # Create entities
        entities = {
            "brca1": EntityMention(
                kind="gene",
                name="brca1",
                aliases=["BRCA1"],
                quotes=[],
                reasoning="test",
            ),
            "brca": EntityMention(
                kind="gene", name="brca", aliases=["BRCA"], quotes=[], reasoning="test"
            ),
        }

        # Mock LLM (should not be called)
        mock_agent = create_mock_agent_with_override(None)
        import interaction_finder.extraction.nodes as nodes_module

        original_getter = nodes_module.get_entity_merge_agent
        nodes_module.get_entity_merge_agent = lambda config: mock_agent

        try:
            decisions = await node._get_merge_decisions(
                [("brca", "brca1")], entities, ctx
            )

            # Should NOT have made LLM call
            assert not mock_agent.run.called

            # Should return cached decision
            assert len(decisions) == 1
            assert decisions[0].should_merge is True
            assert decisions[0].reasoning == "[Cached decision from previous document]"

            # Metrics should reflect cache hit
            assert ctx.state.merge_cache_hits == 1
            assert ctx.state.merge_cache_misses == 0

        finally:
            nodes_module.get_entity_merge_agent = original_getter

    @pytest.mark.asyncio
    async def test_cache_uses_normalized_names(self, mock_deps):
        """Cache should normalize entity names for key."""
        node = ValidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Pre-populate cache with lowercase key
        cache_key = ("brca", "brca1", "gene")
        ctx.state.merge_decision_cache[cache_key] = True

        # Create entities with different capitalization
        entities = {
            "BRCA1": EntityMention(
                kind="gene",
                name="BRCA1",
                aliases=["BRCA1"],
                quotes=[],
                reasoning="test",
            ),
            "BRCA": EntityMention(
                kind="gene",
                name="BRCA",
                aliases=["BRCA"],
                quotes=[],
                reasoning="test",
            ),
        }

        # Mock LLM (should not be called due to normalized cache hit)
        mock_agent = create_mock_agent_with_override(None)
        import interaction_finder.extraction.nodes as nodes_module

        original_getter = nodes_module.get_entity_merge_agent
        nodes_module.get_entity_merge_agent = lambda config: mock_agent

        try:
            decisions = await node._get_merge_decisions(
                [("BRCA", "BRCA1")], entities, ctx
            )

            # Should NOT call LLM (cache hit despite different case)
            assert not mock_agent.run.called
            assert len(decisions) == 1
            assert ctx.state.merge_cache_hits == 1

        finally:
            nodes_module.get_entity_merge_agent = original_getter

    @pytest.mark.asyncio
    async def test_mixed_cached_and_uncached(self, mock_deps):
        """Should handle mix of cached and uncached pairs efficiently."""
        node = ValidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Pre-populate cache with one decision
        ctx.state.merge_decision_cache[("brca", "brca1", "gene")] = True

        entities = {
            "brca1": EntityMention(
                kind="gene",
                name="brca1",
                aliases=["BRCA1"],
                quotes=[],
                reasoning="test",
            ),
            "brca": EntityMention(
                kind="gene", name="brca", aliases=["BRCA"], quotes=[], reasoning="test"
            ),
            "tp53": EntityMention(
                kind="gene", name="tp53", aliases=["TP53"], quotes=[], reasoning="test"
            ),
            "tp": EntityMention(
                kind="gene", name="tp", aliases=["TP"], quotes=[], reasoning="test"
            ),
        }

        # Mock LLM for uncached pair only
        mock_result = MagicMock()
        mock_result.output = EntityMergeDecisions(
            decisions=[
                EntityMergeDecision(
                    parent_entity="tp",
                    child_entity="tp53",
                    should_merge=False,
                    reasoning="TP and TP53 are different proteins",
                )
            ]
        )
        mock_agent = create_mock_agent_with_override(mock_result)

        import interaction_finder.extraction.nodes as nodes_module

        original_getter = nodes_module.get_entity_merge_agent
        nodes_module.get_entity_merge_agent = lambda config: mock_agent

        try:
            decisions = await node._get_merge_decisions(
                [("brca", "brca1"), ("tp", "tp53")], entities, ctx
            )

            # Should call LLM only for uncached pair
            assert mock_agent.run.called
            call_args = mock_agent.run.call_args
            # Check that prompt only includes TP/TP53, not BRCA/BRCA1
            prompt = call_args[0][0]
            assert "tp53" in prompt.lower()
            assert "brca1" not in prompt.lower()

            # Should return both decisions
            assert len(decisions) == 2

            # Metrics should show 1 hit, 1 miss
            assert ctx.state.merge_cache_hits == 1
            assert ctx.state.merge_cache_misses == 1

        finally:
            nodes_module.get_entity_merge_agent = original_getter


class TestCrossDocumentNormalization:
    """Test post-hoc cross-document merge normalization."""

    def test_applies_merge_across_documents(self, mock_deps):
        """Cached merge decision should apply to all documents."""
        node = ValidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Pre-populate cache: "brca1" should merge into "brca"
        ctx.state.merge_decision_cache[("brca", "brca1", "gene")] = True

        # Doc1 has both entities, Doc2 has both entities
        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "brca": EntityMention(
                    kind="gene",
                    name="brca",
                    aliases=["BRCA"],
                    quotes=[],
                    reasoning="doc1",
                ),
                "brca1": EntityMention(
                    kind="gene",
                    name="brca1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="doc1",
                ),
            },
            resource2: {
                "brca": EntityMention(
                    kind="gene",
                    name="brca",
                    aliases=["BRCA"],
                    quotes=[],
                    reasoning="doc2",
                ),
                "brca1": EntityMention(
                    kind="gene",
                    name="brca1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="doc2",
                ),
            },
        }

        # Apply normalization
        node._normalize_merges_across_documents(ctx)

        # Both documents should have merged brca1 into brca
        for resource_id in [resource1, resource2]:
            entities = ctx.state.validated_entities_by_resource[resource_id]
            assert "brca" in entities
            assert "brca1" not in entities  # Should be merged away
            assert "brca1" in entities["brca"].aliases
            assert "MERGED_POSTHOC" in entities["brca"].reasoning

        # Should have merged 2 entities (one per document)
        assert ctx.state.entities_merged == 2

    def test_only_merges_when_both_present(self, mock_deps):
        """Should only merge when both parent and child exist in document."""
        node = ValidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Cache says to merge brca1 into brca
        ctx.state.merge_decision_cache[("brca", "brca1", "gene")] = True

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        resource2 = ResourceId(url="https://example.com/doc2", counter=1)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "brca": EntityMention(
                    kind="gene",
                    name="brca",
                    aliases=["BRCA"],
                    quotes=[],
                    reasoning="doc1",
                ),
                "brca1": EntityMention(
                    kind="gene",
                    name="brca1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="doc1",
                ),
            },
            resource2: {
                # Only has brca1, not brca - cannot merge
                "brca1": EntityMention(
                    kind="gene",
                    name="brca1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="doc2",
                ),
            },
        }

        # Apply normalization
        node._normalize_merges_across_documents(ctx)

        # Doc1 should have merged
        assert "brca" in ctx.state.validated_entities_by_resource[resource1]
        assert "brca1" not in ctx.state.validated_entities_by_resource[resource1]

        # Doc2 should NOT have merged (no parent to merge into)
        assert "brca1" in ctx.state.validated_entities_by_resource[resource2]
        assert "brca" not in ctx.state.validated_entities_by_resource[resource2]

        # Only 1 entity merged
        assert ctx.state.entities_merged == 1

    def test_respects_reject_decisions(self, mock_deps):
        """Should not merge when cache says should_merge=False."""
        node = ValidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Cache says NOT to merge
        ctx.state.merge_decision_cache[("tp", "tp53", "gene")] = False

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "tp": EntityMention(
                    kind="gene",
                    name="tp",
                    aliases=["TP"],
                    quotes=[],
                    reasoning="doc1",
                ),
                "tp53": EntityMention(
                    kind="gene",
                    name="tp53",
                    aliases=["TP53"],
                    quotes=[],
                    reasoning="doc1",
                ),
            },
        }

        # Apply normalization
        node._normalize_merges_across_documents(ctx)

        # Should NOT have merged
        assert "tp" in ctx.state.validated_entities_by_resource[resource1]
        assert "tp53" in ctx.state.validated_entities_by_resource[resource1]
        assert ctx.state.entities_merged == 0

    def test_handles_multiple_entity_kinds(self, mock_deps):
        """Should handle merges across different entity kinds independently."""
        node = ValidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene", "disease"],
            permitted_pairs=build_permitted_pairs(["gene", "disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Cache decisions for different kinds
        ctx.state.merge_decision_cache[("brca", "brca1", "gene")] = True
        ctx.state.merge_decision_cache[
            ("pah", "pulmonary arterial hypertension pah", "disease")
        ] = True

        resource1 = ResourceId(url="https://example.com/doc1", counter=0)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "brca": EntityMention(
                    kind="gene",
                    name="brca",
                    aliases=["BRCA"],
                    quotes=[],
                    reasoning="doc1",
                ),
                "brca1": EntityMention(
                    kind="gene",
                    name="brca1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="doc1",
                ),
                "pah": EntityMention(
                    kind="disease",
                    name="pah",
                    aliases=["PAH"],
                    quotes=[],
                    reasoning="doc1",
                ),
                "pulmonary arterial hypertension pah": EntityMention(
                    kind="disease",
                    name="pulmonary arterial hypertension pah",
                    aliases=["PAH"],
                    quotes=[],
                    reasoning="doc1",
                ),
            },
        }

        # Apply normalization
        node._normalize_merges_across_documents(ctx)

        entities = ctx.state.validated_entities_by_resource[resource1]

        # Both merges should have happened
        assert "brca" in entities
        assert "brca1" not in entities
        assert "pah" in entities
        assert "pulmonary arterial hypertension pah" not in entities

        # Should have merged 2 entities
        assert ctx.state.entities_merged == 2

    def test_empty_cache_does_nothing(self, mock_deps):
        """Should handle empty cache gracefully."""
        node = ValidateEntitiesNode()
        state = State(
            topic="test",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # No cache entries
        resource1 = ResourceId(url="https://example.com/doc1", counter=0)

        ctx.state.validated_entities_by_resource = {
            resource1: {
                "brca": EntityMention(
                    kind="gene",
                    name="brca",
                    aliases=["BRCA"],
                    quotes=[],
                    reasoning="doc1",
                ),
                "brca1": EntityMention(
                    kind="gene",
                    name="brca1",
                    aliases=["BRCA1"],
                    quotes=[],
                    reasoning="doc1",
                ),
            },
        }

        # Apply normalization (should be no-op)
        node._normalize_merges_across_documents(ctx)

        # Nothing should change
        assert "brca" in ctx.state.validated_entities_by_resource[resource1]
        assert "brca1" in ctx.state.validated_entities_by_resource[resource1]
        assert ctx.state.entities_merged == 0


class TestMetricsTracking:
    """Test that cache metrics are properly tracked."""

    def test_cache_metrics_in_metadata(self, mock_deps):
        """ExtractionMetadata should include cache metrics."""
        from interaction_finder.extraction.models import ExtractionMetadata

        metadata = ExtractionMetadata(
            topic="test",
            resource_count=10,
            total_entities_found=100,
            entities_after_validation=90,
            entities_merged=20,
            merge_cache_hits=15,
            merge_cache_misses=5,
            proximal_sets_found=30,
            total_pairs_found=50,
            pairs_accepted=40,
            pairs_rejected=10,
            quotes_validated=100,
            quotes_failed=5,
        )

        assert metadata.merge_cache_hits == 15
        assert metadata.merge_cache_misses == 5

        # Verify cache hit rate calculation
        total = metadata.merge_cache_hits + metadata.merge_cache_misses
        hit_rate = metadata.merge_cache_hits / total if total > 0 else 0
        assert hit_rate == 0.75  # 15/20 = 75%
