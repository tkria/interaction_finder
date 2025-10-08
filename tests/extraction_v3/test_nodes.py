"""
Tests for extraction graph V3 nodes.

Covers ExtractEntities node with caching, quote validation, and parallelism.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_graph import GraphRunContext

from interaction_finder.extraction_graph_v3.nodes import ExtractEntities
from interaction_finder.extraction_graph_v3.state import ExtractionStateV3
from interaction_finder.extraction_graph_v3.deps import ExtractionDepsV3
from interaction_finder.extraction_graph_v3.cache import SemanticCacheManager
from interaction_finder.extraction_graph_v2.models import (
    EntityWithQuotes,
    SimpleEntityOut,
    SimpleEntityListOut,
)
from interaction_finder.resources import ResourcePool
from interaction_finder.models import Term


@pytest.fixture
def mock_config():
    """Create a minimal mock config."""
    config = MagicMock()
    config.task.get_kind_names.return_value = ["gene", "disease"]
    config.task.relation = "gene-disease interaction"
    config.task.context = "biomedical gene-disease interactions"
    config.agents.get.return_value = MagicMock(llm="openai:gpt-4o-mini")
    return config


@pytest.fixture
def mock_page_fetcher():
    """Create a minimal mock page fetcher."""
    return MagicMock()


@pytest.fixture
def sample_resource_pool():
    """Create a resource pool with sample documents."""
    pool = ResourcePool()

    # Add first document about BRCA1 and breast cancer
    pool.add(
        url="https://example.com/doc1",
        title="BRCA1 in breast cancer",
        document_text="BRCA1 is a tumor suppressor gene associated with breast cancer. "
        "Mutations in BRCA1 significantly increase the risk of developing breast cancer.",
    )

    # Add second document about BRCA1 and ovarian cancer
    pool.add(
        url="https://example.com/doc2",
        title="BRCA1 in ovarian cancer",
        document_text="BRCA1 mutations are also linked to ovarian cancer. "
        "Women with BRCA1 mutations have elevated ovarian cancer risk.",
    )

    return pool


@pytest.fixture
def deps_v3(mock_config, mock_page_fetcher):
    """Create ExtractionDepsV3 with test configuration."""
    return ExtractionDepsV3(
        model="openai:gpt-4o-mini",
        config=mock_config,
        page_fetcher=mock_page_fetcher,
        target_term=Term(name="BRCA1", kind="gene"),
        extraction_parallelism=2,  # Test with limited parallelism
        semantic_cache_enabled=False,  # Start with cache disabled
    )


@pytest.fixture
def state_v3(sample_resource_pool):
    """Create ExtractionStateV3 with sample resources."""
    state = ExtractionStateV3()
    state.resource_pool = sample_resource_pool
    return state


@pytest.fixture
def mock_agent_result():
    """Create mock agent result with entities."""
    return SimpleEntityListOut(
        entities=[
            SimpleEntityOut(
                name="BRCA1",
                kind="gene",
                aliases=["BRCA1"],
                quotes=[
                    "BRCA1 is a tumor suppressor gene associated with breast cancer.",
                ],
            ),
            SimpleEntityOut(
                name="breast cancer",
                kind="disease",
                aliases=["breast cancer"],
                quotes=[
                    "Mutations in BRCA1 significantly increase the risk of developing breast cancer.",
                ],
            ),
        ],
        entity_kinds=["gene", "disease"],
        reasoning="Extracted BRCA1 gene and breast cancer disease from document",
    )


class TestExtractEntitiesBasic:
    """Test basic entity extraction functionality."""

    @pytest.mark.asyncio
    async def test_extract_entities_without_cache(
        self, state_v3, deps_v3, mock_agent_result
    ):
        """Test entity extraction without caching."""
        # Create context
        ctx = GraphRunContext(state=state_v3, deps=deps_v3)

        # Mock agent creation and execution
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_agent_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_entity_extractor_v3",
            return_value=mock_agent,
        ):
            # Run extraction
            node = ExtractEntities()
            await node.run(ctx)

        # Verify entities were extracted
        assert len(state_v3.entities_found) >= 1
        assert state_v3.metrics.entities_extraction_calls > 0
        assert state_v3.metrics.cache_misses_extraction == 0  # Cache disabled

    @pytest.mark.asyncio
    async def test_extract_entities_with_cache_miss(
        self, state_v3, deps_v3, mock_agent_result, tmp_path
    ):
        """Test entity extraction with cache miss."""
        # Enable caching
        deps_v3.semantic_cache_enabled = True
        state_v3.cache = SemanticCacheManager(tmp_path / "cache")

        # Create context
        ctx = GraphRunContext(state=state_v3, deps=deps_v3)

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_agent_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_entity_extractor_v3",
            return_value=mock_agent,
        ):
            node = ExtractEntities()
            await node.run(ctx)

        # Verify cache miss recorded
        assert state_v3.metrics.cache_misses_extraction > 0
        assert state_v3.metrics.cache_hits_extraction == 0

    @pytest.mark.asyncio
    async def test_extract_entities_with_cache_hit(
        self, state_v3, deps_v3, sample_resource_pool, tmp_path
    ):
        """Test entity extraction with cache hit."""
        # Enable caching
        deps_v3.semantic_cache_enabled = True
        cache = SemanticCacheManager(tmp_path / "cache")
        state_v3.cache = cache

        # Pre-populate cache with entities
        resource = list(sample_resource_pool.resources)[0]
        cached_entities = [
            EntityWithQuotes(
                name="BRCA1",
                kind="gene",
                aliases=["BRCA1"],
                quotes=[resource.quote("BRCA1 is a tumor suppressor gene")],
                confidence=1.0,
            )
        ]

        # Compute cache key and save
        from interaction_finder.extraction_graph_v3.cache import (
            compute_extraction_cache_key,
        )

        cache_key = compute_extraction_cache_key(
            full_doc_text=resource.text,
            entity_kinds=deps_v3.get_entity_kinds(),
            model_version="openai:gpt-4o-mini",
            prompt_version="v3_2025-10",
        )
        await cache.set_extraction(cache_key, cached_entities)

        # Create context
        ctx = GraphRunContext(state=state_v3, deps=deps_v3)

        # Mock agent (should not be called for cache hit)
        mock_agent = AsyncMock()

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_entity_extractor_v3",
            return_value=mock_agent,
        ):
            node = ExtractEntities()
            await node.run(ctx)

        # Verify cache hit
        assert state_v3.metrics.cache_hits_extraction > 0
        # Agent should not be called for cached documents
        # (may be called for second document which isn't cached)


class TestEntityDeduplication:
    """Test entity deduplication across documents."""

    @pytest.mark.asyncio
    async def test_entity_deduplication_merges_quotes(self, state_v3, deps_v3):
        """Test that same entity from multiple documents has quotes merged."""
        # Create two documents with overlapping entity
        pool = ResourcePool()
        pool.add(
            url="https://example.com/doc1",
            title="Doc 1",
            document_text="BRCA1 is important. BRCA1 has many functions.",
        )
        pool.add(
            url="https://example.com/doc2",
            title="Doc 2",
            document_text="BRCA1 plays a critical role. BRCA1 is well-studied.",
        )
        state_v3.resource_pool = pool

        # Create context
        ctx = GraphRunContext(state=state_v3, deps=deps_v3)

        # Mock agent to return same entity for both documents
        mock_agent = AsyncMock()

        def make_result(doc_text):
            if "Doc 1" in doc_text or "important" in doc_text:
                quote = "BRCA1 is important."
            else:
                quote = "BRCA1 plays a critical role."

            return SimpleEntityListOut(
                entities=[
                    SimpleEntityOut(
                        name="BRCA1",
                        kind="gene",
                        aliases=["BRCA1"],
                        quotes=[quote],
                    ),
                ],
                entity_kinds=["gene"],
                reasoning="Found BRCA1",
            )

        mock_agent.run.side_effect = lambda text, deps: make_result(text)

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_entity_extractor_v3",
            return_value=mock_agent,
        ):
            node = ExtractEntities()
            await node.run(ctx)

        # Verify entity was merged
        assert "BRCA1" in state_v3.entities_found
        entity = state_v3.entities_found["BRCA1"]

        # Should have quotes from both documents
        assert len(entity.quotes) >= 2


class TestQuoteValidation:
    """Test quote validation with fuzzy matching."""

    @pytest.mark.asyncio
    async def test_quote_validation_with_exact_match(
        self, state_v3, deps_v3, mock_agent_result
    ):
        """Test quote validation succeeds with exact match."""
        ctx = GraphRunContext(state=state_v3, deps=deps_v3)

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_agent_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_entity_extractor_v3",
            return_value=mock_agent,
        ):
            node = ExtractEntities()
            await node.run(ctx)

        # Should have entities with valid quotes
        assert len(state_v3.entities_found) > 0
        for entity in state_v3.entities_found.values():
            assert len(entity.quotes) > 0

    @pytest.mark.asyncio
    async def test_quote_validation_with_fuzzy_match(self, state_v3, deps_v3):
        """Test quote validation with fuzzy matching logs errors below threshold."""
        # Create document
        pool = ResourcePool()
        pool.add(
            url="https://example.com/doc",
            title="Test Doc",
            document_text="BRCA1 is a tumor suppressor gene.",
        )
        state_v3.resource_pool = pool

        ctx = GraphRunContext(state=state_v3, deps=deps_v3)

        # Mock agent with slightly different quote (should fuzzy match but be below auto-correct)
        mock_result = SimpleEntityListOut(
            entities=[
                SimpleEntityOut(
                    name="BRCA1",
                    kind="gene",
                    aliases=["BRCA1"],
                    quotes=[
                        "BRCA1 is tumor suppressor gene",  # Missing "a"
                    ],
                ),
            ],
            entity_kinds=["gene"],
            reasoning="Found BRCA1",
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_entity_extractor_v3",
            return_value=mock_agent,
        ):
            node = ExtractEntities()
            await node.run(ctx)

        # Entity should not be included (fuzzy match below auto-correct threshold)
        # but error should be logged
        assert "BRCA1" not in state_v3.entities_found
        assert len(deps_v3.quote_error_log) > 0
        assert deps_v3.quote_error_log[0].error_type == "paraphrased"

    @pytest.mark.asyncio
    async def test_quote_validation_failure_logged(self, state_v3, deps_v3):
        """Test quote validation failure is logged."""
        pool = ResourcePool()
        pool.add(
            url="https://example.com/doc",
            title="Test Doc",
            document_text="BRCA1 is a gene.",
        )
        state_v3.resource_pool = pool

        ctx = GraphRunContext(state=state_v3, deps=deps_v3)

        # Mock agent with invalid quote
        mock_result = SimpleEntityListOut(
            entities=[
                SimpleEntityOut(
                    name="BRCA1",
                    kind="gene",
                    aliases=["BRCA1"],
                    quotes=[
                        "This quote does not exist in the document at all.",
                    ],
                ),
            ],
            entity_kinds=["gene"],
            reasoning="Found BRCA1",
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_entity_extractor_v3",
            return_value=mock_agent,
        ):
            node = ExtractEntities()
            await node.run(ctx)

        # Entity should not be included (no valid quotes)
        assert "BRCA1" not in state_v3.entities_found

        # Quote error should be logged
        assert len(deps_v3.quote_error_log) > 0


class TestParallelProcessing:
    """Test parallel document processing."""

    @pytest.mark.asyncio
    async def test_parallel_processing_multiple_documents(self, deps_v3):
        """Test multiple documents processed in parallel."""
        # Create state with multiple documents
        pool = ResourcePool()
        for i in range(5):
            pool.add(
                url=f"https://example.com/doc{i}",
                title=f"Doc {i}",
                document_text=f"BRCA1 in document {i}. BRCA1 is important in doc {i}.",
            )

        state = ExtractionStateV3()
        state.resource_pool = pool

        ctx = GraphRunContext(state=state, deps=deps_v3)

        # Track calls to verify parallelism
        call_times = []

        async def mock_agent_run(text, deps):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate LLM call
            return SimpleEntityListOut(
                entities=[
                    SimpleEntityOut(
                        name="BRCA1",
                        kind="gene",
                        aliases=["BRCA1"],
                        quotes=["BRCA1 is important"],
                    ),
                ],
                entity_kinds=["gene"],
                reasoning="Found BRCA1",
            )

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = mock_agent_run

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_entity_extractor_v3",
            return_value=mock_agent,
        ):
            node = ExtractEntities()
            await node.run(ctx)

        # Verify all documents were processed
        assert state.metrics.entities_extraction_calls == 5


class TestErrorHandling:
    """Test error handling during extraction."""

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_crash_pipeline(self, state_v3, deps_v3):
        """Test that LLM failure doesn't crash entire pipeline."""
        ctx = GraphRunContext(state=state_v3, deps=deps_v3)

        # Mock agent that fails
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("LLM API error")

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_entity_extractor_v3",
            return_value=mock_agent,
        ):
            node = ExtractEntities()
            # Should not raise exception
            await node.run(ctx)

        # Metrics should track failure
        assert state_v3.metrics.entities_extraction_calls > 0

    @pytest.mark.asyncio
    async def test_partial_document_failure(self, deps_v3):
        """Test that failure on some documents doesn't prevent others."""
        # Create multiple documents
        pool = ResourcePool()
        pool.add(
            url="https://example.com/doc1",
            title="Doc 1",
            document_text="BRCA1 is a gene.",
        )
        pool.add(
            url="https://example.com/doc2",
            title="Doc 2",
            document_text="BRCA2 is another gene.",
        )

        state = ExtractionStateV3()
        state.resource_pool = pool

        ctx = GraphRunContext(state=state, deps=deps_v3)

        # Mock agent that fails on first doc, succeeds on second
        call_count = [0]

        async def mock_agent_run(text, deps):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First document failed")

            return SimpleEntityListOut(
                entities=[
                    SimpleEntityOut(
                        name="BRCA2",
                        kind="gene",
                        aliases=["BRCA2"],
                        quotes=["BRCA2 is another gene."],
                    ),
                ],
                entity_kinds=["gene"],
                reasoning="Found BRCA2",
            )

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = mock_agent_run

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_entity_extractor_v3",
            return_value=mock_agent,
        ):
            node = ExtractEntities()
            await node.run(ctx)

        # Second document should still be processed
        assert "BRCA2" in state.entities_found


class TestCheckpointCallback:
    """Test checkpoint callback invocation."""

    @pytest.mark.asyncio
    async def test_checkpoint_callback_invoked(
        self, state_v3, deps_v3, mock_agent_result
    ):
        """Test that checkpoint callback is invoked after extraction."""
        # Create checkpoint callback
        checkpoint_calls = []

        async def checkpoint_callback(stage: str, state):
            checkpoint_calls.append((stage, state))

        deps_v3.checkpoint_callback = checkpoint_callback

        ctx = GraphRunContext(state=state_v3, deps=deps_v3)

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_agent_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_entity_extractor_v3",
            return_value=mock_agent,
        ):
            node = ExtractEntities()
            await node.run(ctx)

        # Verify checkpoint was called
        assert len(checkpoint_calls) == 1
        assert checkpoint_calls[0][0] == "extraction"
        assert checkpoint_calls[0][1] is state_v3
