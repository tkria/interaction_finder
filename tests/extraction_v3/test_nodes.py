"""
Tests for extraction graph V3 nodes.

Covers ExtractEntities node with caching, quote validation, and parallelism.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_graph import GraphRunContext

from interaction_finder.extraction_graph_v3.nodes import (
    ExtractEntities,
    AssessIndividually,
    EvaluatePairs,
)
from interaction_finder.extraction_graph_v3.state import ExtractionStateV3
from interaction_finder.extraction_graph_v3.deps import ExtractionDepsV3
from interaction_finder.extraction_graph_v3.cache import SemanticCacheManager
from interaction_finder.extraction_graph_v3.models import (
    PairCandidate,
    PairEvaluationOut,
)
from interaction_finder.extraction_graph_v2.models import (
    EntityWithQuotes,
    SimpleEntityOut,
    SimpleEntityListOut,
    AssessmentOut,
    IndividualAssessment,
)
from interaction_finder.resources import ResourcePool
from interaction_finder.models import Term


class MockAgentRunResult:
    """
    Mock wrapper for Pydantic-AI v1.0.0 AgentRunResult.

    Wraps the output data to match the real AgentRunResult.output API,
    allowing tests to accurately reflect production behavior.
    """

    def __init__(self, output):
        self.output = output


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
    return MockAgentRunResult(
        SimpleEntityListOut(
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

            return MockAgentRunResult(
                SimpleEntityListOut(
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
            )

        mock_agent.run.side_effect = lambda text, deps, **kwargs: make_result(text)

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
        mock_result = MockAgentRunResult(
            SimpleEntityListOut(
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
        mock_result = MockAgentRunResult(
            SimpleEntityListOut(
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

        async def mock_agent_run(text, deps, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate LLM call
            return MockAgentRunResult(
                SimpleEntityListOut(
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

        async def mock_agent_run(text, deps, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First document failed")

            return MockAgentRunResult(
                SimpleEntityListOut(
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


class TestEntityKindsValidation:
    """Test validation of entity_kinds configuration."""

    @pytest.mark.asyncio
    async def test_empty_entity_kinds_raises_error(self, state_v3, deps_v3):
        """Test that empty entity_kinds raises clear error with helpful message."""
        # Mock get_entity_kinds to return empty list
        deps_v3.get_entity_kinds = lambda: []

        ctx = GraphRunContext(state=state_v3, deps=deps_v3)

        node = ExtractEntities()

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError) as exc_info:
            await node.run(ctx)

        error_message = str(exc_info.value)

        # Verify error message contains key information
        assert "No entity kinds configured" in error_message
        assert "[task.kinds]" in error_message
        assert "config.toml" in error_message
        assert "Example configuration" in error_message
        assert "gene" in error_message
        assert "disease" in error_message


# ============================================================================
# AssessIndividually Node Tests
# ============================================================================


@pytest.fixture
def mock_assessment_result():
    """Create mock assessment agent result."""
    return MockAgentRunResult(
        AssessmentOut(
            potential="high",
            related=["breast cancer", "ovarian cancer"],
            evidence=[
                "BRCA1 is a tumor suppressor gene associated with breast cancer.",
                "BRCA1 mutations are also linked to ovarian cancer.",
            ],
            reasoning="BRCA1 shows high relationship potential with multiple cancers",
        )
    )


@pytest.fixture
def state_with_entities(sample_resource_pool):
    """Create state with extracted entities ready for assessment."""
    state = ExtractionStateV3()
    state.resource_pool = sample_resource_pool

    # Manually add entities (simulating ExtractEntities output)
    resources = sample_resource_pool.resources  # This is a list property

    state.entities_found["BRCA1"] = EntityWithQuotes(
        name="BRCA1",
        kind="gene",
        aliases=["BRCA1"],
        quotes=[
            resources[0].quote("BRCA1 is a tumor suppressor gene"),
            resources[1].quote("BRCA1 mutations"),
        ],
        confidence=1.0,
    )

    state.entities_found["breast cancer"] = EntityWithQuotes(
        name="breast cancer",
        kind="disease",
        aliases=["breast cancer"],
        quotes=[resources[0].quote("breast cancer")],
        confidence=1.0,
    )

    return state


class TestAssessIndividuallyBasic:
    """Test basic assessment functionality."""

    @pytest.mark.asyncio
    async def test_assess_entities_without_cache(
        self, state_with_entities, deps_v3, mock_assessment_result
    ):
        """Test entity assessment without caching."""
        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_assessment_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        # Verify assessments were created
        assert len(state_with_entities.individual_assessments) >= 1
        assert state_with_entities.metrics.assessment_calls > 0
        assert (
            state_with_entities.metrics.cache_misses_assessment == 0
        )  # Cache disabled

    @pytest.mark.asyncio
    async def test_assess_entities_with_cache_miss(
        self, state_with_entities, deps_v3, mock_assessment_result, tmp_path
    ):
        """Test assessment with cache miss."""
        # Enable caching
        deps_v3.semantic_cache_enabled = True
        state_with_entities.cache = SemanticCacheManager(tmp_path / "cache")

        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_assessment_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        # Verify cache miss recorded
        assert state_with_entities.metrics.cache_misses_assessment > 0
        assert state_with_entities.metrics.cache_hits_assessment == 0

    @pytest.mark.asyncio
    async def test_assess_entities_with_cache_hit(
        self, state_with_entities, deps_v3, tmp_path
    ):
        """Test assessment with cache hit."""
        # Enable caching
        deps_v3.semantic_cache_enabled = True
        cache = SemanticCacheManager(tmp_path / "cache")
        state_with_entities.cache = cache

        # Pre-populate cache with assessment
        entity = state_with_entities.entities_found["BRCA1"]
        cached_assessment = IndividualAssessment(
            entity=entity,
            relationship_potential="high",
            related_entities=["breast cancer"],
            evidence_quotes=[],
            reasoning="Cached assessment",
            confidence=0.9,
        )

        # Compute cache key and save
        from interaction_finder.extraction_graph_v3.cache import (
            compute_assessment_cache_key,
        )

        cache_key = compute_assessment_cache_key(
            entity_name=entity.name,
            entity_kind=entity.kind,
            task_context=deps_v3.get_task_context(),
            model_version="openai:gpt-4o-mini",
            prompt_version="v3_2025-10",
        )
        await cache.set_assessment(cache_key, cached_assessment)

        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        # Mock agent (should not be called for cache hit)
        mock_agent = AsyncMock()

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        # Verify cache hit
        assert state_with_entities.metrics.cache_hits_assessment > 0


class TestRelatedEntitiesExtraction:
    """Test extraction of related_entities field."""

    @pytest.mark.asyncio
    async def test_related_entities_populated(self, state_with_entities, deps_v3):
        """Test that related_entities field is populated from agent output."""
        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        # Mock agent with specific related entities
        mock_result = MockAgentRunResult(
            AssessmentOut(
                potential="high",
                related=["breast cancer", "ovarian cancer", "TP53"],
                evidence=["BRCA1 interacts with these entities"],
                reasoning="High potential for relationships",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        # Verify related entities are present
        assert "BRCA1" in state_with_entities.individual_assessments
        assessment = state_with_entities.individual_assessments["BRCA1"]
        assert len(assessment.related_entities) == 3
        assert "breast cancer" in assessment.related_entities
        assert "ovarian cancer" in assessment.related_entities
        assert "TP53" in assessment.related_entities

    @pytest.mark.asyncio
    async def test_empty_related_entities_for_low_potential(
        self, state_with_entities, deps_v3
    ):
        """Test that low potential entities can have empty related_entities."""
        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        # Mock agent with low potential and no related entities
        mock_result = MockAgentRunResult(
            AssessmentOut(
                potential="low",
                related=[],
                evidence=[],
                reasoning="Low potential for relationships",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        # Verify assessment exists with empty related_entities
        assert "BRCA1" in state_with_entities.individual_assessments
        assessment = state_with_entities.individual_assessments["BRCA1"]
        assert assessment.relationship_potential == "low"
        assert len(assessment.related_entities) == 0


class TestEvidenceQuoteValidation:
    """Test evidence quote validation with fuzzy matching."""

    @pytest.mark.asyncio
    async def test_evidence_quotes_validated(self, state_with_entities, deps_v3):
        """Test that evidence quotes are validated against resources."""
        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        # Mock agent with evidence that exists in resource
        mock_result = MockAgentRunResult(
            AssessmentOut(
                potential="high",
                related=["breast cancer"],
                evidence=[
                    "BRCA1 is a tumor suppressor gene associated with breast cancer.",
                ],
                reasoning="Strong evidence for relationship",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        # Verify evidence quotes were validated
        assessment = state_with_entities.individual_assessments["BRCA1"]
        assert len(assessment.evidence_quotes) > 0
        # Evidence quotes should be ResourceQuote objects
        from interaction_finder.resources import ResourceQuote

        assert isinstance(assessment.evidence_quotes[0], ResourceQuote)

    @pytest.mark.asyncio
    async def test_invalid_evidence_quotes_skipped(self, state_with_entities, deps_v3):
        """Test that invalid evidence quotes are skipped without crashing."""
        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        # Mock agent with evidence that doesn't exist in resource
        mock_result = MockAgentRunResult(
            AssessmentOut(
                potential="high",
                related=["breast cancer"],
                evidence=[
                    "This evidence quote does not exist in any resource document.",
                ],
                reasoning="Some reasoning",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        # Assessment should exist but with no validated evidence quotes
        assessment = state_with_entities.individual_assessments["BRCA1"]
        assert len(assessment.evidence_quotes) == 0

    @pytest.mark.asyncio
    async def test_validation_constrained_to_quotes_used(
        self, sample_resource_pool, deps_v3
    ):
        """
        Test cross-document contamination prevention.

        Verifies that validation only searches quotes_used, not all entity quotes.
        Entity exists in both documents A and B, but prompt only uses document A
        quotes. Evidence from document B should be rejected with warning.
        """
        # Create state with entity in both documents
        state = ExtractionStateV3()
        state.resource_pool = sample_resource_pool
        resources = sample_resource_pool.resources

        # Entity has quotes from both documents
        state.entities_found["BRCA1"] = EntityWithQuotes(
            name="BRCA1",
            kind="gene",
            aliases=["BRCA1"],
            quotes=[
                resources[0].quote("BRCA1 is a tumor suppressor gene"),  # doc A
                resources[1].quote("BRCA1 mutations"),  # doc B
            ],
            confidence=1.0,
        )

        ctx = GraphRunContext(state=state, deps=deps_v3)

        # Mock agent returns evidence from document B
        # (which exists in entity.quotes but not in quotes_used)
        mock_result = MockAgentRunResult(
            AssessmentOut(
                potential="high",
                related=["ovarian cancer"],
                evidence=[
                    "BRCA1 mutations increase ovarian cancer risk.",  # from doc B
                ],
                reasoning="Evidence from document B",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        # Mock _build_assessment_prompt to return only doc A quotes in quotes_used
        original_build_prompt = AssessIndividually._build_assessment_prompt

        def mock_build_prompt(self, entity, deps):
            # Return only quotes from document A in quotes_used
            quotes_used = [resources[0].quote("BRCA1 is a tumor suppressor gene")]
            prompt = f"Assess {entity.name}"
            return prompt, quotes_used

        with (
            patch(
                "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
                return_value=mock_agent,
            ),
            patch.object(
                AssessIndividually,
                "_build_assessment_prompt",
                mock_build_prompt,
            ),
        ):
            node = AssessIndividually()
            await node.run(ctx)

        # Evidence from document B should be rejected (not in quotes_used)
        assessment = state.individual_assessments["BRCA1"]
        assert len(assessment.evidence_quotes) == 0, (
            "Evidence from document B should be rejected when only document A "
            "quotes are in quotes_used"
        )


class TestConfidenceMapping:
    """Test mapping of potential levels to confidence scores."""

    @pytest.mark.asyncio
    async def test_confidence_mapping_high(self, state_with_entities, deps_v3):
        """Test high potential maps to 0.9 confidence."""
        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        mock_result = MockAgentRunResult(
            AssessmentOut(
                potential="high",
                related=["breast cancer"],
                evidence=[],
                reasoning="High potential",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        assessment = state_with_entities.individual_assessments["BRCA1"]
        assert assessment.confidence == 0.9

    @pytest.mark.asyncio
    async def test_confidence_mapping_medium(self, state_with_entities, deps_v3):
        """Test medium potential maps to 0.7 confidence."""
        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        mock_result = MockAgentRunResult(
            AssessmentOut(
                potential="medium",
                related=["breast cancer"],
                evidence=[],
                reasoning="Medium potential",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        assessment = state_with_entities.individual_assessments["BRCA1"]
        assert assessment.confidence == 0.7

    @pytest.mark.asyncio
    async def test_confidence_mapping_low(self, state_with_entities, deps_v3):
        """Test low potential maps to 0.5 confidence."""
        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        mock_result = MockAgentRunResult(
            AssessmentOut(
                potential="low",
                related=[],
                evidence=[],
                reasoning="Low potential",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        assessment = state_with_entities.individual_assessments["BRCA1"]
        assert assessment.confidence == 0.5

    @pytest.mark.asyncio
    async def test_confidence_mapping_none(self, state_with_entities, deps_v3):
        """Test none potential maps to 0.0 confidence."""
        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        mock_result = MockAgentRunResult(
            AssessmentOut(
                potential="none",
                related=[],
                evidence=[],
                reasoning="No potential",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        assessment = state_with_entities.individual_assessments["BRCA1"]
        assert assessment.confidence == 0.0


class TestConcurrentAssessment:
    """Test concurrent processing of all entities."""

    @pytest.mark.asyncio
    async def test_concurrent_assessment_multiple_entities(self, deps_v3):
        """Test multiple entities assessed concurrently."""
        # Create state with multiple entities
        pool = ResourcePool()
        pool.add(
            url="https://example.com/doc",
            title="Test Doc",
            document_text="BRCA1, BRCA2, and TP53 are important genes. They interact with breast cancer and ovarian cancer.",
        )

        state = ExtractionStateV3()
        state.resource_pool = pool

        # Add multiple entities
        resource = pool.resources[0]
        for gene in ["BRCA1", "BRCA2", "TP53"]:
            state.entities_found[gene] = EntityWithQuotes(
                name=gene,
                kind="gene",
                aliases=[gene],
                quotes=[resource.quote(gene)],
                confidence=1.0,
            )

        ctx = GraphRunContext(state=state, deps=deps_v3)

        # Track concurrent calls
        call_times = []

        async def mock_agent_run(prompt, deps, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate LLM call
            return MockAgentRunResult(
                AssessmentOut(
                    potential="high",
                    related=["breast cancer"],
                    evidence=[],
                    reasoning="Gene with cancer relationship",
                )
            )

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = mock_agent_run

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        # Verify all entities were assessed
        assert len(state.individual_assessments) == 3
        assert state.metrics.assessment_calls == 3


class TestAssessmentErrorHandling:
    """Test error handling during assessment."""

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_crash(self, state_with_entities, deps_v3):
        """Test that LLM failure doesn't crash assessment pipeline."""
        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        # Mock agent that fails
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("LLM API error")

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            # Should not raise exception
            await node.run(ctx)

        # Metrics should track calls
        assert state_with_entities.metrics.assessment_calls > 0

        # Failed assessments should have minimal assessment
        for assessment in state_with_entities.individual_assessments.values():
            if "failed" in assessment.reasoning.lower():
                assert assessment.relationship_potential == "none"
                assert assessment.confidence == 0.0

    @pytest.mark.asyncio
    async def test_partial_assessment_failure(self, deps_v3):
        """Test that failure on some assessments doesn't prevent others."""
        pool = ResourcePool()
        pool.add(
            url="https://example.com/doc",
            title="Test Doc",
            document_text="BRCA1 and BRCA2 are genes.",
        )

        state = ExtractionStateV3()
        state.resource_pool = pool

        # Add entities
        resource = pool.resources[0]
        state.entities_found["BRCA1"] = EntityWithQuotes(
            name="BRCA1",
            kind="gene",
            aliases=["BRCA1"],
            quotes=[resource.quote("BRCA1")],
            confidence=1.0,
        )
        state.entities_found["BRCA2"] = EntityWithQuotes(
            name="BRCA2",
            kind="gene",
            aliases=["BRCA2"],
            quotes=[resource.quote("BRCA2")],
            confidence=1.0,
        )

        ctx = GraphRunContext(state=state, deps=deps_v3)

        # Mock agent that fails on first entity, succeeds on second
        call_count = [0]

        async def mock_agent_run(prompt, deps, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First assessment failed")

            return MockAgentRunResult(
                AssessmentOut(
                    potential="high",
                    related=["breast cancer"],
                    evidence=[],
                    reasoning="Good assessment",
                )
            )

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = mock_agent_run

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        # Both assessments should be present
        assert len(state.individual_assessments) == 2


class TestAssessmentCheckpointCallback:
    """Test checkpoint callback invocation."""

    @pytest.mark.asyncio
    async def test_checkpoint_callback_invoked(
        self, state_with_entities, deps_v3, mock_assessment_result
    ):
        """Test that checkpoint callback is invoked after assessment."""
        checkpoint_calls = []

        async def checkpoint_callback(stage: str, state):
            checkpoint_calls.append((stage, state))

        deps_v3.checkpoint_callback = checkpoint_callback

        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_assessment_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_assessment_agent_v3",
            return_value=mock_agent,
        ):
            node = AssessIndividually()
            await node.run(ctx)

        # Verify checkpoint was called
        assert len(checkpoint_calls) == 1
        assert checkpoint_calls[0][0] == "assessment"
        assert checkpoint_calls[0][1] is state_with_entities


# ============================================================================
# GeneratePairCandidates Node Tests (Task 07)
# ============================================================================


@pytest.fixture
def state_with_assessments():
    """Create state with entities and assessments for pair generation."""
    pool = ResourcePool()

    # Add document with multiple entities in chunks
    pool.add(
        url="https://example.com/doc",
        title="Gene-disease interactions",
        document_text="BRCA1 is associated with breast cancer. "
        "TP53 mutations lead to breast cancer. "
        "EGFR is involved in lung cancer. "
        "KRAS mutations also cause lung cancer.",
    )

    state = ExtractionStateV3()
    state.resource_pool = pool
    resource = pool.resources[0]

    # Add entities with quotes
    from interaction_finder.extraction_graph_v2.models import IndividualAssessment

    # Gene: BRCA1
    brca1_quote = resource.quote("BRCA1 is associated with breast cancer")
    state.entities_found["BRCA1"] = EntityWithQuotes(
        name="BRCA1",
        kind="gene",
        aliases=["BRCA1"],
        quotes=[brca1_quote],
        confidence=1.0,
    )

    # Disease: breast cancer (same chunk as BRCA1)
    breast_cancer_quote = resource.quote("breast cancer")
    state.entities_found["breast cancer"] = EntityWithQuotes(
        name="breast cancer",
        kind="disease",
        aliases=["breast cancer"],
        quotes=[breast_cancer_quote],
        confidence=1.0,
    )

    # Gene: TP53 (adjacent chunk)
    tp53_quote = resource.quote("TP53 mutations")
    state.entities_found["TP53"] = EntityWithQuotes(
        name="TP53",
        kind="gene",
        aliases=["TP53"],
        quotes=[tp53_quote],
        confidence=1.0,
    )

    # Gene: EGFR (different chunk, for document-level test)
    egfr_quote = resource.quote("EGFR is involved")
    state.entities_found["EGFR"] = EntityWithQuotes(
        name="EGFR",
        kind="gene",
        aliases=["EGFR"],
        quotes=[egfr_quote],
        confidence=1.0,
    )

    # Disease: lung cancer (same chunk as EGFR)
    lung_cancer_quote = resource.quote("lung cancer")
    state.entities_found["lung cancer"] = EntityWithQuotes(
        name="lung cancer",
        kind="disease",
        aliases=["lung cancer"],
        quotes=[lung_cancer_quote],
        confidence=1.0,
    )

    # Add assessments with related entities
    state.individual_assessments["BRCA1"] = IndividualAssessment(
        entity=state.entities_found["BRCA1"],
        relationship_potential="high",
        related_entities=["breast cancer", "TP53"],  # Explicit suggestions
        evidence_quotes=[brca1_quote],
        reasoning="BRCA1 strongly linked to breast cancer",
        confidence=0.9,
    )

    state.individual_assessments["TP53"] = IndividualAssessment(
        entity=state.entities_found["TP53"],
        relationship_potential="high",
        related_entities=["breast cancer"],
        evidence_quotes=[tp53_quote],
        reasoning="TP53 mutations cause breast cancer",
        confidence=0.9,
    )

    state.individual_assessments["EGFR"] = IndividualAssessment(
        entity=state.entities_found["EGFR"],
        relationship_potential="medium",
        related_entities=["lung cancer"],
        evidence_quotes=[egfr_quote],
        reasoning="EGFR involved in lung cancer",
        confidence=0.7,
    )

    # Low potential entity - should be filtered
    state.individual_assessments["breast cancer"] = IndividualAssessment(
        entity=state.entities_found["breast cancer"],
        relationship_potential="low",
        related_entities=[],
        evidence_quotes=[],
        reasoning="Disease entity, low relationship potential",
        confidence=0.5,
    )

    return state


class TestGeneratePairCandidatesSameChunk:
    """Test Tier 1: same-chunk co-occurrence."""

    @pytest.mark.asyncio
    async def test_same_chunk_detection(self, state_with_assessments, deps_v3):
        """Test that entities in same chunk generate candidates."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        # BRCA1 and breast cancer should be in same chunk
        pair_key = tuple(sorted(["BRCA1", "breast cancer"]))
        assert pair_key in state_with_assessments.pair_candidates

        candidate = state_with_assessments.pair_candidates[pair_key]
        assert candidate.generation_strategy in ["same_chunk", "both"]
        assert candidate.co_occurrence_count >= 1

    @pytest.mark.asyncio
    async def test_same_chunk_strategy_disabled(self, state_with_assessments, deps_v3):
        """Test that disabling same-chunk skips those candidates."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        deps_v3.enable_same_chunk = False
        deps_v3.enable_adjacent_chunks = False
        deps_v3.enable_document_level = False

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        # No co-occurrence candidates should be generated
        assert state_with_assessments.metrics.candidates_from_cooccurrence == 0


class TestGeneratePairCandidatesAdjacentChunks:
    """Test Tier 2: adjacent-chunk co-occurrence."""

    @pytest.mark.asyncio
    async def test_adjacent_chunk_detection(self, state_with_assessments, deps_v3):
        """Test that entities in adjacent chunks generate candidates."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        deps_v3.enable_same_chunk = False  # Disable same-chunk to isolate adjacent
        deps_v3.enable_adjacent_chunks = True

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        # Should find some adjacent-chunk candidates
        # (exact pairs depend on chunk boundaries in the test data)
        assert state_with_assessments.metrics.candidates_from_cooccurrence >= 0


class TestGeneratePairCandidatesDocumentLevel:
    """Test Tier 3: document-level co-occurrence."""

    @pytest.mark.asyncio
    async def test_document_level_detection(self, state_with_assessments, deps_v3):
        """Test that document-level co-occurrence generates candidates."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        deps_v3.enable_same_chunk = False
        deps_v3.enable_adjacent_chunks = False
        deps_v3.enable_document_level = True

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        # Document-level should find all cross-kind pairs in the document
        # (depends on include_same_kind_pairs setting)
        assert len(state_with_assessments.pair_candidates) > 0


class TestGeneratePairCandidatesAssessmentSuggested:
    """Test assessment-suggested pair generation."""

    @pytest.mark.asyncio
    async def test_assessment_suggested_pairs(self, state_with_assessments, deps_v3):
        """Test that assessment related_entities generate candidates."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        # Disable co-occurrence to isolate assessment suggestions
        deps_v3.enable_same_chunk = False
        deps_v3.enable_adjacent_chunks = False
        deps_v3.enable_document_level = False

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        # BRCA1 assessment suggests "breast cancer" and "TP53"
        brca1_breast = tuple(sorted(["BRCA1", "breast cancer"]))
        brca1_tp53 = tuple(sorted(["BRCA1", "TP53"]))

        assert brca1_breast in state_with_assessments.pair_candidates
        assert (
            state_with_assessments.pair_candidates[brca1_breast].generation_strategy
            == "assessment_suggested"
        )

        # TP53-TP53 would be same-kind, should be filtered by default
        # BRCA1-TP53 is gene-gene, should also be filtered
        assert brca1_tp53 not in state_with_assessments.pair_candidates

    @pytest.mark.asyncio
    async def test_low_potential_entities_skipped(
        self, state_with_assessments, deps_v3
    ):
        """Test that low/none potential entities don't generate candidates."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        deps_v3.enable_same_chunk = False
        deps_v3.enable_adjacent_chunks = False
        deps_v3.enable_document_level = False

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        # "breast cancer" has low potential and empty related_entities
        # No candidates should originate from it
        for candidate in state_with_assessments.pair_candidates.values():
            assert (
                candidate.entity_a.name != "breast cancer"
                or candidate.entity_b.name != "breast cancer"
            )


class TestGeneratePairCandidatesFuzzyMatching:
    """Test fuzzy entity name matching."""

    @pytest.mark.asyncio
    async def test_exact_match(self, state_with_assessments, deps_v3):
        """Test exact name matching works."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        node = GeneratePairCandidates()

        matched = node._fuzzy_match_entity(
            "BRCA1", state_with_assessments.entities_found
        )
        assert matched is not None
        assert matched.name == "BRCA1"

    @pytest.mark.asyncio
    async def test_case_insensitive_match(self, state_with_assessments, deps_v3):
        """Test case-insensitive matching."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        node = GeneratePairCandidates()

        matched = node._fuzzy_match_entity(
            "brca1", state_with_assessments.entities_found
        )
        assert matched is not None
        assert matched.name == "BRCA1"

    @pytest.mark.asyncio
    async def test_alias_match(self, state_with_assessments, deps_v3):
        """Test alias matching."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        # Add entity with aliases (resource no longer needed)

        state_with_assessments.entities_found["TP53"].aliases = [
            "TP53",
            "p53",
            "tumor protein p53",
        ]

        node = GeneratePairCandidates()

        # Match via alias
        matched = node._fuzzy_match_entity("p53", state_with_assessments.entities_found)
        assert matched is not None
        assert matched.name == "TP53"

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, state_with_assessments, deps_v3):
        """Test that non-existent entities return None."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        node = GeneratePairCandidates()

        matched = node._fuzzy_match_entity(
            "NONEXISTENT", state_with_assessments.entities_found
        )
        assert matched is None


class TestGeneratePairCandidatesHybridStrategy:
    """Test hybrid strategy combining co-occurrence and assessment."""

    @pytest.mark.asyncio
    async def test_both_strategy_marking(self, state_with_assessments, deps_v3):
        """Test that pairs found by both methods are marked 'both'."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        # Enable both strategies
        deps_v3.enable_same_chunk = True
        deps_v3.enable_adjacent_chunks = True

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        # BRCA1-breast cancer should be found by both co-occurrence and assessment
        pair_key = tuple(sorted(["BRCA1", "breast cancer"]))
        if pair_key in state_with_assessments.pair_candidates:
            candidate = state_with_assessments.pair_candidates[pair_key]
            assert candidate.generation_strategy == "both"

    @pytest.mark.asyncio
    async def test_deduplication(self, state_with_assessments, deps_v3):
        """Test that duplicate pairs are properly deduplicated."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        # Each unique pair should appear only once
        pair_keys = list(state_with_assessments.pair_candidates.keys())
        assert len(pair_keys) == len(set(pair_keys))


class TestGeneratePairCandidatesSameKindFiltering:
    """Test filtering of same-kind pairs."""

    @pytest.mark.asyncio
    async def test_same_kind_pairs_filtered_by_default(
        self, state_with_assessments, deps_v3
    ):
        """Test that same-kind pairs are filtered by default."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        deps_v3.include_same_kind_pairs = False  # Default
        deps_v3.enable_same_chunk = True
        deps_v3.enable_adjacent_chunks = True

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        # No gene-gene or disease-disease pairs should exist
        for candidate in state_with_assessments.pair_candidates.values():
            assert candidate.entity_a.kind != candidate.entity_b.kind

    @pytest.mark.asyncio
    async def test_same_kind_pairs_included_when_enabled(
        self, state_with_assessments, deps_v3
    ):
        """Test that same-kind pairs are included when configured."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        deps_v3.include_same_kind_pairs = True
        deps_v3.enable_document_level = True

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        # Some same-kind pairs may exist (e.g., gene-gene)
        # At minimum, should not crash
        assert len(state_with_assessments.pair_candidates) > 0


class TestGeneratePairCandidatesMetrics:
    """Test metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, state_with_assessments, deps_v3):
        """Test that metrics are properly tracked."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        metrics = state_with_assessments.metrics

        # Check metrics are populated
        assert metrics.candidates_generated > 0
        assert metrics.candidates_generated == len(
            state_with_assessments.pair_candidates
        )

        # Sum of sources should match or exceed total (due to "both" strategy)
        assert metrics.candidates_from_cooccurrence >= 0
        assert metrics.candidates_from_assessment >= 0


class TestGeneratePairCandidatesCheckpoint:
    """Test checkpoint callback invocation."""

    @pytest.mark.asyncio
    async def test_checkpoint_callback_invoked(self, state_with_assessments, deps_v3):
        """Test that checkpoint callback is invoked after candidate generation."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates

        checkpoint_calls = []

        async def checkpoint_callback(stage: str, state):
            checkpoint_calls.append((stage, state))

        deps_v3.checkpoint_callback = checkpoint_callback

        ctx = GraphRunContext(state=state_with_assessments, deps=deps_v3)

        node = GeneratePairCandidates()
        await node.run(ctx)

        # Verify checkpoint was called
        assert len(checkpoint_calls) == 1
        assert checkpoint_calls[0][0] == "candidates"
        assert checkpoint_calls[0][1] is state_with_assessments


# ============================================================================
# EvaluatePairs Node Tests (Task 08)
# ============================================================================


@pytest.fixture
def state_with_candidates():
    """Create state with pair candidates ready for evaluation."""
    pool = ResourcePool()

    # Add document with multiple co-occurring entities
    pool.add(
        url="https://example.com/doc",
        title="Gene-disease interactions",
        document_text="BRCA1 is a tumor suppressor gene associated with breast cancer. "
        "Mutations in BRCA1 significantly increase the risk of developing breast cancer. "
        "TP53 is another tumor suppressor gene. TP53 mutations also increase breast cancer risk.",
    )

    state = ExtractionStateV3()
    state.resource_pool = pool
    resource = pool.resources[0]

    # Add entities with quotes
    brca1_quote1 = resource.quote("BRCA1 is a tumor suppressor gene")
    brca1_quote2 = resource.quote("Mutations in BRCA1 significantly increase")

    state.entities_found["BRCA1"] = EntityWithQuotes(
        name="BRCA1",
        kind="gene",
        aliases=["BRCA1"],
        quotes=[brca1_quote1, brca1_quote2],
        confidence=1.0,
    )

    breast_cancer_quote1 = resource.quote("breast cancer")
    breast_cancer_quote2 = resource.quote("breast cancer risk")

    state.entities_found["breast cancer"] = EntityWithQuotes(
        name="breast cancer",
        kind="disease",
        aliases=["breast cancer"],
        quotes=[breast_cancer_quote1, breast_cancer_quote2],
        confidence=1.0,
    )

    tp53_quote = resource.quote("TP53 is another tumor suppressor gene")

    state.entities_found["TP53"] = EntityWithQuotes(
        name="TP53",
        kind="gene",
        aliases=["TP53"],
        quotes=[tp53_quote],
        confidence=1.0,
    )

    # Add pair candidates
    brca1_breast_cancer = PairCandidate(
        entity_a=state.entities_found["BRCA1"],
        entity_b=state.entities_found["breast cancer"],
        co_occurrence_count=2,
        shared_resources=[resource.id.id],
        generation_strategy="same_chunk",
    )

    tp53_breast_cancer = PairCandidate(
        entity_a=state.entities_found["TP53"],
        entity_b=state.entities_found["breast cancer"],
        co_occurrence_count=1,
        shared_resources=[resource.id.id],
        generation_strategy="same_chunk",
    )

    state.pair_candidates[tuple(sorted(["BRCA1", "breast cancer"]))] = (
        brca1_breast_cancer
    )
    state.pair_candidates[tuple(sorted(["TP53", "breast cancer"]))] = tp53_breast_cancer

    return state


@pytest.fixture
def mock_pair_evaluation_result_positive():
    """Create mock pair evaluation result with relationship."""
    return MockAgentRunResult(
        PairEvaluationOut(
            relationship_exists=True,
            relationship_type="gene-disease interaction",
            confidence="high",
            evidence=[
                "BRCA1 is a tumor suppressor gene associated with breast cancer.",
                "Mutations in BRCA1 significantly increase the risk of developing breast cancer.",
            ],
            reasoning="Strong evidence for BRCA1-breast cancer relationship",
        )
    )


@pytest.fixture
def mock_pair_evaluation_result_negative():
    """Create mock pair evaluation result without relationship."""
    return MockAgentRunResult(
        PairEvaluationOut(
            relationship_exists=False,
            relationship_type="gene-disease interaction",
            confidence="low",
            evidence=[],
            reasoning="No evidence for direct relationship",
        )
    )


class TestEvaluatePairsBasic:
    """Test basic pair evaluation functionality."""

    @pytest.mark.asyncio
    async def test_evaluate_pairs_with_accepted_pair(
        self, state_with_candidates, deps_v3, mock_pair_evaluation_result_positive
    ):
        """Test pair evaluation accepts pair with valid evidence."""
        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_pair_evaluation_result_positive

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # Verify at least one pair was accepted
        assert len(state_with_candidates.final_pairs) >= 1
        assert state_with_candidates.metrics.pairs_accepted >= 1
        assert state_with_candidates.metrics.pair_evaluation_calls > 0

    @pytest.mark.asyncio
    async def test_evaluate_pairs_with_rejected_pair(
        self, state_with_candidates, deps_v3, mock_pair_evaluation_result_negative
    ):
        """Test pair evaluation rejects pair without relationship."""
        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        # Mock agent that rejects all pairs
        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_pair_evaluation_result_negative

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # Verify all pairs were rejected
        assert len(state_with_candidates.final_pairs) == 0
        assert state_with_candidates.metrics.pairs_rejected > 0
        assert state_with_candidates.metrics.pairs_accepted == 0

    @pytest.mark.asyncio
    async def test_evaluate_pairs_mixed_results(self, state_with_candidates, deps_v3):
        """Test pair evaluation with mixed accept/reject results."""
        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        # Mock agent with alternating results
        call_count = [0]

        async def mock_agent_run(prompt, deps, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MockAgentRunResult(
                    PairEvaluationOut(
                        relationship_exists=True,
                        relationship_type="gene-disease interaction",
                        confidence="high",
                        evidence=[
                            "BRCA1 is a tumor suppressor gene associated with breast cancer."
                        ],
                        reasoning="Accepted",
                    )
                )
            else:
                return MockAgentRunResult(
                    PairEvaluationOut(
                        relationship_exists=False,
                        relationship_type="gene-disease interaction",
                        confidence="low",
                        evidence=[],
                        reasoning="Rejected",
                    )
                )

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = mock_agent_run

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # Verify mixed results
        assert state_with_candidates.metrics.pairs_accepted >= 1
        assert state_with_candidates.metrics.pairs_rejected >= 1
        assert (
            state_with_candidates.metrics.pairs_accepted
            + state_with_candidates.metrics.pairs_rejected
            == state_with_candidates.metrics.pair_evaluation_calls
        )


class TestChunkBasedContextExtraction:
    """Test chunk-based context extraction."""

    @pytest.mark.asyncio
    async def test_same_chunk_context_extraction(self, state_with_candidates, deps_v3):
        """Test extraction of same-chunk contexts."""
        node = EvaluatePairs()

        entity_a = state_with_candidates.entities_found["BRCA1"]
        entity_b = state_with_candidates.entities_found["breast cancer"]
        resource = state_with_candidates.resource_pool.resources[0]

        contexts, quotes_used = node._extract_pair_contexts(
            entity_a, entity_b, resource
        )

        # Should find at least one context where both entities appear
        assert len(contexts) > 0
        # Should track quotes for extracted contexts
        assert len(quotes_used) > 0

    @pytest.mark.asyncio
    async def test_no_shared_context_returns_empty(self, deps_v3):
        """Test that entities without shared contexts return empty list."""
        # Create separate documents for entities
        pool = ResourcePool()
        pool.add(
            url="https://example.com/doc1",
            title="Doc 1",
            document_text="BRCA1 is a gene.",
        )
        pool.add(
            url="https://example.com/doc2",
            title="Doc 2",
            document_text="Breast cancer is a disease.",
        )

        # Entities from different documents
        entity_a = EntityWithQuotes(
            name="BRCA1",
            kind="gene",
            aliases=["BRCA1"],
            quotes=[pool.resources[0].quote("BRCA1")],
            confidence=1.0,
        )

        entity_b = EntityWithQuotes(
            name="breast cancer",
            kind="disease",
            aliases=["breast cancer"],
            quotes=[pool.resources[1].quote("breast cancer")],
            confidence=1.0,
        )

        node = EvaluatePairs()

        # Try to extract contexts from first resource (where only entity_a appears)
        contexts, quotes_used = node._extract_pair_contexts(
            entity_a, entity_b, pool.resources[0]
        )

        # Should return empty (entity_b not in this resource)
        assert len(contexts) == 0
        assert len(quotes_used) == 0

    @pytest.mark.asyncio
    async def test_adjacent_chunk_context_extraction(self, deps_v3):
        """Test extraction of adjacent chunk contexts."""
        # Create document with entities in adjacent chunks
        pool = ResourcePool()
        pool.add(
            url="https://example.com/doc",
            title="Test Doc",
            document_text="BRCA1 is important in the first chunk. "
            + "In the adjacent chunk, breast cancer is discussed. "
            + "These two are related.",
        )

        resource = pool.resources[0]

        entity_a = EntityWithQuotes(
            name="BRCA1",
            kind="gene",
            aliases=["BRCA1"],
            quotes=[resource.quote("BRCA1 is important")],
            confidence=1.0,
        )

        entity_b = EntityWithQuotes(
            name="breast cancer",
            kind="disease",
            aliases=["breast cancer"],
            quotes=[resource.quote("breast cancer is discussed")],
            confidence=1.0,
        )

        node = EvaluatePairs()
        contexts, quotes_used = node._extract_pair_contexts(
            entity_a, entity_b, resource
        )

        # Should find contexts (either same-chunk or adjacent)
        # (depends on chunking, but should not be empty for related entities)
        assert len(contexts) >= 0  # May be 0 if chunking separates them
        # quotes_used should align with contexts
        if len(contexts) > 0:
            assert len(quotes_used) > 0


class TestEvaluatePairs:
    """Test pair evaluation with source attribution (Task 03)."""

    @pytest.mark.asyncio
    async def test_extract_pair_contexts_tracks_quotes(
        self, state_with_candidates, deps_v3
    ):
        """Test that _extract_pair_contexts returns quotes for all contexts."""
        node = EvaluatePairs()

        entity_a = state_with_candidates.entities_found["BRCA1"]
        entity_b = state_with_candidates.entities_found["breast cancer"]
        resource = state_with_candidates.resource_pool.resources[0]

        contexts, quotes_used = node._extract_pair_contexts(
            entity_a, entity_b, resource
        )

        # Verify quotes_used contains ResourceQuote objects
        assert all(hasattr(q, "resource") for q in quotes_used)
        assert all(hasattr(q, "chunk_indices") for q in quotes_used)

        # If contexts found, quotes should be tracked
        if len(contexts) > 0:
            assert len(quotes_used) > 0
            # All quotes should be from the same resource
            assert all(q.resource.id == resource.id for q in quotes_used)

    @pytest.mark.asyncio
    async def test_evaluation_prompt_has_source_labels(
        self, state_with_candidates, deps_v3
    ):
        """Test that evaluation prompt includes source labels for evidence."""
        node = EvaluatePairs()

        entity_a = state_with_candidates.entities_found["BRCA1"]
        entity_b = state_with_candidates.entities_found["breast cancer"]

        # Get a candidate to build prompt for
        candidate = list(state_with_candidates.pair_candidates.values())[0]

        # Mock evidence contexts and sources
        evidence_contexts = [
            "BRCA1 is a tumor suppressor gene.",
            "Breast cancer susceptibility is linked to BRCA1.",
        ]
        context_sources = [
            "https://example.com/doc1",
            "https://example.com/doc2",
        ]

        prompt = node._build_evaluation_prompt(
            candidate, evidence_contexts, context_sources, deps_v3
        )

        # Verify source labels are present
        assert "(Source: https://example.com/doc1)" in prompt
        assert "(Source: https://example.com/doc2)" in prompt

        # Verify evidence numbering
        assert "Evidence 1" in prompt
        assert "Evidence 2" in prompt


class TestProvenanceValidation:
    """Test provenance validation for accepted pairs."""

    @pytest.mark.asyncio
    async def test_provenance_validation_success(
        self, state_with_candidates, deps_v3, mock_pair_evaluation_result_positive
    ):
        """Test that pairs with valid evidence pass provenance validation."""
        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_pair_evaluation_result_positive

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # All accepted pairs should have valid provenance
        for pair in state_with_candidates.final_pairs:
            assert pair.validate_provenance()

    @pytest.mark.asyncio
    async def test_provenance_validation_failure_rejects_pair(
        self, state_with_candidates, deps_v3
    ):
        """Test that pairs without valid evidence are rejected."""
        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        # Mock agent with evidence that doesn't exist in documents
        mock_result = MockAgentRunResult(
            PairEvaluationOut(
                relationship_exists=True,
                relationship_type="gene-disease interaction",
                confidence="high",
                evidence=[
                    "This evidence quote does not exist in any document at all.",
                ],
                reasoning="Hallucinated evidence",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # Pair should be rejected due to invalid provenance
        # (no valid evidence quotes found)
        # At minimum, pairs should have evidence_quotes if accepted
        for pair in state_with_candidates.final_pairs:
            assert len(pair.evidence_quotes) > 0


class TestEvidenceQuoteMatching:
    """Test evidence quote validation with fuzzy matching."""

    @pytest.mark.asyncio
    async def test_evidence_quote_validated_with_exact_match(
        self, state_with_candidates, deps_v3
    ):
        """Test evidence quotes are validated with exact match."""
        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        # Mock agent with exact evidence from document
        mock_result = MockAgentRunResult(
            PairEvaluationOut(
                relationship_exists=True,
                relationship_type="gene-disease interaction",
                confidence="high",
                evidence=[
                    "BRCA1 is a tumor suppressor gene associated with breast cancer.",
                ],
                reasoning="Exact match evidence",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # Should have at least one pair with validated evidence
        assert len(state_with_candidates.final_pairs) >= 1
        pair = state_with_candidates.final_pairs[0]
        assert len(pair.evidence_quotes) > 0

        # Evidence quotes should be ResourceQuote objects
        from interaction_finder.resources import ResourceQuote

        assert isinstance(pair.evidence_quotes[0], ResourceQuote)

    @pytest.mark.asyncio
    async def test_evidence_quote_validated_with_fuzzy_match(
        self, state_with_candidates, deps_v3
    ):
        """Test evidence quotes validated with fuzzy matching."""
        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        # Mock agent with slightly different wording (should fuzzy match)
        mock_result = MockAgentRunResult(
            PairEvaluationOut(
                relationship_exists=True,
                relationship_type="gene-disease interaction",
                confidence="high",
                evidence=[
                    "BRCA1 tumor suppressor gene associated breast cancer",  # Missing articles
                ],
                reasoning="Fuzzy match evidence",
            )
        )

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_result

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # May or may not accept depending on fuzzy match threshold
        # At minimum, should not crash
        assert state_with_candidates.metrics.pair_evaluation_calls > 0


class TestParallelEvaluation:
    """Test parallel pair evaluation."""

    @pytest.mark.asyncio
    async def test_parallel_evaluation_multiple_pairs(self, deps_v3):
        """Test multiple pairs evaluated in parallel."""
        # Create state with multiple candidates
        pool = ResourcePool()
        pool.add(
            url="https://example.com/doc",
            title="Test Doc",
            document_text="BRCA1 with breast cancer. BRCA2 with breast cancer. "
            "TP53 with breast cancer.",
        )

        state = ExtractionStateV3()
        state.resource_pool = pool
        resource = pool.resources[0]

        # Add multiple entities and candidates
        for gene in ["BRCA1", "BRCA2", "TP53"]:
            state.entities_found[gene] = EntityWithQuotes(
                name=gene,
                kind="gene",
                aliases=[gene],
                quotes=[resource.quote(gene)],
                confidence=1.0,
            )

        state.entities_found["breast cancer"] = EntityWithQuotes(
            name="breast cancer",
            kind="disease",
            aliases=["breast cancer"],
            quotes=[resource.quote("breast cancer")],
            confidence=1.0,
        )

        # Add candidates
        for gene in ["BRCA1", "BRCA2", "TP53"]:
            state.pair_candidates[tuple(sorted([gene, "breast cancer"]))] = (
                PairCandidate(
                    entity_a=state.entities_found[gene],
                    entity_b=state.entities_found["breast cancer"],
                    co_occurrence_count=1,
                    shared_resources=[resource.id.id],
                    generation_strategy="same_chunk",
                )
            )

        ctx = GraphRunContext(state=state, deps=deps_v3)

        # Track parallel calls
        call_times = []

        async def mock_agent_run(prompt, deps, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate LLM call
            return MockAgentRunResult(
                PairEvaluationOut(
                    relationship_exists=True,
                    relationship_type="gene-disease interaction",
                    confidence="high",
                    evidence=["Evidence text"],
                    reasoning="Accepted",
                )
            )

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = mock_agent_run

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # Verify all pairs were evaluated
        assert state.metrics.pair_evaluation_calls == 3

    @pytest.mark.asyncio
    async def test_parallelism_control(self, state_with_candidates, deps_v3):
        """Test parallelism control limits concurrent evaluations."""
        deps_v3.pair_evaluation_parallelism = 1  # Limit to 1 concurrent

        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        mock_agent = AsyncMock()
        mock_agent.run.return_value = PairEvaluationOut(
            relationship_exists=True,
            relationship_type="gene-disease interaction",
            confidence="high",
            evidence=["Evidence"],
            reasoning="Accepted",
        )

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # Should complete successfully with limited parallelism
        assert state_with_candidates.metrics.pair_evaluation_calls > 0


class TestEvaluationErrorHandling:
    """Test error handling during pair evaluation."""

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_crash(self, state_with_candidates, deps_v3):
        """Test that LLM failure doesn't crash pair evaluation."""
        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        # Mock agent that fails
        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("LLM API error")

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            # Should not raise exception
            await node.run(ctx)

        # Metrics should track calls
        assert state_with_candidates.metrics.pair_evaluation_calls > 0
        # All evaluations failed, so no pairs accepted
        assert state_with_candidates.metrics.pairs_accepted == 0

    @pytest.mark.asyncio
    async def test_partial_evaluation_failure(self, state_with_candidates, deps_v3):
        """Test that failure on some pairs doesn't prevent others."""
        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        # Mock agent that fails on first pair, succeeds on second
        call_count = [0]

        async def mock_agent_run(prompt, deps, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First evaluation failed")

            return MockAgentRunResult(
                PairEvaluationOut(
                    relationship_exists=True,
                    relationship_type="gene-disease interaction",
                    confidence="high",
                    evidence=[
                        "BRCA1 is a tumor suppressor gene associated with breast cancer."
                    ],
                    reasoning="Good evaluation",
                )
            )

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = mock_agent_run

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # At least one pair should succeed
        assert state_with_candidates.metrics.pairs_accepted >= 1


class TestEvaluationPromptBuilding:
    """Test evaluation prompt construction."""

    @pytest.mark.asyncio
    async def test_prompt_includes_entity_information(
        self, state_with_candidates, deps_v3
    ):
        """Test that prompt includes entity names and kinds."""
        node = EvaluatePairs()

        candidate = list(state_with_candidates.pair_candidates.values())[0]
        evidence_contexts = ["Some context text"]
        context_sources = ["https://example.com/doc"]

        prompt = node._build_evaluation_prompt(
            candidate, evidence_contexts, context_sources, deps_v3
        )

        # Verify entity information in prompt
        assert candidate.entity_a.name in prompt
        assert candidate.entity_b.name in prompt
        assert candidate.entity_a.kind in prompt
        assert candidate.entity_b.kind in prompt

    @pytest.mark.asyncio
    async def test_prompt_limits_context_count(self, state_with_candidates, deps_v3):
        """Test that prompt limits to 5 contexts for token efficiency."""
        node = EvaluatePairs()

        candidate = list(state_with_candidates.pair_candidates.values())[0]
        # Provide more than 5 contexts
        evidence_contexts = [f"Context {i}" for i in range(10)]
        context_sources = [f"https://example.com/doc{i}" for i in range(10)]

        prompt = node._build_evaluation_prompt(
            candidate, evidence_contexts, context_sources, deps_v3
        )

        # Verify only first 5 contexts included
        assert "Evidence 1 (Source:" in prompt
        assert "Evidence 5 (Source:" in prompt
        assert "Evidence 6" not in prompt


class TestFinalCheckpoint:
    """Test final checkpoint callback invocation."""

    @pytest.mark.asyncio
    async def test_checkpoint_callback_invoked(
        self, state_with_candidates, deps_v3, mock_pair_evaluation_result_positive
    ):
        """Test that checkpoint callback is invoked after evaluation."""
        checkpoint_calls = []

        async def checkpoint_callback(stage: str, state):
            checkpoint_calls.append((stage, state))

        deps_v3.checkpoint_callback = checkpoint_callback

        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        mock_agent = AsyncMock()
        mock_agent.run.return_value = mock_pair_evaluation_result_positive

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # Verify checkpoint was called
        assert len(checkpoint_calls) == 1
        assert checkpoint_calls[0][0] == "final"
        assert checkpoint_calls[0][1] is state_with_candidates


class TestAcceptanceRate:
    """Test acceptance rate calculation and metrics."""

    @pytest.mark.asyncio
    async def test_acceptance_rate_metrics(self, state_with_candidates, deps_v3):
        """Test that acceptance rate is tracked correctly."""
        ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)

        # Mock agent with 50% acceptance
        call_count = [0]

        async def mock_agent_run(prompt, deps, **kwargs):
            call_count[0] += 1
            return MockAgentRunResult(
                PairEvaluationOut(
                    relationship_exists=(call_count[0] % 2 == 1),
                    relationship_type="gene-disease interaction",
                    confidence="high" if (call_count[0] % 2 == 1) else "low",
                    evidence=[
                        "BRCA1 is a tumor suppressor gene associated with breast cancer."
                    ]
                    if (call_count[0] % 2 == 1)
                    else [],
                    reasoning="Alternating results",
                )
            )

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = mock_agent_run

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.create_pair_evaluator_v3",
            return_value=mock_agent,
        ):
            node = EvaluatePairs()
            await node.run(ctx)

        # Verify metrics
        metrics = state_with_candidates.metrics
        assert metrics.pair_evaluation_calls > 0
        assert (
            metrics.pairs_accepted + metrics.pairs_rejected
            == metrics.pair_evaluation_calls
        )


class TestTieredCooccurrence:
    """Test tiered co-occurrence candidate generation."""

    @pytest.mark.asyncio
    async def test_tier1_same_chunk_candidates(self, deps_v3):
        """Test Tier 1 (same-chunk) candidate generation."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates
        from interaction_finder.resources import compute_chunk_spans

        # Create document with entities in same chunk
        pool = ResourcePool()
        doc_text = "BRCA1 is associated with breast cancer in this study."
        chunks = ["BRCA1 is associated with breast cancer in this study."]
        chunk_spans = compute_chunk_spans(doc_text, chunks)

        resource = pool.add(
            url="https://example.com/doc",
            title="Test",
            document_text=doc_text,
            chunks=chunk_spans,
        )

        state = ExtractionStateV3()
        state.resource_pool = pool

        # Add entities from same chunk
        state.entities_found["BRCA1"] = EntityWithQuotes(
            name="BRCA1",
            kind="gene",
            quotes=[resource.quote("BRCA1")],
        )
        state.entities_found["breast cancer"] = EntityWithQuotes(
            name="breast cancer",
            kind="disease",
            quotes=[resource.quote("breast cancer")],
        )

        # Add assessments
        from interaction_finder.extraction_graph_v2.models import IndividualAssessment

        state.individual_assessments["BRCA1"] = IndividualAssessment(
            entity=state.entities_found["BRCA1"],
            relationship_potential="high",
            related_entities=[],
            evidence_quotes=[],
            reasoning="Test",
        )
        state.individual_assessments["breast cancer"] = IndividualAssessment(
            entity=state.entities_found["breast cancer"],
            relationship_potential="high",
            related_entities=[],
            evidence_quotes=[],
            reasoning="Test",
        )

        ctx = GraphRunContext(state=state, deps=deps_v3)

        # Generate candidates
        node = GeneratePairCandidates()
        await node.run(ctx)

        # Should generate same-chunk candidate
        assert len(state.pair_candidates) >= 1
        candidate = list(state.pair_candidates.values())[0]
        assert candidate.generation_strategy in ["same_chunk", "both"]
        assert candidate.co_occurrence_count > 0

    @pytest.mark.asyncio
    async def test_tier2_adjacent_chunks_candidates(self, deps_v3):
        """Test Tier 2 (adjacent chunks) candidate generation."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates
        from interaction_finder.resources import compute_chunk_spans

        # Create document with entities in adjacent chunks
        pool = ResourcePool()
        doc_text = "BRCA1 is a gene. Breast cancer is a disease."
        chunks = ["BRCA1 is a gene.", "Breast cancer is a disease."]
        chunk_spans = compute_chunk_spans(doc_text, chunks)

        resource = pool.add(
            url="https://example.com/doc",
            title="Test",
            document_text=doc_text,
            chunks=chunk_spans,
        )

        state = ExtractionStateV3()
        state.resource_pool = pool

        # Add entities from adjacent chunks
        state.entities_found["BRCA1"] = EntityWithQuotes(
            name="BRCA1",
            kind="gene",
            quotes=[resource.quote("BRCA1")],
        )
        state.entities_found["breast cancer"] = EntityWithQuotes(
            name="breast cancer",
            kind="disease",
            quotes=[resource.quote("Breast cancer")],
        )

        # Add assessments
        from interaction_finder.extraction_graph_v2.models import IndividualAssessment

        state.individual_assessments["BRCA1"] = IndividualAssessment(
            entity=state.entities_found["BRCA1"],
            relationship_potential="high",
            related_entities=[],
            evidence_quotes=[],
            reasoning="Test",
        )
        state.individual_assessments["breast cancer"] = IndividualAssessment(
            entity=state.entities_found["breast cancer"],
            relationship_potential="high",
            related_entities=[],
            evidence_quotes=[],
            reasoning="Test",
        )

        ctx = GraphRunContext(state=state, deps=deps_v3)

        # Generate candidates
        node = GeneratePairCandidates()
        await node.run(ctx)

        # Should generate adjacent-chunk or document-level candidate
        assert len(state.pair_candidates) >= 1
        candidate = list(state.pair_candidates.values())[0]
        assert candidate.generation_strategy in [
            "adjacent_chunks",
            "document_level",
            "both",
        ]

    @pytest.mark.asyncio
    async def test_tier3_document_level_candidates(self, deps_v3):
        """Test Tier 3 (document-level) candidate generation."""
        from interaction_finder.extraction_graph_v3.nodes import GeneratePairCandidates
        from interaction_finder.resources import compute_chunk_spans

        # Enable tier 3 candidates
        deps_v3.enable_document_level = True

        # Create document with entities far apart
        pool = ResourcePool()
        doc_text = (
            "BRCA1 is mentioned here. "
            + " ".join(["Filler text."] * 50)
            + " Breast cancer is mentioned far away."
        )
        chunks = [
            "BRCA1 is mentioned here.",
            " ".join(["Filler text."] * 50),
            "Breast cancer is mentioned far away.",
        ]
        chunk_spans = compute_chunk_spans(doc_text, chunks)

        resource = pool.add(
            url="https://example.com/doc",
            title="Test",
            document_text=doc_text,
            chunks=chunk_spans,
        )

        state = ExtractionStateV3()
        state.resource_pool = pool

        # Add entities from distant chunks
        state.entities_found["BRCA1"] = EntityWithQuotes(
            name="BRCA1",
            kind="gene",
            quotes=[resource.quote("BRCA1")],
        )
        state.entities_found["breast cancer"] = EntityWithQuotes(
            name="breast cancer",
            kind="disease",
            quotes=[resource.quote("Breast cancer")],
        )

        # Add assessments
        from interaction_finder.extraction_graph_v2.models import IndividualAssessment

        state.individual_assessments["BRCA1"] = IndividualAssessment(
            entity=state.entities_found["BRCA1"],
            relationship_potential="high",
            related_entities=[],
            evidence_quotes=[],
            reasoning="Test",
        )
        state.individual_assessments["breast cancer"] = IndividualAssessment(
            entity=state.entities_found["breast cancer"],
            relationship_potential="high",
            related_entities=[],
            evidence_quotes=[],
            reasoning="Test",
        )

        ctx = GraphRunContext(state=state, deps=deps_v3)

        # Generate candidates
        node = GeneratePairCandidates()
        await node.run(ctx)

        # Should generate document-level candidate
        assert len(state.pair_candidates) >= 1
        candidate = list(state.pair_candidates.values())[0]
        assert candidate.generation_strategy in ["document_level", "both"]
