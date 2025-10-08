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


# ============================================================================
# AssessIndividually Node Tests
# ============================================================================


from interaction_finder.extraction_graph_v3.nodes import AssessIndividually
from interaction_finder.extraction_graph_v2.models import (
    AssessmentOut,
    IndividualAssessment,
)


@pytest.fixture
def mock_assessment_result():
    """Create mock assessment agent result."""
    return AssessmentOut(
        potential="high",
        related=["breast cancer", "ovarian cancer"],
        evidence=[
            "BRCA1 is a tumor suppressor gene associated with breast cancer.",
            "BRCA1 mutations are also linked to ovarian cancer.",
        ],
        reasoning="BRCA1 shows high relationship potential with multiple cancers",
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
        mock_result = AssessmentOut(
            potential="high",
            related=["breast cancer", "ovarian cancer", "TP53"],
            evidence=["BRCA1 interacts with these entities"],
            reasoning="High potential for relationships",
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
        mock_result = AssessmentOut(
            potential="low",
            related=[],
            evidence=[],
            reasoning="Low potential for relationships",
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
        mock_result = AssessmentOut(
            potential="high",
            related=["breast cancer"],
            evidence=[
                "BRCA1 is a tumor suppressor gene associated with breast cancer.",
            ],
            reasoning="Strong evidence for relationship",
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
        mock_result = AssessmentOut(
            potential="high",
            related=["breast cancer"],
            evidence=[
                "This evidence quote does not exist in any resource document.",
            ],
            reasoning="Some reasoning",
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


class TestConfidenceMapping:
    """Test mapping of potential levels to confidence scores."""

    @pytest.mark.asyncio
    async def test_confidence_mapping_high(self, state_with_entities, deps_v3):
        """Test high potential maps to 0.9 confidence."""
        ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)

        mock_result = AssessmentOut(
            potential="high",
            related=["breast cancer"],
            evidence=[],
            reasoning="High potential",
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

        mock_result = AssessmentOut(
            potential="medium",
            related=["breast cancer"],
            evidence=[],
            reasoning="Medium potential",
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

        mock_result = AssessmentOut(
            potential="low",
            related=[],
            evidence=[],
            reasoning="Low potential",
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

        mock_result = AssessmentOut(
            potential="none",
            related=[],
            evidence=[],
            reasoning="No potential",
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

        async def mock_agent_run(prompt, deps):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate LLM call
            return AssessmentOut(
                potential="high",
                related=["breast cancer"],
                evidence=[],
                reasoning="Gene with cancer relationship",
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

        async def mock_agent_run(prompt, deps):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First assessment failed")

            return AssessmentOut(
                potential="high",
                related=["breast cancer"],
                evidence=[],
                reasoning="Good assessment",
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
