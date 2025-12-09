"""Tests for assess_document_pairs deduplication and progress tracking."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from interaction_finder.extraction.document_pipeline import assess_document_pairs
from interaction_finder.extraction.models import (
    EntityMention,
    EntityRef,
    PairEvidenceJudgment,
)
from interaction_finder.extraction.progress import create_extraction_progress
from interaction_finder.resources import ResourcePool, ResourceQuote


def make_entity_ref(name: str, kind: str) -> EntityRef:
    """Create a minimal EntityRef for testing."""
    return EntityRef(
        canonical=name,
        mentions=[
            EntityMention(
                kind=kind,
                name=name,
                aliases=[],
                quotes=[],
                reasoning="Test entity",
            )
        ],
    )


@pytest.fixture
def resource_pool():
    """Create a resource pool with a test document."""
    pool = ResourcePool()
    pool.add(
        url="https://example.com/doc1",
        title="Test Document",
        document_text="BRCA1 interacts with TP53. BRCA1 causes Cancer progression.",
        chunks=[(0, 30), (30, 60)],
    )
    return pool


def make_quote(resource, text: str) -> ResourceQuote:
    """Create a ResourceQuote for testing (text must exist in resource)."""
    return ResourceQuote(resource, text=text)


@pytest.fixture
def mock_deps(resource_pool):
    """Create mock dependencies."""
    deps = MagicMock()
    deps.config = MagicMock()
    deps.logger = MagicMock()
    deps.resource_pool = resource_pool
    deps.progress = None
    deps.agent_semaphore = asyncio.Semaphore(10)
    return deps


@pytest.fixture
def entities():
    """Create test entities."""
    return {
        "BRCA1": make_entity_ref("BRCA1", "gene"),
        "TP53": make_entity_ref("TP53", "gene"),
        "Cancer": make_entity_ref("Cancer", "disease"),
    }


def create_mock_agent(judgment: PairEvidenceJudgment):
    """Create a mock agent that returns the given judgment."""
    mock_result = MagicMock()
    mock_result.output = judgment
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_result)
    mock_agent._name = "mock_pair_judge"
    return mock_agent


class TestAssessDocumentPairsDeduplication:
    """Test that duplicate pairs are merged before assessment."""

    @pytest.mark.asyncio
    async def test_duplicate_pairs_assessed_once(
        self, resource_pool, mock_deps, entities
    ):
        """Same entity pair appearing multiple times should only be assessed once."""
        resource = list(resource_pool.resources)[0]
        quote1 = make_quote(resource, "BRCA1 interacts with TP53")
        quote2 = make_quote(resource, "BRCA1 causes Cancer")
        # Create duplicate pairs (same entities, different occurrences)
        pairs = [
            ("BRCA1", "TP53", {"interacts_with"}, [quote1]),
            ("BRCA1", "TP53", {"binds_to"}, [quote1]),  # Duplicate pair
            ("TP53", "BRCA1", {"regulates"}, [quote2]),  # Same pair, reversed order
        ]
        judgment = PairEvidenceJudgment(
            relationship="interacts_with",
            confidence="high",
            reasoning="Strong evidence of interaction between these genes.",
            supporting_quote_ids=[0],
        )
        mock_agent = create_mock_agent(judgment)
        with patch(
            "interaction_finder.extraction.document_pipeline.get_pair_judge_agent",
            return_value=mock_agent,
        ):
            assessments = await assess_document_pairs(
                pairs,
                entities,
                resource,
                topic="gene interactions",
                region_padding_chunks=1,
                config=mock_deps.config,
                deps=mock_deps,
            )
        # Should have exactly 1 assessment (all 3 pairs are the same entity pair)
        assert len(assessments) == 1
        # Agent should have been called exactly once
        assert mock_agent.run.call_count == 1

    @pytest.mark.asyncio
    async def test_distinct_pairs_assessed_separately(
        self, resource_pool, mock_deps, entities
    ):
        """Different entity pairs should each be assessed."""
        resource = list(resource_pool.resources)[0]
        quote1 = make_quote(resource, "BRCA1 interacts with TP53")
        quote2 = make_quote(resource, "BRCA1 causes Cancer")
        # Create two distinct pairs
        pairs = [
            ("BRCA1", "TP53", {"interacts_with"}, [quote1]),
            ("Cancer", "BRCA1", {"causes"}, [quote2]),
        ]
        judgment = PairEvidenceJudgment(
            relationship="associated_with",
            confidence="medium",
            reasoning="Evidence supports association between entities.",
            supporting_quote_ids=[0],
        )
        mock_agent = create_mock_agent(judgment)
        with patch(
            "interaction_finder.extraction.document_pipeline.get_pair_judge_agent",
            return_value=mock_agent,
        ):
            assessments = await assess_document_pairs(
                pairs,
                entities,
                resource,
                topic="gene-disease associations",
                region_padding_chunks=1,
                config=mock_deps.config,
                deps=mock_deps,
            )
        # Should have 2 assessments (distinct pairs)
        assert len(assessments) == 2
        # Agent should have been called twice
        assert mock_agent.run.call_count == 2

    @pytest.mark.asyncio
    async def test_relationship_candidates_merged(
        self, resource_pool, mock_deps, entities
    ):
        """Duplicate pairs should have their relationship candidates merged."""
        resource = list(resource_pool.resources)[0]
        quote = make_quote(resource, "BRCA1 interacts with TP53")
        # Same pair with different relationship candidates
        pairs = [
            ("BRCA1", "TP53", {"interacts_with"}, [quote]),
            ("BRCA1", "TP53", {"regulates"}, [quote]),
        ]
        captured_prompts = []

        def capture_prompt(*args, **kwargs):
            if args:
                captured_prompts.append(args[0])
            result = MagicMock()
            result.output = PairEvidenceJudgment(
                relationship="interacts_with",
                confidence="high",
                reasoning="Evidence supports interaction between these genes.",
                supporting_quote_ids=[0],
            )
            return result

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=capture_prompt)
        mock_agent._name = "mock_pair_judge"
        with patch(
            "interaction_finder.extraction.document_pipeline.get_pair_judge_agent",
            return_value=mock_agent,
        ):
            await assess_document_pairs(
                pairs,
                entities,
                resource,
                topic="gene interactions",
                region_padding_chunks=1,
                config=mock_deps.config,
                deps=mock_deps,
            )
        # Should be called once with merged candidates
        assert mock_agent.run.call_count == 1
        # Check that all relationship candidates appear in the prompt
        prompt = captured_prompts[0]
        assert "interacts_with" in prompt
        assert "regulates" in prompt


class TestAssessDocumentPairsProgress:
    """Test that progress counter is updated correctly after deduplication."""

    @pytest.mark.asyncio
    async def test_progress_total_reflects_deduplicated_count(
        self, resource_pool, mock_deps, entities
    ):
        """Progress total should be set to deduplicated pair count, not raw count."""
        resource = list(resource_pool.resources)[0]
        quote1 = make_quote(resource, "BRCA1 interacts with TP53")
        quote2 = make_quote(resource, "BRCA1 causes Cancer")
        # 5 raw pairs that deduplicate to 2 unique pairs
        pairs = [
            ("BRCA1", "TP53", {"interacts_with"}, [quote1]),
            ("BRCA1", "TP53", {"binds"}, [quote1]),  # Duplicate
            ("TP53", "BRCA1", {"regulates"}, [quote1]),  # Duplicate (reversed)
            ("BRCA1", "Cancer", {"causes"}, [quote2]),
            ("Cancer", "BRCA1", {"associated_with"}, [quote2]),  # Duplicate (reversed)
        ]
        # Set up progress tracking
        progress = create_extraction_progress()
        mock_deps.progress = progress
        initial_total = progress["Pairs assessed"].total
        judgment = PairEvidenceJudgment(
            relationship="associated_with",
            confidence="medium",
            reasoning="Evidence supports association between these entities.",
            supporting_quote_ids=[0],
        )
        mock_agent = create_mock_agent(judgment)
        with patch(
            "interaction_finder.extraction.document_pipeline.get_pair_judge_agent",
            return_value=mock_agent,
        ):
            assessments = await assess_document_pairs(
                pairs,
                entities,
                resource,
                topic="gene interactions",
                region_padding_chunks=1,
                config=mock_deps.config,
                deps=mock_deps,
            )
        # Should have 2 unique pairs
        assert len(assessments) == 2
        # Progress total should reflect deduplicated count (2), not raw count (5)
        assert progress["Pairs assessed"].total == initial_total + 2
        # Completed should match total (all pairs assessed)
        assert progress["Pairs assessed"].completed == 2
        # In-progress should be 0 (all done)
        assert progress["Pairs assessed"].in_progress == 0
