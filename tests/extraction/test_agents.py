"""Tests for extraction agents.

Tests agent configuration and output schema compliance using TestModel.
"""

import pytest
from pydantic_ai.models.test import TestModel

from interaction_finder.extraction.assess_entity import get_entity_assessor_agent
from interaction_finder.extraction.assess_pair import get_pair_assessor_agent
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.extract import get_entity_extractor_agent
from interaction_finder.extraction.extract_pairs import get_pair_extractor_agent
from interaction_finder.extraction.judge import get_final_judge_agent
from interaction_finder.extraction.models import (
    EntityEvidenceAssessment,
    EntityExtractionOut,
    FinalJudgmentOut,
    PairEvidenceAssessment,
    PairExtractionOut,
)


class TestEntityExtractorAgent:
    """Tests for entity_extractor_agent configuration and output."""

    def test_agent_configuration(self):
        """Test agent has correct configuration."""
        agent = get_entity_extractor_agent()
        assert agent._deps_type == Deps
        # Note: pydantic-ai agents don't expose output_type directly
        assert agent._system_prompts is not None

    @pytest.mark.asyncio
    async def test_agent_returns_correct_type(self):
        """Test agent returns EntityExtractionOut."""
        test_model = TestModel()
        agent = get_entity_extractor_agent()
        with agent.override(model=test_model):
            result = await agent.run(
                "Extract entities from: BRCA1 is associated with breast cancer.",
                deps=None,
            )
            assert isinstance(result.output, EntityExtractionOut)
            assert isinstance(result.output.entities, dict)
            assert isinstance(result.output.reasoning, str)


class TestPairExtractorAgent:
    """Tests for pair_extractor_agent configuration and output."""

    def test_agent_configuration(self):
        """Test agent has correct configuration."""
        agent = get_pair_extractor_agent()
        assert agent._deps_type == Deps
        assert agent._system_prompts is not None

    @pytest.mark.asyncio
    async def test_agent_returns_correct_type(self):
        """Test agent returns PairExtractionOut."""
        test_model = TestModel()
        agent = get_pair_extractor_agent()
        with agent.override(model=test_model):
            result = await agent.run(
                "Extract pairs from: BRCA1 is associated with breast cancer.", deps=None
            )
            assert isinstance(result.output, PairExtractionOut)
            assert isinstance(result.output.pairs, list)
            assert isinstance(result.output.reasoning, str)


class TestEntityAssessorAgent:
    """Tests for entity_assessor_agent configuration and output."""

    def test_agent_configuration(self):
        """Test agent has correct configuration."""
        agent = get_entity_assessor_agent()
        assert agent._deps_type == Deps
        assert agent._system_prompts is not None

    @pytest.mark.asyncio
    async def test_agent_returns_correct_type(self):
        """Test agent returns EntityEvidenceAssessment."""
        test_model = TestModel()
        agent = get_entity_assessor_agent()
        with agent.override(model=test_model):
            result = await agent.run(
                "Assess: BRCA1 mentioned in context of breast cancer.", deps=None
            )
            assert isinstance(result.output, EntityEvidenceAssessment)
            assert result.output.strength in ["none", "weak", "strong"]
            assert isinstance(result.output.rationale, str)
            assert isinstance(result.output.supporting_quote_ids, list)


class TestPairAssessorAgent:
    """Tests for pair_assessor_agent configuration and output."""

    def test_agent_configuration(self):
        """Test agent has correct configuration."""
        agent = get_pair_assessor_agent()
        assert agent._deps_type == Deps
        assert agent._system_prompts is not None

    @pytest.mark.asyncio
    async def test_agent_returns_correct_type(self):
        """Test agent returns PairEvidenceAssessment."""
        test_model = TestModel()
        agent = get_pair_assessor_agent()
        with agent.override(model=test_model):
            result = await agent.run(
                "Assess: BRCA1 associated with breast cancer.", deps=None
            )
            assert isinstance(result.output, PairEvidenceAssessment)
            assert result.output.strength in ["none", "weak", "strong"]
            assert isinstance(result.output.rationale, str)
            assert isinstance(result.output.supporting_quote_ids, list)


class TestFinalJudgeAgent:
    """Tests for final_judge_agent configuration and output."""

    def test_agent_configuration(self):
        """Test agent has correct configuration."""
        agent = get_final_judge_agent()
        assert agent._deps_type == Deps
        assert agent._system_prompts is not None

    @pytest.mark.asyncio
    async def test_agent_returns_correct_type(self):
        """Test agent returns FinalJudgmentOut."""
        test_model = TestModel()
        agent = get_final_judge_agent()
        with agent.override(model=test_model):
            result = await agent.run(
                "Judge: BRCA1-breast cancer association across documents.", deps=None
            )
            assert isinstance(result.output, FinalJudgmentOut)
            assert isinstance(result.output.accepted, bool)
            assert result.output.confidence in ["high", "medium", "low"]
            assert isinstance(result.output.rationale, str)
