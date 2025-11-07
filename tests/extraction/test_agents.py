"""Tests for extraction agents.

Tests agent configuration and output schema compliance using TestModel.
"""

import pytest
from pydantic_ai.models.test import TestModel

from interaction_finder.extraction.assess_entity import entity_assessor_agent
from interaction_finder.extraction.assess_pair import pair_assessor_agent
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.extract import entity_extractor_agent
from interaction_finder.extraction.extract_pairs import pair_extractor_agent
from interaction_finder.extraction.judge import final_judge_agent
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
        assert entity_extractor_agent._deps_type == Deps
        # Note: pydantic-ai agents don't expose output_type directly
        assert entity_extractor_agent._system_prompts is not None

    @pytest.mark.asyncio
    async def test_agent_returns_correct_type(self):
        """Test agent returns EntityExtractionOut."""
        test_model = TestModel()
        with entity_extractor_agent.override(model=test_model):
            result = await entity_extractor_agent.run(
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
        assert pair_extractor_agent._deps_type == Deps
        assert pair_extractor_agent._system_prompts is not None

    @pytest.mark.asyncio
    async def test_agent_returns_correct_type(self):
        """Test agent returns PairExtractionOut."""
        test_model = TestModel()
        with pair_extractor_agent.override(model=test_model):
            result = await pair_extractor_agent.run(
                "Extract pairs from: BRCA1 is associated with breast cancer.", deps=None
            )
            assert isinstance(result.output, PairExtractionOut)
            assert isinstance(result.output.pairs, list)
            assert isinstance(result.output.reasoning, str)


class TestEntityAssessorAgent:
    """Tests for entity_assessor_agent configuration and output."""

    def test_agent_configuration(self):
        """Test agent has correct configuration."""
        assert entity_assessor_agent._deps_type == Deps
        assert entity_assessor_agent._system_prompts is not None

    @pytest.mark.asyncio
    async def test_agent_returns_correct_type(self):
        """Test agent returns EntityEvidenceAssessment."""
        test_model = TestModel()
        with entity_assessor_agent.override(model=test_model):
            result = await entity_assessor_agent.run(
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
        assert pair_assessor_agent._deps_type == Deps
        assert pair_assessor_agent._system_prompts is not None

    @pytest.mark.asyncio
    async def test_agent_returns_correct_type(self):
        """Test agent returns PairEvidenceAssessment."""
        test_model = TestModel()
        with pair_assessor_agent.override(model=test_model):
            result = await pair_assessor_agent.run(
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
        assert final_judge_agent._deps_type == Deps
        assert final_judge_agent._system_prompts is not None

    @pytest.mark.asyncio
    async def test_agent_returns_correct_type(self):
        """Test agent returns FinalJudgmentOut."""
        test_model = TestModel()
        with final_judge_agent.override(model=test_model):
            result = await final_judge_agent.run(
                "Judge: BRCA1-breast cancer association across documents.", deps=None
            )
            assert isinstance(result.output, FinalJudgmentOut)
            assert isinstance(result.output.accepted, bool)
            assert result.output.confidence in ["high", "medium", "low"]
            assert isinstance(result.output.rationale, str)
