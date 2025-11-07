"""Tests for extraction pipeline nodes.

Tests node structure, methods, and dataclass configuration.
"""

from dataclasses import is_dataclass

from interaction_finder.extraction.nodes import (
    AssessEntitiesNode,
    AssessPairsNode,
    ExtractFromDocumentsNode,
    FinalizeNode,
    JudgePairsNode,
)


class TestNodeStructure:
    """Test that all nodes have correct structure."""

    def test_extract_from_documents_node(self):
        """Test ExtractFromDocumentsNode structure."""
        node = ExtractFromDocumentsNode()
        assert is_dataclass(node)
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_assess_entities_node(self):
        """Test AssessEntitiesNode structure."""
        node = AssessEntitiesNode()
        assert is_dataclass(node)
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_assess_pairs_node(self):
        """Test AssessPairsNode structure."""
        node = AssessPairsNode()
        assert is_dataclass(node)
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_judge_pairs_node(self):
        """Test JudgePairsNode structure."""
        node = JudgePairsNode()
        assert is_dataclass(node)
        assert hasattr(node, "run")
        assert callable(node.run)

    def test_finalize_node(self):
        """Test FinalizeNode structure."""
        node = FinalizeNode()
        assert is_dataclass(node)
        assert hasattr(node, "run")
        assert callable(node.run)


class TestJudgePairsNodeLogic:
    """Test JudgePairsNode deterministic judgment logic."""

    def test_can_judge_deterministically_strong(self):
        """Test high-confidence accept with multiple strong assessments."""
        from interaction_finder.extraction.models import PairAssessment
        from interaction_finder.resources import ResourceId

        node = JudgePairsNode()
        assessments = [
            PairAssessment(
                resource_id=ResourceId(url="url1", counter=0),
                strength="strong",
                rationale="Strong evidence",
                quotes=[],
            ),
            PairAssessment(
                resource_id=ResourceId(url="url2", counter=1),
                strength="strong",
                rationale="Strong evidence",
                quotes=[],
            ),
        ]

        can_judge, judgment = node._can_judge_deterministically(assessments)

        assert can_judge is True
        assert judgment is not None
        assert judgment.accepted is True
        assert judgment.confidence == "high"

    def test_can_judge_deterministically_none(self):
        """Test high-confidence reject with multiple none assessments."""
        from interaction_finder.extraction.models import PairAssessment
        from interaction_finder.resources import ResourceId

        node = JudgePairsNode()
        assessments = [
            PairAssessment(
                resource_id=ResourceId(url="url1", counter=0),
                strength="none",
                rationale="No evidence",
                quotes=[],
            ),
            PairAssessment(
                resource_id=ResourceId(url="url2", counter=0),
                strength="none",
                rationale="No evidence",
                quotes=[],
            ),
        ]

        can_judge, judgment = node._can_judge_deterministically(assessments)

        assert can_judge is True
        assert judgment is not None
        assert judgment.accepted is False
        assert judgment.confidence == "high"

    def test_cannot_judge_deterministically_mixed(self):
        """Test that mixed assessments require LLM judgment."""
        from interaction_finder.extraction.models import PairAssessment
        from interaction_finder.resources import ResourceId

        node = JudgePairsNode()
        assessments = [
            PairAssessment(
                resource_id=ResourceId(url="url1", counter=0),
                strength="strong",
                rationale="Strong evidence",
                quotes=[],
            ),
            PairAssessment(
                resource_id=ResourceId(url="url2", counter=0),
                strength="weak",
                rationale="Weak evidence",
                quotes=[],
            ),
        ]

        can_judge, judgment = node._can_judge_deterministically(assessments)

        assert can_judge is False
        assert judgment is None

    def test_cannot_judge_deterministically_single_strong(self):
        """Test that single strong assessment requires LLM judgment."""
        from interaction_finder.extraction.models import PairAssessment
        from interaction_finder.resources import ResourceId

        node = JudgePairsNode()
        assessments = [
            PairAssessment(
                resource_id=ResourceId(url="url1", counter=0),
                strength="strong",
                rationale="Strong evidence",
                quotes=[],
            ),
        ]

        can_judge, judgment = node._can_judge_deterministically(assessments)

        assert can_judge is False
        assert judgment is None
