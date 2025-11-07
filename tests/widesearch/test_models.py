"""Tests for widesearch Pydantic models."""

import pytest
from pydantic import ValidationError

from interaction_finder.widesearch.models import (
    QueryGenerationOut,
    ReflectionOut,
    ResultSelectionOut,
    SubjectGoalsOut,
)


class TestSubjectGoalsOut:
    """Tests for SubjectGoalsOut model."""

    def test_valid_goals(self):
        """Test valid subject goals output."""
        output = SubjectGoalsOut(
            goals=["methodology", "applications", "clinical trials"],
            reasoning="These areas cover the core research landscape comprehensively and will guide our search",
        )
        assert len(output.goals) == 3
        assert output.goals[0] == "methodology"

    def test_minimum_goals_validation(self):
        """Test minimum goals constraint."""
        with pytest.raises(ValidationError):
            SubjectGoalsOut(
                goals=["only", "two"],  # Too few
                reasoning="This should fail",
            )

    def test_maximum_goals_validation(self):
        """Test maximum goals constraint."""
        with pytest.raises(ValidationError):
            SubjectGoalsOut(
                goals=[f"goal_{i}" for i in range(20)],  # Too many
                reasoning="This should fail",
            )

    def test_reasoning_minimum_length(self):
        """Test reasoning minimum length constraint."""
        with pytest.raises(ValidationError):
            SubjectGoalsOut(
                goals=["a", "b", "c"],
                reasoning="short",  # Too short
            )


class TestQueryGenerationOut:
    """Tests for QueryGenerationOut model."""

    def test_valid_queries(self):
        """Test valid query generation output."""
        output = QueryGenerationOut(
            queries=["diabetes treatment", "insulin resistance mechanisms"],
            reasoning="These queries target different aspects of the topic",
        )
        assert len(output.queries) == 2

    def test_single_query(self):
        """Test single query is valid."""
        output = QueryGenerationOut(
            queries=["comprehensive review of diabetes"],
            reasoning="Broad query to start the search and find comprehensive review articles",
        )
        assert len(output.queries) == 1

    def test_too_many_queries(self):
        """Test maximum queries constraint."""
        with pytest.raises(ValidationError):
            QueryGenerationOut(
                queries=[f"query {i}" for i in range(15)],  # Too many
                reasoning="This should fail",
            )


class TestResultSelectionOut:
    """Tests for ResultSelectionOut model."""

    def test_valid_selection(self):
        """Test valid result selection output."""
        output = ResultSelectionOut(
            selected_indices=[0, 2, 5, 7],
            covered_topics_summary="These results cover methodology and clinical applications comprehensively",
            reasoning="Selected based on relevance and diversity to ensure comprehensive coverage",
        )
        assert len(output.selected_indices) == 4
        assert 0 in output.selected_indices

    def test_empty_selection(self):
        """Test empty selection is valid."""
        output = ResultSelectionOut(
            selected_indices=[],
            covered_topics_summary="No relevant results found in this round after reviewing all search results",
            reasoning="All results were off-topic or not relevant to the research goals",
        )
        assert len(output.selected_indices) == 0

    def test_summary_minimum_length(self):
        """Test covered topics summary minimum length."""
        with pytest.raises(ValidationError):
            ResultSelectionOut(
                selected_indices=[1, 2],
                covered_topics_summary="short",  # Too short
                reasoning="This should fail",
            )


class TestReflectionOut:
    """Tests for ReflectionOut model."""

    def test_continue_decision(self):
        """Test reflection output with continue decision."""
        output = ReflectionOut(
            satisfied_goals=["methodology", "clinical trials"],
            new_goals=["economic impact"],
            should_continue=True,
            reasoning="We have covered basic areas but discovered a new important domain that should be explored in the next round",
        )
        assert output.should_continue is True
        assert len(output.satisfied_goals) == 2
        assert len(output.new_goals) == 1

    def test_stop_decision(self):
        """Test reflection output with stop decision."""
        output = ReflectionOut(
            satisfied_goals=["methodology", "clinical trials", "applications"],
            new_goals=[],
            should_continue=False,
            reasoning="All major subject goals have been well-covered and additional searching is unlikely to add significant value",
        )
        assert output.should_continue is False
        assert len(output.new_goals) == 0

    def test_reasoning_minimum_length(self):
        """Test reasoning minimum length constraint."""
        with pytest.raises(ValidationError):
            ReflectionOut(
                satisfied_goals=["a"],
                new_goals=[],
                should_continue=False,
                reasoning="too short",  # Too short (< 100 chars)
            )

    def test_empty_satisfied_goals(self):
        """Test empty satisfied goals is valid."""
        output = ReflectionOut(
            satisfied_goals=[],
            new_goals=["new area discovered"],
            should_continue=True,
            reasoning="No goals satisfied yet but discovered a new important area that should be explored during the search",
        )
        assert len(output.satisfied_goals) == 0
        assert output.should_continue is True
