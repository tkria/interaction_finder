"""Pydantic output models for widesearch agents.

All agents return structured Pydantic models. These models define the
contract between agents and nodes, providing automatic validation and
clear intent specification.
"""

from pydantic import BaseModel, Field


class SubjectGoalsOut(BaseModel):
    """Output model for goal planning agent.

    Identifies subject areas and research domains that should be covered
    during the search session to ensure comprehensive literature discovery.
    """

    goals: list[str] = Field(
        ...,
        min_length=3,
        max_length=15,
        description="Subject areas and research domains to cover during search",
    )
    reasoning: str = Field(
        ...,
        min_length=50,
        description="Brief explanation of why these goals were selected",
    )


class QueryGenerationOut(BaseModel):
    """Output model for query generation agent.

    Produces diverse search queries targeting unsatisfied subject goals
    and leveraging provided keyphrases.
    """

    queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Search queries targeting unsatisfied subject goals",
    )
    reasoning: str = Field(
        ...,
        min_length=50,
        description="Explanation of query generation strategy and targeting",
    )


class ResultSelectionOut(BaseModel):
    """Output model for result selection agent.

    Selects which search results are most relevant for the topic and
    summarizes what subject areas are well-covered by the selected results.
    """

    selected_indices: list[int] = Field(
        ...,
        description="Indices of search results selected for fetching (0-based)",
    )
    covered_topics_summary: str = Field(
        ...,
        min_length=50,
        description="Summary of subject areas and topics covered by selected results",
    )
    reasoning: str = Field(
        ...,
        min_length=50,
        description="Explanation of selection criteria and decisions",
    )


class ReflectionOut(BaseModel):
    """Output model for reflection agent.

    Evaluates search coverage, determines if goals are satisfied, and
    decides whether to continue searching or stop.
    """

    satisfied_goals: list[str] = Field(
        ...,
        description="Subject goals that are now satisfied based on search results",
    )
    new_goals: list[str] = Field(
        default_factory=list,
        description="Additional subject areas discovered that should be explored",
    )
    should_continue: bool = Field(
        ...,
        description="Whether to perform another round of searching",
    )
    reasoning: str = Field(
        ...,
        min_length=100,
        description="Detailed explanation of coverage assessment and decision",
    )
