"""Pydantic-AI agents for widesearch pipeline.

Each agent has a specific role in the pipeline and returns structured
Pydantic models. Agents are configured with appropriate system prompts
and output types.
"""

from typing import Type

from pydantic import BaseModel
from pydantic_ai import Agent

from interaction_finder.widesearch.deps import Deps
from interaction_finder.widesearch.models import (
    QueryGenerationOut,
    ReflectionOut,
    ResultSelectionOut,
    SubjectGoalsOut,
)


def resolve_model(model_name: str):
    """Resolve model name to pydantic-ai model specification.

    Handles both cloud models (e.g., "openai:gpt-4o") and local models.
    For simplicity, we pass model names directly and rely on pydantic-ai's
    built-in resolution.

    Parameters:
        model_name: str — model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        Model specification for pydantic-ai Agent
    """
    # Pass through model names directly - pydantic-ai handles cloud model resolution
    return model_name


def mk_agent(
    model_name: str,
    output: Type[BaseModel],
    *,
    system_prompt: str,
) -> Agent[Deps, BaseModel]:
    """Create an agent with consistent configuration.

    Parameters:
        model_name: str — model to use (e.g., "openai:gpt-4o-mini")
        output: Type[BaseModel] — Pydantic model for structured output
        system_prompt: str — role-specific system prompt

    Returns:
        Configured Agent instance
    """
    return Agent(
        model=resolve_model(model_name),
        deps_type=Deps,
        output_type=output,
        retries=2,
        system_prompt=system_prompt,
    )


# Cached agent instances by (agent_type, model_name)
_agent_cache: dict[tuple[str, str], Agent] = {}


def get_goal_planner_agent(model_name: str = "openai:gpt-4o-mini") -> Agent:
    """Get or create goal planner agent.

    Lazy initialization with per-model caching.

    Parameters:
        model_name: Model identifier

    Returns:
        Configured Agent instance
    """
    cache_key = ("goal_planner", model_name)
    if cache_key not in _agent_cache:
        _agent_cache[cache_key] = mk_agent(
            model_name,
            SubjectGoalsOut,
            system_prompt="""You are an expert research strategist planning comprehensive literature searches.

Your task is to identify subject areas and research domains that should be covered to ensure comprehensive literature discovery on a given topic.

Guidelines:
- Consider the core topic and its direct applications
- Identify related methodologies, techniques, and approaches
- Think about adjacent research areas and intersecting domains
- Consider both theoretical foundations and practical applications
- Include areas that might use different terminology for similar concepts
- Aim for 5-10 well-defined subject goals that span the research landscape

Your goals should guide query generation to ensure diverse, comprehensive coverage without drifting into irrelevance.""",
        )
    return _agent_cache[cache_key]


def get_query_generator_agent(model_name: str = "openai:gpt-4o-mini") -> Agent:
    """Get or create query generator agent.

    Lazy initialization with per-model caching.

    Parameters:
        model_name: Model identifier

    Returns:
        Configured Agent instance
    """
    cache_key = ("query_generator", model_name)
    if cache_key not in _agent_cache:
        _agent_cache[cache_key] = mk_agent(
            model_name,
            QueryGenerationOut,
            system_prompt="""You are an expert at generating search queries for academic literature discovery.

Your task is to create diverse, targeted search queries that will find relevant papers and articles. You will be given:
- A research topic
- A list of keyphrases to incorporate
- Subject goals to target (unsatisfied areas needing coverage)

Guidelines:
- Generate 2-5 queries per round, prioritizing quality over quantity
- Target unsatisfied subject goals explicitly
- Incorporate provided keyphrases naturally into queries
- Use variations in terminology, synonyms, and related concepts
- Balance specificity (to stay relevant) with breadth (to find diverse results)
- Consider different angles: methodologies, applications, reviews, comparisons
- Avoid redundancy with previous queries

Your queries should be suitable for academic search engines (PubMed, Google Scholar, etc.) and find papers that advance coverage of unsatisfied subject goals.""",
        )
    return _agent_cache[cache_key]


def get_result_selector_agent(model_name: str = "openai:gpt-4o-mini") -> Agent:
    """Get or create result selector agent.

    Lazy initialization with per-model caching.

    Parameters:
        model_name: Model identifier

    Returns:
        Configured Agent instance
    """
    cache_key = ("result_selector", model_name)
    if cache_key not in _agent_cache:
        _agent_cache[cache_key] = mk_agent(
            model_name,
            ResultSelectionOut,
            system_prompt="""You are an expert at evaluating search results for relevance and coverage.

Your task is to select which search results are most relevant to the research topic and summarize what subject areas they cover.

Selection criteria:
- Prioritize results highly relevant to the topic
- Look for comprehensive coverage (reviews, surveys, meta-analyses)
- Favor results that appear to discuss multiple aspects or connections
- Prefer authoritative sources and recent publications when titles/snippets suggest quality
- Select 5-15 results depending on quality and diversity
- Ensure selected results advance coverage of subject goals

Coverage summary requirements:
- Identify what subject areas and topics are well-covered by selected results
- Note any connections, methodologies, or perspectives represented
- Be specific about what aspects of the topic these results address

Your selection and summary will guide the reflection process to determine if more searching is needed.""",
        )
    return _agent_cache[cache_key]


def get_reflector_agent(model_name: str = "openai:gpt-4o-mini") -> Agent:
    """Get or create reflector agent.

    Lazy initialization with per-model caching.

    Parameters:
        model_name: Model identifier

    Returns:
        Configured Agent instance
    """
    cache_key = ("reflector", model_name)
    if cache_key not in _agent_cache:
        _agent_cache[cache_key] = mk_agent(
            model_name,
            ReflectionOut,
            system_prompt="""You are an expert at evaluating literature search coverage and deciding when sufficient breadth has been achieved.

Your task is to reflect on search results collected so far and decide whether to continue searching or stop.

You will be given:
- The research topic
- Original subject goals to cover
- Currently satisfied goals
- Summaries of search results from each round

Evaluation criteria:
- Assess which subject goals are now well-covered (mark as satisfied)
- Identify any new important subject areas discovered that should be explored
- Determine if another search round would likely add significant value
- Consider the quality and diversity of results found so far
- Balance comprehensiveness with diminishing returns

Decision guidelines:
- Continue if major subject goals remain unsatisfied and more searching likely helps
- Continue if new important areas were discovered that warrant exploration
- Stop if subject goals are well-covered and additional searching unlikely to add value
- Stop if results are becoming repetitive or quality is declining
- Be decisive - avoid unnecessary additional rounds once good coverage is achieved

Your decision will determine whether the search continues or produces final output.""",
        )
    return _agent_cache[cache_key]
