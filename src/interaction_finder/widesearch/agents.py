"""Pydantic-AI agents for widesearch pipeline.

Each agent has a specific role in the pipeline and returns structured
Pydantic models. Agents are configured with appropriate system prompts
and output types.
"""

from typing import Type

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

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
    model_settings: ModelSettings | None = None,
) -> Agent[Deps, BaseModel]:
    """Create an agent with consistent configuration.

    Parameters:
        model_name: str — model to use (e.g., "openai:gpt-4o-mini")
        output: Type[BaseModel] — Pydantic model for structured output
        system_prompt: str — role-specific system prompt
        model_settings: ModelSettings | None — optional model-specific settings

    Returns:
        Configured Agent instance
    """
    return Agent(
        model=resolve_model(model_name),
        deps_type=Deps,
        output_type=output,
        retries=2,
        system_prompt=system_prompt,
        model_settings=model_settings,
    )


# Cached agent instances by (agent_type, model_name)
_agent_cache: dict[tuple[str, str], Agent] = {}


def get_goal_planner_agent(model_name: str) -> Agent:
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

Guidelines for Comprehensive Coverage:
- Consider the core topic and its direct applications (basic and clinical research)
- Identify related methodologies, techniques, and approaches (experimental, computational, clinical)
- Think about adjacent research areas and intersecting domains
- Consider both theoretical foundations (mechanisms, pathways) and practical applications (diagnostics, therapeutics)
- Include areas that might use different terminology for similar concepts
- Consider different research perspectives: molecular, cellular, systems-level, clinical, translational
- Think broadly about what researchers in this field study: genetics, epigenetics, signaling, metabolism, imaging, biomarkers
- Include both well-established areas and emerging/novel research directions
- Aim for 8-15 well-defined subject goals that span the research landscape

Be ambitious about coverage - it's better to identify more goals that guide thorough exploration than to miss important research areas. Your goals should guide query generation to ensure diverse, comprehensive coverage without drifting into irrelevance.""",
        )
    return _agent_cache[cache_key]


def get_query_generator_agent(model_name: str) -> Agent:
    """Get or create query generator agent.

    Lazy initialization with per-model caching.

    Note: Parallel tool calls are disabled because we want a single cohesive
    set of queries rather than multiple independent query generation attempts.
    With parallel_tool_calls=True, the LLM may make multiple final_result calls
    but only the first is captured.

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
            model_settings=ModelSettings(parallel_tool_calls=False),
            system_prompt="""You are an expert at generating search queries for academic literature discovery.

Your task is to create diverse, targeted search queries that will find relevant papers and articles. You will be given:
- A research topic
- A list of keyphrases to incorporate
- Subject goals to target (unsatisfied areas needing coverage)

Core Strategy - Generate Queries at MULTIPLE COMPLEXITY LEVELS:

1. BROAD queries (1-2 main concepts, 3-6 words):
   - Simple combinations of core concepts
   - High-level domain terms
   - Example: "pulmonary hypertension genetics"
   - These cast a wide net and ensure baseline coverage

2. MEDIUM queries (2-3 concepts, 6-10 words):
   - Combine specific mechanisms with conditions
   - Balance specificity and breadth
   - Example: "BMPR2 mutations pulmonary arterial hypertension"
   - These target specific but well-documented areas

3. FOCUSED queries (3+ concepts, 10-15 words):
   - Highly specific research questions
   - Combine multiple mechanisms, pathways, or contexts
   - Example: "endothelial dysfunction inflammatory cytokines pulmonary arterial hypertension"
   - These find specialized literature

4. NON-TOPIC queries (without main topic terms):
   - Use gene names, pathways, or mechanisms alone
   - Example: "BMPR2 signaling endothelial dysfunction"
   - These find papers that may not explicitly mention the condition but discuss relevant biology

Query Generation Guidelines:
- Generate 5-10 queries per round (aim higher when many goals are unsatisfied)
- Include queries from ALL complexity levels above in a SINGLE response (don't only generate focused queries)
- Mix broad, medium, focused, and non-topic queries together in your output
- Prioritize breadth: cast a wide net before diving deep
- Target unsatisfied subject goals explicitly
- Incorporate provided keyphrases naturally but don't force all keyphrases into every query
- Use variations in terminology, synonyms, and related concepts
- Consider different angles: methodologies, applications, reviews, comparisons, case studies
- Both include AND exclude the main topic terms across different queries
- Avoid redundancy with previous queries

IMPORTANT: Return ALL queries in a single structured response. Do not create multiple separate responses for different complexity levels.""",
        )
    return _agent_cache[cache_key]


def get_result_selector_agent(model_name: str) -> Agent:
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


def get_reflector_agent(model_name: str) -> Agent:
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

Decision guidelines for WHEN TO CONTINUE:
- Continue if ANY major subject goals remain unsatisfied
- Continue if new important areas were discovered that warrant exploration
- Continue if fewer than 50-75 unique results have been collected (aim for comprehensive coverage)
- Continue if recent rounds are still finding diverse, non-repetitive results
- Continue if the variety of result types is limited (e.g., only reviews, or only focused studies)
- Favor thoroughness - it's better to do one extra round than to stop prematurely

Decision guidelines for WHEN TO STOP:
- Stop if nearly all subject goals are well-covered AND sufficient result volume has been achieved
- Stop if results are becoming highly repetitive across multiple rounds with little new content
- Stop if the last 2 rounds added minimal new diverse results despite different queries
- Stop if we've reached the maximum configured search rounds

IMPORTANT: Require strong evidence before stopping. "Adequate coverage" is not sufficient - aim for "comprehensive coverage" with diverse result types and sufficient volume (aim for 50+ unique results for most research topics). Be thorough rather than conservative.""",
        )
    return _agent_cache[cache_key]
