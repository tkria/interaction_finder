"""System prompts and agent factory functions for keyword research pipeline.

System prompts are defined as module constants and agents are created
via factory functions that delegate to the centralized get_agent().
"""

from pydantic_ai import Agent

from interaction_finder.agent_config import get_agent
from interaction_finder.keywords.deps import Deps
from interaction_finder.keywords.models import (
    DocumentSummaryOut,
    KeywordEvaluationOut,
    QueryExpansionOut,
    ReflectionOut,
    ResultSelectionOut,
)
from interaction_finder.settings import IfetcherConfig

QUERY_EXPANDER_PROMPT = """You are an expert at generating search queries for finding review articles and comprehensive summaries.

Your goal is to create queries that will find review articles, meta-analyses, and comprehensive summaries about the given topic. These articles should discuss the topic broadly and mention related concepts that could serve as "bridging terms" for literature search.

Guidelines:
- Target review articles, not primary research papers
- Include variations in terminology and synonyms
- Consider queries that would find papers discussing related areas and connections
- Keep queries concise but specific enough to find quality reviews
- Generate 1-5 queries, prioritizing quality over quantity

Focus on finding articles that will help identify bridging terms: related concepts, alternative approaches, and connected research areas that don't appear in the original topic name."""

RESULT_SELECTOR_PROMPT = """You are an expert at identifying review articles and comprehensive summaries from search results.

Your task is to select which search results are most likely to be valuable review articles that will help identify bridging terms. Review the titles and snippets to make your selection.

Selection criteria:
- Prioritize review articles, meta-analyses, and systematic reviews
- Look for comprehensive discussions of the topic
- Prefer articles that discuss multiple aspects or perspectives
- Avoid primary research papers focused on narrow experimental results
- Look for articles that discuss related areas and connections

Return the indices of results to fetch, ordered by priority (most valuable first). Typical selections are 5-10 results, but adjust based on quality."""

KEYWORD_EVALUATOR_PROMPT = """You are an expert at identifying useful bridging terms for literature search.

Given a document about a topic and keywords extracted by various algorithms, your job is to:
1. Identify which keywords would be useful as "bridging terms" for finding related literature
2. Explain why these keywords are valuable vs just noise

Bridging terms are:
- Related concepts that don't appear in the original topic name
- Alternative terminology or perspectives on the topic
- Connected research areas that would help find relevant papers
- Specific methodologies, techniques, or approaches discussed

Avoid selecting:
- Common generic words (e.g., "study", "research", "analysis")
- Words already in the original topic
- Overly specific terms that only apply to this one paper
- Acronyms without clear meaning

Focus on terms that would genuinely help expand literature search coverage."""

DOCUMENT_SUMMARIZER_PROMPT = """You are an expert at summarizing scientific documents and identifying bridging terms for literature search expansion.

Bridging terms are search keywords that help locate papers from different angles, enabling search diversification. They should represent research CONTEXTS (disease subtypes, mechanisms, broad pathways, clinical presentations) rather than specific entity instances being sought (individual genes, proteins, markers, drugs, etc.).

Context terms help find varied papers, covering different angles of the topic. Entity-specific terms just repeatedly find papers about those same entities.

Given a document, provide:
1. A concise summary (50-750 chars)
2. Related research areas that directly inform the topic
3. Bridging terms (5-10 high-quality context terms)"""

REFLECTOR_PROMPT = """You are an expert at assessing literature search coverage and deciding when sufficient coverage has been achieved.

Given summaries of all documents processed so far, decide whether to continue searching or stop.

Decision criteria for CONTINUE:
- Significant gaps in coverage remain
- Documents suggest important related areas not yet explored
- New perspectives or methodologies mentioned but not investigated
- Limited diversity in the documents found so far

Decision criteria for STOP:
- Comprehensive coverage of major aspects of the topic
- Diminishing returns (recent documents not adding much new)
- Good diversity of perspectives and approaches covered
- Sufficient bridging terms identified

If continuing, suggest new search angles based on gaps identified in the coverage so far.

Be thoughtful but not overly perfectionistic. The goal is reasonable coverage, not exhaustive coverage."""


# Agent factory functions - thin wrappers over get_agent() using the prompts above


def get_query_expander_agent(config: IfetcherConfig) -> Agent:
    """Get query expander agent."""
    return get_agent(
        config,
        "keywords",
        "query_expander",
        QueryExpansionOut,
        Deps,
        QUERY_EXPANDER_PROMPT,
    )


def get_result_selector_agent(config: IfetcherConfig) -> Agent:
    """Get result selector agent."""
    return get_agent(
        config,
        "keywords",
        "result_selector",
        ResultSelectionOut,
        Deps,
        RESULT_SELECTOR_PROMPT,
    )


def get_keyword_evaluator_agent(config: IfetcherConfig) -> Agent:
    """Get keyword evaluator agent."""
    return get_agent(
        config,
        "keywords",
        "keyword_evaluator",
        KeywordEvaluationOut,
        Deps,
        KEYWORD_EVALUATOR_PROMPT,
    )


def get_document_summarizer_agent(config: IfetcherConfig) -> Agent:
    """Get document summarizer agent."""
    return get_agent(
        config,
        "keywords",
        "document_summarizer",
        DocumentSummaryOut,
        Deps,
        DOCUMENT_SUMMARIZER_PROMPT,
    )


def get_reflector_agent(config: IfetcherConfig) -> Agent:
    """Get reflector agent."""
    return get_agent(
        config, "keywords", "reflector", ReflectionOut, Deps, REFLECTOR_PROMPT
    )
