"""Graph nodes for keyword research pipeline.

Each node represents a stage in the pipeline. Nodes control flow via
return type annotations. All branching and looping happens in nodes;
agents only produce typed data.
"""

from dataclasses import dataclass
from typing import Union

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_ai.usage import RunUsage

from interaction_finder.keywords.agents import (
    document_summarizer_agent,
    keyword_evaluator_agent,
    query_expander_agent,
    reflector_agent,
    result_selector_agent,
)
from interaction_finder.keywords.deps import Deps
from interaction_finder.keywords.models import BridgingTermsOut
from interaction_finder.keywords.state import State
from interaction_finder.search.models import SearchQuery


@dataclass
class ExpandQueryNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Generate search queries for finding review articles.

    Uses query_expander_agent to create queries tailored for finding
    review articles and comprehensive summaries.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "SearchNode":
        """Generate search queries and increment round counter."""
        # Increment round counter
        ctx.state.current_round += 1
        # Use query expander agent
        usage = RunUsage()
        result = await query_expander_agent.run(
            f"Generate search queries to find review articles about: {ctx.state.topic}",
            deps=ctx.deps,
            usage=usage,
        )
        # Store queries in state
        ctx.state.search_queries = result.output.queries
        return SearchNode()


@dataclass
class SearchNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Execute searches for all queries generated in ExpandQuery.

    Uses the search backend from deps to fetch results for each query.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "RerankNode":
        """Execute searches and store results."""
        # Execute all searches
        all_results = []
        for query_text in ctx.state.search_queries:
            query = SearchQuery(
                query=query_text,
                max_results=ctx.deps.config.get("max_results_per_query", 20),
            )
            results = await ctx.deps.search_backend.search(query)
            all_results.extend(results)
        # Store in state
        ctx.state.all_search_results = all_results
        return RerankNode()


@dataclass
class RerankNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Rerank search results by semantic similarity to topic.

    Uses the reranker from deps to improve result ordering.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "SelectResultsNode":
        """Rerank results and update state."""
        if not ctx.state.all_search_results:
            # No results to rerank, skip to selection
            return SelectResultsNode()
        # Rerank using topic as query
        reranked = ctx.deps.reranker.rerank(
            ctx.state.topic, ctx.state.all_search_results
        )
        # Update state with reranked results
        ctx.state.all_search_results = reranked
        return SelectResultsNode()


@dataclass
class SelectResultsNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Select which search results to fetch and process.

    Uses result_selector_agent to choose the most promising results.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "FetchDocumentsNode":
        """Select results to fetch."""
        if not ctx.state.all_search_results:
            # No results available, skip to finalization
            return FinalizeNode()
        # Prepare context for agent
        results_context = "\n\n".join(
            [
                f"[{i}] {r.title}\n{r.snippet or '(no snippet)'}"
                for i, r in enumerate(ctx.state.all_search_results)
            ]
        )
        prompt = f"""Review these search results and select which ones to fetch for keyword extraction.

Topic: {ctx.state.topic}

Search Results:
{results_context}

Select the indices of results that are most likely to be valuable review articles."""
        # Use result selector agent
        usage = RunUsage()
        result = await result_selector_agent.run(prompt, deps=ctx.deps, usage=usage)
        # Get selected results
        max_to_fetch = ctx.deps.config.get("max_documents_to_fetch", 10)
        selected_indices = result.output.selected_indices[:max_to_fetch]
        ctx.state.selected_results = [
            ctx.state.all_search_results[i]
            for i in selected_indices
            if i < len(ctx.state.all_search_results)
        ]
        return FetchDocumentsNode()


@dataclass
class FetchDocumentsNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Fetch markdown content for selected results.

    Uses fetcher from deps to retrieve and convert documents.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "ExtractKeywordsNode":
        """Fetch documents and store content."""
        if not ctx.state.selected_results:
            # Nothing to fetch
            return FinalizeNode()
        # Extract URLs
        urls = [r.url for r in ctx.state.selected_results]
        # Fetch all documents concurrently
        contents = await ctx.deps.fetcher.get_markdown(
            urls, progress=False, fail_fast=False, retry=False
        )
        # Store successful fetches
        for url, content in zip(urls, contents):
            if content:  # Skip failures
                ctx.state.fetched_content[url] = content
        return ExtractKeywordsNode()


@dataclass
class ExtractKeywordsNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Extract keywords from all fetched documents using multiple methods.

    Runs all extractors (RAKE, YAKE, TF-IDF, KeyBERT) on each document.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "EvaluateKeywordsNode":
        """Extract keywords from all documents."""
        if not ctx.state.fetched_content:
            # No content to process
            return FinalizeNode()
        max_keywords = ctx.deps.config.get("max_keywords_per_method", 30)
        # Extract keywords for each document
        for url, content in ctx.state.fetched_content.items():
            doc_keywords = []
            # Run all extractors
            for name, extractor in ctx.deps.extractors.items():
                try:
                    keywords = extractor.extract(content, max_keywords=max_keywords)
                    doc_keywords.extend(keywords)
                except Exception:
                    # Skip extractor if it fails
                    continue
            # Store keywords for this document
            ctx.state.extracted_keywords[url] = doc_keywords
        return EvaluateKeywordsNode()


@dataclass
class EvaluateKeywordsNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Evaluate keywords and summarize each document.

    Uses keyword_evaluator_agent and document_summarizer_agent to:
    1. Identify useful bridging terms
    2. Summarize document content
    3. Assess coverage contribution
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "ReflectNode":
        """Evaluate keywords and summarize documents."""
        if not ctx.state.extracted_keywords:
            # No keywords to evaluate
            return ReflectNode()
        # Process each document
        for url, keywords in ctx.state.extracted_keywords.items():
            content = ctx.state.fetched_content.get(url, "")
            if not content:
                continue
            # Format keywords for agent
            keywords_text = ", ".join(set(kw.keyword for kw in keywords[:50]))
            # Summarize document
            summary_prompt = f"""Summarize this document about '{ctx.state.topic}' and identify bridging terms.

Document content:
{content[:3000]}...

Extracted keywords: {keywords_text}

Provide a summary, related research areas, and bridging terms that would help find related literature."""
            usage = RunUsage()
            summary_result = await document_summarizer_agent.run(
                summary_prompt, deps=ctx.deps, usage=usage
            )
            # Store summary
            ctx.state.document_summaries.append(summary_result.output)
        return ReflectNode()


@dataclass
class ReflectNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Decide whether to continue searching or finalize results.

    Uses reflector_agent to assess coverage and decide next action.
    Loops back to ExpandQuery if continuing, or moves to Finalize if stopping.
    """

    async def run(
        self, ctx: GraphRunContext[State, Deps]
    ) -> Union["ExpandQueryNode", "FinalizeNode"]:
        """Reflect on coverage and decide whether to continue."""
        # Check iteration limit
        if ctx.state.current_round >= ctx.state.max_rounds:
            return FinalizeNode()
        # Check if we have any summaries
        if not ctx.state.document_summaries:
            # No documents processed, stop
            return FinalizeNode()
        # Prepare summaries for agent
        summaries_text = "\n\n".join(
            [
                f"Document {i + 1}:\nSummary: {s.summary}\n"
                f"Related areas: {', '.join(s.related_areas)}\n"
                f"Bridging terms: {', '.join(s.bridging_terms)}\n"
                f"Coverage: {s.coverage_contribution}"
                for i, s in enumerate(ctx.state.document_summaries)
            ]
        )
        prompt = f"""Review the documents processed so far and decide whether to continue searching.

Topic: {ctx.state.topic}
Current round: {ctx.state.current_round}/{ctx.state.max_rounds}
Documents processed: {len(ctx.state.document_summaries)}

Document Summaries:
{summaries_text}

Decide whether coverage is sufficient (stop) or more searches are needed (continue)."""
        # Use reflector agent
        usage = RunUsage()
        result = await reflector_agent.run(prompt, deps=ctx.deps, usage=usage)
        # Make decision
        if result.output.decision == "stop":
            return FinalizeNode()
        # Continue with new search angles
        # Note: In a full implementation, we'd use new_search_angles to guide the next ExpandQuery
        return ExpandQueryNode()


@dataclass
class FinalizeNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Collect and deduplicate all bridging terms.

    Returns the final BridgingTermsOut result.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> End[BridgingTermsOut]:
        """Finalize results and return."""
        # Collect all bridging terms from summaries
        all_terms = set()
        for summary in ctx.state.document_summaries:
            all_terms.update(summary.bridging_terms)
        # Sort alphabetically
        final_terms = sorted(all_terms)
        # Create final assessment
        assessment = (
            f"Completed {ctx.state.current_round} search round(s). "
            f"Processed {len(ctx.state.document_summaries)} documents. "
            f"Identified {len(final_terms)} unique bridging terms."
        )
        # Return final result
        return End(
            BridgingTermsOut(
                terms=final_terms,
                total_documents_processed=len(ctx.state.document_summaries),
                rounds_completed=ctx.state.current_round,
                coverage_assessment=assessment,
            )
        )
