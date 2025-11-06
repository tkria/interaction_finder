"""Graph nodes for keyword research pipeline.

Each node represents a stage in the pipeline. Nodes control flow via
return type annotations. All branching and looping happens in nodes;
agents only produce typed data.
"""

from dataclasses import dataclass
from typing import Union

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_ai.usage import RunUsage

from interaction_finder.logging import logfire
from interaction_finder.keywords.agents import (
    document_summarizer_agent,
    query_expander_agent,
    reflector_agent,
    result_selector_agent,
)
from interaction_finder.keywords.deps import Deps
from interaction_finder.keywords.models import BridgingTermsOut
from interaction_finder.keywords.normalization import normalize_term_for_deduplication
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
        with logfire.span(
            "ExpandQueryNode",
            topic=ctx.state.topic,
            round=ctx.state.current_round + 1,
        ):
            # Increment round counter
            ctx.state.current_round += 1
            logfire.info(
                f"Expanding queries for round {ctx.state.current_round}: {ctx.state.topic}"
            )
            # Use query expander agent
            usage = RunUsage()
            result = await query_expander_agent.run(
                f"Generate search queries to find review articles about: {ctx.state.topic}",
                deps=ctx.deps,
                usage=usage,
            )
            # Store queries in state
            ctx.state.search_queries = result.output.queries
            logfire.info(f"Generated {len(result.output.queries)} search queries")
            return SearchNode()


@dataclass
class SearchNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Execute searches for all queries generated in ExpandQuery.

    Uses the search backend from deps to fetch results for each query.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "RerankNode":
        """Execute searches and store results."""
        with logfire.span("SearchNode", num_queries=len(ctx.state.search_queries)):
            logfire.info(f"Executing {len(ctx.state.search_queries)} search queries")
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
            logfire.info(f"Found {len(all_results)} total search results")
            return RerankNode()


@dataclass
class RerankNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Rerank search results by semantic similarity to topic.

    Uses the reranker from deps to improve result ordering.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "SelectResultsNode":
        """Rerank results and update state."""
        with logfire.span("RerankNode", num_results=len(ctx.state.all_search_results)):
            if not ctx.state.all_search_results:
                logfire.info("No results to rerank, skipping")
                # No results to rerank, skip to selection
                return SelectResultsNode()
            logfire.info(f"Reranking {len(ctx.state.all_search_results)} results")
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
        with logfire.span(
            "SelectResultsNode", num_results=len(ctx.state.all_search_results)
        ):
            if not ctx.state.all_search_results:
                logfire.info("No results available, skipping to finalization")
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
            logfire.info(f"Selected {len(ctx.state.selected_results)} results to fetch")
            return FetchDocumentsNode()


@dataclass
class FetchDocumentsNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Fetch markdown content for selected results.

    Uses fetcher from deps to retrieve and convert documents. Deduplicates
    by checking ResourcePool before fetching.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "ExtractKeywordsNode":
        """Fetch documents and add to resource pool."""
        with logfire.span(
            "FetchDocumentsNode", num_selected=len(ctx.state.selected_results)
        ):
            if not ctx.state.selected_results:
                logfire.info("No selected results, skipping to finalization")
                return FinalizeNode()
            # Separate new URLs from already-fetched
            urls_to_fetch = []
            resource_ids_map = {}
            for result in ctx.state.selected_results:
                if result.url not in ctx.deps.resource_pool:
                    # Register new resource
                    rid = ctx.deps.resource_pool.register(result.url)
                    urls_to_fetch.append((result.url, result.title))
                    resource_ids_map[result.url] = rid
            # Skip fetching if all URLs already processed
            if not urls_to_fetch:
                logfire.info("All URLs already fetched, skipping to extraction")
                return ExtractKeywordsNode()
            logfire.info(f"Fetching {len(urls_to_fetch)} new documents")
            # Fetch only new URLs
            urls = [url for url, _ in urls_to_fetch]
            contents = await ctx.deps.fetcher.get_markdown(
                urls, progress=False, fail_fast=False, retry=False
            )
            # Add content to resource pool
            fetched_count = 0
            for (url, title), content in zip(urls_to_fetch, contents):
                if content:
                    rid = resource_ids_map[url]
                    ctx.deps.resource_pool.add_content(rid, title, content)
                    fetched_count += 1
            logfire.info(
                f"Successfully fetched {fetched_count}/{len(urls_to_fetch)} documents"
            )
            return ExtractKeywordsNode()


@dataclass
class ExtractKeywordsNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Extract keywords from all fetched documents using multiple methods.

    Runs all extractors (RAKE, YAKE, TF-IDF, KeyBERT) on each document.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "EvaluateKeywordsNode":
        """Extract keywords from all documents in resource pool."""
        resources = ctx.deps.resource_pool.resources
        with logfire.span("ExtractKeywordsNode", num_resources=len(resources)):
            if not resources:
                logfire.info("No resources available, skipping to finalization")
                return FinalizeNode()
            max_keywords = ctx.deps.config.get("max_keywords_per_method", 30)
            # Track already-processed URLs to avoid re-extraction
            already_processed = set(ctx.state.extracted_keywords.keys())
            new_resources = [r for r in resources if r.id.url not in already_processed]
            logfire.info(
                f"Extracting keywords from {len(new_resources)} new documents "
                f"({len(already_processed)} already processed)"
            )
            # Extract keywords for each resource
            for resource in new_resources:
                doc_keywords = []
                # Run all extractors
                for name, extractor in ctx.deps.extractors.items():
                    try:
                        keywords = extractor.extract(
                            resource.text, max_keywords=max_keywords
                        )
                        doc_keywords.extend(keywords)
                    except Exception as e:
                        logfire.warning(
                            f"Extractor {name} failed for {resource.id.url}: {e}"
                        )
                        continue
                # Store keywords for this document
                ctx.state.extracted_keywords[resource.id.url] = doc_keywords
            total_keywords = sum(
                len(kws) for kws in ctx.state.extracted_keywords.values()
            )
            logfire.info(f"Extracted {total_keywords} total keywords")
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
        with logfire.span(
            "EvaluateKeywordsNode", num_docs=len(ctx.state.extracted_keywords)
        ):
            if not ctx.state.extracted_keywords:
                logfire.info("No extracted keywords, skipping to reflection")
                return ReflectNode()
            logfire.info(
                f"Evaluating keywords for {len(ctx.state.extracted_keywords)} documents"
            )
            # Process each document
            for url, keywords in ctx.state.extracted_keywords.items():
                # Get content from resource pool
                resource = ctx.deps.resource_pool.get(url)
                if not resource:
                    continue
                # Format keywords for agent
                keywords_text = ", ".join(set(kw.keyword for kw in keywords[:50]))
                # Summarize document
                summary_prompt = f"""Summarize this document about '{ctx.state.topic}' and identify bridging terms.

Document content:
{resource.text[:3000]}...

Extracted keywords: {keywords_text}

Provide a summary, related research areas, and bridging terms that would help find related literature."""
                usage = RunUsage()
                summary_result = await document_summarizer_agent.run(
                    summary_prompt, deps=ctx.deps, usage=usage
                )
                # Store summary
                ctx.state.document_summaries.append(summary_result.output)
            logfire.info(f"Generated {len(ctx.state.document_summaries)} summaries")
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
        with logfire.span(
            "ReflectNode",
            round=ctx.state.current_round,
            max_rounds=ctx.state.max_rounds,
            num_summaries=len(ctx.state.document_summaries),
        ):
            # Check iteration limit
            if ctx.state.current_round >= ctx.state.max_rounds:
                logfire.info(f"Reached max rounds ({ctx.state.max_rounds}), finalizing")
                return FinalizeNode()
            # Check if we have any summaries
            if not ctx.state.document_summaries:
                logfire.info("No document summaries, finalizing")
                # No documents processed, stop
                return FinalizeNode()
            logfire.info(
                f"Reflecting on {len(ctx.state.document_summaries)} documents in round {ctx.state.current_round}"
            )
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
            decision = result.output.decision
            logfire.info(f"Reflection decision: {decision}")
            if decision == "stop":
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
        with logfire.span(
            "FinalizeNode", num_summaries=len(ctx.state.document_summaries)
        ):
            # Collect all bridging terms with advanced normalization for deduplication
            # Normalize using: abbreviation stripping, text normalization, lemmatization, suffix stripping
            # Track both normalized keys and original terms (preserving first occurrence)
            terms_by_normalized = {}
            for summary in ctx.state.document_summaries:
                for term in summary.bridging_terms:
                    normalized_key = normalize_term_for_deduplication(term)
                    # Keep first occurrence (preserves original casing and formatting)
                    if normalized_key not in terms_by_normalized:
                        terms_by_normalized[normalized_key] = term
            all_terms = list(terms_by_normalized.values())
            # Handle empty results
            if not all_terms:
                logfire.info("No bridging terms found")
                return End(
                    BridgingTermsOut(
                        terms=[],
                        scores=[],
                        total_documents_processed=len(ctx.state.document_summaries),
                        rounds_completed=ctx.state.current_round,
                        coverage_assessment=(
                            f"Completed {ctx.state.current_round} search round(s). "
                            f"Processed {len(ctx.state.document_summaries)} documents. "
                            f"No bridging terms identified."
                        ),
                        resources=ctx.deps.resource_pool,
                    )
                )
            # Rerank terms by semantic similarity to topic
            logfire.info(
                f"Reranking {len(all_terms)} bridging terms by topic relevance"
            )
            scored_terms = ctx.deps.reranker.rerank_terms(ctx.state.topic, all_terms)
            # Extract terms and scores
            final_terms = [term for term, _ in scored_terms]
            final_scores = [score for _, score in scored_terms]
            # Create final assessment
            assessment = (
                f"Completed {ctx.state.current_round} search round(s). "
                f"Processed {len(ctx.state.document_summaries)} documents. "
                f"Identified {len(final_terms)} unique bridging terms, "
                f"ranked by semantic relevance to topic."
            )
            logfire.info(
                f"Finalized: {len(final_terms)} bridging terms from {len(ctx.state.document_summaries)} documents"
            )
            # Return final result
            return End(
                BridgingTermsOut(
                    terms=final_terms,
                    scores=final_scores,
                    total_documents_processed=len(ctx.state.document_summaries),
                    rounds_completed=ctx.state.current_round,
                    coverage_assessment=assessment,
                    resources=ctx.deps.resource_pool,
                )
            )
