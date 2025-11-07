"""Graph nodes for keyword research pipeline.

Each node represents a stage in the pipeline. Nodes control flow via
return type annotations. All branching and looping happens in nodes;
agents only produce typed data.
"""

import asyncio
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
from interaction_finder.keywords.extractors.base import ScoredKeyword
from interaction_finder.keywords.models import BridgingTermsOut
from interaction_finder.keywords.normalization import normalize_term_for_deduplication
from interaction_finder.keywords.state import State
from interaction_finder.search.models import SearchQuery


def _clean_and_rerank_keywords_for_display(
    keywords: list[ScoredKeyword],
    topic: str,
    reranker,
    max_keywords: int = 50,
) -> str:
    """Clean, deduplicate, and rerank keywords by topic similarity.

    Filters noise, deduplicates variants, then reranks by semantic similarity
    to the target topic. Returns top-ranked keywords with scores for LLM review.

    Parameters:
        keywords: list[ScoredKeyword] — keywords from all extraction methods
        topic: str — target research topic
        reranker — reranker instance with rerank_terms method
        max_keywords: int — maximum keywords to return (default: 50)

    Returns:
        str — formatted keyword list with similarity scores
    """
    import re

    noise_patterns = [
        r"copyright|©|disclaimer",
        r"^(figure|fig|table)\b",
        r"^[*#\d\s.()]+$",
    ]
    # Deduplicate using normalization (keep highest-scoring variant)
    terms_by_normalized = {}
    for kw in keywords:
        keyword = kw.keyword.strip()
        # Skip very short/long
        if len(keyword) < 3 or len(keyword) > 80:
            continue
        # Skip if matches noise
        if any(re.search(p, keyword, re.IGNORECASE) for p in noise_patterns):
            continue
        # Deduplicate
        normalized_key = normalize_term_for_deduplication(keyword)
        if (
            normalized_key not in terms_by_normalized
            or kw.score > terms_by_normalized[normalized_key][1]
        ):
            terms_by_normalized[normalized_key] = (keyword, kw.score)
    # Extract unique terms
    unique_terms = [kw for kw, _ in terms_by_normalized.values()]
    if not unique_terms:
        return "(no keywords extracted)"
    # Rerank by semantic similarity to topic
    reranked = reranker.rerank_terms(topic, unique_terms, top_k=max_keywords)
    # Return just the terms (already ranked by relevance)
    terms_only = [term for term, _ in reranked]
    return ", ".join(terms_only)


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
            # Use query expander agent
            usage = RunUsage()
            result = await query_expander_agent.run(
                f"Generate search queries to find review articles about: {ctx.state.topic}",
                deps=ctx.deps,
                usage=usage,
            )
            # Store queries in state
            ctx.state.search_queries = result.output.queries
            logfire.info(
                f"Generated {len(result.output.queries)} queries for round {ctx.state.current_round}",
                queries=result.output.queries,
                reasoning=result.output.reasoning[:200],
            )
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
                logfire.info("No search results available, skipping to finalization")
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
            logfire.info(
                f"Selected {len(ctx.state.selected_results)} results from {len(ctx.state.all_search_results)} available",
                selected_titles=[r.title[:60] for r in ctx.state.selected_results],
                reasoning=result.output.reasoning[:200],
            )
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
                logfire.info(
                    f"All {len(ctx.state.selected_results)} documents cached, proceeding to extraction"
                )
                return ExtractKeywordsNode()
            # Fetch only new URLs
            urls = [url for url, _ in urls_to_fetch]
            contents = await ctx.deps.fetcher.get_markdown(
                urls, progress=False, fail_fast=False, retry=False
            )
            # Add content to resource pool and track failures
            fetched_count = 0
            failed_count = 0
            for (url, title), content in zip(urls_to_fetch, contents):
                if content:
                    rid = resource_ids_map[url]
                    ctx.deps.resource_pool.add_content(rid, title, content)
                    fetched_count += 1
                else:
                    failed_count += 1
            cached_count = len(ctx.state.selected_results) - len(urls_to_fetch)
            logfire.info(
                f"Fetched {fetched_count}/{len(urls_to_fetch)} new documents ({cached_count} from cache, {failed_count} failed)"
            )
            return ExtractKeywordsNode()


async def _extract_keywords_async(extractor, text: str, max_keywords: int):
    """Run single extractor in thread pool (extractors are CPU-bound).

    Parameters:
        extractor — extractor instance with extract() method
        text: str — document text
        max_keywords: int — maximum keywords per extractor

    Returns:
        list[ScoredKeyword] — extracted keywords, or empty list on error
    """
    try:
        # Run CPU-bound extraction in thread pool to avoid blocking event loop
        return await asyncio.to_thread(extractor.extract, text, max_keywords)
    except Exception as e:
        # Return error info for caller to log
        return e


async def _extract_from_resource(resource, extractors: dict, max_keywords: int):
    """Extract keywords from single resource using all extractors (parallel).

    Parameters:
        resource — resource with .id.url, .title, .text attributes
        extractors: dict — {name: extractor} mapping
        max_keywords: int — maximum keywords per extractor

    Returns:
        tuple[str, list[ScoredKeyword], list[tuple[str, Exception]]] —
            (url, combined_keywords, failed_extractors)
    """
    # Run all extractors in parallel for this resource
    tasks = [
        _extract_keywords_async(extractor, resource.text, max_keywords)
        for extractor in extractors.values()
    ]
    results = await asyncio.gather(*tasks)
    # Separate successful extractions from failures
    doc_keywords = []
    failed = []
    for (name, _), result in zip(extractors.items(), results):
        if isinstance(result, Exception):
            failed.append((name, result))
        else:
            doc_keywords.extend(result)
    return (resource.id.url, resource.title, doc_keywords, failed)


async def _summarize_document(url: str, keywords: list, ctx):
    """Summarize single document and extract bridging terms.

    Parameters:
        url: str — document URL
        keywords: list[ScoredKeyword] — extracted keywords
        ctx — GraphRunContext with state and deps

    Returns:
        DocumentSummaryOut — summary with bridging terms
    """
    # Get content from resource pool
    resource = ctx.deps.resource_pool.get(url)
    if not resource:
        # Return None to indicate skip (caller will filter)
        return None
    # Clean, deduplicate, and rerank keywords for LLM review
    max_keywords_for_llm = ctx.deps.config.get("max_keywords_for_llm", 50)
    keywords_text = _clean_and_rerank_keywords_for_display(
        keywords, ctx.state.topic, ctx.deps.reranker, max_keywords_for_llm
    )
    # Get document context length from config
    context_chars = ctx.deps.config.get("document_context_chars", 12000)
    # Summarize document with strict filtering instructions
    summary_prompt = f"""Review this document and identify HIGH-QUALITY bridging terms.

**Target topic:** {ctx.state.topic}

**Document content:**
{resource.text[:context_chars]}

**Extracted keywords (ranked by relevance to topic):**
{keywords_text}

---

**Your task:**
1. Summary: Concisely describe what this document contributes to understanding the target topic
2. Related research areas: List research areas that connect to the target topic
3. Bridging terms: Identify 5-10 HIGH-QUALITY bridging terms

The extracted keywords above are suggestions—you may use them directly, combine them, or identify better terms from the document content.

**BRIDGING TERM REQUIREMENTS:**

Each bridging term must:
  - Be a specific concept, mechanism, pathway, gene, protein, or biological entity
  - Be directly relevant to "{ctx.state.topic}" (not to tangential topics mentioned in the document)
  - Use precise scientific terminology (e.g., "BMPR2 gene" not "genetic mutations")
  - Be a term that commonly appears in scientific literature about the target topic

**EXCLUDE:**
  - The target topic itself or obvious rewordings
  - Generic research terms: "genetic factors", "molecular mechanisms", "risk factors", "clinical outcomes", "biomarkers", "pathogenesis"
  - Methodological terms: "genome-wide association studies", "next-generation sequencing", "statistical analysis"
  - Multi-word descriptive phrases: prefer concise established terms (e.g., "endothelial dysfunction" not "dysfunction of endothelial cells")

**Test:** For each term, ask "Would this term appear frequently in papers specifically about {ctx.state.topic}?" If no, exclude it.

Select fewer, higher-quality terms rather than reaching for quantity."""
    usage = RunUsage()
    summary_result = await document_summarizer_agent.run(
        summary_prompt, deps=ctx.deps, usage=usage
    )
    return summary_result.output


@dataclass
class ExtractKeywordsNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Extract keywords from all fetched documents using multiple methods.

    Runs all extractors (RAKE, YAKE, TF-IDF, KeyBERT) concurrently on each
    document, with two-level parallelism:
    - Resource-level: process multiple documents in parallel
    - Extractor-level: run all extractors on each document in parallel
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "EvaluateKeywordsNode":
        """Extract keywords from all documents in resource pool (parallel)."""
        resources = ctx.deps.resource_pool.resources
        with logfire.span("ExtractKeywordsNode", num_resources=len(resources)):
            if not resources:
                logfire.info("No resources available, skipping to finalization")
                return FinalizeNode()
            max_keywords = ctx.deps.config.get("max_keywords_per_method", 30)
            # Track already-processed URLs to avoid re-extraction
            already_processed = set(ctx.state.extracted_keywords.keys())
            new_resources = [r for r in resources if r.id.url not in already_processed]
            if not new_resources and not already_processed:
                logfire.info(
                    "No resources to extract keywords from, skipping to finalization"
                )
                return FinalizeNode()
            # Extract keywords in parallel: both resources AND extractors within each resource
            tasks = [
                _extract_from_resource(resource, ctx.deps.extractors, max_keywords)
                for resource in new_resources
            ]
            results = await asyncio.gather(*tasks)
            # Process results and log any failures
            for url, title, doc_keywords, failed in results:
                ctx.state.extracted_keywords[url] = doc_keywords
                # Log any extractor failures
                for name, error in failed:
                    logfire.warning(
                        f"Extractor {name} failed for document",
                        extractor=name,
                        url=url,
                        title=title[:60],
                        error=str(error),
                    )
            total_keywords = sum(
                len(kws) for kws in ctx.state.extracted_keywords.values()
            )
            logfire.info(
                f"Extracted {total_keywords} keywords from {len(new_resources)} new documents ({len(already_processed)} previously processed)"
            )
            return EvaluateKeywordsNode()


@dataclass
class EvaluateKeywordsNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Evaluate keywords and summarize each document (parallel).

    Uses document_summarizer_agent to:
    1. Identify useful bridging terms
    2. Summarize document content
    3. Assess coverage contribution

    All documents are processed in parallel for maximum throughput.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "ReflectNode":
        """Evaluate keywords and summarize documents (parallel)."""
        with logfire.span(
            "EvaluateKeywordsNode", num_docs=len(ctx.state.extracted_keywords)
        ):
            if not ctx.state.extracted_keywords:
                logfire.info("No extracted keywords, skipping to reflection")
                return ReflectNode()
            # Create tasks for all documents (parallel processing)
            tasks = [
                _summarize_document(url, keywords, ctx)
                for url, keywords in ctx.state.extracted_keywords.items()
            ]
            # Execute all summarization tasks in parallel
            results = await asyncio.gather(*tasks)
            # Filter out None results (skipped documents) and add to state
            summaries = [r for r in results if r is not None]
            ctx.state.document_summaries.extend(summaries)
            # Calculate stats
            total_bridging = sum(len(s.bridging_terms) for s in summaries)
            logfire.info(
                f"Generated {len(summaries)} summaries with {total_bridging} bridging terms total"
            )
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
                logfire.info(f"Max rounds reached ({ctx.state.max_rounds}), finalizing")
                return FinalizeNode()
            # Check if we have any summaries
            if not ctx.state.document_summaries:
                logfire.info("No document summaries available, finalizing")
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
            decision = result.output.decision
            logfire.info(
                f"Reflection: {decision} after round {ctx.state.current_round}",
                decision=decision,
                reasoning=result.output.reasoning,
                new_search_angles=result.output.new_search_angles,
            )
            if decision == "stop":
                return FinalizeNode()
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
                logfire.info("No bridging terms found after deduplication")
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
                f"Finalized: {len(final_terms)} bridging terms from {len(ctx.state.document_summaries)} documents",
                top_5=[
                    (t, f"{s:.3f}") for t, s in zip(final_terms[:5], final_scores[:5])
                ],
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
