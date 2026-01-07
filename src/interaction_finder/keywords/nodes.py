"""Graph nodes for keyword research pipeline.

Each node represents a stage in the pipeline. Nodes control flow via
return type annotations. All branching and looping happens in nodes;
agents only produce typed data.
"""

import asyncio
from dataclasses import dataclass
from typing import Union

from pydantic_graph import BaseNode, End, GraphRunContext

from interaction_finder.agent_utils import rename_agent
from interaction_finder.logging import get_logger
from interaction_finder.usage import record_usage

logger = get_logger(__name__)
from interaction_finder.keywords.agents import (
    get_document_summarizer_agent,
    get_query_expander_agent,
    get_reflector_agent,
    get_result_selector_agent,
)
from interaction_finder.keywords.deps import Deps
from interaction_finder.keywords.extractors.base import ScoredKeyword
from interaction_finder.keywords.models import BridgingTermsOut
from interaction_finder.keywords.normalization import normalize_term_for_deduplication
from interaction_finder.keywords.state import State
from interaction_finder.logging import logfire
from interaction_finder.resources import compute_chunk_spans
from interaction_finder.search.models import SearchQuery


def _clean_and_rerank_keywords_for_display(
    keywords: list[ScoredKeyword],
    topic: str,
    reranker,
    max_keywords: int = 50,
) -> str:
    """Clean, deduplicate, and rerank keywords by topic similarity.

    Filters noise, deduplicates variants, then reranks by semantic similarity
    to the target topic (if reranker provided). Returns top-ranked keywords
    with scores for LLM review.

    Parameters:
        keywords: list[ScoredKeyword] — keywords from all extraction methods
        topic: str — target research topic
        reranker — reranker instance with rerank_terms method (or None to skip reranking)
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
    # Rerank by semantic similarity to topic (if reranker available)
    if reranker is not None:
        reranked = reranker.rerank_terms(topic, unique_terms, top_k=max_keywords)
        terms_only = [term for term, _ in reranked]
    else:
        # No reranking: take top-k by original scores
        sorted_by_score = sorted(
            terms_by_normalized.items(), key=lambda x: x[1][1], reverse=True
        )
        terms_only = [kw for _, (kw, _) in sorted_by_score[:max_keywords]]
    return ", ".join(terms_only)


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
        # Update progress (round counter)
        if ctx.deps.progress:
            ctx.deps.progress["Round"].total = ctx.state.max_rounds
            ctx.deps.progress["Round"].completed = ctx.state.current_round
            ctx.deps.progress["Round"].activate()
        # Use query expander agent with backend-specific prompt
        backend_name = ctx.deps.search_backend.name
        agent = get_query_expander_agent(ctx.deps.config, backend_name)
        with rename_agent(agent, f"ExpandQueryNode (round {ctx.state.current_round})"):
            result = await agent.run(
                f"Generate search queries to find review articles about: {ctx.state.topic}",
                deps=ctx.deps,
            )
        record_usage(ctx.deps.usage, "query_expander", agent, result)
        # Store queries in state
        ctx.state.search_queries = result.output.queries
        logger.info(
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
        # Update progress to show searching status
        if ctx.deps.progress:
            backend_name = type(ctx.deps.search_backend).__name__.replace("Backend", "")
            ctx.deps.progress["Searches run"].activate()
            ctx.deps.progress["Results found"].activate()
            ctx.deps.progress.set_status(f"Searching {backend_name}")
        with logfire.span(
            f"Keyword search: {ctx.state.topic}",
            topic=ctx.state.topic,
            num_queries=len(ctx.state.search_queries),
        ):
            # Execute all searches
            all_results = []
            for query_text in ctx.state.search_queries:
                query = SearchQuery(
                    query=query_text,
                    max_results=ctx.deps.config.stage.keywords.max_results_per_query,
                )
                results = await ctx.deps.search_backend.search(query)
                all_results.extend(results)
                # Update progress counters
                if ctx.deps.progress:
                    ctx.deps.progress["Searches run"].add()
                    ctx.deps.progress["Results found"].add(len(results))
            # Store in state
            ctx.state.all_search_results = all_results

            # Count unique URLs
            unique_urls = len(set(r.url for r in all_results))

            logger.info(
                f"Fetched {len(all_results)} results ({unique_urls} unique)",
                queries=ctx.state.search_queries,
                total_results=len(all_results),
                unique_urls=unique_urls,
                results=all_results,
            )
            return RerankNode()


@dataclass
class RerankNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Rerank search results by semantic similarity to topic.

    Uses the reranker from deps to improve result ordering. Returns
    top-k results based on configuration.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "SelectResultsNode":
        """Rerank results and update state (or skip if disabled)."""
        with logfire.span(
            f"Rerank: {len(ctx.state.all_search_results)} results",
            num_results=len(ctx.state.all_search_results),
        ):
            if not ctx.state.all_search_results:
                logger.info(
                    "No results to rerank",
                    input_count=0,
                    output_count=0,
                    results=[],
                )
                return SelectResultsNode()
            # Check if reranking is enabled (rerank_top_k > 0)
            top_k = ctx.deps.config.stage.keywords.rerank_top_k
            if top_k == 0 or ctx.deps.reranker is None:
                logger.info(
                    f"Reranking disabled, passing {len(ctx.state.all_search_results)} results unchanged",
                    input_count=len(ctx.state.all_search_results),
                    output_count=len(ctx.state.all_search_results),
                    reranking_enabled=False,
                )
                return SelectResultsNode()
            # Update progress to show reranking
            if ctx.deps.progress:
                ctx.deps.progress.set_status("Reranking results")
            # Rerank using topic as query
            reranked = ctx.deps.reranker.rerank(
                ctx.state.topic, ctx.state.all_search_results, top_k=top_k
            )
            # Update state with reranked results
            ctx.state.all_search_results = reranked

            logger.info(
                f"Reranked {len(reranked)} results",
                input_count=len(ctx.state.all_search_results),
                output_count=len(reranked),
                top_k=top_k,
                results=reranked,
            )
            return SelectResultsNode()


@dataclass
class SelectResultsNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Select which search results to fetch and process.

    Uses result_selector_agent to choose the most promising results.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "FetchDocumentsNode":
        """Select results to fetch."""
        if not ctx.state.all_search_results:
            logger.info("No search results available, skipping to finalization")
            return FinalizeNode()
        # Update progress to show selection
        if ctx.deps.progress:
            ctx.deps.progress.set_status("Selecting results")
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
        # Use result selector agent with renamed span
        agent = get_result_selector_agent(ctx.deps.config)
        with rename_agent(agent, "SelectResultsNode"):
            result = await agent.run(prompt, deps=ctx.deps)
        record_usage(ctx.deps.usage, "result_selector", agent, result)
        # Get selected results
        max_to_fetch = ctx.deps.config.stage.keywords.max_documents_to_fetch
        selected_indices = result.output.selected_indices[:max_to_fetch]
        ctx.state.selected_results = [
            ctx.state.all_search_results[i]
            for i in selected_indices
            if i < len(ctx.state.all_search_results)
        ]
        # Set Fetched total now that we know how many documents to fetch
        if ctx.deps.progress:
            ctx.deps.progress["Fetched"].total = len(ctx.state.selected_results)
            ctx.deps.progress["Fetched"].activate()
        logger.info(
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
        # Update progress to show fetching
        if ctx.deps.progress:
            ctx.deps.progress.set_status("Fetching documents")
        with logfire.span(
            f"Fetch {len(ctx.state.selected_results)} documents",
            num_selected=len(ctx.state.selected_results),
        ):
            if not ctx.state.selected_results:
                logger.info("No selected results, skipping to finalization")
                return FinalizeNode()
            # Separate new URLs from already-fetched (cached in resource pool)
            urls_to_fetch = []
            resource_ids_map = {}
            cached_count = 0
            for result in ctx.state.selected_results:
                if result.url not in ctx.deps.resource_pool:
                    # Register new resource
                    rid = ctx.deps.resource_pool.register(result.url)
                    urls_to_fetch.append((result.url, result.title))
                    resource_ids_map[result.url] = rid
                else:
                    cached_count += 1
            # Update progress: cached documents are already complete
            if ctx.deps.progress and cached_count > 0:
                ctx.deps.progress["Fetched"].completed = cached_count
            # Skip fetching if all URLs already processed
            if not urls_to_fetch:
                logger.info(
                    f"All {len(ctx.state.selected_results)} documents cached, proceeding to extraction"
                )
                if ctx.deps.progress:
                    ctx.deps.progress["Fetched"].complete()
                return ExtractKeywordsNode()
            # Fetch only new URLs
            urls = [url for url, _ in urls_to_fetch]
            # Fetch documents with DOI metadata
            documents = await ctx.deps.fetcher.fetch_documents(
                urls, progress=False, fail_fast=False, retry=False
            )
            # Fetch chunks for all URLs in batch
            chunk_lists = await ctx.deps.fetcher.get_chunks(
                urls, progress=False, fail_fast=False, retry=False
            )
            # Add content to resource pool and track failures
            fetched_count = 0
            failed_count = 0
            for (url, title), chunks in zip(urls_to_fetch, chunk_lists):
                doc = documents.get(url)
                if doc:
                    rid = resource_ids_map[url]
                    # Compute chunk spans from full text and chunk texts
                    chunk_spans = (
                        compute_chunk_spans(doc.content_markdown, chunks)
                        if chunks
                        else None
                    )
                    ctx.deps.resource_pool.add_content(
                        rid,
                        title,
                        doc.content_markdown,
                        chunks=chunk_spans,
                        doi=doc.doi,
                        publication_date=doc.publication_date,
                    )
                    fetched_count += 1
                else:
                    failed_count += 1
            # Update progress: mark fetching complete
            if ctx.deps.progress:
                ctx.deps.progress["Fetched"].completed = cached_count + fetched_count
                ctx.deps.progress["Fetched"].complete()
            logger.info(
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
        # Update progress to show extraction
        if ctx.deps.progress:
            ctx.deps.progress.set_status("Extracting keywords")
        resources = ctx.deps.resource_pool.resources
        with logfire.span("ExtractKeywordsNode", num_resources=len(resources)):
            if not resources:
                logger.info("No resources available, skipping to finalization")
                return FinalizeNode()
            max_keywords = ctx.deps.config.stage.keywords.max_keywords_per_method
            # Track already-processed URLs to avoid re-extraction
            already_processed = set(ctx.state.extracted_keywords.keys())
            new_resources = [r for r in resources if r.id.url not in already_processed]
            if not new_resources and not already_processed:
                logger.info(
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
                    logger.warning(
                        f"Extractor {name} failed for document",
                        extractor=name,
                        url=url,
                        title=title[:60],
                        error=str(error),
                    )
            total_keywords = sum(
                len(kws) for kws in ctx.state.extracted_keywords.values()
            )
            # Update progress: set Processed total (documents to evaluate) and Keywords total
            if ctx.deps.progress:
                ctx.deps.progress["Processed"].total = len(ctx.state.extracted_keywords)
                ctx.deps.progress["Processed"].activate()
                ctx.deps.progress["Keywords"].total = total_keywords
                ctx.deps.progress["Keywords"].activate()
            logger.info(
                f"Extracted {total_keywords} keywords from {len(new_resources)} new documents ({len(already_processed)} previously processed)"
            )
            return EvaluateKeywordsNode()


async def _evaluate_single_document(
    url: str,
    keywords: list[ScoredKeyword],
    topic: str,
    config,
    reranker,
    resource_pool,
    deps,
):
    """Evaluate keywords and summarize a single document.

    Parameters:
        url: str — document URL
        keywords: list[ScoredKeyword] — extracted keywords for this document
        topic: str — research topic
        config — configuration object
        reranker — reranker instance (or None)
        resource_pool — resource pool to fetch document from
        deps — agent dependencies

    Returns:
        tuple[DocumentSummary, int] | None — (summary, bridging_count) or None on failure
    """
    try:
        # Get content from resource pool
        resource = resource_pool.get(url)
        if not resource:
            return None
        # Clean, deduplicate, and rerank keywords for LLM review
        max_keywords_for_llm = config.stage.keywords.max_keywords_for_llm
        keywords_text = _clean_and_rerank_keywords_for_display(
            keywords, topic, reranker, max_keywords_for_llm
        )
        # Summarize document with strict filtering instructions
        summary_prompt = f"""Review this document and identify HIGH-QUALITY bridging terms.

**Target topic:** {topic}

**Document content:**
{resource.text}

**Extracted keywords (ranked by relevance to topic):**
{keywords_text}

---

**Your task:**
1. Summary: Concisely describe what this document contributes to understanding the target topic
2. Related research areas: List only areas that directly inform mechanisms, contexts, or biological frameworks connected to the topic
3. Bridging terms: Identify 5-10 HIGH-QUALITY bridging terms

**What are bridging terms?**

Bridging terms are search keywords for the next literature discovery stage. They help locate papers containing relevant information from different angles—including papers that might not use your primary search terms directly.

They should represent research CONTEXTS (disease subtypes, mechanisms, broad pathways, clinical presentations, related conditions) rather than specific entity instances being sought, because:
- Context terms enable search diversification across different perspectives
- Entity-specific terms just repeatedly find papers focused on those entities
- Goal: discover information through varied lenses, not enumerate known entities

**Identify what the topic is asking for**
When the topic seeks specific entity types (genes, proteins, drugs, markers, ligands, antibodies):
→ Extract contexts around those entities (disease subtypes, pathways, mechanisms, clinical presentations)
→ Do not extract the specific entity instances themselves

**Examples:**
Topic: "genes associated with pulmonary hypertension"
✓ GOOD: "familial PAH", "Eisenmenger syndrome", "BMP signaling pathway", "pulmonary vascular remodeling"
✗ BAD: "BMPR2", "ACVRL1", "ENG" (specific genes being sought)
✗ BAD: "BMPR2 signaling" (still specific to one gene, not a broad pathway)

Topic: "cell markers for regulatory T cells"
✓ GOOD: "Foxp3+ Tregs", "immune suppression", "peripheral tolerance"
✗ BAD: "CD25", "CD127" (specific markers being sought)

**Selection criteria:**

Include terms that are:
- Research contexts: disease subtypes, mechanisms, broad pathways, clinical presentations, related conditions, cell types, tissue contexts
- Directly relevant to "{topic}" (not tangential topics in the document)
- Established biomedical terminology from the field
- Terms that commonly appear in literature about the target topic

Exclude:
- Specific instances of the entity type being sought (if seeking genes→no gene names; if seeking markers→no marker names)
- The target topic itself or trivial rewordings
- Generic terms: "genetic factors", "molecular mechanisms", "risk factors", "clinical outcomes", "biomarkers", "pathogenesis"
- Methodological terms: "genome-wide association studies", "next-generation sequencing", "statistical analysis"
- Ad hoc descriptive phrases; prefer established biomedical terms even if multi-word

Use only concepts supported by the document or widely established in the field; do not invent terms.

**Quality test:**
1. Is this a context/mechanism rather than a specific answer to what I'm searching for?
2. Would this term appear frequently in papers about {topic}?

If "no" to either question, exclude it. Select fewer, higher-quality terms rather than reaching for quantity."""
        # Use document summarizer agent with renamed span (include doc title for context)
        agent = get_document_summarizer_agent(config)
        with rename_agent(agent, f"EvaluateKeywordsNode: {resource.title[:60]}"):
            summary_result = await agent.run(summary_prompt, deps=deps)
        record_usage(deps.usage, "document_summarizer", agent, summary_result)
        # Return summary and bridging term count
        bridging_count = len(summary_result.output.bridging_terms)
        return (summary_result.output, bridging_count)
    except Exception as e:
        # Log error and return None (following codebase pattern)
        logger.warning(
            f"Document evaluation failed",
            url=url,
            error=str(e),
        )
        return None


@dataclass
class EvaluateKeywordsNode(BaseNode[State, Deps, BridgingTermsOut]):
    """Evaluate keywords and summarize each document.

    Uses keyword_evaluator_agent and document_summarizer_agent to:
    1. Identify useful bridging terms
    2. Summarize document content
    3. Assess coverage contribution

    Processes all documents concurrently for improved performance.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "ReflectNode":
        """Evaluate keywords and summarize documents (parallel processing)."""
        # Update progress to show evaluation
        if ctx.deps.progress:
            ctx.deps.progress.set_status("Evaluating keywords")
        if not ctx.state.extracted_keywords:
            logger.info("No extracted keywords, skipping to reflection")
            return ReflectNode()
        # Process all documents in parallel
        tasks = [
            _evaluate_single_document(
                url,
                keywords,
                ctx.state.topic,
                ctx.deps.config,
                ctx.deps.reranker,
                ctx.deps.resource_pool,
                ctx.deps,
            )
            for url, keywords in ctx.state.extracted_keywords.items()
        ]
        results = await asyncio.gather(*tasks)
        # Filter out failures (None) and extract summaries
        successful_results = [r for r in results if r is not None]
        total_bridging = sum(count for _, count in successful_results)
        # Add summaries to state and update progress
        for summary, _ in successful_results:
            ctx.state.document_summaries.append(summary)
        # Update progress: mark Processed and Keywords complete
        if ctx.deps.progress:
            ctx.deps.progress["Processed"].completed = len(successful_results)
            ctx.deps.progress["Processed"].complete()
            ctx.deps.progress["Keywords"].completed = total_bridging
            ctx.deps.progress["Keywords"].complete()
        failed_count = len(results) - len(successful_results)
        logger.info(
            f"Generated {len(successful_results)} summaries with {total_bridging} bridging terms total ({failed_count} failed)"
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
        # Update progress to show reflection
        if ctx.deps.progress:
            ctx.deps.progress.set_status("Reflecting on coverage")
        # Check iteration limit
        if ctx.state.current_round >= ctx.state.max_rounds:
            logger.info(f"Max rounds reached ({ctx.state.max_rounds}), finalizing")
            return FinalizeNode()
        # Check if we have any summaries
        if not ctx.state.document_summaries:
            logger.info("No document summaries available, finalizing")
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
        # Use reflector agent with renamed span
        agent = get_reflector_agent(ctx.deps.config)
        with rename_agent(
            agent,
            name=f"ReflectNode (round {ctx.state.current_round}/{ctx.state.max_rounds})",
        ):
            result = await agent.run(prompt, deps=ctx.deps)
        record_usage(ctx.deps.usage, "reflector", agent, result)
        # Make decision
        decision = result.output.decision
        logger.info(
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
                logger.info("No bridging terms found after deduplication")
                return End(
                    BridgingTermsOut(
                        topic=ctx.state.topic,
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
            # Rerank terms by semantic similarity to topic (if reranker available)
            if ctx.deps.reranker is not None:
                scored_terms = ctx.deps.reranker.rerank_terms(
                    ctx.state.topic, all_terms
                )
                final_terms = [term for term, _ in scored_terms]
                final_scores = [score for _, score in scored_terms]
            else:
                # No reranking: use terms as-is with placeholder scores
                final_terms = all_terms
                final_scores = [1.0] * len(all_terms)
            # Create final assessment
            assessment = (
                f"Completed {ctx.state.current_round} search round(s). "
                f"Processed {len(ctx.state.document_summaries)} documents. "
                f"Identified {len(final_terms)} unique bridging terms, "
                f"ranked by semantic relevance to topic."
            )
            logger.info(
                f"Finalized: {len(final_terms)} bridging terms from {len(ctx.state.document_summaries)} documents",
                top_5=[
                    (t, f"{s:.3f}") for t, s in zip(final_terms[:5], final_scores[:5])
                ],
            )
            # Return final result
            return End(
                BridgingTermsOut(
                    topic=ctx.state.topic,
                    terms=final_terms,
                    scores=final_scores,
                    total_documents_processed=len(ctx.state.document_summaries),
                    rounds_completed=ctx.state.current_round,
                    coverage_assessment=assessment,
                    resources=ctx.deps.resource_pool,
                )
            )
