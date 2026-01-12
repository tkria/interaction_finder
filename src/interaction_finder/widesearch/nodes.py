"""Graph nodes for widesearch pipeline.

Each node represents a stage in the pipeline. Nodes control flow via
return type annotations. All branching and looping happens in nodes;
agents only produce typed data.
"""

import asyncio
from dataclasses import dataclass
from typing import Union

from pydantic_graph import BaseNode, End, GraphRunContext
from interaction_finder.agent_utils import rename_agent
from interaction_finder.logging import logfire, get_logger
from interaction_finder.usage import record_usage

logger = get_logger(__name__)
from interaction_finder.search.models import SearchQuery, SearchResult
from interaction_finder.widesearch.agents import (
    get_goal_planner_agent,
    get_query_generator_agent,
    get_reflector_agent,
    get_result_selector_agent,
)
from interaction_finder.widesearch.deps import Deps
from interaction_finder.widesearch.state import State


@dataclass
class PlanGoalsNode(BaseNode[State, Deps, list[SearchResult]]):
    """Initial planning node: identifies subject goals to cover.

    Uses goal_planner_agent to identify subject areas and research domains
    that should be covered during the search session for comprehensive
    literature discovery.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "GenerateQueriesNode":
        """Plan subject goals and initialize the search session."""
        logger.info(f"Planning subject goals for topic: {ctx.state.topic}")
        # Use goal planner agent with renamed span
        agent = get_goal_planner_agent(ctx.deps.config)
        prompt = f"""Research topic: {ctx.state.topic}

Keyphrases available: {", ".join(ctx.state.keyphrases)}

Identify subject areas and research domains that should be covered to ensure comprehensive literature discovery."""
        with rename_agent(agent, "PlanGoalsNode"):
            result = await agent.run(prompt, deps=ctx.deps)
        record_usage(ctx.deps.usage, "goal_planner", agent, result)
        # Store goals in state
        ctx.state.subject_goals = result.output.goals
        logger.info(
            f"Identified {len(result.output.goals)} subject goals",
            goals=result.output.goals,
            reasoning=result.output.reasoning[:200],
        )
        return GenerateQueriesNode()


@dataclass
class GenerateQueriesNode(BaseNode[State, Deps, list[SearchResult]]):
    """Generate search queries targeting unsatisfied subject goals.

    Uses get_query_generator_agent to create diverse queries that target
    unsatisfied subject goals and incorporate provided keyphrases.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "SearchNode":
        """Generate queries for current round."""
        # Increment round counter
        ctx.state.current_round += 1
        # Reset per-round counters and update progress display
        if ctx.deps.progress:
            ctx.deps.progress["Searches run"].completed = 0
            ctx.deps.progress["Searches run"].in_progress = 0
            ctx.deps.progress["Searches run"].total = 0
            ctx.deps.progress["Round"].total = ctx.state.max_rounds
            ctx.deps.progress["Round"].completed = ctx.state.current_round
            ctx.deps.progress["Round"].activate()
        logger.info(f"Starting round {ctx.state.current_round}/{ctx.state.max_rounds}")
        # Prepare context for agent
        unsatisfied = [
            g for g in ctx.state.subject_goals if g not in ctx.state.satisfied_goals
        ]
        prompt = f"""Research topic: {ctx.state.topic}

Keyphrases to incorporate: {", ".join(ctx.state.keyphrases)}

Subject goals (unsatisfied): {", ".join(unsatisfied) if unsatisfied else "(all goals satisfied)"}

Round {ctx.state.current_round} of {ctx.state.max_rounds}

Generate search queries that target unsatisfied subject goals and incorporate the keyphrases."""
        # Use query generator agent with backend-specific prompt
        backend_name = ctx.deps.search_backend.name
        agent = get_query_generator_agent(ctx.deps.config, backend_name)
        with rename_agent(
            agent, name=f"GenerateQueriesNode (round {ctx.state.current_round})"
        ):
            result = await agent.run(prompt, deps=ctx.deps)
        record_usage(ctx.deps.usage, "query_generator", agent, result)
        # Store queries in state
        ctx.state.current_queries = result.output.queries
        ctx.state.all_queries.extend(result.output.queries)
        logger.info(
            f"Generated {len(result.output.queries)} queries "
            f"(broad={len(result.output.broad_queries)}, "
            f"medium={len(result.output.medium_queries)}, "
            f"focused={len(result.output.focused_queries)}, "
            f"indirect={len(result.output.indirect_queries)})",
            queries=result.output.queries,
            broad_queries=result.output.broad_queries,
            medium_queries=result.output.medium_queries,
            focused_queries=result.output.focused_queries,
            indirect_queries=result.output.indirect_queries,
            reasoning=result.output.reasoning[:200],
        )
        return SearchNode()


@dataclass
class SearchNode(BaseNode[State, Deps, list[SearchResult]]):
    """Execute searches for all queries generated in GenerateQueries.

    Uses the search backend from deps to fetch results for each query.
    Aggregates all results for the current round.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "RerankNode":
        """Execute searches and store results."""
        with logfire.span(
            "SearchNode",
            num_queries=len(ctx.state.current_queries),
            round=ctx.state.current_round,
        ):
            # Set phase to searching with backend name and initialize progress
            if ctx.deps.progress:
                backend_name = ctx.deps.search_backend.name
                num_queries = len(ctx.state.current_queries)
                ctx.deps.progress["Searches run"].total = num_queries
                ctx.deps.progress["Searches run"].work(num_queries)
                ctx.deps.progress["Searches run"].activate()
                ctx.deps.progress.set_status(f"Searching {backend_name}")

            # Execute all searches concurrently, tracking progress as they complete
            async def execute_search(query_text: str) -> list[SearchResult]:
                query = SearchQuery(
                    query=query_text,
                    max_results=ctx.deps.config.stage.search.results_per_query,
                )
                try:
                    return await ctx.deps.search_backend.search(query)
                except Exception as e:
                    logger.warning(
                        f"Search failed, skipping query: {query_text!r}",
                        error=str(e),
                    )
                    return []

            all_results: list[SearchResult] = []
            tasks = [execute_search(q) for q in ctx.state.current_queries]
            for coro in asyncio.as_completed(tasks):
                results = await coro
                all_results.extend(results)

                # Update progress display
                if ctx.deps.progress:
                    ctx.deps.progress["Searches run"].done()
                    ctx.deps.progress["Results found"].add(len(results))

            # Store in state
            ctx.state.current_results = all_results

            # Count unique URLs
            unique_urls = len(set(r.url for r in all_results))

            logger.info(
                f"Fetched {len(all_results)} results ({unique_urls} unique)",
                total_results=len(all_results),
                unique_urls=unique_urls,
                results=all_results,
            )

            return RerankNode()


@dataclass
class RerankNode(BaseNode[State, Deps, list[SearchResult]]):
    """Rerank search results by semantic similarity to topic.

    Uses the reranker from deps to improve result ordering. Returns
    top-k results based on configuration.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "SelectResultsNode":
        """Rerank results and update state (or skip if disabled)."""
        with logfire.span(
            f"Rerank: {len(ctx.state.current_results)} results",
            num_results=len(ctx.state.current_results),
            round=ctx.state.current_round,
        ):
            if not ctx.state.current_results:
                logger.info(
                    "No results to rerank",
                    input_count=0,
                    output_count=0,
                    results=[],
                )
                return SelectResultsNode()

            # Check if reranking is enabled (rerank_top_k > 0)
            top_k = ctx.deps.config.stage.search.rerank_top_k

            if top_k == 0 or ctx.deps.reranker is None:
                logger.info(
                    f"Reranking disabled, passing {len(ctx.state.current_results)} results unchanged",
                    input_count=len(ctx.state.current_results),
                    output_count=len(ctx.state.current_results),
                    reranking_enabled=False,
                )
                return SelectResultsNode()
            # Set phase to reranking
            if ctx.deps.progress:
                ctx.deps.progress.set_status("Reranking results")
            # Rerank using topic as query
            reranked = ctx.deps.reranker.rerank(
                ctx.state.topic, ctx.state.current_results, top_k=top_k
            )

            # Update state with reranked results
            ctx.state.current_results = reranked

            logger.info(
                f"Reranked {len(reranked)} results",
                input_count=len(ctx.state.current_results),
                output_count=len(reranked),
                top_k=top_k,
                results=reranked,
            )

            return SelectResultsNode()


@dataclass
class SelectResultsNode(BaseNode[State, Deps, list[SearchResult]]):
    """Select which search results to fetch and register with ResourcePool.

    Uses get_result_selector_agent to choose the most promising results and
    summarize what subject areas they cover. Registers selected URLs with
    the ResourcePool. Supports batching for processing large result sets.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "ReflectNode":
        """Select results and register with ResourcePool."""
        with logfire.span(
            "SelectResultsNode",
            num_results=len(ctx.state.current_results),
            round=ctx.state.current_round,
        ):
            if not ctx.state.current_results:
                logger.info("No results available, skipping to reflection")
                # Add empty summary for this round
                ctx.state.search_summaries.append("(no results found this round)")
                return ReflectNode()

            # Determine batching strategy
            batch_size = ctx.deps.config.stage.search.batch_size
            if batch_size == 0 or len(ctx.state.current_results) <= batch_size:
                # Process all results in a single batch
                await self._process_batch(
                    ctx, ctx.state.current_results, batch_offset=0
                )
            else:
                # Split into batches and process each
                results = ctx.state.current_results
                num_batches = (len(results) + batch_size - 1) // batch_size
                logger.info(
                    f"Processing {len(results)} results in {num_batches} batches of size {batch_size}"
                )

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(results))
                    batch = results[start_idx:end_idx]
                    await self._process_batch(ctx, batch, batch_offset=start_idx)

            return ReflectNode()

    async def _process_batch(
        self,
        ctx: GraphRunContext[State, Deps],
        batch: list[SearchResult],
        batch_offset: int,
    ) -> None:
        """Process a single batch of results with the result selector agent.

        Parameters:
            ctx: Graph run context with state and deps
            batch: list[SearchResult] — batch of results to process
            batch_offset: int — offset for mapping indices back to full result list
        """
        # Set phase to selecting
        if ctx.deps.progress:
            ctx.deps.progress["Results selected"].activate()
            ctx.deps.progress.set_status("Selecting results")
        # Prepare context for agent
        results_context = "\n\n".join(
            [
                f"[{i}] {r.title}\n{r.snippet or '(no snippet)'}\nURL: {r.url}"
                for i, r in enumerate(batch)
            ]
        )
        prompt = f"""Research topic: {ctx.state.topic}

Subject goals: {", ".join(ctx.state.subject_goals)}

Search results from round {ctx.state.current_round}:
{results_context}

Select the most relevant results and summarize what subject areas they cover."""
        # Use result selector agent with renamed span
        agent = get_result_selector_agent(ctx.deps.config)
        with rename_agent(
            agent, name=f"SelectResultsNode (batch {batch_offset // len(batch) + 1})"
        ):
            result = await agent.run(prompt, deps=ctx.deps)
        record_usage(ctx.deps.usage, "result_selector", agent, result)
        # Register selected results with ResourcePool
        selected_urls = []
        registered_count = 0
        for idx in result.output.selected_indices:
            # Map batch-local index to full result list
            if 0 <= idx < len(batch):
                search_result = batch[idx]
                selected_urls.append(search_result.url)
                # Store full SearchResult for metadata preservation
                ctx.state.selected_search_results[search_result.url] = search_result
                # Register URL with resource pool (may raise ValueError if duplicate)
                try:
                    ctx.deps.resource_pool.register(search_result.url)
                    registered_count += 1
                except ValueError:
                    # URL already registered, skip
                    pass
        # Update progress display with selected count
        if ctx.deps.progress:
            ctx.deps.progress["Results selected"].add(
                len(result.output.selected_indices)
            )
        # Track selected URLs per query
        for query in ctx.state.current_queries:
            if query not in ctx.state.selected_results:
                ctx.state.selected_results[query] = []
            ctx.state.selected_results[query].extend(selected_urls)
        # Accumulate coverage summaries
        ctx.state.search_summaries.append(result.output.covered_topics_summary)
        # Compute rejected indices for logging
        all_indices = set(range(len(batch)))
        selected_indices_set = set(result.output.selected_indices)
        rejected_indices = sorted(all_indices - selected_indices_set)
        # Build selected and rejected result info for logging
        selected_results_info = [
            {"index": idx, "title": batch[idx].title, "url": batch[idx].url}
            for idx in result.output.selected_indices
            if 0 <= idx < len(batch)
        ]
        rejected_results_info = [
            {"index": idx, "title": batch[idx].title, "url": batch[idx].url}
            for idx in rejected_indices
        ]
        logger.info(
            f"Batch processed: selected {len(result.output.selected_indices)} results ({registered_count} new URLs registered)",
            batch_size=len(batch),
            batch_offset=batch_offset,
            selected_count=len(result.output.selected_indices),
            registered_count=registered_count,
            rejected_count=len(rejected_indices),
            covered_topics=result.output.covered_topics_summary[:200],
            reasoning=result.output.reasoning,
            selected_results=selected_results_info,
            rejected_results=rejected_results_info,
        )


@dataclass
class ReflectNode(BaseNode[State, Deps, list[SearchResult]]):
    """Reflect on search coverage and decide whether to continue.

    Uses get_reflector_agent to evaluate coverage, update satisfied goals,
    identify new goals, and decide whether to perform another search round
    or stop. Enforces max_rounds limit.
    """

    async def run(
        self, ctx: GraphRunContext[State, Deps]
    ) -> Union["GenerateQueriesNode", End[list[SearchResult]]]:
        """Reflect on coverage and decide next action."""
        # Check if we've reached max_rounds
        if ctx.state.current_round >= ctx.state.max_rounds:
            logger.info(f"Reached max_rounds ({ctx.state.max_rounds}), stopping")
            # Return all unique results collected, preserving metadata
            all_registered = [
                url for urls in ctx.state.selected_results.values() for url in urls
            ]
            unique_urls = list(set(all_registered))
            # Retrieve full SearchResult objects from state
            final_results = [
                ctx.state.selected_search_results[url] for url in unique_urls
            ]
            return End(final_results)
        # Prepare context for agent
        satisfied_str = (
            ", ".join(ctx.state.satisfied_goals)
            if ctx.state.satisfied_goals
            else "(none yet)"
        )
        summaries_str = "\n\n".join(
            [
                f"Round {i + 1}: {summary}"
                for i, summary in enumerate(ctx.state.search_summaries)
            ]
        )
        prompt = f"""Research topic: {ctx.state.topic}

Subject goals: {", ".join(ctx.state.subject_goals)}

Satisfied goals: {satisfied_str}

Search summaries so far:
{summaries_str}

Current round: {ctx.state.current_round} of {ctx.state.max_rounds}

Evaluate coverage and decide whether to continue searching or stop."""
        # Use reflector agent with renamed span
        agent = get_reflector_agent(ctx.deps.config)
        with rename_agent(
            agent,
            name=f"ReflectNode (round {ctx.state.current_round}/{ctx.state.max_rounds})",
        ):
            result = await agent.run(prompt, deps=ctx.deps)
        record_usage(ctx.deps.usage, "reflector", agent, result)
        # Update satisfied goals
        ctx.state.satisfied_goals.extend(result.output.satisfied_goals)
        # Deduplicate satisfied goals
        ctx.state.satisfied_goals = list(set(ctx.state.satisfied_goals))
        # Add any new goals discovered
        if result.output.new_goals:
            ctx.state.subject_goals.extend(result.output.new_goals)
            logger.info(
                f"Added {len(result.output.new_goals)} new subject goals",
                new_goals=result.output.new_goals,
            )
        # Update continue flag
        ctx.state.should_continue = result.output.should_continue
        logger.info(
            f"Reflection complete: {'continue' if result.output.should_continue else 'stop'}",
            satisfied_goals=len(ctx.state.satisfied_goals),
            total_goals=len(ctx.state.subject_goals),
            decision=result.output.should_continue,
            reasoning=result.output.reasoning[:200],
        )
        # Decide next action
        if result.output.should_continue:
            return GenerateQueriesNode()
        else:
            # Return all unique results collected, preserving metadata
            all_registered = [
                url for urls in ctx.state.selected_results.values() for url in urls
            ]
            unique_urls = list(set(all_registered))
            # Retrieve full SearchResult objects from state
            final_results = [
                ctx.state.selected_search_results[url] for url in unique_urls
            ]
            logger.info(
                f"Search complete: collected {len(unique_urls)} unique URLs across {ctx.state.current_round} rounds"
            )
            return End(final_results)
