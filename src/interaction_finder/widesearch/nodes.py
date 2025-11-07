"""Graph nodes for widesearch pipeline.

Each node represents a stage in the pipeline. Nodes control flow via
return type annotations. All branching and looping happens in nodes;
agents only produce typed data.
"""

from dataclasses import dataclass
from typing import Union

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_ai.usage import RunUsage

from interaction_finder.logging import logfire
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
        with logfire.span("PlanGoalsNode", topic=ctx.state.topic):
            logfire.info(f"Planning subject goals for topic: {ctx.state.topic}")

            # Use goal planner agent
            usage = RunUsage()
            prompt = f"""Research topic: {ctx.state.topic}

Keyphrases available: {", ".join(ctx.state.keyphrases)}

Identify subject areas and research domains that should be covered to ensure comprehensive literature discovery."""
            result = await get_goal_planner_agent().run(prompt, deps=ctx.deps, usage=usage)

            # Store goals in state
            ctx.state.subject_goals = result.output.goals

            logfire.info(
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
        with logfire.span(
            "GenerateQueriesNode",
            round=ctx.state.current_round + 1,
            unsatisfied_goals=len(ctx.state.subject_goals)
            - len(ctx.state.satisfied_goals),
        ):
            # Increment round counter
            ctx.state.current_round += 1

            logfire.info(
                f"Starting round {ctx.state.current_round}/{ctx.state.max_rounds}"
            )

            # Prepare context for agent
            unsatisfied = [
                g for g in ctx.state.subject_goals if g not in ctx.state.satisfied_goals
            ]

            prompt = f"""Research topic: {ctx.state.topic}

Keyphrases to incorporate: {", ".join(ctx.state.keyphrases)}

Subject goals (unsatisfied): {", ".join(unsatisfied) if unsatisfied else "(all goals satisfied)"}

Round {ctx.state.current_round} of {ctx.state.max_rounds}

Generate search queries that target unsatisfied subject goals and incorporate the keyphrases."""

            # Use query generator agent
            usage = RunUsage()
            result = await get_query_generator_agent.run(prompt, deps=ctx.deps, usage=usage)

            # Store queries in state
            ctx.state.current_queries = result.output.queries
            ctx.state.all_queries.extend(result.output.queries)

            logfire.info(
                f"Generated {len(result.output.queries)} queries",
                queries=result.output.queries,
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
            logfire.info(f"Executing {len(ctx.state.current_queries)} search queries")

            # Execute all searches
            all_results = []
            for query_text in ctx.state.current_queries:
                query = SearchQuery(
                    query=query_text,
                    max_results=ctx.deps.config.get("results_per_query", 50),
                )
                results = await ctx.deps.search_backend.search(query)
                all_results.extend(results)

            # Store in state
            ctx.state.current_results = all_results

            # Count unique URLs
            unique_urls = len(set(r.url for r in all_results))

            logfire.info(
                f"Found {len(all_results)} total results ({unique_urls} unique URLs)"
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
            "RerankNode",
            num_results=len(ctx.state.current_results),
            round=ctx.state.current_round,
        ):
            if not ctx.state.current_results:
                logfire.info("No results to rerank, skipping")
                return SelectResultsNode()

            # Check if reranking is enabled
            enable_reranking = ctx.deps.config.get("enable_reranking", True)

            if not enable_reranking:
                logfire.info(
                    f"Reranking disabled, passing {len(ctx.state.current_results)} results unchanged"
                )
                return SelectResultsNode()

            logfire.info(f"Reranking {len(ctx.state.current_results)} results")

            # Rerank using topic as query
            top_k = ctx.deps.config.get("rerank_top_k", 20)
            reranked = ctx.deps.reranker.rerank(
                ctx.state.topic, ctx.state.current_results, top_k=top_k
            )

            # Update state with reranked results
            ctx.state.current_results = reranked

            return SelectResultsNode()


@dataclass
class SelectResultsNode(BaseNode[State, Deps, list[SearchResult]]):
    """Select which search results to fetch and register with ResourcePool.

    Uses get_result_selector_agent to choose the most promising results and
    summarize what subject areas they cover. Registers selected URLs with
    the ResourcePool.
    """

    async def run(self, ctx: GraphRunContext[State, Deps]) -> "ReflectNode":
        """Select results and register with ResourcePool."""
        with logfire.span(
            "SelectResultsNode",
            num_results=len(ctx.state.current_results),
            round=ctx.state.current_round,
        ):
            if not ctx.state.current_results:
                logfire.info("No results available, skipping to reflection")
                # Add empty summary for this round
                ctx.state.search_summaries.append("(no results found this round)")
                return ReflectNode()

            # Prepare context for agent
            results_context = "\n\n".join(
                [
                    f"[{i}] {r.title}\n{r.snippet or '(no snippet)'}\nURL: {r.url}"
                    for i, r in enumerate(ctx.state.current_results)
                ]
            )

            prompt = f"""Research topic: {ctx.state.topic}

Subject goals: {", ".join(ctx.state.subject_goals)}

Search results from round {ctx.state.current_round}:
{results_context}

Select the most relevant results and summarize what subject areas they cover."""

            # Use result selector agent
            usage = RunUsage()
            result = await get_result_selector_agent.run(prompt, deps=ctx.deps, usage=usage)

            # Register selected results with ResourcePool
            selected_urls = []
            registered_count = 0
            for idx in result.output.selected_indices:
                if 0 <= idx < len(ctx.state.current_results):
                    search_result = ctx.state.current_results[idx]
                    selected_urls.append(search_result.url)
                    # Register URL with resource pool (may raise ValueError if duplicate)
                    try:
                        ctx.deps.resource_pool.register(search_result.url)
                        registered_count += 1
                    except ValueError:
                        # URL already registered, skip
                        pass

            # Track selected URLs per query
            for query in ctx.state.current_queries:
                if query not in ctx.state.selected_results:
                    ctx.state.selected_results[query] = []
                ctx.state.selected_results[query].extend(selected_urls)

            # Store coverage summary for this round
            ctx.state.search_summaries.append(result.output.covered_topics_summary)

            logfire.info(
                f"Selected {len(result.output.selected_indices)} results ({registered_count} new URLs registered)",
                selected_count=len(result.output.selected_indices),
                registered_count=registered_count,
                covered_topics=result.output.covered_topics_summary[:200],
            )

            return ReflectNode()


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
        with logfire.span(
            "ReflectNode",
            round=ctx.state.current_round,
            max_rounds=ctx.state.max_rounds,
        ):
            # Check if we've reached max_rounds
            if ctx.state.current_round >= ctx.state.max_rounds:
                logfire.info(f"Reached max_rounds ({ctx.state.max_rounds}), stopping")
                # Return all unique results collected
                all_registered = [
                    url for urls in ctx.state.selected_results.values() for url in urls
                ]
                unique_urls = list(set(all_registered))
                final_results = [
                    SearchResult(title="", url=url, snippet=None) for url in unique_urls
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

            # Use reflector agent
            usage = RunUsage()
            result = await get_reflector_agent.run(prompt, deps=ctx.deps, usage=usage)

            # Update satisfied goals
            ctx.state.satisfied_goals.extend(result.output.satisfied_goals)
            # Deduplicate satisfied goals
            ctx.state.satisfied_goals = list(set(ctx.state.satisfied_goals))

            # Add any new goals discovered
            if result.output.new_goals:
                ctx.state.subject_goals.extend(result.output.new_goals)
                logfire.info(
                    f"Added {len(result.output.new_goals)} new subject goals",
                    new_goals=result.output.new_goals,
                )

            # Update continue flag
            ctx.state.should_continue = result.output.should_continue

            logfire.info(
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
                # Return all unique results collected
                all_registered = [
                    url for urls in ctx.state.selected_results.values() for url in urls
                ]
                unique_urls = list(set(all_registered))
                final_results = [
                    SearchResult(title="", url=url, snippet=None) for url in unique_urls
                ]
                logfire.info(
                    f"Search complete: collected {len(unique_urls)} unique URLs across {ctx.state.current_round} rounds"
                )
                return End(final_results)
