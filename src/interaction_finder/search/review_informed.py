"""
Review-informed search mode for enhanced query generation using review papers.

This module implements a sophisticated search strategy that first discovers review
papers, analyzes them to extract key concepts and research directions, then uses
this knowledge to generate highly targeted search queries for primary research.
"""

import asyncio
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from .base import SearchBackend, SearchQuery, SearchResults, SearchResult
from ..fetcher import PageFetcher

# Console for styled output
console = Console()


@dataclass
class ReviewKnowledge:
    """Knowledge extracted from a single review paper."""

    title: str
    key_concepts: List[str]  # Important entities, phenomena, concepts
    methodologies: List[str]  # Research methods and approaches mentioned
    terminology: List[str]  # Domain-specific terms and abbreviations
    research_gaps: List[str]  # Identified gaps and future directions
    seminal_papers: List[str]  # Important citations or author names
    emerging_themes: List[str]  # New research directions mentioned


@dataclass
class InformedQuery:
    """A query generated from review analysis with metadata."""

    query: str
    focus: str  # Research angle (e.g., "methodology_focus", "emerging_theme")
    rationale: str  # Why this query strategy is valuable
    source_reviews: List[str]  # Titles of reviews that informed this query
    weight: float = 1.0  # Relative importance weight


class ReviewKnowledgeResponse(BaseModel):
    """LLM response for review knowledge extraction."""

    key_concepts: List[str] = Field(
        description="Important biological entities, phenomena, or concepts discussed"
    )
    methodologies: List[str] = Field(
        description="Research methodologies, techniques, or experimental approaches"
    )
    terminology: List[str] = Field(
        description="Domain-specific terms, abbreviations, or technical vocabulary"
    )
    research_gaps: List[str] = Field(
        description="Identified research gaps, limitations, or future directions"
    )
    seminal_papers: List[str] = Field(
        description="Important authors, landmark studies, or influential papers mentioned"
    )
    emerging_themes: List[str] = Field(
        description="New research directions, emerging technologies, or recent developments"
    )


class InformedQueryResponse(BaseModel):
    """LLM response for informed query generation."""

    class QueryStrategy(BaseModel):
        query: str = Field(description="The specific search query")
        focus: str = Field(description="Research focus or angle")
        rationale: str = Field(description="Why this strategy is valuable")

    queries: List[QueryStrategy] = Field(
        description="List of informed search strategies"
    )
    synthesis_notes: str = Field(
        description="How the review knowledge was synthesized for query generation"
    )


class ReviewAnalyzer:
    """Analyzes review papers to extract research knowledge."""

    def __init__(
        self, model_name: str = "openai:gpt-4o-mini", temperature: float = 0.3
    ):
        self.model_name = model_name

        analysis_instructions = """
        You are a research analyst specialized in extracting key knowledge from academic
        review papers to inform literature search strategies.

        When analyzing a review paper, identify:

        1. KEY CONCEPTS: Important biological entities, phenomena, diseases, or concepts
           that are central to the field (not just mentioned in passing)

        2. METHODOLOGIES: Specific research methods, experimental techniques,
           computational approaches, or analytical frameworks discussed

        3. TERMINOLOGY: Domain-specific terms, abbreviations, alternative names,
           or technical vocabulary that researchers in this field would use

        4. RESEARCH GAPS: Explicitly mentioned limitations, unanswered questions,
           or areas identified as needing future research

        5. SEMINAL PAPERS: Important author names, landmark studies, or foundational
           papers that are frequently cited or discussed as influential

        6. EMERGING THEMES: Recent developments, new technologies, novel approaches,
           or trending research directions mentioned as promising or growing

        Focus on actionable knowledge that would help generate better search queries.
        Be specific and avoid overly generic terms.
        """

        self.agent = Agent(
            model_name,
            output_type=ReviewKnowledgeResponse,
            instructions=analysis_instructions,
        )

    async def analyze_review(
        self, title: str, content: str, max_tokens: int = 8000
    ) -> ReviewKnowledge:
        """Analyze a single review paper to extract knowledge."""

        # Truncate content if too long
        if len(content) > max_tokens * 4:  # Rough token estimation
            content = content[: max_tokens * 4] + "..."

        prompt = f"""
        Analyze this review paper to extract key research knowledge:

        TITLE: {title}

        CONTENT:
        {content}

        Extract the most important and actionable knowledge that would help
        generate targeted literature search queries in this research domain.
        """

        try:
            result = await self.agent.run(prompt)
            response = result.output

            return ReviewKnowledge(
                title=title,
                key_concepts=response.key_concepts,
                methodologies=response.methodologies,
                terminology=response.terminology,
                research_gaps=response.research_gaps,
                seminal_papers=response.seminal_papers,
                emerging_themes=response.emerging_themes,
            )
        except Exception as e:
            print(f"Failed to analyze review '{title}': {e}")
            # Return minimal knowledge on failure
            return ReviewKnowledge(
                title=title,
                key_concepts=[],
                methodologies=[],
                terminology=[],
                research_gaps=[],
                seminal_papers=[],
                emerging_themes=[],
            )


class InformedQueryGenerator:
    """Generates targeted queries based on synthesized review knowledge."""

    def __init__(
        self, model_name: str = "openai:gpt-4o-mini", temperature: float = 0.3
    ):
        self.model_name = model_name

        generation_instructions = """
        You are a literature search strategist who designs targeted search queries
        based on insights extracted from review papers.

        CRITICAL: Your queries must stay within the scope and intent of the ORIGINAL query.
        Do not drift into tangential topics, even if they appear in the review papers.

        Your goal is to generate diverse, highly-targeted search queries that would
        find research that directly addresses the original query's intent, using
        insights from review papers to make the searches more effective.

        Use the synthesized knowledge from review papers to create queries that:

        1. Stay focused on the ORIGINAL query's main topic and constraints
        2. Use domain-specific terminology and alternative phrasings discovered
        3. Target specific aspects or subtopics within the original scope
        4. Explore research gaps within the original query's domain
        5. Focus on methodologies relevant to the original query's intent
        6. Target work by relevant authors or research groups mentioned
        7. Use precise biological entities or mechanisms from the original context

        Each query should:
        - DIRECTLY serve the original query's intent and scope
        - Be specific enough to find targeted, relevant research
        - Use precise scientific terminology when appropriate
        - Target a distinct research angle WITHIN the original scope
        - Avoid significant overlap with other queries
        - Preserve any constraints from the original query (e.g., if it asks for "reviews", focus on reviews)

        If the original query specifies a particular study type (reviews, meta-analyses, etc.)
        or specific focus area, ensure your generated queries maintain that focus.
        """

        self.agent = Agent(
            model_name,
            output_type=InformedQueryResponse,
            instructions=generation_instructions,
        )

    async def generate_queries(
        self,
        original_query: str,
        review_knowledge: List[ReviewKnowledge],
        target_queries: int = 15,
    ) -> List[InformedQuery]:
        """Generate informed search queries from review knowledge."""

        if not review_knowledge:
            # Fallback to basic query if no review knowledge
            return [
                InformedQuery(
                    query=original_query,
                    focus="original",
                    rationale="Fallback to original query due to no review knowledge",
                    source_reviews=[],
                    weight=1.0,
                )
            ]

        # Synthesize knowledge from all reviews
        synthesis_context = self._build_knowledge_synthesis(review_knowledge)

        prompt = f"""
        Generate {target_queries} diverse, targeted search queries based on insights from review papers.

        ORIGINAL RESEARCH QUERY: "{original_query}"

        SYNTHESIZED KNOWLEDGE FROM REVIEW PAPERS:
        {synthesis_context}

        Create search queries that would find complementary primary research that goes
        beyond what the review papers already cover. Each query should target a specific
        research angle informed by the knowledge extracted from the reviews.

        Ensure the queries are:
        - Scientifically precise and use appropriate terminology
        - Diverse in their research approaches and targets
        - Likely to find different types of valuable research
        - Based on specific insights from the review knowledge provided
        """

        try:
            result = await self.agent.run(prompt)
            response = result.output

            # Convert response to InformedQuery objects
            informed_queries = []
            source_titles = [rk.title for rk in review_knowledge]

            for query_strategy in response.queries:
                informed_queries.append(
                    InformedQuery(
                        query=query_strategy.query,
                        focus=query_strategy.focus.lower()
                        .replace(" ", "_")
                        .replace("-", "_"),
                        rationale=query_strategy.rationale,
                        source_reviews=source_titles,
                        weight=1.0,
                    )
                )

            # Shuffle to avoid systematic bias
            random.shuffle(informed_queries)
            return informed_queries

        except Exception as e:
            print(f"Failed to generate informed queries: {e}")
            # Fallback to original query
            return [
                InformedQuery(
                    query=original_query,
                    focus="original_fallback",
                    rationale="Fallback due to query generation failure",
                    source_reviews=[rk.title for rk in review_knowledge],
                    weight=1.0,
                )
            ]

    def _build_knowledge_synthesis(
        self, review_knowledge: List[ReviewKnowledge]
    ) -> str:
        """Build a synthesized knowledge context from multiple reviews."""

        synthesis_parts = []

        # Aggregate all knowledge categories
        all_concepts = []
        all_methodologies = []
        all_terminology = []
        all_gaps = []
        all_papers = []
        all_themes = []

        for rk in review_knowledge:
            all_concepts.extend(rk.key_concepts)
            all_methodologies.extend(rk.methodologies)
            all_terminology.extend(rk.terminology)
            all_gaps.extend(rk.research_gaps)
            all_papers.extend(rk.seminal_papers)
            all_themes.extend(rk.emerging_themes)

        # Remove duplicates while preserving order
        def dedupe(lst):
            seen = set()
            return [x for x in lst if not (x.lower() in seen or seen.add(x.lower()))]

        if all_concepts:
            synthesis_parts.append(
                f"Key Concepts: {', '.join(dedupe(all_concepts)[:15])}"
            )
        if all_methodologies:
            synthesis_parts.append(
                f"Methodologies: {', '.join(dedupe(all_methodologies)[:10])}"
            )
        if all_terminology:
            synthesis_parts.append(
                f"Terminology: {', '.join(dedupe(all_terminology)[:15])}"
            )
        if all_gaps:
            synthesis_parts.append(f"Research Gaps: {', '.join(dedupe(all_gaps)[:8])}")
        if all_papers:
            synthesis_parts.append(
                f"Important References: {', '.join(dedupe(all_papers)[:10])}"
            )
        if all_themes:
            synthesis_parts.append(
                f"Emerging Themes: {', '.join(dedupe(all_themes)[:8])}"
            )

        # Add source review information
        review_titles = [rk.title for rk in review_knowledge]
        synthesis_parts.append(
            f"Source Reviews ({len(review_titles)}): {'; '.join(review_titles)}"
        )

        return "\n\n".join(synthesis_parts)


class ReviewInformedOrchestrator:
    """Orchestrates the complete review-informed search process."""

    def __init__(
        self,
        backend: SearchBackend,
        fetcher: PageFetcher,
        review_config: Dict[str, Any],
        llm_config: Dict[str, Any],
        agent_spec: Optional["AgentSpec"] = None,
    ):
        self.backend = backend
        self.fetcher = fetcher
        self.review_config = review_config
        self.llm_config = llm_config

        # Get model name from AgentSpec or LLM config
        model_name = llm_config.get("model_name", "openai:gpt-4o-mini")
        if agent_spec and agent_spec.llm:
            model_name = str(agent_spec.llm)

        # Use LLM config temperature, fallback to advanced config, then default
        temperature = llm_config.get("temperature", 0.3)

        # Initialize components
        self.analyzer = ReviewAnalyzer(
            model_name=model_name,
            temperature=temperature,
        )

        self.query_generator = InformedQueryGenerator(
            model_name=model_name,
            temperature=temperature,
        )

    async def perform_review_informed_search(
        self,
        original_query: str,
        max_results_per_query: int = 20,
        verbose: bool = False,
    ) -> SearchResults:
        """Perform the complete review-informed search process."""

        if verbose:
            print(f"🔍 Starting review-informed search for: '{original_query}'")

        # Phase 1: Find review papers
        if verbose:
            print("📖 Phase 1: Discovering review papers...")

        # Find enough reviews to achieve target (search for more to ensure good selection)
        target_reviews = self.review_config.get("target_reviews", 3)
        search_limit = max(
            target_reviews * 2, 10
        )  # Search for more to get better selection

        review_results = await self._find_review_papers(
            original_query, search_limit, verbose
        )

        if not review_results.results:
            if verbose:
                print("⚠️  No review papers found, falling back to original query")
                print(f"   📝 Fallback query: '{original_query}'")
            # Fallback to single query
            fallback_query = SearchQuery(
                query=original_query, max_results=max_results_per_query
            )
            return await self.backend.search(fallback_query)

        if verbose:
            print(f"📚 Found {len(review_results.results)} review papers")

        # Phase 2: Analyze top reviews
        if verbose:
            print("🧠 Phase 2: Analyzing review papers...")

        review_knowledge = await self._analyze_reviews(
            review_results.results[:target_reviews], verbose
        )

        if verbose:
            total_concepts = sum(len(rk.key_concepts) for rk in review_knowledge)
            total_methods = sum(len(rk.methodologies) for rk in review_knowledge)
            print(
                f"💡 Extracted {total_concepts} key concepts and {total_methods} methodologies"
            )

        # Phase 3: Generate informed queries
        if verbose:
            print("🎯 Phase 3: Generating informed search queries...")

        # Use target_searches from main expansion config for consistency
        target_queries = self.llm_config.get("max_terms", 15)
        informed_queries = await self.query_generator.generate_queries(
            original_query,
            review_knowledge,
            target_queries=target_queries,
        )

        if verbose:
            print(f"🔎 Generated {len(informed_queries)} targeted search queries:")
            for i, iq in enumerate(informed_queries, 1):
                console.print(f"   📝 {i}: '{iq.query}' ", end="")
                console.print(f"({iq.focus})", style="dim")

        # Phase 4: Execute informed searches
        if verbose:
            print("🚀 Phase 4: Executing informed searches...")

        primary_results = await self._execute_informed_searches(
            informed_queries,
            max_results_per_query,
            verbose,
        )

        # Phase 5: Combine results
        if verbose:
            print("🔗 Phase 5: Combining results...")

        if self.review_config.get("include_reviews_in_results", True):
            combined_results = self._combine_results(
                review_results, primary_results, original_query
            )
        else:
            combined_results = primary_results

        if verbose:
            print(
                f"✅ Review-informed search complete: {len(combined_results.results)} total results"
            )

        return combined_results

    async def _find_review_papers(
        self, query: str, max_results: int, verbose: bool = False
    ) -> SearchResults:
        """Find review papers related to the query."""

        # Build review-specific search query
        if (
            hasattr(self.backend, "backend_name")
            and "pubmed" in self.backend.backend_name.lower()
        ):
            # For PubMed, use publication type filters
            review_query = SearchQuery(
                query=query,
                max_results=max_results,
                filters={
                    "publication_type": ["Review", "Systematic Review", "Meta-Analysis"]
                },
            )
            if verbose:
                print(
                    f"   📝 Review search query: '{query}' [PubMed filters: Review, Systematic Review, Meta-Analysis]"
                )
        else:
            # For other backends, modify the query text
            enhanced_query = f'{query} AND (review OR survey OR "systematic review" OR "meta-analysis" OR "state of the art")'
            review_query = SearchQuery(query=enhanced_query, max_results=max_results)
            if verbose:
                print(f"   📝 Review search query: '{enhanced_query}'")

        return await self.backend.search(review_query)

    async def _analyze_reviews(
        self, review_results: List[SearchResult], verbose: bool = False
    ) -> List[ReviewKnowledge]:
        """Fetch and analyze review paper content."""

        review_urls = [result.url for result in review_results]

        if verbose:
            print(f"   📄 Fetching content from {len(review_urls)} review papers...")

        # Fetch markdown content for reviews
        try:
            review_contents = await self.fetcher.get_markdown(review_urls)
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Failed to fetch review content: {e}")
            else:
                print(f"Failed to fetch review content: {e}")
            return []

        if verbose:
            successful_fetches = sum(1 for content in review_contents if content)
            print(
                f"   ✅ Successfully fetched {successful_fetches}/{len(review_urls)} reviews"
            )

        # Analyze each review
        review_knowledge = []
        max_tokens = 8000  # Use sensible default for review content

        if verbose and review_results:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Analyzing reviews...", total=len(review_results)
                )

                for i, (result, content) in enumerate(
                    zip(review_results, review_contents), 1
                ):
                    # Update progress description with current review
                    short_title = (
                        result.title[:40] + "..."
                        if len(result.title) > 40
                        else result.title
                    )
                    progress.update(task, description=f"Analyzing: {short_title}")

                    if content:  # Only analyze if content was successfully fetched
                        try:
                            knowledge = await self.analyzer.analyze_review(
                                title=result.title,
                                content=content,
                                max_tokens=max_tokens,
                            )
                            review_knowledge.append(knowledge)
                        except Exception as e:
                            console.print(f"   ⚠️  Failed to analyze review {i}: {e}")
                    else:
                        console.print(f"   ⏭️  Skipping review {i} (no content)")

                    # Advance progress
                    progress.advance(task)

                progress.update(task, description="Analysis complete")
        else:
            # Non-verbose mode - just analyze without progress bar
            for result, content in zip(review_results, review_contents):
                if content:
                    try:
                        knowledge = await self.analyzer.analyze_review(
                            title=result.title,
                            content=content,
                            max_tokens=max_tokens,
                        )
                        review_knowledge.append(knowledge)
                    except Exception as e:
                        console.print(f"Failed to analyze review: {e}")

        return review_knowledge

    async def _execute_informed_searches(
        self,
        informed_queries: List[InformedQuery],
        max_results_per_query: int,
        verbose: bool,
    ) -> SearchResults:
        """Execute all informed search queries and combine results."""

        # Let each query return as many results as the backend provides
        # We'll deduplicate and rerank at the end instead of artificially constraining each query
        if verbose:
            print(
                f"   🔍 Executing {len(informed_queries)} queries (no per-query limits)"
            )

        # Execute searches concurrently
        search_tasks = []
        for iq in informed_queries:
            # Let each query use the backend's natural result limit (no max_results constraint)
            search_query = SearchQuery(
                query=iq.query
            )  # max_results=None -> backend decides
            search_tasks.append(self.backend.search(search_query))

        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Combine results and deduplicate while tracking frequency
        result_map = {}  # url -> (search_result, frequency)

        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                if verbose:
                    print(f"   ⚠️  Query {i + 1} failed: {result}")
                continue

            if verbose:
                print(f"   📑 Query {i + 1}: {len(result.results)} results")

            for search_result in result.results:
                if search_result.url in result_map:
                    # Increment frequency for duplicate
                    existing_result, freq = result_map[search_result.url]
                    result_map[search_result.url] = (existing_result, freq + 1)
                else:
                    # First occurrence
                    result_map[search_result.url] = (search_result, 1)

        # Extract results and add frequency to metadata
        all_results = []
        for search_result, frequency in result_map.values():
            # Add frequency to metadata
            search_result.metadata = search_result.metadata.copy()
            search_result.metadata["frequency"] = frequency
            all_results.append(search_result)

        if verbose:
            # Show deduplication stats
            multi_query_results = [
                r for r in all_results if r.metadata.get("frequency", 1) > 1
            ]
            print(
                f"   🔗 Combined: {len(all_results)} unique results after deduplication"
            )
            if multi_query_results:
                max_freq = max(
                    r.metadata.get("frequency", 1) for r in multi_query_results
                )
                print(
                    f"   📊 {len(multi_query_results)} results found by multiple queries (max: {max_freq} queries)"
                )

        # Rerank by relevance score and limit to target
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)
        final_results = all_results[:max_results_per_query]

        if verbose and len(final_results) < len(all_results):
            print(f"   🎯 Limited to top {len(final_results)} results by relevance")

        # Create combined SearchResults object
        original_query = (
            informed_queries[0].query if informed_queries else "review-informed search"
        )
        combined_query = SearchQuery(
            query=original_query, max_results=len(final_results)
        )

        return SearchResults(
            query=combined_query,
            results=final_results,
            total_found=len(all_results),
            search_time=0.0,  # Would need to track timing
            backend=self.backend.backend_name,
        )

    def _combine_results(
        self,
        review_results: SearchResults,
        primary_results: SearchResults,
        original_query: str,
    ) -> SearchResults:
        """Combine review papers with primary research results."""

        # Start with review papers (higher weight)
        combined_results = []
        seen_urls = set()

        # Add review papers first
        for result in review_results.results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                # Boost relevance for review papers
                result.relevance_score = min(1.0, result.relevance_score + 0.1)
                combined_results.append(result)

        # Add primary research results
        for result in primary_results.results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                combined_results.append(result)

        # Sort by relevance score
        combined_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Create combined SearchResults
        combined_query = SearchQuery(
            query=original_query, max_results=len(combined_results)
        )

        return SearchResults(
            query=combined_query,
            results=combined_results,
            total_found=len(combined_results),
            search_time=review_results.search_time + primary_results.search_time,
            backend=f"review_informed_{self.backend.backend_name}",
        )


# Factory function for easy integration
async def create_review_informed_search(
    original_query: str,
    backend: SearchBackend,
    fetcher: PageFetcher,
    review_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    agent_spec: Optional["AgentSpec"] = None,
    max_results_per_query: int = 20,
    verbose: bool = False,
) -> SearchResults:
    """
    Factory function to perform review-informed search.

    Args:
        original_query: The user's research query
        backend: Search backend to use
        fetcher: PageFetcher for retrieving review content
        review_config: Review-informed specific configuration
        llm_config: LLM expansion configuration (reused)
        agent_spec: Agent specification for LLM configuration
        max_results_per_query: Maximum results per search
        verbose: Whether to show progress information

    Returns:
        SearchResults combining review papers and informed primary research
    """
    orchestrator = ReviewInformedOrchestrator(
        backend, fetcher, review_config, llm_config, agent_spec
    )
    return await orchestrator.perform_review_informed_search(
        original_query, max_results_per_query, verbose
    )
