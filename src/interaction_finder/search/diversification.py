"""
LLM-driven query diversification for strategic deep search coverage.

This module uses LLM reasoning capabilities to generate diverse, comprehensive
search strategies rather than relying on rigid rule-based categorization.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import random

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .base import EnhancedExpandedQuery, EnhancedExpansionTerm


@dataclass
class DiversifiedQuery:
    """A single diversified query with strategic metadata."""

    query: str
    focus: str  # Strategic angle (e.g., "molecular_mechanisms", "clinical_research")
    rationale: str  # Why this query strategy is valuable
    terms_used: List[str]  # Key terms incorporated
    weight: float = 1.0  # Relative importance weight


class SearchStrategy(BaseModel):
    """Individual search strategy from LLM reasoning."""

    query: str = Field(description="The specific search query")
    angle: str = Field(description="Research angle or perspective")
    rationale: str = Field(description="Why this strategy is valuable")
    specificity: str = Field(description="broad, focused, or specific")


class StrategicDiversificationResponse(BaseModel):
    """LLM response for strategic query diversification."""

    strategies: List[SearchStrategy] = Field(
        description="List of diverse search strategies"
    )
    coverage_analysis: str = Field(
        description="Analysis of how these strategies provide comprehensive coverage"
    )


class RefinementResponse(BaseModel):
    """LLM response for query refinement and validation."""

    class Improvement(BaseModel):
        original_query: str = Field(description="Original query to improve")
        improved_query: str = Field(description="Improved version")
        reason: str = Field(description="Why this improvement was made")

    final_queries: List[str] = Field(description="Final set of 8 diverse queries")
    improvements: List[Improvement] = Field(description="Any improvements made")
    coverage_gaps: List[str] = Field(
        description="Any remaining coverage gaps identified"
    )


class LLMQueryDiversifier:
    """LLM-driven query diversifier using strategic reasoning."""

    def __init__(self, model_name: str = "openai:gpt-4o-mini", target_queries: int = 8):
        """
        Initialize LLM diversifier.

        Args:
            model_name: Model to use for diversification reasoning
            target_queries: Number of diverse queries to generate
        """
        self.model_name = model_name
        self.target_queries = target_queries

        # Initialize strategic diversification agent
        strategy_instructions = """
        You are a research strategist specializing in comprehensive literature searches.
        Your expertise is in designing diverse search strategies that maximize research coverage
        while minimizing redundancy.

        When given a research query, analyze it strategically and design multiple complementary
        search approaches that would together provide comprehensive topic coverage.

        Consider different:
        - Research perspectives (molecular, clinical, computational, epidemiological)
        - Specificity levels (broad context, focused studies, specific mechanisms)
        - Temporal aspects (recent advances, foundational research, longitudinal studies)
        - Methodological approaches (experimental, observational, meta-analytic)
        - Related domains that inform the main topic
        - Different ways researchers might phrase similar concepts

        Focus on creating queries that would find different types of valuable research
        while maintaining scientific relevance and avoiding unnecessary overlap.
        """

        self.strategy_agent = Agent(
            model_name,
            output_type=StrategicDiversificationResponse,
            instructions=strategy_instructions,
        )

        # Initialize refinement agent
        refinement_instructions = """
        You are a research librarian expert in optimizing literature search strategies.

        Your role is to review sets of search queries for comprehensiveness, specificity,
        and diversity. You identify gaps, redundancies, and opportunities for improvement.

        When reviewing search strategies, ensure they:
        - Cover the topic comprehensively from multiple valid angles
        - Avoid unnecessary redundancy while maintaining complementary coverage
        - Use appropriate specificity for their intended research angle
        - Would realistically find different valuable research in academic databases

        Make targeted improvements to maximize research discovery potential.
        """

        self.refinement_agent = Agent(
            model_name,
            output_type=RefinementResponse,
            instructions=refinement_instructions,
        )

    async def diversify_query(
        self,
        expanded_query: EnhancedExpandedQuery,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[DiversifiedQuery]:
        """
        Generate diverse queries using LLM strategic reasoning.

        Args:
            expanded_query: Results from advanced query expansion
            context: Optional context for diversification

        Returns:
            List of diversified queries for comprehensive search
        """
        if context is None:
            context = {}

        # Build rich context for LLM reasoning
        diversification_context = self._build_diversification_context(
            expanded_query, context
        )

        try:
            # Phase 1: Generate strategic candidates
            candidates = await self._generate_strategic_candidates(
                expanded_query.original_query, diversification_context
            )

            # Phase 2: Refine and select final queries
            final_queries = await self._refine_and_select_queries(
                candidates, diversification_context
            )

            return final_queries

        except Exception as e:
            print(f"LLM diversification failed: {e}")
            # Fallback to original query
            return [
                DiversifiedQuery(
                    query=expanded_query.original_query,
                    focus="original",
                    rationale="Fallback to original query due to diversification failure",
                    terms_used=[expanded_query.original_query],
                    weight=1.0,
                )
            ]

    def _build_diversification_context(
        self, expanded_query: EnhancedExpandedQuery, context: Dict[str, Any]
    ) -> str:
        """Build rich context string for LLM reasoning."""

        context_parts = []

        # Domain and research area
        domain = context.get("domain", "academic research")
        context_parts.append(f"Research domain: {domain}")

        # Available expansion terms for context
        if expanded_query.expansion_terms:
            expansion_terms = [
                term.term for term in expanded_query.expansion_terms[:10]
            ]
            context_parts.append(
                f"Related concepts identified: {', '.join(expansion_terms)}"
            )

        # Intent context if available
        if expanded_query.intent_card:
            context_parts.append(f"Research intent: {expanded_query.intent_card}")

        # HyDE context for conceptual understanding
        if expanded_query.hyde_text:
            # Use first 200 chars of HyDE for context
            hyde_excerpt = (
                expanded_query.hyde_text[:200] + "..."
                if len(expanded_query.hyde_text) > 200
                else expanded_query.hyde_text
            )
            context_parts.append(f"Conceptual context: {hyde_excerpt}")

        return "\n".join(context_parts)

    async def _generate_strategic_candidates(
        self, original_query: str, context_str: str
    ) -> List[SearchStrategy]:
        """Generate strategic search candidates using LLM reasoning."""

        prompt = f"""
        Design {self.target_queries + 2} diverse search strategies for comprehensive literature coverage of this research topic:

        RESEARCH QUERY: "{original_query}"

        CONTEXT:
        {context_str}

        Create search strategies that approach this topic from different valuable research angles.
        Each strategy should target different types of research that would provide complementary insights.

        Consider generating strategies for:
        - Different research methodologies or approaches
        - Different levels of biological/conceptual specificity
        - Different research communities or perspectives
        - Related but distinct research questions
        - Different temporal focuses (recent vs foundational)

        Each search query should be specific enough to find targeted research while being
        broad enough to capture multiple relevant papers.

        Ensure the strategies together would provide comprehensive topic coverage while
        minimizing redundant results.
        """

        result = await self.strategy_agent.run(prompt)
        return result.output.strategies

    async def _refine_and_select_queries(
        self, candidates: List[SearchStrategy], context_str: str
    ) -> List[DiversifiedQuery]:
        """Refine candidates and select final diverse queries."""

        # Convert candidates to query strings for review
        candidate_queries = [
            f"Query: {strategy.query}\nAngle: {strategy.angle}\nRationale: {strategy.rationale}"
            for strategy in candidates
        ]

        refinement_prompt = f"""
        Review these search strategy candidates and select/refine the best {self.target_queries}
        for comprehensive research coverage:

        ORIGINAL RESEARCH TOPIC: First candidate represents the original query focus.

        CANDIDATES:
        {chr(10).join(f"{i + 1}. {query}" for i, query in enumerate(candidate_queries))}

        CONTEXT:
        {context_str}

        Your task:
        1. Select {self.target_queries} strategies that maximize comprehensive coverage
        2. Improve any queries that could be more effective
        3. Ensure minimal redundancy while maintaining complementary approaches
        4. Identify any important research angles not covered

        Prioritize:
        - Strategies that would find different types of valuable research
        - Appropriate specificity for each research angle
        - Realistic queries that would work well in academic search databases
        - Balanced coverage across the research landscape
        """

        result = await self.refinement_agent.run(refinement_prompt)
        refinement_output = result.output

        # Convert refined queries back to DiversifiedQuery objects
        diversified_queries = []

        for i, query in enumerate(refinement_output.final_queries):
            # Try to match with original candidate for metadata
            matching_candidate = None
            for candidate in candidates:
                if candidate.query in query or query in candidate.query:
                    matching_candidate = candidate
                    break

            focus = (
                matching_candidate.angle if matching_candidate else f"strategy_{i + 1}"
            )
            rationale = (
                matching_candidate.rationale
                if matching_candidate
                else f"Strategic approach {i + 1}"
            )

            diversified_queries.append(
                DiversifiedQuery(
                    query=query,
                    focus=focus.lower().replace(" ", "_").replace("-", "_"),
                    rationale=rationale,
                    terms_used=query.split()[:5],  # First 5 words as key terms
                    weight=1.0,
                )
            )

        # Shuffle to avoid systematic bias in search order
        random.shuffle(diversified_queries)

        return diversified_queries


# Factory function for backwards compatibility
async def create_diversified_queries(
    expanded_query: EnhancedExpandedQuery,
    target_queries: int = 8,
    context: Optional[Dict[str, Any]] = None,
    model_name: str = "openai:gpt-4o-mini",
) -> List[DiversifiedQuery]:
    """
    Factory function to create diversified queries using LLM reasoning.

    Args:
        expanded_query: Results from advanced query expansion
        target_queries: Number of diverse queries to generate
        context: Optional context for diversification
        model_name: LLM model to use for reasoning

    Returns:
        List of strategically diversified queries
    """
    diversifier = LLMQueryDiversifier(
        model_name=model_name, target_queries=target_queries
    )
    return await diversifier.diversify_query(expanded_query, context)
