"""
Advanced query expansion with diversity-driven term selection and HyDE generation.

This module implements a sophisticated multi-phase query expansion system inspired
by modern retrieval research, featuring intent normalization, diverse term selection,
and surrogate document generation for enhanced search recall.
"""

import asyncio
import json
import math
import random
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..base import QueryExpander, EnhancedExpandedQuery, EnhancedExpansionTerm


class IntentCardResponse(BaseModel):
    """Response model for intent card generation."""

    purpose_scope: str = Field(description="Purpose and scope of the research")
    likely_entities: str = Field(description="Likely entities of interest")
    likely_methods: str = Field(description="Likely methods or processes")
    time_bounds: str = Field(description="Explicit or implicit time bounds")


class ExpansionTermResponse(BaseModel):
    """Individual expansion term response."""

    term: str = Field(description="The expansion term")
    category: str = Field(description="Term category")
    rationale: str = Field(description="Why this term is relevant")


class TwoAngleExpansionResponse(BaseModel):
    """Response model for two-angle expansion generation."""

    A: List[ExpansionTermResponse] = Field(description="Entity/synonym-centric terms")
    B: List[ExpansionTermResponse] = Field(description="Method/timeline-centric terms")


class ScoredTerm:
    """Term with computed score for selection."""

    def __init__(self, term: EnhancedExpansionTerm, score: float):
        self.term = term
        self.score = score


class AdvancedQueryExpander(QueryExpander):
    """Advanced query expander with diversity-driven selection and HyDE generation."""

    # Category scoring bonuses
    CATEGORY_BONUSES = {
        "synonym": 0.3,
        "entity": 0.3,
        "method": 0.2,
        "timeframe": 0.2,
        "abbreviation": 0.1,
        "broader": 0.1,
        "narrower": 0.1,
    }

    def __init__(
        self,
        model_name: str = "openai:gpt-4o-mini",
        max_terms: int = 10,
        per_category_cap: int = 3,
        mmr_lambda: float = 0.7,
        temperature: float = 0.2,
        deterministic_seed: Optional[int] = 42,
    ):
        """
        Initialize advanced query expander.

        Args:
            model_name: Model to use for LLM calls
            max_terms: Maximum total expansion terms
            per_category_cap: Maximum terms per category
            mmr_lambda: Lambda parameter for MMR diversity (0.7 = balanced)
            temperature: LLM temperature for deterministic outputs
            deterministic_seed: Seed for reproducible results
        """
        self.model_name = model_name
        self.max_terms = max_terms
        self.per_category_cap = per_category_cap
        self.mmr_lambda = mmr_lambda
        self.temperature = temperature

        if deterministic_seed is not None:
            random.seed(deterministic_seed)
            np.random.seed(deterministic_seed)

        # Initialize agents for different phases
        self._init_agents()

    def _init_agents(self):
        """Initialize specialized agents for each expansion phase."""

        # Intent card agent
        intent_system = """
        You are a careful research assistant. Your task is to analyze a research query
        and extract its core intent in a structured format. Focus on understanding
        the research purpose without proposing any expansion terms.
        """

        self.intent_agent = Agent(
            self.model_name,
            output_type=IntentCardResponse,
            instructions=intent_system,
        )

        # Two-angle expansion agent
        expansion_system = """
        You are a terminology expert for academic literature search. Generate diverse
        expansion terms from two complementary angles to maximize search recall.

        Guidelines:
        - Focus on terms likely to appear in academic abstracts and titles
        - Prefer short phrases (2-4 tokens) over long descriptions
        - Assign categories: synonym, entity, method, timeframe, abbreviation
        - Provide brief rationales for term relevance
        - Ensure diversity within and between lists
        """

        self.expansion_agent = Agent(
            self.model_name,
            output_type=TwoAngleExpansionResponse,
            instructions=expansion_system,
        )

        # HyDE surrogate generation agent
        hyde_system = """
        You are an expert academic writer. Generate plausible research abstracts
        that would answer the given query. Write in a cautious, objective tone
        typical of scientific literature. Include likely entities, methods, and
        timeframes without making specific claims or citing sources.
        """

        self.hyde_agent = Agent(
            self.model_name,
            output_type=str,
            instructions=hyde_system,
        )

    @property
    def expansion_method(self) -> str:
        """Name of the expansion method."""
        return "advanced"

    def supports_context(self) -> List[str]:
        """Return list of supported context keys."""
        return [
            "domain",
            "time_hints",
            "entity_types",
            "max_terms",
            "per_category_cap",
            "mmr_lambda",
            "idf_probe",
            "embedding_fn",
        ]

    async def expand_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> EnhancedExpandedQuery:
        """
        Expand query using advanced multi-phase pipeline.

        Args:
            query: Original search query
            context: Optional context with domain, constraints, etc.

        Returns:
            EnhancedExpandedQuery with structured expansion results
        """
        if not query or not query.strip():
            return EnhancedExpandedQuery(
                original_query=query,
                intent_card="",
                expansion_terms=[],
                hyde_text="",
                expansion_method=self.expansion_method,
                total_confidence=0.0,
            )

        if not context:
            context = {}

        try:
            # Step A: Intent normalization
            intent_card = await self._generate_intent_card(query)

            # Step B: Two-angle expansion generation
            expansion_candidates = await self._generate_diverse_expansions(
                query, context
            )

            # Step C: HyDE surrogate generation
            hyde_text = await self._generate_hyde_surrogate(query, context)

            # Step D: Selection & weighting with diversity
            selected_terms = self._select_diverse_terms(expansion_candidates, context)

            # Calculate total confidence
            total_confidence = (
                sum(term.confidence for term in selected_terms)
                / max(len(selected_terms), 1)
                if selected_terms
                else 1.0
            )

            return EnhancedExpandedQuery(
                original_query=query,
                intent_card=intent_card,
                expansion_terms=selected_terms,
                hyde_text=hyde_text,
                expansion_method=self.expansion_method,
                total_confidence=min(1.0, total_confidence),
            )

        except Exception as e:
            # Fallback to minimal expansion on error
            print(f"Advanced expansion failed: {e}")
            return EnhancedExpandedQuery(
                original_query=query,
                intent_card="",
                expansion_terms=[],
                hyde_text="",
                expansion_method=self.expansion_method,
                total_confidence=0.0,
            )

    async def _generate_intent_card(self, query: str) -> str:
        """Generate structured intent card for the query."""
        prompt = f"""
        Analyze this research query in exactly 4 bullets:
        1) Purpose & scope, 2) Likely entities, 3) Likely methods/processes, 4) Time bounds.
        Do not propose expansion terms.

        Query: "{query}"
        """

        try:
            result = await self.intent_agent.run(prompt)
            intent_data = result.output

            intent_card = f"""1) Purpose & scope: {intent_data.purpose_scope}
2) Likely entities: {intent_data.likely_entities}
3) Likely methods: {intent_data.likely_methods}
4) Time bounds: {intent_data.time_bounds}"""

            return intent_card

        except Exception as e:
            print(f"Intent card generation failed: {e}")
            return f"Research query analysis for: {query}"

    async def _generate_diverse_expansions(
        self, query: str, context: Dict[str, Any]
    ) -> List[EnhancedExpansionTerm]:
        """Generate diverse expansion terms using two-angle approach."""

        # Build context hints
        domain_hint = context.get("domain", "biomedical research")
        time_hint = context.get("time_hints", "")

        context_text = f"Domain: {domain_hint}"
        if time_hint:
            context_text += f"\nTime context: {time_hint}"

        prompt = f"""
        Generate two short expansion lists for literature retrieval:
        (A) entity/synonym-centric terms (up to 6 items)
        (B) method/timeline-centric terms (up to 6 items)

        For each term provide: {{"term": "...", "category": "...", "rationale": "..."}}
        Categories: synonym, entity, method, timeframe, abbreviation

        Hard caps: ≤6 items per list; terse phrases (2-4 words); no duplicates

        Query: "{query}"
        Context: {context_text}
        """

        try:
            result = await self.expansion_agent.run(prompt)
            expansion_data = result.output

            # Convert to EnhancedExpansionTerm objects
            candidates = []

            for term_resp in expansion_data.A:
                candidates.append(
                    EnhancedExpansionTerm(
                        term=term_resp.term,
                        category=term_resp.category,
                        rationale=term_resp.rationale,
                        source="LLM-A",
                    )
                )

            for term_resp in expansion_data.B:
                candidates.append(
                    EnhancedExpansionTerm(
                        term=term_resp.term,
                        category=term_resp.category,
                        rationale=term_resp.rationale,
                        source="LLM-B",
                    )
                )

            return candidates

        except Exception as e:
            print(f"Expansion generation failed: {e}")
            return []

    async def _generate_hyde_surrogate(
        self, query: str, context: Dict[str, Any]
    ) -> str:
        """Generate HyDE surrogate abstract."""

        domain_hint = context.get("domain", "biomedical research")

        prompt = f"""
        Write a 120-180 word plausible research abstract that would answer:
        "{query}"

        Context: {domain_hint}

        Guidelines:
        - Mention likely entities, methods, and timeframes
        - Use cautious, objective scientific tone
        - No specific citations or references
        - Focus on methodology and findings that would be relevant
        """

        try:
            result = await self.hyde_agent.run(prompt)
            return result.output.strip()

        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return f"Research investigating {query} using appropriate methodologies."

    def _select_diverse_terms(
        self, candidates: List[EnhancedExpansionTerm], context: Dict[str, Any]
    ) -> List[EnhancedExpansionTerm]:
        """Apply diversity-driven term selection with MMR-style scoring."""

        if not candidates:
            return []

        # Step 1: Score each candidate
        scored_terms = self._score_terms(candidates, context)

        # Step 2: Apply MMR diversity selection
        selected = self._mmr_selection(scored_terms, context)

        # Step 3: Enforce category caps and assign weight hints
        final_terms = self._enforce_category_caps_and_weights(selected, context)

        return final_terms

    def _score_terms(
        self, candidates: List[EnhancedExpansionTerm], context: Dict[str, Any]
    ) -> List[ScoredTerm]:
        """Score terms using IDF + category bonus - length penalty."""

        # Apply confidence filtering with automatic fallback
        min_confidence = context.get("min_confidence", 0.5)
        min_expansion_terms = context.get("min_expansion_terms", 1)

        # First, try with the main confidence threshold
        if min_confidence > 0.0:
            high_confidence_candidates = [
                term for term in candidates if term.confidence >= min_confidence
            ]
        else:
            high_confidence_candidates = candidates

        # If we don't have enough terms, automatically use the best available terms
        if len(high_confidence_candidates) < min_expansion_terms:
            # Sort all candidates by confidence (best first)
            candidates.sort(key=lambda x: -x.confidence)
            # Take the best candidates up to the minimum required
            filtered_candidates = candidates[:min_expansion_terms]
        else:
            filtered_candidates = high_confidence_candidates

        scored_terms = []
        idf_probe = context.get("idf_probe")

        for term in filtered_candidates:
            # IDF score (or fallback to inverse length)
            if idf_probe and callable(idf_probe):
                idf_score = idf_probe(term.term)
            else:
                # Fallback: inverse normalized length
                idf_score = 1.0 / (len(term.term.split()) + 1)

            # Category bonus
            category_bonus = self.CATEGORY_BONUSES.get(term.category, 0.0)

            # Length penalty for very long terms
            length_penalty = max(0, (len(term.term.split()) - 4) * 0.1)

            # Combined score
            base_score = idf_score + category_bonus - length_penalty

            scored_terms.append(ScoredTerm(term, base_score))

        return scored_terms

    def _mmr_selection(
        self, scored_terms: List[ScoredTerm], context: Dict[str, Any]
    ) -> List[ScoredTerm]:
        """Apply MMR-style diversity selection."""

        if not scored_terms:
            return []

        # Get parameters
        max_terms = context.get("max_terms", self.max_terms)
        lambda_param = context.get("mmr_lambda", self.mmr_lambda)
        embedding_fn = context.get("embedding_fn")

        # Sort by base score initially
        scored_terms.sort(key=lambda x: x.score, reverse=True)

        selected = []
        remaining = scored_terms[:]

        # Greedy MMR selection
        while len(selected) < max_terms and remaining:
            if not selected:
                # First term: highest base score
                best_term = remaining.pop(0)
                selected.append(best_term)
                continue

            # MMR scoring for remaining terms
            best_mmr_score = float("-inf")
            best_idx = 0

            for i, candidate in enumerate(remaining):
                # Relevance component
                relevance = candidate.score

                # Diversity component (max similarity to selected)
                if embedding_fn and callable(embedding_fn):
                    try:
                        candidate_emb = embedding_fn(candidate.term.term)
                        max_sim = 0.0

                        for selected_term in selected:
                            selected_emb = embedding_fn(selected_term.term.term)
                            # Cosine similarity
                            sim = np.dot(candidate_emb, selected_emb) / (
                                np.linalg.norm(candidate_emb)
                                * np.linalg.norm(selected_emb)
                            )
                            max_sim = max(max_sim, sim)

                    except Exception:
                        # Fallback to string similarity
                        max_sim = self._string_similarity(
                            candidate.term.term,
                            max([s.term.term for s in selected], key=len),
                        )
                else:
                    # Fallback to string similarity
                    max_sim = max(
                        self._string_similarity(candidate.term.term, s.term.term)
                        for s in selected
                    )

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i

            # Select best MMR term
            best_term = remaining.pop(best_idx)
            selected.append(best_term)

        return selected

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute Jaccard similarity over character 3-grams."""

        def get_trigrams(s: str) -> set:
            s = s.lower().replace(" ", "")
            return set(s[i : i + 3] for i in range(len(s) - 2))

        tri1 = get_trigrams(s1)
        tri2 = get_trigrams(s2)

        if not tri1 and not tri2:
            return 1.0
        if not tri1 or not tri2:
            return 0.0

        intersection = len(tri1 & tri2)
        union = len(tri1 | tri2)

        return intersection / union if union > 0 else 0.0

    def _enforce_category_caps_and_weights(
        self, scored_terms: List[ScoredTerm], context: Dict[str, Any]
    ) -> List[EnhancedExpansionTerm]:
        """Enforce per-category caps and assign weight hints."""

        per_cat_cap = context.get("per_category_cap", self.per_category_cap)

        # Group by category
        by_category: Dict[str, List[ScoredTerm]] = {}
        for scored_term in scored_terms:
            category = scored_term.term.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(scored_term)

        # Apply caps and collect final terms
        final_terms = []
        for category, terms in by_category.items():
            # Sort by score within category and take top N
            terms.sort(key=lambda x: x.score, reverse=True)
            capped_terms = terms[:per_cat_cap]
            final_terms.extend(capped_terms)

        # Sort all final terms by score for weight assignment
        final_terms.sort(key=lambda x: x.score, reverse=True)

        # Assign weight hints based on tertiles
        result = []
        for i, scored_term in enumerate(final_terms):
            term = scored_term.term

            # Weight hint: 3 (top tertile), 2 (middle), 1 (bottom)
            if i < len(final_terms) // 3:
                weight_hint = 3
            elif i < (2 * len(final_terms)) // 3:
                weight_hint = 2
            else:
                weight_hint = 1

            term.weight_hint = weight_hint
            result.append(term)

        return result


def create_advanced_expander(
    model_name: str = "openai:gpt-4o-mini",
    max_terms: int = 10,
    per_category_cap: int = 3,
    mmr_lambda: float = 0.7,
    temperature: float = 0.2,
    deterministic_seed: Optional[int] = 42,
) -> AdvancedQueryExpander:
    """Factory function to create an advanced query expander."""
    return AdvancedQueryExpander(
        model_name=model_name,
        max_terms=max_terms,
        per_category_cap=per_category_cap,
        mmr_lambda=mmr_lambda,
        temperature=temperature,
        deterministic_seed=deterministic_seed,
    )
