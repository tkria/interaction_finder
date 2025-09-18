"""
LLM-based query expansion for intelligent biomedical term generation.

This module provides query expansion using large language models to generate
contextually relevant biomedical terms, synonyms, and related concepts.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..base import QueryExpander, ExpandedQuery, ExpansionTerm


class ExpansionRequest(BaseModel):
    """Request model for LLM expansion."""

    query: str = Field(description="Original search query to expand")
    context: str = Field(
        default="biomedical", description="Domain context for expansion"
    )
    max_terms: int = Field(default=10, description="Maximum number of expansion terms")
    include_synonyms: bool = Field(default=True, description="Include direct synonyms")
    include_related: bool = Field(default=True, description="Include related concepts")
    include_abbreviations: bool = Field(
        default=True, description="Include abbreviations/acronyms"
    )


class ExpansionResponse(BaseModel):
    """Response model for LLM expansion results."""

    class Term(BaseModel):
        term: str = Field(description="The expanded term")
        confidence: float = Field(description="Confidence score 0-1", ge=0.0, le=1.0)
        category: str = Field(
            description="Category: synonym, related, abbreviation, etc."
        )
        explanation: str = Field(description="Why this term is relevant")

    original_query: str = Field(description="The original query")
    expanded_terms: List[Term] = Field(description="List of expansion terms")
    reasoning: str = Field(description="Reasoning for the expansion choices")


class LLMQueryExpander(QueryExpander):
    """Query expander using LLM for intelligent biomedical term expansion."""

    def __init__(self, model_name: str = "openai:gpt-4o-mini", max_terms: int = 15):
        """
        Initialize LLM expander.

        Args:
            model_name: Model to use (e.g., 'openai:gpt-4o-mini')
            max_terms: Maximum number of terms to generate
        """
        self.model_name = model_name
        self.max_terms = max_terms

        # Initialize the agent with biomedical expertise
        # We'll set the system prompt dynamically based on domain context
        self.base_system_prompt = """
            You are a terminology expert specializing in query expansion for academic
            and research literature search. Your task is to expand search queries
            with relevant terms, synonyms, abbreviations, and related concepts.

            Guidelines:
            1. Focus on terms that would help find relevant academic literature
            2. Include official terminology, alternative names, and common aliases
            3. Consider abbreviations and acronyms commonly used in the domain
            4. Include both more specific and more general related terms
            5. Assign realistic confidence scores based on relevance and commonality
            6. Provide brief explanations for why each term is relevant
            7. Prioritize terms that are likely to appear in academic abstracts and titles

            Categories:
            - "synonym": Direct synonyms or alternative names
            - "related": Related concepts or associated terms
            - "abbreviation": Acronyms or abbreviated forms
            - "broader": More general terms
            - "narrower": More specific terms
            """

        self.agent = Agent(
            model_name,
            result_type=ExpansionResponse,
            system_prompt=self.base_system_prompt,
        )

    @property
    def expansion_method(self) -> str:
        """Name of the expansion method."""
        return "llm"

    def supports_context(self) -> List[str]:
        """Return list of supported context keys."""
        return [
            "entity_types",
            "domain",
            "max_terms",
            "include_synonyms",
            "include_related",
            "include_abbreviations",
            "min_confidence",
        ]

    async def expand_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ExpandedQuery:
        """
        Expand query using LLM-generated biomedical terms.

        Args:
            query: Original search query
            context: Optional context including entity types, domain, etc.

        Returns:
            ExpandedQuery with LLM-generated expansion terms
        """
        if not query or not query.strip():
            return ExpandedQuery(
                original_query=query,
                expanded_terms=[],
                expansion_method=self.expansion_method,
            )

        try:
            # Build expansion request
            request = self._build_expansion_request(query, context)

            # Get LLM expansion
            result = await self.agent.run(
                f"Expand the biomedical search query: '{request.query}'\n"
                f"Context: {request.context}\n"
                f"Max terms: {request.max_terms}\n"
                f"Include synonyms: {request.include_synonyms}\n"
                f"Include related concepts: {request.include_related}\n"
                f"Include abbreviations: {request.include_abbreviations}"
            )

            # Convert to ExpansionTerm objects
            expansion_terms = []
            for term_data in result.data.expanded_terms:
                expansion_terms.append(
                    ExpansionTerm(
                        term=term_data.term,
                        confidence=term_data.confidence,
                        source="llm",
                        category=term_data.category,
                    )
                )

            # Apply context-based filtering
            expansion_terms = self._apply_context_filters(expansion_terms, context)

            # Calculate total confidence
            total_confidence = (
                sum(term.confidence for term in expansion_terms)
                / max(len(expansion_terms), 1)
                if expansion_terms
                else 1.0
            )

            return ExpandedQuery(
                original_query=query,
                expanded_terms=expansion_terms,
                expansion_method=self.expansion_method,
                total_confidence=min(1.0, total_confidence),
            )

        except Exception as e:
            # Fallback to empty expansion on error
            print(f"LLM expansion failed: {e}")
            return ExpandedQuery(
                original_query=query,
                expanded_terms=[],
                expansion_method=self.expansion_method,
                total_confidence=0.0,
            )

    def _build_expansion_request(
        self, query: str, context: Optional[Dict[str, Any]]
    ) -> ExpansionRequest:
        """Build expansion request from query and context."""
        if not context:
            context = {}

        # Determine domain context
        domain_context = "biomedical research"
        entity_types = context.get("entity_types", [])

        if "gene" in entity_types:
            domain_context += " focusing on genes and proteins"
        if "disease" in entity_types:
            domain_context += " focusing on diseases and disorders"
        if "drug" in entity_types:
            domain_context += " focusing on drugs and therapeutics"

        return ExpansionRequest(
            query=query,
            context=domain_context,
            max_terms=min(context.get("max_terms", self.max_terms), 20),
            include_synonyms=context.get("include_synonyms", True),
            include_related=context.get("include_related", True),
            include_abbreviations=context.get("include_abbreviations", True),
        )

    def _apply_context_filters(
        self, expansion_terms: List[ExpansionTerm], context: Optional[Dict[str, Any]]
    ) -> List[ExpansionTerm]:
        """Apply context-based filtering to expansion terms."""
        if not context:
            return expansion_terms

        filtered = list(expansion_terms)

        # Filter by minimum confidence
        min_confidence = context.get("min_confidence", 0.3)  # Higher default for LLM
        if min_confidence > 0.0:
            filtered = [term for term in filtered if term.confidence >= min_confidence]

        # Filter by entity types if specified
        entity_types = context.get("entity_types", [])
        if entity_types:
            # Keep terms that match the requested entity types
            relevant_categories = set()
            if "gene" in entity_types:
                relevant_categories.update(["synonym", "related", "abbreviation"])
            if "disease" in entity_types:
                relevant_categories.update(
                    ["synonym", "related", "broader", "narrower"]
                )

            if relevant_categories:
                filtered = [
                    term for term in filtered if term.category in relevant_categories
                ]

        # Sort by confidence (highest first)
        filtered.sort(key=lambda x: -x.confidence)

        # Apply max terms limit
        max_terms = context.get("max_expansion_terms", len(filtered))
        return filtered[:max_terms]


def create_llm_expander(
    model_name: str = "openai:gpt-4o-mini", max_terms: int = 15
) -> LLMQueryExpander:
    """Factory function to create an LLM query expander."""
    return LLMQueryExpander(model_name=model_name, max_terms=max_terms)
