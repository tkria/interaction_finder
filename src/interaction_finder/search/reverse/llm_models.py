"""
Pydantic models for LLM-based query generation.

Defines structured output types for the LLM agent that generates
search queries from paper content.
"""

from pydantic import BaseModel, Field


class LLMQueryResponse(BaseModel):
    """
    Structured response from LLM query generation agent.

    The LLM analyzes paper content and hint fields to generate 1-3
    precision-focused search queries that can recover the paper.
    """

    queries: list[str] = Field(
        min_length=1,
        max_length=3,
        description="Generated search queries (1-3 queries, specific and targeted)",
    )
    reasoning: str = Field(
        description="Brief explanation of query generation strategy and why these queries are effective"
    )
    backend_specific: bool = Field(
        default=False,
        description="Whether queries use backend-specific syntax (e.g., PubMed field tags like [Title])",
    )

    def validate_query_count(self) -> bool:
        """Ensure we have at least one query and no more than three."""
        return 1 <= len(self.queries) <= 3

    def get_top_queries(self, n: int) -> list[str]:
        """Get top N queries, respecting the limit."""
        return self.queries[:n]


class LLMQueryConstructionResponse(BaseModel):
    """
    Structured response from LLM query construction agent.

    The LLM analyzes extracted keywords, keyword scores, hint terms, and
    optionally full resource content to construct a single, optimized
    search query.

    This model is used for Stage 2 of the two-stage pipeline (query construction),
    where the LLM receives both extracted keywords (structural guidance) AND
    full content (additional context) to build sophisticated queries.

    Fields:
        query: str - The constructed search query
        reasoning: str - Brief explanation of construction strategy (optional, for debugging)

    Example:
        >>> response = LLMQueryConstructionResponse(
        ...     query='BRCA1[Title] AND ("breast cancer" OR "mammary carcinoma")',
        ...     reasoning="Combined specific gene with synonymous disease terms for precision"
        ... )
    """

    query: str = Field(
        min_length=1,
        description="The constructed search query (non-empty)",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of construction strategy (optional, for debugging)",
    )
