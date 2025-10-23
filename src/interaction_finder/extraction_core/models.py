"""
Configuration models for unified extraction core.

Provides Pydantic models for configuring the extraction, analysis, and evaluation
workflows. These models ensure type safety, validation, and clear documentation
of configuration options.
"""

from enum import Enum
from typing import Literal, Union
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai.models import Model


class CooccurrenceStrategy(str, Enum):
    """
    Strategy for finding entity pairs via co-occurrence analysis.

    Defines how aggressively to search for entity relationships based on
    document structure and proximity. Strategies are ordered from most
    restrictive to most permissive.
    """

    TIERED = "tiered"
    """Use tiered approach: same chunk → adjacent chunks → document level, stopping when sufficient pairs found"""

    SAME_CHUNK = "same_chunk"
    """Only consider entities appearing in the same chunk/section"""

    ADJACENT = "adjacent"
    """Consider entities in same or adjacent chunks"""

    DOCUMENT = "document"
    """Consider any entities appearing in the same document"""


class AnalysisConfig(BaseModel):
    """
    Configuration for individual entity analysis.

    Used in gene-disease workflow to analyze each entity's relevance and
    associations with the target condition. Supports flexible LLM model
    specification for experimentation and cost optimization.

    Example:
        ```python
        config = AnalysisConfig(
            analysis_type="gene_association",
            target_context="pulmonary arterial hypertension",
            max_contexts=5,
            model="openai:gpt-4o"
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    analysis_type: Literal["gene_association", "relationship_potential"] = Field(
        description="Type of analysis to perform on the entity"
    )

    target_context: str = Field(
        description="Disease, condition, or biological context for the analysis (e.g., 'PAH', 'breast cancer')"
    )

    max_contexts: int = Field(
        default=10,
        ge=1,
        description="Maximum number of entity occurrence contexts to include in analysis prompt",
    )

    model: Union[Model, str] = Field(
        description="LLM model to use (pydantic-ai Model instance or string like 'openai:gpt-4o')"
    )


class EvaluationConfig(BaseModel):
    """
    Configuration for entity pair evaluation.

    Used in relationship extraction workflow to assess whether entity pairs
    have specific relationship types (interaction, regulation, etc.). Supports
    filtering pairs based on shared document resources.

    Example:
        ```python
        config = EvaluationConfig(
            relationship_type="protein-protein interaction",
            task_context="cancer biology",
            model="openai:gpt-4o-mini",
            require_shared_resources=True
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    relationship_type: str = Field(
        description="Type of relationship to evaluate (e.g., 'interaction', 'regulation', 'phosphorylation')"
    )

    task_context: str = Field(
        description="Biological or research context for evaluation (e.g., 'cancer', 'signaling pathways')"
    )

    model: Union[Model, str] = Field(
        description="LLM model to use (pydantic-ai Model instance or string like 'openai:gpt-4o')"
    )

    require_shared_resources: bool = Field(
        default=True,
        description="Only evaluate pairs that co-occur in at least one document (strongly recommended)",
    )
