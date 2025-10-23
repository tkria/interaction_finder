"""
Individual entity analysis for gene-disease associations.

Provides pure functional interface for analyzing individual entities to assess
their relevance and associations with target conditions.
"""

from typing import Optional

# Import with TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction_graph_v2.models import EntityWithQuotes, IndividualAssessment

from .models import AnalysisConfig


async def analyze_entity(
    entity: "EntityWithQuotes",
    config: AnalysisConfig,
) -> Optional["IndividualAssessment"]:
    """
    Analyze an entity for disease associations or relationship potential.

    Uses LLM to assess entity's relevance to the target condition and identify
    potential relationships. Samples entity contexts to stay within token limits
    while maximizing coverage of entity mentions.

    Args:
        entity: Entity to analyze with all occurrence quotes
        config: Analysis configuration (type, target context, model, max contexts)

    Returns:
        IndividualAssessment with relationship potential and evidence, or None if
        entity has no relevant associations

    Raises:
        RuntimeError: If LLM analysis fails after retries

    Example:
        ```python
        config = AnalysisConfig(
            analysis_type="gene_association",
            target_context="pulmonary arterial hypertension",
            max_contexts=5,
            model="openai:gpt-4o"
        )

        assessment = await analyze_entity(entity, config)
        if assessment:
            print(f"{entity.name}: {assessment.relationship_potential}")
            print(f"Related entities: {assessment.related_entities}")
            print(f"Evidence: {len(assessment.evidence_quotes)} quotes")
        ```
    """
    raise NotImplementedError("analyze_entity will be implemented in Task 04")
