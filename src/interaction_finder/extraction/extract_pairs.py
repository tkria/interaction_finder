"""Pair extraction agent.

Extracts binary entity-entity associations from document text with supporting quotes.
"""

from pydantic_ai import Agent

from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import PairExtractionOut


def resolve_model(model_name: str):
    """Resolve model name to pydantic-ai model specification.

    Parameters:
        model_name: Model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        Model specification for pydantic-ai Agent
    """
    return model_name


pair_extractor_agent = Agent(
    model=resolve_model("openai:gpt-4o-mini"),
    deps_type=Deps,
    output_type=PairExtractionOut,
    retries=2,
    system_prompt="""You are an expert at identifying biological associations in scientific text.

Your task is to extract binary relationships between biological entities that
are relevant to the given topic. Each association should connect exactly two
entities with a specific relationship type.

Guidelines:
- Extract only associations clearly stated or strongly implied in the text
- Each pair should connect exactly two entities (binary relationships)
- Use canonical entity names (e.g., "BRCA1", "breast cancer")
- Specify the relationship type (e.g., "associated_with", "regulates", "inhibits")
- Provide exact quotes that support each association
- Focus on associations relevant to the topic
- Be conservative: only extract well-supported associations

Common relationship types:
- "associated_with" - general association or correlation
- "regulates" - regulatory relationship
- "activates" / "inhibits" - directional control
- "interacts_with" - physical or functional interaction
- "causes" - causal relationship
- "treats" - therapeutic relationship

Output requirements:
- Use canonical entity names
- Specify clear relationship types
- Provide multiple supporting quotes when available
- Give brief reasoning for your extraction choices""",
)
