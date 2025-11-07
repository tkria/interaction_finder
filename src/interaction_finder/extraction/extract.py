"""Entity extraction agent.

Extracts entities of specified types from document text with supporting quotes.
"""

from pydantic_ai import Agent

from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import EntityExtractionOut


def resolve_model(model_name: str):
    """Resolve model name to pydantic-ai model specification.

    Handles both cloud models (e.g., "openai:gpt-4o") and local models.
    For simplicity, we pass model names directly and rely on pydantic-ai's
    built-in resolution.

    Parameters:
        model_name: Model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        Model specification for pydantic-ai Agent
    """
    return model_name


# Cached agent instances by model name
_entity_extractor_agents: dict[str, Agent] = {}


def get_entity_extractor_agent(model_name: str) -> Agent:
    """Get or create entity extractor agent instance.

    Lazy initialization to avoid requiring API keys at import time.
    Caches agents per model name for reuse.

    Parameters:
        model_name: Model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        Configured Agent instance
    """
    if model_name not in _entity_extractor_agents:
        _entity_extractor_agents[model_name] = Agent(
            model=resolve_model(model_name),
            deps_type=Deps,
            output_type=EntityExtractionOut,
            retries=2,
            system_prompt="""You are an expert biomedical entity extraction specialist.

Your task is to identify and extract biological entities from scientific text.
Focus on entities of the specified types and ensure they are relevant to the
given topic.

Guidelines:
- Extract entities only of the requested types (genes, diseases, proteins, etc.)
- Use canonical names when possible (e.g., "BRCA1" not "BRCA-1")
- Include all verbatim names as they appear in the text
- Provide exact quotes that mention each entity
- Focus on entities clearly relevant to the topic
- Be conservative: only extract entities with clear textual support

Output requirements:
- Use canonical names as dictionary keys
- Include entity type for each entity
- List all verbatim names found in text
- Provide multiple supporting quotes when available
- Give brief reasoning for your extraction choices""",
        )
    return _entity_extractor_agents[model_name]
