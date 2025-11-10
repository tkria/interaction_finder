"""Entity extraction agent.

Extracts entities of specified types from document text with supporting quotes.
"""

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

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
            model_settings=ModelSettings(parallel_tool_calls=False),
            system_prompt="""You are an expert biomedical entity extraction specialist.

Your task: identify and extract biological entities from scientific text.

**Extraction rules:**
1. Extract only entities of the requested types (gene, disease, protein, etc.)
2. Use canonical names (e.g., "BRCA1" not "BRCA-1")
3. Include all verbatim names as they appear in the text
4. Provide exact quotes supporting each entity
5. Only extract entities clearly relevant to the topic
6. Require clear textual support for every entity
7. Explain your reasoning for each entity extraction

**If no entities are found:** Return an empty list with appropriate structure.""",
        )
    return _entity_extractor_agents[model_name]
