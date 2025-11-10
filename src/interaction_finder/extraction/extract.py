"""Entity extraction agent.

Extracts entities of specified types from document text with supporting quotes.
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import EntityExtractionOut


get_entity_extractor_agent = agent_getter(
    "extraction",
    "entity",
    EntityExtractionOut,
    Deps,
    """You are an expert biomedical entity extraction specialist.

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
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
