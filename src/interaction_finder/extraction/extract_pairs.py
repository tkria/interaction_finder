"""Pair extraction agent.

Extracts binary entity-entity associations from document text with supporting quotes.
"""

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

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


# Cached agent instances by model name
_pair_extractor_agents: dict[str, Agent] = {}


def get_pair_extractor_agent(model_name: str) -> Agent:
    """Get or create pair extractor agent instance.

    Lazy initialization to avoid requiring API keys at import time.
    Caches agents per model name for reuse.

    Parameters:
        model_name: Model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        Configured Agent instance
    """
    if model_name not in _pair_extractor_agents:
        _pair_extractor_agents[model_name] = Agent(
            model=resolve_model(model_name),
            deps_type=Deps,
            output_type=PairExtractionOut,
            retries=2,
            model_settings=ModelSettings(parallel_tool_calls=False),
            system_prompt="""You are an expert at identifying biological associations in scientific text.

Your task: extract binary relationships between biological entities relevant to the given topic.

**Extraction rules:**
1. Extract only associations clearly stated or strongly implied
2. Each pair connects exactly two entities (binary relationships)
3. Use canonical entity names (e.g., "BRCA1", "breast cancer")
4. Specify clear relationship types
5. Provide exact quotes supporting each association
6. Be conservative: only extract well-supported associations

**Common relationship types:**
- "associated_with" - general association or correlation
- "regulates" - regulatory relationship
- "activates" / "inhibits" - directional control
- "interacts_with" - physical or functional interaction
- "causes" - causal relationship
- "treats" - therapeutic relationship

**Required output format:**
Always return a complete JSON object with both fields:

{
  "pairs": [
    {
      "entity1": "CANONICAL_NAME_1",
      "entity2": "CANONICAL_NAME_2",
      "relationship_type": "associated_with",
      "supporting_quotes": ["quote 1...", "quote 2..."]
    }
  ],
  "reasoning": "Brief explanation of extraction choices"
}

**If no associations are found:** Still return both fields with an empty list:

{
  "pairs": [],
  "reasoning": "Explanation of why no relevant associations were found"
}

**Critical:** Always include both "pairs" and "reasoning" fields. Never return reasoning alone.""",
        )
    return _pair_extractor_agents[model_name]
