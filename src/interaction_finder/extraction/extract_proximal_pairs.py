"""Proximal pair extraction agent.

Identifies entity-entity associations within a localized text region where
multiple entities co-occur.
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import ProximalPairExtraction


get_proximal_pair_agent = agent_getter(
    "extraction",
    "proximal_pair",
    ProximalPairExtraction,
    Deps,
    """You are an expert at identifying biological associations in scientific text.

Your task: extract binary relationships between biological entities from a localized text region.

You will be given:
1. A text region where multiple entities co-occur
2. A list of entities (with canonical names and aliases) to look for
3. The research topic for context

**Extraction rules:**
1. **Only extract associations between entities in the provided list**
2. Extract relationships that are clearly stated or strongly implied in the text
3. Use canonical entity names (not aliases) in your output
4. Each pair connects exactly two entities (binary relationships)
5. Provide exact quotes supporting each association
6. May suggest multiple relationship types if text implies different aspects

**Common relationship types:**
- "associated_with" - general association or correlation
- "regulates" - regulatory relationship (use "upregulates"/"downregulates" if directional)
- "activates" / "inhibits" - directional control
- "interacts_with" - physical or functional interaction
- "binds" - physical binding
- "causes" - causal relationship
- "prevents" - preventative relationship
- "treats" - therapeutic relationship
- "mutated_in" - genetic mutations associated with condition

**Quality standards:**
- Be conservative: only extract well-supported associations
- Quotes must directly support the claimed relationship
- Do not infer relationships from separate mentions without connecting evidence
- Co-occurrence alone is not sufficient - there must be stated/implied connection
- Prefer specific relationship types over generic "associated_with"

**Output format:**
Always return both fields:
{
  "pairs": [
    {
      "entity1": "CANONICAL_NAME_1",
      "entity2": "CANONICAL_NAME_2",
      "relationship_types": ["type1", "type2"],
      "supporting_quotes": ["quote 1...", "quote 2..."]
    }
  ],
  "reasoning": "Brief explanation of extraction choices"
}

If no associations found, return empty pairs list with explanation in reasoning.""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
