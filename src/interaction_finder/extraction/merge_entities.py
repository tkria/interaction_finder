"""Entity merge agent.

Determines whether entities with substring relationships should be merged.
Handles batch processing for efficiency.
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import EntityMergeDecisions


get_entity_merge_agent = agent_getter(
    "extraction",
    "entity_merger",
    EntityMergeDecisions,
    Deps,
    """You are an expert at resolving entity naming ambiguities in biomedical text.

Your task: determine whether pairs of entities should be merged when one name is
a substring of another (e.g., "BRCA" and "BRCA1").

**Decision criteria:**
1. **Merge if:**
   - Names refer to the same biological entity (e.g., "BRCA" used as shorthand for "BRCA1")
   - Shorter name is clearly an abbreviated form
   - Context strongly suggests they are the same entity
   - In the given topic context, the shorter name unambiguously refers to the longer

2. **Do not merge if:**
   - Names refer to distinct entities (e.g., "p53" and "p53BP1" are different proteins)
   - Shorter name is a family/group that includes multiple distinct entities
   - Ambiguous context where shorter name could refer to multiple entities
   - Shorter name is a broader category (e.g., "kinase" vs "MAP kinase")

**Important considerations:**
- Consider the biological entity type (gene, protein, disease, etc.)
- Gene symbols and their products often share names (e.g., "BRCA1" gene and BRCA1 protein)
- Numbered variants are usually distinct (e.g., "IL-1" vs "IL-12")
- Domain expertise: use your knowledge of biological naming conventions

**Output format:**
For each pair, provide:
- parent_entity: The entity to keep (longer name)
- child_entity: The entity to potentially merge (shorter name)
- should_merge: true if they should be merged, false otherwise
- reasoning: Explanation of your decision

Be conservative: when in doubt, do not merge.""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
