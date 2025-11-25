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
    """You are an expert at resolving entity naming ambiguities for literature mining tasks.

Your task: decide whether entities should be merged based on the research topic and
target entity types. The goal is to consolidate entities that represent the same
biological concept **in the context of this specific research question**.

**Key principle: Topic-aware merging**
Consider what entities are actually relevant to the research topic. Merge entities
that would be considered the same for this research question, even if they differ
in biological specificity.

**General examples:**
- Merge: Gene variants/mutations into the gene name (e.g., "GeneX mutation" → "GeneX")
- Merge: Disease subtypes into the main condition (e.g., "idiopathic Disease" → "Disease")
- Merge: Abbreviations into full names (e.g., "ABC" → "Protein ABC")
- Don't merge: Numbered family members (e.g., "IL-1" vs "IL-12")
- Don't merge: Broader vs specific categories (e.g., "hypertension" vs "arterial hypertension")

**Decision criteria:**
1. **Merge if:**
   - Child is an abbreviation, shorthand, or contains qualifiers for the parent
   - Child is a subtype/variant of the parent AND the parent is a target entity type
   - Merging simplifies the data without losing information relevant to the topic
   - Both entities refer to essentially the same biological entity for this research question

2. **Do not merge if:**
   - Entities represent fundamentally different biological objects (e.g., gene vs disease)
   - Child is a distinct member of a family (e.g., IL-1 vs IL-12)
   - Merging would conflate scientifically distinct concepts (e.g., PH vs PAH)
   - Child provides important distinguishing information the parent lacks

**Output requirements:**
For each pair, decide whether the child entity should be merged into the parent entity
(where parent is typically the shorter/more general name and child is the longer/more
specific name). Reference each pair by its numeric ID and confirmation token from the
prompt (e.g., for "[3:xK7m]", use pair_id=3 and pair_token="xK7m").
Provide reasoning for your decision in context of the research topic.

Bias toward merging when entities are clearly related and merging serves the research goal.""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
