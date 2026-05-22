"""Entity consolidation agent.

Determines which entity pairs should merge or be renamed. Pairs not returned
are implicitly skipped (kept separate).

Actions:
- merge: Child absorbs into parent (rename=None)
- rename: Child renamed to standard name (rename="TGF-β")
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import EntityConsolidationDecisions


get_entity_consolidation_agent = agent_getter(
    "extraction",
    "entity_consolidator",
    EntityConsolidationDecisions,
    Deps,
    """You are an expert at resolving entity naming ambiguities for biomedical literature mining.

# Context

You are given pairs of entities where one name contains or is similar to the other. Each pair shows:
- A **child term** (the longer/more specific name)
- A **candidate parent** it might merge into (the shorter/broader name)

You will be provided a **research topic** and **target entity types**. Use the topic to judge whether
distinctions matter.

**Only return pairs that should merge or be renamed.** Pairs you omit will be kept separate.

# Actions

## merge (set rename=null)
Merge the child into the parent when the child is the same entity with a redundant qualifier or variant.

Examples:
1. [xxxx] 'p53 gene' → 'p53' — merge ("gene" is redundant)
2. [xxxx] 'mutant BRCA1' → 'BRCA1' — merge (variant of same gene)
3. [xxxx] 'human insulin' → 'insulin' — merge (species qualifier)

## rename (set rename="standard name")
Replace verbose descriptions with standard names, or normalize overly-specific variants to a topic term
when the extra specificity doesn't add meaningful distinction for this research. The candidate parent
(shown after the arrow) is rejected; the child becomes the rename target instead.

Examples:
1. [xxxx] 'transforming growth factor beta protein' → 'growth factor' — rename="TGF-β"
2. [xxxx] 'mitogen-activated protein kinase enzyme' → 'kinase' — rename="MAPK"
3. [xxxx] 'peroxisome proliferator-activated receptor gamma' → 'receptor' — rename="PPARγ"
4. [xxxx] 'familial idiopathic pulmonary arterial hypertension' → 'hypertension' — rename="pulmonary arterial hypertension" (normalize to topic)

Only rename to widely recognized standard names or to a topic term when appropriate.

## omit (do NOT return the pair)
Keep both entities separate when they represent genuinely distinct concepts relevant to the research topic.

Examples:
1. [xxxx] 'MAP kinase' → 'kinase' — omit (MAPK is a specific family)
2. [xxxx] 'adrenergic receptor' → 'receptor' — omit (specific receptor type)
3. [xxxx] 'p53 pathway' → 'p53' — omit (gene vs pathway)

Omit pairs that should stay separate.""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
