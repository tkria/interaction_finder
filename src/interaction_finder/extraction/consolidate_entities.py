"""Entity consolidation agent.

Determines how to handle entity pairs with substring/similarity relationships:
skip (keep separate), merge (into parent), or rename (simplify verbose names).
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
distinctions matter: keep separate (skip) if both are relevant and distinct in this context, merge if
one is merely a redundant variant, or rename when appropriate.

# Actions

## skip
Keep both entities separate. Use when they represent genuinely distinct concepts relevant to the research topic.

Examples:
1. [xxxx] 'MAP kinase' → 'kinase' — skip (MAPK is a specific family)
2. [xxxx] 'adrenergic receptor' → 'receptor' — skip (specific receptor type)
3. [xxxx] 'p53 pathway' → 'p53' — skip (gene vs pathway)

## merge
Merge the term into the parent. Use when the term is the same entity with a redundant qualifier or variant.

Examples:
1. [xxxx] 'p53 gene' → 'p53' — merge ("gene" is redundant)
2. [xxxx] 'mutant BRCA1' → 'BRCA1' — merge (variant of same gene)
3. [xxxx] 'human insulin' → 'insulin' — merge (species qualifier)

## rename
Replace verbose descriptions with standard names, or normalize overly-specific variants to a topic term
when the extra specificity doesn't add meaningful distinction for this research.

Examples:
1. [xxxx] 'transforming growth factor beta protein' → 'growth factor' — rename to "TGF-β"
2. [xxxx] 'mitogen-activated protein kinase enzyme' → 'kinase' — rename to "MAPK"
3. [xxxx] 'peroxisome proliferator-activated receptor gamma' → 'receptor' — rename to "PPARγ"
4. [xxxx] 'familial idiopathic pulmonary arterial hypertension' → 'hypertension' — rename to "pulmonary arterial hypertension" (normalize to topic)

Only rename to widely recognized standard names or to a topic term when appropriate.

# Output

For each pair, provide pair_id, pair_token, action, target (for rename only), and reasoning.""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
