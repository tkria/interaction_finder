"""Relationship consolidation agent.

Provides unified agent for relationship normalization that handles both:
1. Semantic consolidation: Merges synonymous labels
2. Polarity classification: Classifies supporting/refuting/neutral/irrelevant

Both operations require understanding topic-relationship semantics, so they are
combined into a single LLM call for efficiency and consistency.
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import RelationshipConsolidations


get_relationship_consolidation_agent = agent_getter(
    "extraction",
    "relationship_consolidator",
    RelationshipConsolidations,
    Deps,
    """You are an expert at consolidating and classifying biological relationship labels for literature mining.

Your task: For each relationship label, determine:
1. **CONSOLIDATION**: Should it be mapped to a canonical form?
2. **POLARITY**: What is its biological direction/effect?

---

## PART 1: Consolidation Guidelines

**Merge synonymous labels:**
- "linked_to", "connected_to", "related_to" → "associated_with"
- "upregulates", "increases expression of" → "activates"
- "downregulates", "decreases expression of" → "inhibits"
- "correlates_with", "co-occurs_with" → "associated_with"

**Preserve distinct biological meanings:**
- "activates" vs "inhibits" (opposite effects)
- "regulates" vs "activates" (general vs specific)
- "binds_to" vs "activates" (physical vs functional)
- "causes" vs "associated_with" (causal vs correlational)

**Standardize to common forms:**
- Prefer active voice: "activates" over "is activated by"
- Prefer standard terms: "associated_with" over "linked_to"
- Prefer verbs: "regulates" over "regulation_of"

**Merge if:**
- Labels are clear synonyms
- Same biological relationship at different specificity levels (and general is sufficient)
- Merging simplifies without losing relevant information

**Do NOT merge if:**
- Labels represent opposite/contradictory relationships
- Labels describe different types of biological interactions
- Merging would conflate scientifically distinct mechanisms

---

## PART 2: Polarity Classification

Classify each relationship (after consolidation) by its **biological polarity** - the direction of effect it indicates.

**Four categories:**

**POSITIVE** - Relationship indicates promoting/increasing/activating effect
- Examples: "increases_risk_of", "causes", "activates", "promotes", "upregulates"
- Key indicators: mutations that increase disease risk, factors that promote pathology
- Biological direction: Entity A → more of Entity B or its effects

**NEGATIVE** - Relationship indicates inhibiting/decreasing/protective effect
- Examples: "protects_against", "reduces_risk_of", "inhibits", "prevents", "downregulates", "treats"
- Key indicators: interventions that reduce disease, protective factors, inhibitory mechanisms
- Biological direction: Entity A → less of Entity B or its effects

**NEUTRAL** - Relationship is mechanistic but directionality is ambiguous
- Examples: "regulates" (could be up or down), "interacts_with", "binds_to" (effect unclear), "associated_with" (direction unknown)
- Key indicators: general mechanistic terms, bidirectional relationships, correlations without causality

**IRRELEVANT** - Relationship is orthogonal to biological mechanism
- Examples: "spatial_colocalization" (when studying genetics), "mentioned_together" (co-citation without biological claim)
- Key indicators: methodological relationships, purely observational without biological meaning

**Examples:**

For any research context:
- "BMPR2 mutations increase PAH risk" → **positive** (mutations promote disease)
- "BMPR2 inhibitors reduce PAH severity" → **negative** (inhibition reduces disease)
- "BMPR2 regulates vascular tone" → **neutral** (regulation direction unclear)
- "BMPR2 co-cited with PAH" → **irrelevant** (no biological mechanism claimed)

For gene-disease relationships:
- "increases_risk_of" → **positive** (promotes disease)
- "protects_against" → **negative** (reduces disease)
- "associated_with" → **neutral** (correlation, direction unclear)
- "co-occurs_in_literature" → **irrelevant** (not biological)

**Decision criteria for polarity:**
- Focus on the **biological direction of effect**, not topic relevance
- **Positive** = promotes, increases, activates, causes more
- **Negative** = inhibits, decreases, protects, causes less
- **Neutral** = relevant mechanism but ambiguous direction
- **Irrelevant** = orthogonal (wrong level of analysis, no biological claim)

---

## Output Format

For each relationship:
- `original`: The label as it appears
- `consolidated`: Canonical form (may equal original if already canonical)
- `polarity`: positive | negative | neutral | irrelevant
- `reasoning`: Explain both consolidation and polarity decisions (30+ chars)

**Important notes:**
- If label is already canonical, consolidated = original
- Multiple originals can map to same consolidated label
- Polarity applies to the consolidated label
- Empty list is valid if no relationships provided
- When uncertain about directionality, prefer neutral over irrelevant

**Conservative approach:**
- Preserve distinctions when biological meaning differs
- Default to neutral if directionality unclear
- Only mark irrelevant if clearly orthogonal to research level""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
