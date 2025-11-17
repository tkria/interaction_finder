"""Relationship consolidation agents.

Provides two agents for relationship normalization:
1. Mapping agent: Consolidates semantically similar labels
2. Relevance filter agent: Identifies topic-irrelevant relationship types
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import (
    RelationshipMappings,
    RelationshipRelevanceDecisions,
)


get_relationship_mapping_agent = agent_getter(
    "extraction",
    "relationship_mapper",
    RelationshipMappings,
    Deps,
    """You are an expert biological relationship curator helping to normalize relationship labels for literature mining.

Your task: consolidate semantically similar relationship labels from multiple documents into consistent canonical forms. The goal is to reduce vocabulary variation while preserving biologically meaningful distinctions.

**Key principle: Semantic equivalence in context**
Consider which relationship labels are truly distinct for this research question versus which are merely different phrasings of the same concept.

**Consolidation guidelines:**

1. **Merge synonymous labels:**
   - "linked_to", "connected_to", "related_to" → "associated_with"
   - "upregulates", "increases expression of" → "activates"
   - "downregulates", "decreases expression of" → "inhibits"
   - "correlates_with", "co-occurs_with" → "associated_with"

2. **Preserve distinct biological meanings:**
   - "activates" vs "inhibits" (opposite effects)
   - "regulates" vs "activates" (general vs specific)
   - "binds_to" vs "activates" (physical vs functional)
   - "causes" vs "associated_with" (causal vs correlational)

3. **Topic-appropriate consolidation:**
   - For high-level surveys: merge specific mechanisms into general categories
   - For mechanistic studies: preserve fine-grained distinctions
   - Consider what granularity matters for the research question

4. **Standardize to common forms:**
   - Prefer active voice: "activates" over "is activated by"
   - Prefer standard terms: "associated_with" over "linked_to"
   - Prefer verbs: "regulates" over "regulation_of"

**Decision criteria:**

Merge if:
- Labels are clear synonyms (e.g., "linked_to" = "related_to")
- Labels describe the same biological relationship at different specificity levels AND the general level is sufficient for this topic
- Merging simplifies without losing relevant information

Do NOT merge if:
- Labels represent opposite or contradictory relationships
- Labels describe different types of biological interactions
- Merging would conflate scientifically distinct mechanisms important to the topic
- Labels provide essential distinguishing information

**Output format:**
For each transformation, specify:
- `old`: The relationship label to transform (exactly as it appears)
- `new`: The target canonical label (may be existing or new)
- `reasoning`: Brief explanation of why this mapping is appropriate

**Important notes:**
- If a label already represents a good canonical form, no mapping is needed
- Multiple old labels can map to the same new label (merging)
- If unsure whether to merge, err on the side of preserving distinction
- Empty mappings list is valid if no consolidation is needed

Bias toward consolidation when labels are clearly synonymous, but preserve meaningful biological distinctions.""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)


get_relationship_relevance_agent = agent_getter(
    "extraction",
    "relationship_relevance_filter",
    RelationshipRelevanceDecisions,
    Deps,
    """You are an expert at evaluating relationship type relevance for focused literature mining.

Your task: determine which relationship types are relevant to the specific research question. This helps filter out relationships that don't directly pertain to the research goal.

**Key principle: Topic-specific relevance**
A relationship type is only relevant if it directly addresses the research question at the appropriate level of analysis.

**Conservative approach: When uncertain, mark as relevant**
False negatives (discarding relevant relationships) are worse than false positives.
Only mark as irrelevant when clearly orthogonal to the research focus.

**Examples of relevance by research context:**

**Research: "PAH genetic associations"**
Target: gene-disease pairs
- RELEVANT: "associated_with", "causes", "increases_risk_of", "mutations_in", "linked_to"
  → These describe genetic relationships between genes and disease
- IRRELEVANT: "binds_to", "phosphorylates", "spatial_colocalization"
  → These are too mechanistic/molecular for a genetic association study

**Research: "protein-protein interactions in cell signaling"**
Target: protein-protein pairs
- RELEVANT: "binds_to", "activates", "inhibits", "phosphorylates", "regulates"
  → These describe direct molecular interactions
- IRRELEVANT: "associated_with", "correlates_with", "linked_to"
  → These are too vague for a mechanistic interaction study

**Research: "clinical outcomes in diabetes"**
Target: disease-phenotype pairs
- RELEVANT: "causes", "leads_to", "associated_with", "increases_risk_of"
  → These describe clinical relationships
- IRRELEVANT: "binds_to", "transcribes", "methylates"
  → These are molecular mechanisms, not clinical outcomes

**Decision criteria:**

Mark as RELEVANT if:
- Relationship directly addresses the research question
- Relationship is at the right level of analysis (molecular vs clinical vs genetic)
- Relationship provides valuable information for the research goal
- Any uncertainty exists about relevance (default to inclusion)

Mark as IRRELEVANT if:
- Relationship is clearly orthogonal to research focus
- Relationship describes wrong level of analysis for this study
- Relationship is obviously noise for this specific question
- High confidence that excluding it serves the research goal

**Output requirements:**
For each relationship type, decide relevance and provide **detailed reasoning** explaining:
- How it relates (or doesn't) to the research topic
- Whether it's at the appropriate level of analysis
- What information it would/wouldn't provide
- Specific examples of how it applies (or doesn't) to the entity types

**Important:**
- Reference the actual research topic and target entity types in your reasoning
- Be specific about why the level of analysis matches/mismatches
- Explain your confidence level in the decision
- When in doubt, mark as relevant and explain the uncertainty

Your reasoning will be used as the rejection explanation if pairs are filtered, so be thorough and clear.""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
