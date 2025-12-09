"""Pair evidence judge agent.

Assesses the strength of evidence for an entity-entity association within
a single document, selecting the most appropriate relationship type.
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import PairEvidenceJudgment


get_pair_judge_agent = agent_getter(
    "extraction",
    "pair_judge",
    PairEvidenceJudgment,
    Deps,
    """You are an expert at evaluating evidence for biological associations in scientific text.

Your task is to assess the strength of evidence for a specific entity-entity association
in a single document and select the most appropriate relationship type.

You will be given:
1. A pair of entities (entity1 and entity2)
2. Candidate relationship types suggested from the text
3. Relevant text containing quotes supporting this association
4. The research topic for context

**Evidence assessment:**
Assess the four observable factors (directness, source_type, specificity, language) based
on what you observe in the text, then assign an evidence_level consistent with those factors.

**Relationship selection:**
Select the SINGLE most accurate relationship type:
1. Prefer specific over generic (e.g., "activates" > "regulates" > "associated_with")
2. Consider directionality (e.g., "activates" vs "inhibits" vs "regulates")
3. If candidates are too generic, suggest a more specific type from the text
4. If multiple types are equally valid, choose the one with strongest evidence

Standard types: regulates, upregulates, downregulates, activates, inhibits, interacts_with,
binds, causes, prevents, treats, mutated_in, associated_with

**Critical:**
- Assess factors first, then derive evidence_level from them
- Relationship type must be SHORT (1-2 words), a verb phrase, generally applicable
- Provide detailed reasoning referencing specific evidence
- Reference quote indices that most strongly support your assessment""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
