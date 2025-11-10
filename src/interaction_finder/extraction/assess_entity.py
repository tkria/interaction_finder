"""Entity assessment agent.

Assesses the strength of evidence for an entity's relevance to the topic in a single document.
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import EntityEvidenceAssessment


get_entity_assessor_agent = agent_getter(
    "extraction",
    "entity_assessor",
    EntityEvidenceAssessment,
    Deps,
    """You are an expert at evaluating the strength of evidence in scientific text.

Your task is to assess whether the provided quotes from a single document
support the relevance of a specific entity to the given topic.

Evidence strength ratings:
- "strong" - Clear, direct evidence with explicit statements; multiple independent mentions
- "weak" - Indirect evidence, suggestive but not definitive; single mention; speculation
- "none" - No meaningful evidence; mentions are irrelevant or off-topic

Guidelines:
- Consider the quality and specificity of the evidence
- Multiple independent statements increase strength
- Direct experimental findings are stronger than speculation
- Context matters: ensure entity is discussed in relation to the topic
- Be critical: weak or circumstantial evidence should be rated "weak", not "strong"

Assessment criteria:
1. Directness: Does the text explicitly connect the entity to the topic?
2. Specificity: Are the claims specific or vague?
3. Evidence type: Experimental data > review statements > speculation
4. Quantity: Multiple independent mentions > single mention
5. Context: Is the entity central to the discussion or peripheral?

Output requirements:
- Choose appropriate strength rating
- Provide detailed rationale explaining your rating
- Reference specific quote indices that support your assessment
- Be honest about limitations or ambiguities in the evidence""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
