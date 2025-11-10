"""Entity assessment agent.

Assesses the strength of evidence for an entity's relevance to the topic in a single document.
"""

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import EntityEvidenceAssessment


def resolve_model(model_name: str):
    """Resolve model name to pydantic-ai model specification.

    Parameters:
        model_name: Model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        Model specification for pydantic-ai Agent
    """
    return model_name


# Cached agent instances by model name
_entity_assessor_agents: dict[str, Agent] = {}


def get_entity_assessor_agent(model_name: str) -> Agent:
    """Get or create entity assessor agent instance.

    Lazy initialization to avoid requiring API keys at import time.
    Caches agents per model name for reuse.

    Parameters:
        model_name: Model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        Configured Agent instance
    """
    if model_name not in _entity_assessor_agents:
        _entity_assessor_agents[model_name] = Agent(
            model=resolve_model(model_name),
            deps_type=Deps,
            output_type=EntityEvidenceAssessment,
            retries=2,
            model_settings=ModelSettings(parallel_tool_calls=False),
            system_prompt="""You are an expert at evaluating the strength of evidence in scientific text.

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
        )
    return _entity_assessor_agents[model_name]
