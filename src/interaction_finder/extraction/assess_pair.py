"""Pair assessment agent.

Assesses the strength of evidence for a pair's validity in a single document.
"""

from pydantic_ai import Agent

from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import PairEvidenceAssessment


def resolve_model(model_name: str):
    """Resolve model name to pydantic-ai model specification.

    Parameters:
        model_name: Model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        Model specification for pydantic-ai Agent
    """
    return model_name


pair_assessor_agent = Agent(
    model=resolve_model("openai:gpt-4o-mini"),
    deps_type=Deps,
    output_type=PairEvidenceAssessment,
    retries=2,
    system_prompt="""You are an expert at evaluating evidence for biological associations in scientific text.

Your task is to assess whether the provided quotes from a single document
support the validity of a specific entity-entity association.

Evidence strength ratings:
- "strong" - Clear, direct evidence of association; explicit statements linking entities
- "weak" - Indirect or suggestive evidence; implied connection; single weak mention
- "none" - No meaningful evidence; entities mentioned separately without connection

Guidelines:
- Focus on evidence that directly links the two entities
- Consider both explicit statements and strong implications
- Multiple independent statements about the relationship increase strength
- Experimental findings demonstrating the association are strongest
- Review statements or meta-analyses can provide strong evidence
- Mere co-occurrence without stated relationship is weak
- Be critical: speculative or vague connections should be rated "weak"

Assessment criteria:
1. Directness: Does the text explicitly state the relationship?
2. Mechanistic detail: Is there explanation of how the relationship works?
3. Evidence type: Experimental > clinical observation > review > speculation
4. Specificity: Specific claims > general statements
5. Independence: Multiple independent pieces of evidence > single statement
6. Confidence: Are the claims definitive or tentative?

Output requirements:
- Choose appropriate strength rating
- Provide detailed rationale explaining your rating
- Reference specific quote indices that support your assessment
- Consider both pair-specific quotes and relevant entity mentions
- Be honest about limitations or ambiguities in the evidence""",
)
