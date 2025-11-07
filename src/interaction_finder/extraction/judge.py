"""Final judgment agent.

Makes cross-document judgments on pair validity by synthesizing evidence from all resources.
"""

from pydantic_ai import Agent

from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import FinalJudgmentOut


def resolve_model(model_name: str):
    """Resolve model name to pydantic-ai model specification.

    Parameters:
        model_name: Model identifier (e.g., "openai:gpt-4o-mini")

    Returns:
        Model specification for pydantic-ai Agent
    """
    return model_name


# Cached agent instances by model name
_final_judge_agents: dict[str, Agent] = {}


def get_final_judge_agent(model_name: str) -> Agent:
    """Get or create final judge agent instance.

    Lazy initialization to avoid requiring API keys at import time.
    Caches agents per model name for reuse.

    Parameters:
        model_name: Model identifier (e.g., "openai:gpt-4o")

    Returns:
        Configured Agent instance
    """
    if model_name not in _final_judge_agents:
        _final_judge_agents[model_name] = Agent(
            model=resolve_model(model_name),
            deps_type=Deps,
            output_type=FinalJudgmentOut,
            retries=2,
            system_prompt="""You are an expert scientific reviewer synthesizing evidence across multiple documents.

Your task is to make a final judgment on whether an entity-entity association
is valid based on evidence from multiple sources. You will receive:
1. Quotes from all documents supporting the association
2. Per-document evidence assessments
3. The topic and entity types

Decision criteria:
- Accept if there is strong, consistent evidence across sources
- Reject if evidence is consistently weak, contradictory, or absent
- Accept with medium confidence if evidence is mixed but leans positive
- Reject with medium confidence if evidence is mixed but leans negative

Confidence levels:
- "high" - Strong consensus; clear, consistent evidence; no significant contradictions
- "medium" - Evidence leans one direction but has gaps or minor contradictions
- "low" - Evidence is mixed or ambiguous; judgment call required

Guidelines for acceptance:
1. Multiple independent sources with strong evidence → accept with high confidence
2. Single strong source + supportive weak sources → accept with medium confidence
3. Multiple weak sources with consistent message → accept with low confidence
4. Mixed strong/none assessments → examine quotes carefully, decide on balance
5. Consistent "none" assessments → reject with high confidence
6. Mostly weak evidence with no strong support → reject with medium confidence

Important considerations:
- Quality over quantity: one high-quality study beats many weak mentions
- Consistency matters: contradictions reduce confidence
- Mechanistic evidence is stronger than correlation
- Primary research is stronger than secondary citations
- Recent evidence may supersede older findings
- Absence of evidence is not evidence of absence (be cautious with rejection)

Output requirements:
- Make clear accept/reject decision
- Choose appropriate confidence level
- Provide detailed rationale referencing specific evidence
- Acknowledge contradictions or limitations
- Explain what tipped the balance of judgment""",
        )
    return _final_judge_agents[model_name]
