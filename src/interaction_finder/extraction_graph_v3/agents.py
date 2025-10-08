"""
Agent factories for extraction graph V3.

Provides factory functions for creating pydantic-ai agents used in the V3 pipeline:
- Entity extraction from full documents
- Individual entity assessment with relationship potential
- Pair evaluation for candidate relationships

Following pydantic-ai patterns:
- Agents produce structured, typed outputs
- System prompts emphasize exact quote extraction
- Output validators ensure data quality
"""

from typing import List, Union
from pydantic_ai import Agent
from pydantic_ai.models import Model

from .deps import ExtractionDepsV3
from .models import PairEvaluationOut
from ..extraction_graph_v2.models import SimpleEntityListOut, AssessmentOut
from ..models import Term


def create_entity_extractor_v3(
    model: Union[str, Model],
    entity_kinds: List[str],
    task_context: str,
    target_term: Term,
) -> Agent[ExtractionDepsV3, SimpleEntityListOut]:
    """
    Create entity extraction agent for V3 pipeline.

    Extracts entities from full documents (not chunks) with emphasis on
    complete document context and exact quote extraction.

    Args:
        model: LLM model identifier (e.g., 'openai:gpt-4o-mini')
        entity_kinds: List of entity types to extract (e.g., ['gene', 'disease'])
        task_context: Task context from configuration (e.g., 'gene-disease interactions')
        target_term: Target term for extraction context

    Returns:
        Agent configured for entity extraction with SimpleEntityListOut output
    """
    kinds_str = ", ".join(entity_kinds)
    target_info = (
        f"{target_term.name} ({target_term.kind})"
        if target_term.kind
        else target_term.name
    )

    system_prompt = f"""You are a biomedical entity extractor specialized in {task_context}.

Task: Extract all {kinds_str} entities from the document that are relevant to {target_info}.

CRITICAL REQUIREMENTS:
1. Extract entities that appear in the document and relate to {target_info}
2. Provide the exact name as it appears in the document
3. List alternative names/aliases if mentioned
4. Include supporting quotes (exact text passages from the document)

QUOTE REQUIREMENTS:
• Quotes must be VERBATIM - copy text exactly as written
• Use the same words, spellings, and structure as the original
• Include enough context to be meaningful (typically 1-2 sentences)
• Do NOT combine text from different locations
• Do NOT paraphrase or modify the text

OUTPUT FORMAT:
{{
    "entities": [
        {{
            "name": "EXACT_ENTITY_NAME",
            "kind": "one of: {kinds_str}",
            "aliases": ["alternative_name"],
            "quotes": ["exact quote from document"]
        }}
    ],
    "entity_kinds": ["{kinds_str}"],
    "reasoning": "Brief explanation of extraction process"
}}

Context: {task_context}
"""

    return Agent(
        model=model,
        output_type=SimpleEntityListOut,
        system_prompt=system_prompt,
        deps_type=ExtractionDepsV3,
    )


def create_assessment_agent_v3(
    model: Union[str, Model],
    entity_kinds: List[str],
    relationship_type: str,
) -> Agent[ExtractionDepsV3, AssessmentOut]:
    """
    Create individual entity assessment agent for V3 pipeline.

    Assesses whether an entity has relationship potential and identifies
    related entities. Enhanced from V2 to explicitly request related entity names.

    Args:
        model: LLM model identifier (e.g., 'openai:gpt-4o-mini')
        entity_kinds: List of entity types in the task (e.g., ['gene', 'disease'])
        relationship_type: Type of relationships to assess (e.g., 'gene-disease interactions')

    Returns:
        Agent configured for assessment with AssessmentOut output
    """
    kinds_str = ", ".join(entity_kinds)

    system_prompt = f"""You are evaluating entity relationships in {relationship_type}.

Your task is to assess whether the entity has potential for {relationship_type} relationships.

CRITICAL REQUIREMENTS:
1. Evaluate the entity's relationship potential based on the provided contexts
2. List specific names of related entities (NOT entity types) in the 'related' field
3. Provide exact quotes from contexts as evidence
4. Explain your reasoning clearly

IMPORTANT: The 'related' field must contain specific entity names (e.g., "BRCA1", "breast cancer"),
not entity types (e.g., "genes", "diseases"). These names will be used to generate pair candidates.

Context mentions:
{{contexts}}

Evaluate:
1. Does this entity show {relationship_type} potential?
2. What specific entities (by name) might it relate to?
3. What evidence supports these relationships?

OUTPUT FORMAT:
{{
    "potential": "high|medium|low|none",
    "related": ["specific_entity_name1", "specific_entity_name2"],
    "evidence": ["exact quote from context"],
    "reasoning": "Detailed explanation"
}}

Entity kinds: {kinds_str}
"""

    return Agent(
        model=model,
        output_type=AssessmentOut,
        system_prompt=system_prompt,
        deps_type=ExtractionDepsV3,
    )


def create_pair_evaluator_v3(
    model: Union[str, Model],
    relationship_type: str,
) -> Agent[ExtractionDepsV3, PairEvaluationOut]:
    """
    Create pair evaluation agent for V3 pipeline.

    Evaluates candidate entity pairs to determine if a genuine relationship exists.
    NEW agent not present in V2.

    Args:
        model: LLM model identifier (e.g., 'openai:gpt-4o-mini')
        relationship_type: Type of relationships to evaluate (e.g., 'gene-disease interactions')

    Returns:
        Agent configured for pair evaluation with PairEvaluationOut output
    """

    system_prompt = f"""You are evaluating potential {relationship_type} relationships between entity pairs.

Your task is to determine if a relationship exists based on evidence from shared contexts.

CRITICAL REQUIREMENTS:
1. Analyze whether the evidence supports a direct relationship
2. Consider if the evidence is explicit or implicit
3. Assess your confidence level (high/medium/low)
4. Provide exact quotes supporting the relationship

Evidence evaluation:
- Entity A: {{entity_a_name}} ({{entity_a_kind}})
- Entity B: {{entity_b_name}} ({{entity_b_kind}})

Contexts where both entities appear:
{{evidence_contexts}}

Consider:
1. Do the contexts support a direct {relationship_type} relationship?
2. Is the evidence explicit (directly stated) or implicit (implied)?
3. How confident are you in this evaluation?

OUTPUT FORMAT:
{{
    "relationship_exists": true|false,
    "relationship_type": "{relationship_type}",
    "confidence": "high|medium|low",
    "evidence": ["exact quote supporting relationship"],
    "reasoning": "Detailed explanation of evaluation"
}}

If relationship exists, evidence quotes are required.
"""

    return Agent(
        model=model,
        output_type=PairEvaluationOut,
        system_prompt=system_prompt,
        deps_type=ExtractionDepsV3,
    )
