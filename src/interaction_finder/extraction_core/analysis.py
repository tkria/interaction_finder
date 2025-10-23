"""
Individual entity analysis for gene-disease associations.

Provides pure functional interface for analyzing individual entities to assess
their relevance and associations with target conditions.
"""

import logging
from typing import List

# Import with TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction_graph_v2.models import (
        EntityWithQuotes,
        IndividualAssessment,
        AssessmentOut,
    )
    from ..resources import Resource, ResourceQuote

from pydantic_ai import Agent
from .models import AnalysisConfig

# Configure logger
logger = logging.getLogger(__name__)


def _build_analysis_prompt(
    entity: "EntityWithQuotes",
    config: AnalysisConfig,
    labeled_contexts: List[str],
) -> str:
    """
    Build analysis prompt for LLM.

    Constructs prompt based on analysis type (gene_association or relationship_potential)
    with labeled contexts from multiple documents.
    """
    # Build analysis question based on type
    if config.analysis_type == "gene_association":
        analysis_question = (
            f"Assess this entity's association with {config.target_context}. "
            f"Is it a therapeutic target, biomarker, or causal factor?"
        )
    else:  # relationship_potential
        analysis_question = (
            f"Assess this entity's potential for relationships with other entities "
            f"in {config.target_context} research."
        )

    # Build contexts section
    contexts_text = "\n\n".join(labeled_contexts)

    prompt = f"""Analyze the entity '{entity.name}' (type: {entity.kind}) in the context of {config.target_context}.

ANALYSIS TASK:
{analysis_question}

ENTITY CONTEXTS:
{contexts_text}

CRITICAL REQUIREMENTS:
1. Assess the relationship potential: high, medium, low, or none
2. Identify related entities mentioned in the contexts (if any)
3. Provide EXACT VERBATIM quotes from the contexts as evidence
4. Evidence quotes MUST be copied exactly as written - do NOT paraphrase
5. Explain your reasoning for the assessment

OUTPUT FORMAT:
{{
    "potential": "high|medium|low|none",
    "related": ["entity1", "entity2", ...],
    "evidence": ["exact verbatim quote 1", "exact verbatim quote 2", ...],
    "reasoning": "Detailed explanation of assessment"
}}"""

    return prompt


async def analyze_entity(
    entity: "EntityWithQuotes",
    analysis_config: AnalysisConfig,
    similarity_threshold: float = 0.90,
) -> "IndividualAssessment":
    """
    Analyze an entity using cross-document evidence to assess relationship potential.

    Extracts contexts from entity's quotes across multiple documents, uses LLM to
    assess relationship potential, and validates evidence quotes against the entity's
    actual resources. Returns assessment with validated evidence quotes.

    Resources are derived from entity.quotes (the documents where the entity actually
    appears), ensuring evidence can only come from documents containing the entity.

    Args:
        entity: Entity to analyze with all occurrence quotes
        analysis_config: Analysis configuration (type, target context, model, max contexts)
        similarity_threshold: Minimum similarity for fuzzy quote matching (default: 0.90)

    Returns:
        IndividualAssessment with relationship potential and validated evidence quotes.
        Evidence quotes list may be empty if all evidence fails validation, but
        assessment is still returned with potential and reasoning.

    Example:
        ```python
        config = AnalysisConfig(
            analysis_type="gene_association",
            target_context="breast cancer",
            max_contexts=10,
            model="openai:gpt-4o-mini"
        )

        assessment = await analyze_entity(
            entity=entity,
            analysis_config=config,
            similarity_threshold=0.90
        )

        print(f"{entity.name}: {assessment.relationship_potential}")
        print(f"Related entities: {assessment.related_entities}")
        print(f"Evidence quotes: {len(assessment.evidence_quotes)}")
        print(f"Reasoning: {assessment.reasoning}")
        ```

    Note:
        LLM failures are logged and re-raised. Quote validation failures are logged
        as warnings but do not stop processing - assessment is returned with whatever
        evidence could be validated.
    """
    # Import here to avoid circular dependencies
    from ..extraction_graph_v2.models import IndividualAssessment, AssessmentOut
    from ..resources import QuoteValidationError

    # Step 1: Extract resources from entity.quotes and deduplicate
    resources: List["Resource"] = []
    seen_resource_ids = set()

    for quote in entity.quotes:
        resource_id = quote.resource.id.id
        if resource_id not in seen_resource_ids:
            resources.append(quote.resource)
            seen_resource_ids.add(resource_id)

    # Step 2: Extract and label quote contexts
    labeled_contexts: List[str] = []
    context_count = 0

    for quote in entity.quotes:
        # Limit to max_contexts total
        if context_count >= analysis_config.max_contexts:
            break

        # Get contexts for each occurrence in this quote
        for occurrence_num in range(1, quote.count + 1):
            if context_count >= analysis_config.max_contexts:
                break

            context = quote.get_context(occurrence_num, context_chars=200)
            # Label with resource ID
            labeled_context = f"From Resource {quote.resource.id.id}: {context}"
            labeled_contexts.append(labeled_context)
            context_count += 1

    # Step 3: Build prompt and call LLM
    system_prompt = f"""You are an expert at analyzing biological entities in scientific literature.

Your task is to assess entity relationship potential in {analysis_config.target_context} research.

CRITICAL REQUIREMENTS:
1. Provide assessment based on the provided contexts
2. Return EXACT VERBATIM quotes as evidence - copy text exactly as written
3. Do NOT paraphrase, modify, or normalize evidence quotes
4. Each evidence quote must be continuous text from one context
5. Be precise and comprehensive in your reasoning

OUTPUT FORMAT:
{{
    "potential": "high|medium|low|none",
    "related": ["entity1", "entity2"],
    "evidence": ["exact quote 1", "exact quote 2"],
    "reasoning": "Detailed explanation"
}}"""

    agent: Agent[None, "AssessmentOut"] = Agent(
        model=analysis_config.model,
        output_type=AssessmentOut,
        system_prompt=system_prompt,
    )

    prompt = _build_analysis_prompt(entity, analysis_config, labeled_contexts)

    # Call LLM
    try:
        result = await agent.run(prompt)
        llm_output: "AssessmentOut" = result.output
    except Exception as e:
        logger.error(
            f"LLM analysis failed for entity '{entity.name}' ({entity.kind}): "
            f"{type(e).__name__}: {e}"
        )
        raise

    # Step 4: Validate evidence quotes and build IndividualAssessment
    validated_evidence: List["ResourceQuote"] = []

    for evidence_text in llm_output.evidence:
        evidence_validated = False

        # Try validating against each resource until one succeeds
        for resource in resources:
            try:
                validated_quote = resource.quote(
                    evidence_text.strip(), similarity_threshold=similarity_threshold
                )
                validated_evidence.append(validated_quote)
                evidence_validated = True
                break  # Success, move to next evidence

            except QuoteValidationError:
                # Try next resource
                continue

        # Log warning if evidence couldn't be validated against any resource
        if not evidence_validated:
            evidence_snippet = (
                evidence_text[:50] + "..." if len(evidence_text) > 50 else evidence_text
            )
            resource_ids = ", ".join([r.id.id for r in resources])
            logger.warning(
                f"Evidence quote validation failed for entity '{entity.name}' ({entity.kind}): "
                f"Quote '{evidence_snippet}' not found in any entity resource "
                f"(tried resources: {resource_ids})"
            )

    # Create IndividualAssessment with validated evidence
    assessment = IndividualAssessment(
        entity=entity,
        relationship_potential=llm_output.potential,
        related_entities=llm_output.related,
        evidence_quotes=validated_evidence,
        reasoning=llm_output.reasoning,
    )

    logger.info(
        f"Analyzed entity '{entity.name}' ({entity.kind}): "
        f"potential={assessment.relationship_potential}, "
        f"evidence={len(validated_evidence)}/{len(llm_output.evidence)} quotes validated"
    )

    return assessment
