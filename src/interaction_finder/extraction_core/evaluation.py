"""
Entity pair relationship evaluation.

Provides pure functional interface for evaluating whether entity pairs have
specific relationship types using LLM assessment.
"""

import logging
from typing import Optional, List, Set

# Import with TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction_graph_v2.models import EntityWithQuotes, EntityPairOut
    from ..extraction_graph_v3.models import PairEvaluationOut
    from ..resources import Resource, ResourceQuote

from .models import EvaluationConfig
from pydantic_ai import Agent

# Set up logger
logger = logging.getLogger(__name__)


async def evaluate_pair(
    entity_a: "EntityWithQuotes",
    entity_b: "EntityWithQuotes",
    config: EvaluationConfig,
    similarity_threshold: float = 0.90,
) -> Optional["EntityPairOut"]:
    """
    Evaluate whether an entity pair has the specified relationship type.

    Uses LLM to assess relationship between entities based on cross-document
    evidence from shared resources. Validates all evidence quotes against
    source documents to ensure accuracy.

    This function derives shared resources from the entities' quotes, extracts
    contexts from those resources, and uses an LLM to evaluate whether a
    relationship exists. Evidence quotes from the LLM are validated by trying
    to locate them in the shared resources.

    Args:
        entity_a: First entity with occurrence quotes
        entity_b: Second entity with occurrence quotes
        config: Evaluation configuration (relationship type, context, model)
        similarity_threshold: Minimum similarity for evidence quote validation (default: 0.90)

    Returns:
        EntityPairOut if relationship found with validated evidence, None otherwise.
        Returns None if:
        - No shared resources and require_shared_resources=True
        - LLM determines no relationship exists
        - All evidence quotes fail validation

    Raises:
        RuntimeError: If LLM evaluation fails after retries

    Example:
        ```python
        from interaction_finder.extraction_core import evaluate_pair, EvaluationConfig
        from interaction_finder.extraction_graph_v2.models import EntityWithQuotes
        from interaction_finder.resources import Resource, ResourceId

        # Create test entities with shared resource
        resource = Resource(
            id=ResourceId('http://test.com', 1),
            title='Test',
            text='BRCA1 gene interacts with TP53 tumor suppressor in DNA repair.'
        )

        quote1 = resource.quote('BRCA1 gene', similarity_threshold=0.90)
        quote2 = resource.quote('TP53 tumor suppressor', similarity_threshold=0.90)

        entity_a = EntityWithQuotes(name='BRCA1', kind='gene', quotes=[quote1])
        entity_b = EntityWithQuotes(name='TP53', kind='gene', quotes=[quote2])

        config = EvaluationConfig(
            relationship_type='interaction',
            task_context='DNA repair',
            model='openai:gpt-4o-mini',
            require_shared_resources=True
        )

        pair = await evaluate_pair(entity_a, entity_b, config, similarity_threshold=0.90)
        if pair:
            print(f'Relationship: {pair.relationship}')
            print(f'Confidence: {pair.confidence}')
            print(f'Evidence quotes: {len(pair.evidence_quotes)}')
        ```

    Notes:
        - Shared resources are derived from entity quotes (not passed separately)
        - Only evaluates pairs with shared resources if require_shared_resources=True
        - When require_shared_resources=False, falls back to individual entity contexts
        - All evidence quotes must be validated against shared resources
        - Logs warnings for evidence quotes that fail validation
    """
    from ..extraction_graph_v3.models import PairEvaluationOut
    from ..extraction_graph_v2.models import EntityPairOut
    from ..resources import ResourceId

    # Step 1: Find shared resources from entity quotes
    # Build resource maps (ResourceId -> Resource) for each entity
    resources_a_map = {q.resource.id: q.resource for q in entity_a.quotes}
    resources_b_map = {q.resource.id: q.resource for q in entity_b.quotes}

    # Find shared resource IDs
    shared_resource_ids = set(resources_a_map.keys()) & set(resources_b_map.keys())
    shared_resources: List["Resource"] = [
        resources_a_map[rid] for rid in shared_resource_ids
    ]

    # Check shared resource requirement
    if not shared_resources:
        if config.require_shared_resources:
            logger.info(
                f"No shared resources for entities '{entity_a.name}' and '{entity_b.name}', "
                f"skipping evaluation (require_shared_resources=True)"
            )
            return None
        else:
            # Fall back to all resources from both entities
            logger.info(
                f"No shared resources for entities '{entity_a.name}' and '{entity_b.name}', "
                f"using individual contexts (require_shared_resources=False)"
            )
            all_resource_ids = set(resources_a_map.keys()) | set(resources_b_map.keys())
            shared_resources = [
                resources_a_map.get(rid) or resources_b_map[rid]
                for rid in all_resource_ids
            ]

    # Step 2: Extract contexts from shared resources
    contexts_a = _extract_labeled_contexts(entity_a, shared_resources)
    contexts_b = _extract_labeled_contexts(entity_b, shared_resources)

    # Step 3: Build prompt and call LLM
    prompt = _build_evaluation_prompt(
        entity_a=entity_a,
        entity_b=entity_b,
        contexts_a=contexts_a,
        contexts_b=contexts_b,
        relationship_type=config.relationship_type,
        task_context=config.task_context,
    )

    # Create agent for pair evaluation
    agent: Agent[None, PairEvaluationOut] = Agent(
        model=config.model,
        output_type=PairEvaluationOut,
        system_prompt=(
            "You are an expert at evaluating biological relationships from scientific literature. "
            "Analyze provided contexts to determine if a relationship exists between entities. "
            "Provide exact verbatim quotes from the contexts as evidence."
        ),
    )

    # Call LLM
    try:
        result = await agent.run(prompt)
        evaluation = result.output
    except Exception as e:
        logger.error(f"LLM evaluation failed for {entity_a.name}-{entity_b.name}: {e}")
        raise RuntimeError(f"LLM evaluation failed: {e}") from e

    # If no relationship found, return None
    if not evaluation.relationship_exists:
        logger.debug(
            f"No relationship found between '{entity_a.name}' and '{entity_b.name}'"
        )
        return None

    # Step 4: Validate evidence quotes against shared resources
    validated_evidence: List["ResourceQuote"] = []

    for evidence_text in evaluation.evidence:
        # Try validating against all shared resources
        quote_validated = False

        for resource in shared_resources:
            try:
                validated_quote = resource.quote(evidence_text, similarity_threshold)
                validated_evidence.append(validated_quote)
                quote_validated = True
                break  # Success, no need to try other resources
            except Exception:
                # Quote not found in this resource, try next
                continue

        if not quote_validated:
            # Log warning with resource IDs tried
            resource_ids = [r.id.id for r in shared_resources]
            evidence_snippet = (
                evidence_text[:100] + "..."
                if len(evidence_text) > 100
                else evidence_text
            )
            logger.warning(
                f"Evidence quote validation failed for {entity_a.name}-{entity_b.name}: "
                f"'{evidence_snippet}' not found in resources {resource_ids}"
            )

    # If no evidence quotes validated, return None
    if not validated_evidence:
        logger.warning(
            f"All evidence quotes failed validation for {entity_a.name}-{entity_b.name}, "
            f"rejecting relationship"
        )
        return None

    # Build EntityPairOut with validated evidence
    return EntityPairOut(
        entity_a=entity_a,
        entity_b=entity_b,
        relationship=config.relationship_type,
        confidence=evaluation.confidence,
        evidence_quotes=validated_evidence,
        reasoning=evaluation.reasoning,
    )


def _extract_labeled_contexts(
    entity: "EntityWithQuotes", resources: List["Resource"]
) -> List[str]:
    """
    Extract contexts for entity from specified resources, labeled with resource IDs.

    Args:
        entity: Entity to extract contexts for
        resources: Resources to extract from

    Returns:
        List of context strings labeled with resource IDs
    """
    resource_ids = {r.id for r in resources}
    labeled_contexts = []

    for quote in entity.quotes:
        if quote.resource.id in resource_ids:
            # Get contexts for this quote
            for i in range(quote.count):
                context = quote.get_context(i + 1, context_chars=200)
                resource_id = quote.resource.id.id
                labeled_contexts.append(f"[Resource {resource_id}]: {context}")

    return labeled_contexts


def _build_evaluation_prompt(
    entity_a: "EntityWithQuotes",
    entity_b: "EntityWithQuotes",
    contexts_a: List[str],
    contexts_b: List[str],
    relationship_type: str,
    task_context: str,
) -> str:
    """
    Build prompt for LLM evaluation of entity pair relationship.

    Args:
        entity_a: First entity
        entity_b: Second entity
        contexts_a: Labeled contexts for entity_a
        contexts_b: Labeled contexts for entity_b
        relationship_type: Type of relationship to evaluate
        task_context: Biological/research context

    Returns:
        Formatted prompt string
    """
    contexts_a_str = "\n".join(contexts_a) if contexts_a else "(No contexts available)"
    contexts_b_str = "\n".join(contexts_b) if contexts_b else "(No contexts available)"

    return f"""Evaluate potential {relationship_type} between {entity_a.name} and {entity_b.name} in the context of {task_context}.

Entity A ({entity_a.name}) contexts:
{contexts_a_str}

Entity B ({entity_b.name}) contexts:
{contexts_b_str}

Based on the provided contexts, determine:
1. Does a {relationship_type} exist between {entity_a.name} and {entity_b.name}?
2. If yes, what is your confidence level (high/medium/low)?
3. Provide EXACT VERBATIM quotes from the contexts above that support this relationship.

CRITICAL: Evidence quotes MUST be copied verbatim from the contexts provided above.
Do not paraphrase or modify the text. Use the exact words as they appear.

If no clear {relationship_type} is evident from the contexts, indicate that no relationship exists.
"""
