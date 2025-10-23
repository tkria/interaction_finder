"""
Entity extraction from individual documents.

Provides pure functional interface for extracting named entities from scientific
documents with complete provenance tracking via ResourceQuotes.
"""

import logging
from typing import List, Union
from pydantic_ai import Agent
from pydantic_ai.models import Model

# Import with TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction_graph_v2.models import EntityWithQuotes, SimpleEntityListOut
    from ..resources import Resource, ResourceQuote, QuoteValidationError

# Configure logger
logger = logging.getLogger(__name__)


def _build_extraction_prompt(
    entity_kinds: List[str], task_context: str, resource_text: str
) -> str:
    """
    Build extraction prompt for LLM.

    Truncates document if too long to avoid token limits.
    """
    kinds_str = ", ".join(entity_kinds)

    # Truncate document if too long (keep first 8000 chars)
    max_doc_length = 8000
    if len(resource_text) > max_doc_length:
        doc_text = resource_text[:max_doc_length] + "\n\n[Document truncated...]"
    else:
        doc_text = resource_text

    prompt = f"""Extract all {kinds_str} entities from the following scientific document.

RESEARCH CONTEXT: {task_context}

CRITICAL REQUIREMENTS:
1. Extract EVERY entity of the specified types ({kinds_str})
2. For each entity, provide EXACT quotes from the document (verbatim text)
3. Quotes must be copied EXACTLY as written - do NOT paraphrase or modify
4. Include entity aliases if present (e.g., gene symbol "BRCA1" vs "breast cancer 1")
5. If no entities found, return empty list

DOCUMENT:
{doc_text}

Extract entities with supporting quotes."""

    return prompt


async def extract_from_resource(
    resource: "Resource",
    entity_kinds: List[str],
    model: Union[Model, str],
    task_context: str,
    similarity_threshold: float = 0.90,
) -> List["EntityWithQuotes"]:
    """
    Extract entities of specified kinds from a single resource with quote validation.

    Processes a document to identify and extract biological entities (genes, proteins,
    diseases, etc.) with complete provenance tracking. Each extracted entity includes
    all its occurrences with exact character positions via ResourceQuotes.

    The function performs two-stage processing:
    1. LLM extraction: Identifies entities and provides quote strings
    2. Quote validation: Validates each quote against resource using fuzzy matching

    Only entities with at least one successfully validated quote are returned.
    Failed quote validations are logged but do not stop processing.

    Args:
        resource: Document resource to extract from (with content in markdown format)
        entity_kinds: List of entity types to extract (e.g., ["gene", "disease"])
        model: LLM model for extraction (pydantic-ai Model or string like "openai:gpt-4o")
        task_context: Biological context to guide extraction (e.g., "breast cancer research")
        similarity_threshold: Minimum similarity for fuzzy quote matching (default: 0.90)

    Returns:
        List of EntityWithQuotes, each with validated ResourceQuotes

    Example:
        ```python
        entities = await extract_from_resource(
            resource=doc,
            entity_kinds=["gene", "protein"],
            model="openai:gpt-4o",
            task_context="cancer signaling pathways",
            similarity_threshold=0.90
        )
        for entity in entities:
            print(f"{entity.name} ({entity.kind}): {entity.total_occurrences} occurrences")
        ```

    Note:
        This function handles all errors internally. LLM failures return empty list.
        Quote validation failures are logged and those quotes are skipped.
    """
    # Import here to avoid circular dependencies
    from ..extraction_graph_v2.models import SimpleEntityListOut, EntityWithQuotes
    from ..resources import (
        QuoteValidationError,
        QuoteNotFoundError,
        QuoteNearMatchError,
        ParaphraseError,
    )

    # Create agent for entity extraction
    system_prompt = f"""You are an expert entity extractor for scientific literature.

Extract entities of types: {", ".join(entity_kinds)}

CRITICAL REQUIREMENTS:
1. Return EXACT entity names as they appear in the document
2. Provide VERBATIM quotes from the document (copy text exactly)
3. Do NOT paraphrase, normalize, or modify quotes
4. Each quote must be continuous text from one location
5. Include entity aliases if they appear in the document
6. If no entities found, return empty entities list

OUTPUT FORMAT:
{{
    "entities": [
        {{
            "name": "EXACT_ENTITY_NAME",
            "kind": "one of: {", ".join(entity_kinds)}",
            "aliases": ["alternative_name1"],
            "quotes": ["exact verbatim quote from document"]
        }}
    ],
    "entity_kinds": {entity_kinds},
    "reasoning": "Brief extraction summary"
}}"""

    agent: Agent[None, "SimpleEntityListOut"] = Agent(
        model=model,
        output_type=SimpleEntityListOut,
        system_prompt=system_prompt,
    )

    # Build prompt with document content
    prompt = _build_extraction_prompt(entity_kinds, task_context, resource.text)

    # Call LLM to extract entities
    try:
        result = await agent.run(prompt)
        llm_output: "SimpleEntityListOut" = result.output
    except Exception as e:
        logger.warning(
            f"LLM extraction failed for resource {resource.id.id}: {type(e).__name__}: {e}"
        )
        return []

    # If no entities extracted, return early
    if not llm_output.entities:
        logger.debug(f"No entities extracted from resource {resource.id.id}")
        return []

    # Validate quotes and build EntityWithQuotes list
    validated_entities: List["EntityWithQuotes"] = []

    for entity in llm_output.entities:
        validated_quotes: List["ResourceQuote"] = []

        # Validate each quote string for this entity
        for idx, quote_text in enumerate(entity.quotes, start=1):
            logger.info(
                "Validating quote %s for entity '%s' in resource %s",
                idx,
                entity.name,
                resource.id.id,
            )
            try:
                # Validate quote against resource using fuzzy matching
                validated_quote = resource.quote(
                    quote_text.strip(), similarity_threshold=similarity_threshold
                )
                validated_quotes.append(validated_quote)

            except QuoteNotFoundError as e:
                # Quote not found at all
                logger.warning(
                    f"Quote validation failed for entity '{entity.name}' ({entity.kind}) "
                    f"in resource {resource.id.id}: Quote not found"
                )

            except ParaphraseError as e:
                # Quote is paraphrased (low similarity)
                logger.warning(
                    f"Quote validation failed for entity '{entity.name}' ({entity.kind}) "
                    f"in resource {resource.id.id}: Paraphrased quote "
                    f"(similarity: {e.similarity:.2f})"
                )

            except QuoteNearMatchError as e:
                # Quote has near match with suggestions (word substitutions, etc.)
                logger.warning(
                    f"Quote validation failed for entity '{entity.name}' ({entity.kind}) "
                    f"in resource {resource.id.id}: {type(e).__name__} "
                    f"(similarity: {e.similarity:.2f})"
                )

            except QuoteValidationError as e:
                # Generic validation error
                logger.warning(
                    f"Quote validation failed for entity '{entity.name}' ({entity.kind}) "
                    f"in resource {resource.id.id}: {type(e).__name__}"
                )

        # Only include entity if it has at least one valid quote
        if validated_quotes:
            entity_with_quotes = EntityWithQuotes(
                name=entity.name,
                kind=entity.kind,
                aliases=entity.aliases,
                quotes=validated_quotes,
            )
            validated_entities.append(entity_with_quotes)
        else:
            # Log aggregate warning when all quotes fail
            logger.warning(
                f"Entity '{entity.name}' ({entity.kind}) has no valid quotes "
                f"in resource {resource.id.id}, skipping"
            )

    logger.info(
        f"Extracted {len(validated_entities)}/{len(llm_output.entities)} entities "
        f"with valid quotes from resource {resource.id.id}"
    )

    return validated_entities
