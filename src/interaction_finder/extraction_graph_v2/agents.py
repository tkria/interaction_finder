"""
Agent factories for extraction graph V2.

Following pydantic-graph manual patterns:
- Graphs own control flow; agents produce typed data
- Minimal validation with ModelRetry for high-value checks only
- Clean separation between agents (data) and nodes (control flow)
"""

import re
from typing import Type, Union
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models import Model

from .deps import ExtractionDeps
from .models import EntityListOut, AssessmentOut


def mk_agent(
    model: Union[str, Model], output_type: Type[BaseModel], system_prompt: str
) -> Agent[ExtractionDeps, BaseModel]:
    """
    Factory for creating agents with consistent configuration.

    Following pydantic-graph manual pattern exactly.
    """
    return Agent(
        model,
        deps_type=ExtractionDeps,
        output_type=output_type,
        retries=2,  # retry provider/tool errors
        output_retries=2,  # retry JSON validation errors
        system_prompt=system_prompt,
    )


def create_entity_extractor(
    model: Union[str, Model],
    entity_kinds: list[str],
    context: str,
    target_term: str = None,
) -> Agent[ExtractionDeps, EntityListOut]:
    """Create entity extraction agent."""

    kinds_str = ", ".join(entity_kinds)

    # Build target term context for background information only
    target_context = ""
    if target_term:
        target_context = f"\nRESEARCH CONTEXT: The overall research involves {target_term}, but extract ALL entities of the specified types first. Relationship assessment will be done separately."

    system_prompt = f"""You are an expert entity extractor for scientific literature.

Extract ALL {kinds_str} entities found in the provided scientific text.{target_context}

CRITICAL REQUIREMENTS:
1. Extract EVERY entity of the specified types, regardless of apparent relevance
2. Return EXACT text as it appears in the document
3. Do not paraphrase, normalize, or modify entity names
4. Only extract entities that are clearly identifiable
5. For each entity, provide supporting quotes from the documents
6. Include document source for each quote using the exact Resource IDs provided
7. If no entities of the specified types are found, return an empty entities list - do NOT create fake quotes

Context: {context}

OUTPUT FORMAT - REQUIRED STRUCTURE:
{{
    "entities": [
        {{
            "name": "EXACT_ENTITY_NAME",
            "kind": "gene|disease",
            "quotes": [
                {{
                    "text": "exact quote from document containing the entity", 
                    "source": "Resource 1_a1b2c3d4"
                }}
            ]
        }}
    ],
    "entity_kinds": ["{kinds_str}"],
    "reasoning": "Brief explanation of extraction process"
}}

CRITICAL REQUIREMENTS:
- MUST include "entities" array with ALL extracted entities (empty if none found)
- Each entity MUST have "name", "kind", and "quotes" fields
- Each quote MUST have "text" and "source" fields using exact Resource IDs
- Do NOT put entity names in the reasoning field
- Include at least one supporting quote per entity
- NEVER create fake quotes or explanations - if no entities found, return empty list

EXAMPLE:
{{
    "entities": [
        {{
            "name": "BRCA1",
            "kind": "gene", 
            "quotes": [
                {{
                    "text": "BRCA1 is a tumor suppressor gene",
                    "source": "Resource abc123"
                }}
            ]
        }},
        {{
            "name": "breast cancer",
            "kind": "disease",
            "quotes": [
                {{
                    "text": "mutations in BRCA1 are associated with breast cancer", 
                    "source": "Resource 1_a1b2c3d4"
                }}
            ]
        }}
    ],
    "entity_kinds": ["{kinds_str}"],
    "reasoning": "Extracted entities with supporting quotes based on clear identification"
}}

"""

    agent = mk_agent(model, EntityListOut, system_prompt)

    # Enhanced validator to catch common LLM mistakes
    @agent.output_validator
    def validate_entities(out: EntityListOut) -> EntityListOut:
        """Ensure we provide properly structured entities."""
        # Check if entities are empty but reasoning contains entity names
        if not out.entities and out.reasoning:
            # Look for signs that entities were listed in reasoning instead
            reasoning_lower = out.reasoning.lower()
            entity_indicators = [
                "gene",
                "disease",
                "mutation",
                "syndrome",
                "cancer",
                "deficiency",
            ]

            if any(indicator in reasoning_lower for indicator in entity_indicators):
                raise ModelRetry(
                    "Entities appear to be listed in reasoning field instead of entities list. "
                    'Please put each entity in the \'entities\' list as {"name": "EXACT_TEXT", "kind": "gene|disease"}. '
                    "Use reasoning field only to explain your extraction process."
                )

        if not out.entities and not out.reasoning:
            raise ModelRetry(
                "Must provide extracted entities or explain why no entities were found"
            )

        # Basic validation for entity structure
        for entity in out.entities:
            if not entity.name or not entity.kind:
                raise ModelRetry("All entities must have both 'name' and 'kind' fields")

            # Validate entity kind is in expected kinds
            if entity.kind not in entity_kinds:
                raise ModelRetry(
                    f"Entity kind '{entity.kind}' must be one of: {', '.join(entity_kinds)}"
                )

        return out

    # Enhanced validator to check ResourceQuote creation
    @agent.output_validator
    def validate_quotes_exist(out: EntityListOut) -> EntityListOut:
        """
        Basic validation of entity quotes structure and content.
        """
        for entity in out.entities:
            entity_name = entity.name
            entity_quotes = entity.quotes

            if not entity_quotes:
                raise ModelRetry(
                    f"Entity '{entity_name}' has no supporting quotes. "
                    "Please provide exact quotes from the documents."
                )

            for quote_data in entity_quotes:
                quote_text = quote_data.text.strip()
                quote_source = quote_data.source.strip()

                if not quote_text:
                    raise ModelRetry(f"Empty quote text for entity '{entity_name}'")

                # Check for fake explanatory quotes
                fake_quote_indicators = [
                    "no explicit mention",
                    "not found in",
                    "no mention of",
                    "was not found",
                    "not mentioned",
                    "no reference to",
                    "not present in",
                    "no clear",
                    "no specific",
                    "not identified",
                ]
                quote_lower = quote_text.lower()
                if any(indicator in quote_lower for indicator in fake_quote_indicators):
                    raise ModelRetry(
                        f"Fake explanatory quote detected for entity '{entity_name}'. "
                        "Only provide actual quotes from the documents, not explanations. "
                        "If no relevant entities exist, return an empty entities list."
                    )

                # Validate Resource ID format
                if not re.match(r"^Resource [0-9]+_[a-f0-9]{8}", quote_source):
                    raise ModelRetry(
                        f"Invalid source format for entity '{entity_name}'. "
                        f"Source '{quote_source}' does not match expected Resource ID format. "
                        "Use the exact Resource IDs provided in the documents (e.g., 'Resource 1_a1b2c3d4')."
                    )

        return out

    return agent


def create_assessment_agent(
    model: Union[str, Model], entity_kinds: list[str]
) -> Agent[ExtractionDeps, AssessmentOut]:
    """Create individual assessment agent."""

    kinds_str = ", ".join(entity_kinds)
    system_prompt = f"""You are an expert at assessing biological entity relationships.

Your task is to assess one entity's potential for relationships with other {kinds_str} entities.

CRITICAL REQUIREMENTS:
1. For evidence, quote EXACT phrases from the provided contexts
2. Do not paraphrase or summarize evidence text
3. Only claim high/medium potential if you can provide specific evidence
4. Evidence quotes must be verbatim from the contexts

Assess the relationship potential and provide exact supporting quotes."""

    agent = mk_agent(model, AssessmentOut, system_prompt)

    @agent.output_validator
    def validate_evidence(out: AssessmentOut) -> AssessmentOut:
        """Ensure high/medium potential has evidence."""
        if out.potential in ["high", "medium"] and not out.evidence:
            raise ModelRetry(
                "High or medium relationship potential requires specific evidence quotes"
            )
        return out

    return agent
