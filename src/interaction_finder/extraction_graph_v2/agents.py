"""
Agent factories for extraction graph V2.

Following pydantic-graph manual patterns:
- Graphs own control flow; agents produce typed data
- Minimal validation with ModelRetry for high-value checks only
- Clean separation between agents (data) and nodes (control flow)
"""

from typing import Type, Union, List, Optional, TYPE_CHECKING
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from .deps import ExtractionDeps
from .models import EntityListOut, SimpleEntityListOut, AssessmentOut
from .quote_validation import QuoteValidator, compute_bag_of_words_similarity
from ..resources import normalize_text_for_matching, expand_scientific_shorthand

if TYPE_CHECKING:
    from ..resources import Resource


def mk_agent(
    model: Union[str, Model], output_type: Type[BaseModel], system_prompt: str
) -> Agent[ExtractionDeps, BaseModel]:
    """
    Factory for creating agents with consistent configuration.

    Following pydantic-graph manual pattern exactly.
    """
    return Agent(
        model=model,
        result_type=output_type,
        system_prompt=system_prompt,
        deps_type=ExtractionDeps,
    )


def create_entity_extractor(
    model: Union[str, Model],
    entity_kinds: List[str],
    target_term: Optional[str] = None,
    context: str = "",
) -> Agent[ExtractionDeps, SimpleEntityListOut]:
    """Create entity extraction agent with improved error guidance."""

    kinds_str = ", ".join(entity_kinds)

    # Add target term context if provided
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
5. For each entity, provide supporting quotes from the document
6. If no entities of the specified types are found, return an empty entities list - do NOT create fake quotes
7. If an entity has alternative names (e.g., gene symbol "FBLN4" vs full name "fibulin-4"), include them in the 'aliases' field

QUOTE REQUIREMENTS - ABSOLUTELY CRITICAL:
• Quotes must be VERBATIM - copy the meaningful text content exactly as written
• Use the same words, spellings, and structure as the original document
• Copy the text EXACTLY as written, including any unusual phrasing

ELLIPSIS ("...") USAGE - VERY RESTRICTED:
✅ ALLOWED: Only to skip irrelevant words within the SAME SENTENCE
   • "The protein... plays a crucial role" (skipping adjectives in middle)
   • Must be the same sentence, same location, continuous text

❌ NOT ALLOWED: Almost all other uses are wrong
   • Never join different sentences: "Gene X... causes disease Y" 
   • Never join different paragraphs or sections
   • Never use "..." at the start of quotes
   • If unsure, don't use ellipsis at all - use shorter quotes instead

COMMON ACCURACY ISSUES - UNDERSTAND THE PATTERNS:

❌ SPLIT QUOTES: Combining non-adjacent text (MOST COMMON ERROR - CRITICAL TO AVOID)
   • NEVER EVER combine text that isn't written together in the document
   • If you can't find the complete sentence as written, use a shorter quote
   • DO NOT try to "reconstruct" or "complete" sentences by joining parts
   • DO NOT join "Gene X" from one location with "causes disease Y" from another
   • Each quote must be EXACTLY as written in one continuous location
   • When in doubt, use multiple shorter quotes rather than one combined quote

❌ WORD SUBSTITUTIONS: Using synonyms or similar words
   • "It suggests" instead of "This suggests" 
   • "Consistent with" instead of "In agreement with"
   • "The gene" instead of "This gene"

❌ ALTERNATIVE SPELLINGS: Using different spellings
   • "calcination" instead of "calcification"
   • "immuniprecipitated" instead of "coimmunoprecipitated"

❌ REFERENCE ARTIFACTS: Including document metadata
   • Reference numbers: "protein22" instead of "protein"
   • Citation text: "PubMed Google Scholar" appearing in quotes
   • Figure references when not grammatically integrated

❌ FABRICATED CONTENT: Never invent quotes that don't exist in the document
   • Don't create table-like or structured text that isn't there
   • Don't use "..." as an actual quote - find real text or return empty list
   • If you can't find suitable text, use empty quotes array

❌ INAPPROPRIATE QUOTE SCOPE: Choose meaningful quote boundaries  
   • Don't include entire paragraphs when a sentence captures the entity mention
   • Don't make quotes so narrow they lose essential context
   • Include enough surrounding text to be meaningful but stay focused

✅ CORRECT APPROACH: Copy the core text content with the same words, spellings, and meaning structure.

Context: {context}

OUTPUT FORMAT - REQUIRED STRUCTURE:
{{
    "entities": [
        {{
            "name": "EXACT_ENTITY_NAME",
            "kind": "one of: {kinds_str}",
            "aliases": ["alternative_name1", "alternative_name2"],
            "quotes": [
                "exact quote from document containing the entity",
                "another quote from the same document"
            ]
        }}
    ],
    "entity_kinds": ["{kinds_str}"],
    "reasoning": "Brief explanation of extraction process"
}}

CRITICAL REQUIREMENTS:
- MUST include "entities" array with ALL extracted entities (empty if none found)
- Each entity MUST have "name", "kind", and "quotes" fields
- Each quote MUST be exact text from the document
- Do NOT put entity names in the reasoning field
- Include at least one supporting quote per entity
- NEVER create fake quotes or explanations - if no entities found, return empty list

EXAMPLE (domain-agnostic placeholders):
{{
    "entities": [
        {{
            "name": "ENTITY_A",
            "kind": "KIND_1",
            "aliases": ["ALIAS_A"],
            "quotes": [
                "ENTITY_A appears verbatim in this sentence",
                "ENTITY_A is also mentioned here"
            ]
        }},
        {{
            "name": "ENTITY_B",
            "kind": "KIND_2",
            "aliases": ["ALIAS_B"],
            "quotes": [
                "ENTITY_B is discussed in this context",
                "Another mention of ENTITY_B in this document"
            ]
        }}
    ],
    "entity_kinds": ["{kinds_str}"],
    "reasoning": "Extracted entities with supporting quotes from this single document"
}}

"""

    agent = mk_agent(model, SimpleEntityListOut, system_prompt)

    # Create quote validator for this agent
    quote_validator = QuoteValidator(auto_accept_threshold=85.0)

    # Enhanced output validator using the new quote validation system
    @agent.output_validator
    def validate_resourcequote_creation(
        ctx: RunContext[ExtractionDeps], out: SimpleEntityListOut
    ) -> SimpleEntityListOut:
        """
        Validate ResourceQuote creation using the new quote validation system.
        Handles auto-correction for high-confidence fixes.
        """
        # Access resources from the context
        current_resources = ctx.deps.current_resources
        if not current_resources:
            return out  # Skip validation if no resources available

        # Since we're processing one document at a time, use the first (only) resource
        resource = current_resources[0] if current_resources else None
        if not resource:
            return out

        # Validate quotes for each entity
        for entity in out.entities:
            entity_name = entity.name
            entity_kind = entity.kind

            # Validate each quote for this entity
            for i, quote_text in enumerate(entity.quotes):
                quote_text = quote_text.strip()

                # Use the new validation system
                validated_quote = quote_validator.validate_and_correct_quote(
                    entity_name=entity_name,
                    entity_kind=entity_kind,
                    quote_text=quote_text,
                    resource=resource,
                    quote_error_log=ctx.deps.quote_error_log,
                    current_retry=ctx.retry,
                )

                # If validation/correction succeeded, update the quote
                if validated_quote and validated_quote.query_text != quote_text:
                    entity.quotes[i] = validated_quote.query_text

        return out

    # Success logger to track resolved quote errors
    @agent.output_validator
    def log_successful_quotes(
        ctx: RunContext[ExtractionDeps], out: SimpleEntityListOut
    ) -> SimpleEntityListOut:
        """Log successful quotes to match against previous errors using bag-of-words similarity."""
        current_retry = ctx.retry

        # If this is a retry (retry > 0), mark previous errors as resolved
        if current_retry > 0:
            for error_record in ctx.deps.quote_error_log:
                if (
                    error_record.retry_attempt == current_retry
                    and not error_record.resolved
                ):
                    # Find entity in current successful output
                    for entity in out.entities:
                        if entity.name == error_record.entity_name:
                            # Find best matching quote using bag-of-words similarity
                            best_match = None
                            best_similarity = 0.0
                            similarity_threshold = (
                                0.3  # Require at least 30% word overlap
                            )

                            for quote_text in entity.quotes:
                                similarity = compute_bag_of_words_similarity(
                                    error_record.original_quote, quote_text
                                )
                                if (
                                    similarity > best_similarity
                                    and similarity >= similarity_threshold
                                ):
                                    best_similarity = similarity
                                    best_match = quote_text.strip()

                            # Mark as resolved if we found a good match
                            if best_match:
                                error_record.final_accepted_quote = best_match
                                error_record.resolved = True
                            break

        return out

    # Enhanced validator to catch common LLM mistakes
    @agent.output_validator
    def validate_entities(out: SimpleEntityListOut) -> SimpleEntityListOut:
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

    # Enhanced validator to check that entity quotes contain entity names
    @agent.output_validator
    def validate_quotes_exist(out: SimpleEntityListOut) -> SimpleEntityListOut:
        """
        Basic validation of entity quotes structure and content.
        """
        for entity in out.entities:
            entity_name = entity.name
            entity_quotes = entity.quotes

            if not entity_quotes:
                raise ModelRetry(
                    f"Entity '{entity_name}' has no supporting quotes. "
                    "Please provide at least one quote that mentions this entity."
                )

            # Check that at least one quote contains the entity name or alias
            entity_found_in_quotes = False

            for quote_text in entity_quotes:
                normalized_quote = normalize_text_for_matching(quote_text)
                normalized_entity_name = normalize_text_for_matching(entity_name)

                # Check if entity name appears in quote
                if normalized_entity_name in normalized_quote:
                    entity_found_in_quotes = True
                    break

                # Check aliases if provided
                if entity.aliases:
                    for alias in entity.aliases:
                        normalized_alias = normalize_text_for_matching(alias)
                        if normalized_alias in normalized_quote:
                            entity_found_in_quotes = True
                            break
                    if entity_found_in_quotes:
                        break

            # Fail if NO quotes contain the entity name or aliases
            if not entity_found_in_quotes:
                alias_info = ""
                if entity.aliases:
                    alias_info = f" or aliases {entity.aliases}"
                raise ModelRetry(
                    f"No quotes for entity '{entity_name}' contain the entity name{alias_info}. "
                    f"At least one quote must explicitly mention the entity being described. "
                    "Other quotes can use implicit references like 'the protein' or 'mutant cells'. "
                    "If the entity has alternative names, include them in the 'aliases' field."
                )

        return out

    return agent


def create_assessment_agent(
    model: Union[str, Model], entity_kinds: List[str]
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
