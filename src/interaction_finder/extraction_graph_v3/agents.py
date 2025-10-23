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
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from .deps import ExtractionDepsV3
from .models import PairEvaluationOut
from ..extraction_graph_v2.models import SimpleEntityListOut, AssessmentOut
from ..extraction_graph_v2.quote_validation import (
    QuoteValidator,
    compute_bag_of_words_similarity,
)
from ..resources import normalize_text_for_matching
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

QUOTE REQUIREMENTS - ABSOLUTELY CRITICAL:
• Quotes must be VERBATIM - copy the meaningful text content exactly as written
• Use the same words, spellings, and structure as the original document
• Copy the text EXACTLY as written, including any unusual phrasing
• Include enough context to be meaningful (typically 1-2 sentences)

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

    agent = Agent(
        model=model,
        output_type=SimpleEntityListOut,
        system_prompt=system_prompt,
        deps_type=ExtractionDepsV3,
        retries=5,
    )

    # Create quote validator for this agent
    quote_validator = QuoteValidator(auto_accept_threshold=85.0)

    # Validator 1: Entity structure validation
    # Must run first to ensure basic entity structure before quote validation
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

    # Validator 2: Quote existence validation
    # Checks that each entity has quotes mentioning the entity name or alias
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

    # Validator 3: Quote validation using alignment system
    # Uses QuoteValidator for high-confidence auto-correction
    @agent.output_validator
    def validate_resourcequote_creation(
        ctx: RunContext[ExtractionDepsV3], out: SimpleEntityListOut
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
                    similarity_threshold=ctx.deps.quote_similarity_threshold,
                )

                # If validation/correction succeeded, update the quote
                if validated_quote and validated_quote.query_text != quote_text:
                    entity.quotes[i] = validated_quote.query_text

        return out

    # Validator 4: Success tracking
    # Marks previous errors as resolved when retries succeed
    @agent.output_validator
    def log_successful_quotes(
        ctx: RunContext[ExtractionDepsV3], out: SimpleEntityListOut
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

    return agent


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

EVIDENCE QUOTE REQUIREMENTS - ABSOLUTELY CRITICAL:
• Quotes must be VERBATIM - copy the meaningful text content exactly as written
• Use the same words, spellings, and structure as the original document
• Copy the text EXACTLY as written, including any unusual phrasing

ELLIPSIS ("...") USAGE - VERY RESTRICTED:
✅ ALLOWED: Only to skip irrelevant words within the SAME SENTENCE
   • "The protein... plays a crucial role" (skipping adjectives in middle)
   • Must be the same sentence, same location, continuous text

❌ NOT ALLOWED: Almost all other uses are wrong
   • Never join different sentences with "..."
   • Never join different paragraphs or sections
   • Never use "..." at the start of quotes
   • If unsure, don't use ellipsis at all - use shorter quotes instead

COMMON ACCURACY ISSUES:

❌ SPLIT QUOTES: Combining non-adjacent text (MOST COMMON ERROR)
   • NEVER combine text that isn't written together in the document
   • Each quote must be EXACTLY as written in one continuous location
   • Use multiple shorter quotes rather than one combined quote

❌ WORD SUBSTITUTIONS: Using synonyms or similar words
   • "It suggests" instead of "This suggests"
   • "The gene" instead of "This gene"

❌ ALTERNATIVE SPELLINGS: Using different spellings
   • "calcination" instead of "calcification"

❌ REFERENCE ARTIFACTS: Including document metadata
   • Reference numbers, citation text, figure references

❌ FABRICATED CONTENT: Never invent quotes that don't exist
   • If you can't find suitable text, use empty evidence array

✅ CORRECT APPROACH: Copy the core text content with the same words, spellings, and meaning structure.

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

    agent = Agent(
        model=model,
        output_type=AssessmentOut,
        system_prompt=system_prompt,
        deps_type=ExtractionDepsV3,
        retries=5,
    )

    # Create quote validator for this agent (same threshold as entity extractor)
    quote_validator = QuoteValidator(auto_accept_threshold=85.0)

    # Validator 1: Evidence quote validation using alignment system
    # Validates all evidence quotes from the assessment
    @agent.output_validator
    def validate_assessment_evidence_quotes(
        ctx: RunContext[ExtractionDepsV3], out: AssessmentOut
    ) -> AssessmentOut:
        """
        Validate assessment evidence quotes using the quote validation system.
        Handles auto-correction for high-confidence fixes.
        """
        # Access resources from the context
        current_resources = ctx.deps.current_resources
        if not current_resources or not out.evidence:
            return out  # Skip validation if no resources or no evidence quotes

        # Validate each evidence quote against available resources
        for i, quote_text in enumerate(out.evidence):
            quote_text = quote_text.strip()

            # Try validation against each resource (assessment can span multiple sources)
            for resource in current_resources:
                validated_quote = quote_validator.validate_and_correct_quote(
                    entity_name="assessment evidence",  # Generic context name
                    entity_kind="evidence",
                    quote_text=quote_text,
                    resource=resource,
                    quote_error_log=ctx.deps.quote_error_log,
                    current_retry=ctx.retry,
                    similarity_threshold=ctx.deps.quote_similarity_threshold,
                )

                # If validation succeeded, update the quote and move to next
                if validated_quote and validated_quote.query_text != quote_text:
                    out.evidence[i] = validated_quote.query_text
                    break
                elif validated_quote:
                    # Quote was valid as-is
                    break

            # If validation failed against all resources, let the validator handle it
            # (it will have already raised ModelRetry if needed)

        return out

    # Validator 2: Success tracking
    # Marks previous errors as resolved when retries succeed
    @agent.output_validator
    def log_successful_assessment_quotes(
        ctx: RunContext[ExtractionDepsV3], out: AssessmentOut
    ) -> AssessmentOut:
        """Log successful assessment evidence quotes to match against previous errors."""
        current_retry = ctx.retry

        # If this is a retry (retry > 0), mark previous errors as resolved
        if current_retry > 0:
            for error_record in ctx.deps.quote_error_log:
                if (
                    error_record.retry_attempt == current_retry
                    and not error_record.resolved
                ):
                    # Check if any evidence quote matches the error using bag-of-words similarity
                    best_match = None
                    best_similarity = 0.0
                    similarity_threshold = 0.3  # Require at least 30% word overlap

                    for quote_text in out.evidence:
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

        return out

    return agent


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

EVIDENCE QUOTE REQUIREMENTS - ABSOLUTELY CRITICAL:
• Quotes must be VERBATIM - copy the meaningful text content exactly as written
• Use the same words, spellings, and structure as the original document
• Copy the text EXACTLY as written, including any unusual phrasing

ELLIPSIS ("...") USAGE - VERY RESTRICTED:
✅ ALLOWED: Only to skip irrelevant words within the SAME SENTENCE
   • "The protein... plays a crucial role" (skipping adjectives in middle)
   • Must be the same sentence, same location, continuous text

❌ NOT ALLOWED: Almost all other uses are wrong
   • Never join different sentences with "..."
   • Never join different paragraphs or sections
   • Never use "..." at the start of quotes
   • If unsure, don't use ellipsis at all - use shorter quotes instead

COMMON ACCURACY ISSUES:

❌ SPLIT QUOTES: Combining non-adjacent text (MOST COMMON ERROR)
   • NEVER combine text that isn't written together in the document
   • Each quote must be EXACTLY as written in one continuous location
   • Use multiple shorter quotes rather than one combined quote

❌ WORD SUBSTITUTIONS: Using synonyms or similar words
   • "It suggests" instead of "This suggests"
   • "The gene" instead of "This gene"

❌ ALTERNATIVE SPELLINGS: Using different spellings
   • "calcination" instead of "calcification"

❌ REFERENCE ARTIFACTS: Including document metadata
   • Reference numbers, citation text, figure references

❌ FABRICATED CONTENT: Never invent quotes that don't exist
   • If you can't find suitable text, use empty evidence array

✅ CORRECT APPROACH: Copy the core text content with the same words, spellings, and meaning structure.

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

    agent = Agent(
        model=model,
        output_type=PairEvaluationOut,
        system_prompt=system_prompt,
        deps_type=ExtractionDepsV3,
        retries=5,
    )

    # Create quote validator for this agent (same threshold as entity extractor)
    quote_validator = QuoteValidator(auto_accept_threshold=85.0)

    # Validator 1: Evidence quote validation using alignment system
    # Validates all evidence quotes from pair evaluation
    @agent.output_validator
    def validate_pair_evidence_quotes(
        ctx: RunContext[ExtractionDepsV3], out: PairEvaluationOut
    ) -> PairEvaluationOut:
        """
        Validate pair evaluation evidence quotes using the quote validation system.
        Handles auto-correction for high-confidence fixes.
        """
        # Access resources from the context
        current_resources = ctx.deps.current_resources
        if not current_resources or not out.evidence:
            return out  # Skip validation if no resources or no evidence quotes

        # Validate each evidence quote against available resources
        for i, quote_text in enumerate(out.evidence):
            quote_text = quote_text.strip()

            # Try validation against each resource (pair evaluation can span multiple sources)
            for resource in current_resources:
                validated_quote = quote_validator.validate_and_correct_quote(
                    entity_name="pair evidence",  # Generic context name
                    entity_kind="evidence",
                    quote_text=quote_text,
                    resource=resource,
                    quote_error_log=ctx.deps.quote_error_log,
                    current_retry=ctx.retry,
                    similarity_threshold=ctx.deps.quote_similarity_threshold,
                )

                # If validation succeeded, update the quote and move to next
                if validated_quote and validated_quote.query_text != quote_text:
                    out.evidence[i] = validated_quote.query_text
                    break
                elif validated_quote:
                    # Quote was valid as-is
                    break

            # If validation failed against all resources, let the validator handle it
            # (it will have already raised ModelRetry if needed)

        return out

    # Validator 2: Success tracking
    # Marks previous errors as resolved when retries succeed
    @agent.output_validator
    def log_successful_pair_quotes(
        ctx: RunContext[ExtractionDepsV3], out: PairEvaluationOut
    ) -> PairEvaluationOut:
        """Log successful pair evaluation evidence quotes to match against previous errors."""
        current_retry = ctx.retry

        # If this is a retry (retry > 0), mark previous errors as resolved
        if current_retry > 0:
            for error_record in ctx.deps.quote_error_log:
                if (
                    error_record.retry_attempt == current_retry
                    and not error_record.resolved
                ):
                    # Check if any evidence quote matches the error using bag-of-words similarity
                    best_match = None
                    best_similarity = 0.0
                    similarity_threshold = 0.3  # Require at least 30% word overlap

                    for quote_text in out.evidence:
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

        return out

    return agent
