"""
Pure functional direct extraction: documents → extracted, cited associations.

This module provides a collection of pure functions that transform scientific documents
into extracted entity associations with verified citations:

Main Pipeline:
documents → extract_entities → validate_citations → correct_invalid → create_associations

Key Functions:
- extract_cited_associations(): Main entry point for the complete pipeline
- extract_entities_from_documents(): Bulk entity extraction with quotes
- validate_entity_citations(): Pure quote validation against sources
- create_entity_associations(): Generate associations between entities with evidence

Design Principles:
- Pure functions: no side effects, same input → same output
- Composable: each function does one thing well
- Testable: no hidden dependencies or state
- Efficient: hybrid approach minimizes LLM calls
"""

from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass
import itertools
from pydantic import BaseModel, Field
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai import ToolDefinition

from .models import EntityOut, EntityListOut, EntityPairOut, EntityWithQuotes
from .quote_validation import find_longest_matching_subquote
from ..resources import ResourceQuote, Resource

# No artificial limits - trust LLM capabilities


@dataclass
class ExtractionMetrics:
    """Metrics from the extraction pipeline."""

    total_entities: int
    valid_quotes: int
    invalid_quotes: int
    corrections_made: int
    correction_rounds: int = 0
    quotes_discarded: int = 0
    associations_created: int = 0


@dataclass
class InvalidQuote:
    """Information about a quote that failed validation."""

    entity_name: str
    quote_text: str
    resource_id: str
    error_message: str
    entity_kind: str
    diagnostics: Optional[str] = None
    suggested_correction: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of quote validation process."""

    valid_entities: List[EntityOut]
    invalid_quotes: List[InvalidQuote]
    total_quotes: int
    valid_quotes: int


# Helper functions for config-like behavior
def get_valid_entity_pairs(entity_kinds: List[str]) -> List[Tuple[str, str]]:
    """Generate valid entity pairs (all combinations except same type)."""
    return [(a, b) for a in entity_kinds for b in entity_kinds if a != b]


def get_relationship_mapping(
    entity_kinds: List[str], relationship_type: str = "interaction"
) -> Dict[Tuple[str, str], str]:
    """Generate simple relationship mapping."""
    mapping = {}
    for pair in get_valid_entity_pairs(entity_kinds):
        # Simple mapping rules
        if "gene" in pair and "disease" in pair:
            mapping[pair] = "associated_with"
        elif "protein" in pair and "ligand" in pair:
            mapping[pair] = "interacts_with"
        elif "drug" in pair and "disease" in pair:
            mapping[pair] = "treats"
        else:
            mapping[pair] = relationship_type
    return mapping


class QuoteCorrectRequest(BaseModel):
    """Request model for targeted quote correction."""

    entity_name: str = Field(description="Name of the entity")
    entity_kind: str = Field(description="Kind of entity (gene, disease, etc.)")
    invalid_quote: str = Field(description="The quote that failed validation")
    resource_id: str = Field(description="ID of the resource to quote from")
    error_details: str = Field(description="What went wrong with the quote")
    suggestion: Optional[str] = Field(
        description="Suggested correct quote", default=None
    )


class QuoteCorrectionOut(BaseModel):
    """Output model for quote correction."""

    entity_name: str = Field(description="Name of the entity")
    corrected_quote: str = Field(
        description="The corrected, verbatim quote from the resource"
    )
    reasoning: str = Field(description="Brief explanation of the correction made")


class QuoteCorrectionsListOut(BaseModel):
    """Output model for multiple quote corrections."""

    corrections: List[QuoteCorrectionOut] = Field(
        description="List of quote corrections"
    )
    reasoning: str = Field(description="Overall explanation of the corrections made")


# ============================================================================
# MAIN PURE FUNCTIONAL PIPELINE
# ============================================================================


async def extract_cited_entities(
    documents: List[Resource],
    model: Model,
    entity_kinds: List[str],
) -> Tuple[List[EntityWithQuotes], ExtractionMetrics]:
    """
    Main pure function: Documents → Extracted, Validated Entities

    This is the primary entry point that orchestrates the entire extraction
    and validation pipeline, returning entities with full provenance.

    Args:
        documents: List of source documents to extract from
        model: LLM model for extraction and correction
        entity_kinds: Types of entities to extract

    Returns:
        Tuple of (entities with quotes, metrics) from the extraction pipeline
    """

    # Step 1: Extract entities with quotes
    entities = await extract_entities_from_documents(documents, model, entity_kinds)

    # Step 2: Validate all citations
    validation_result = validate_entity_citations(entities, documents)

    # Step 3: Iterative correction with up to 3 rounds
    corrected_entities = []
    remaining_invalid_quotes = validation_result.invalid_quotes
    correction_round = 0
    max_correction_rounds = 3

    while remaining_invalid_quotes and correction_round < max_correction_rounds:
        correction_round += 1
        print(
            f"🔄 Correction round {correction_round}: attempting to fix {len(remaining_invalid_quotes)} invalid quotes"
        )

        # Attempt corrections for this round
        round_corrected_entities = await correct_invalid_citations(
            remaining_invalid_quotes, documents, model
        )

        if not round_corrected_entities:
            print(f"❌ Round {correction_round}: No corrections produced")
            break

        # Validate the corrected entities to see if corrections worked
        round_validation = validate_entity_citations(
            round_corrected_entities, documents
        )

        # Add successfully corrected entities to our collection
        corrected_entities.extend(round_validation.valid_entities)

        # Update remaining invalid quotes for next round (if any)
        remaining_invalid_quotes = round_validation.invalid_quotes

        print(
            f"✅ Round {correction_round}: {len(round_validation.valid_entities)} entities corrected successfully"
        )
        if remaining_invalid_quotes:
            print(
                f"⚠️  Round {correction_round}: {len(remaining_invalid_quotes)} quotes still invalid"
            )

    # Final cleanup: discard any quotes that couldn't be corrected after max rounds
    if remaining_invalid_quotes:
        discarded_count = len(remaining_invalid_quotes)
        print(
            f"🗑️  Discarding {discarded_count} quotes that couldn't be corrected after {max_correction_rounds} rounds"
        )

        # Log the discarded quotes for debugging
        for invalid_quote in remaining_invalid_quotes:
            print(
                f"   - Discarded: {invalid_quote.entity_name} / {invalid_quote.quote_text[:50]}..."
            )

    print(
        f"🎯 Correction complete: {len(corrected_entities)} entities successfully corrected"
    )

    # Step 4: Merge valid and corrected entities, combining quotes for same entity
    all_valid_entities = merge_entity_quotes(
        validation_result.valid_entities, corrected_entities
    )

    # Step 5: Convert to EntityWithQuotes with full ResourceQuote provenance
    doc_index = {doc.id.id: doc for doc in documents}
    entities_with_quotes = [
        convert_to_entity_with_quotes(entity, doc_index)
        for entity in all_valid_entities
    ]

    # Step 6: Calculate metrics
    metrics = ExtractionMetrics(
        total_entities=len(entities),
        valid_quotes=validation_result.valid_quotes,
        invalid_quotes=len(validation_result.invalid_quotes),
        corrections_made=len(corrected_entities),
        correction_rounds=correction_round,
        quotes_discarded=len(remaining_invalid_quotes)
        if remaining_invalid_quotes
        else 0,
        associations_created=0,  # No longer creating associations here
    )

    return entities_with_quotes, metrics


async def extract_cited_associations(
    documents: List[Resource],
    model: Model,
    entity_kinds: List[str],
    relationship_type: str = "interaction",
) -> Tuple[List[EntityPairOut], ExtractionMetrics]:
    """
    Convenience function: Documents → Extracted Entity Pairs

    Combines entity extraction with association creation for backward compatibility.

    Args:
        documents: List of source documents to extract from
        model: LLM model for extraction and correction
        entity_kinds: Types of entities to extract
        relationship_type: Type of relationship to create between entities

    Returns:
        Tuple of (entity pairs, metrics) from the extraction pipeline
    """
    # Extract entities first
    entities, metrics = await extract_cited_entities(documents, model, entity_kinds)

    # Create associations from entities
    associations = create_entity_associations(entities, entity_kinds, relationship_type)

    # Update metrics with association count
    metrics.associations_created = len(associations)

    return associations, metrics


# ============================================================================
# PURE EXTRACTION FUNCTIONS
# ============================================================================


async def extract_entities_from_single_resource(
    resource: Resource,
    model: Model,
    entity_kinds: List[str],
) -> List[EntityOut]:
    """
    Extract entities from a single resource - eliminates attribution confusion.

    Args:
        resource: Single resource to extract from
        model: LLM model for extraction
        entity_kinds: Types of entities to extract

    Returns:
        List of extracted entities with quotes (all from this resource)
    """
    system_prompt = f"""You are extracting {" and ".join(entity_kinds)} entities from a scientific document.

**CRITICAL QUOTING REQUIREMENTS:**
1. ALL quotes must be VERBATIM - exactly as they appear in this document
2. Do NOT paraphrase, rephrase, or modify any text
3. Do NOT use synonyms (e.g., "Consistent with" vs "In agreement with")
4. Include surrounding context when helpful for understanding
5. Each quote must be from exactly this document

**EXAMPLES OF CORRECT QUOTING:**
❌ Wrong: "Consistent with previous results, BRCA1 mutations cause cancer"
✅ Right: "In agreement with previous results we detected two predominant proteins"

❌ Wrong: "The study shows BRCA1 is important for DNA repair"  
✅ Right: "BRCA1 is a tumor suppressor gene that plays a critical role in DNA repair"

Extract ALL {" and ".join(entity_kinds)} entities you can find from this document.
For each entity:
1. Find the exact name as it appears in text
2. Provide verbatim quotes that mention the entity  
3. Include aliases/alternative names if found

Document Title: {resource.title}
Document Content:
{resource.text}"""

    user_prompt = f"Extract all {' and '.join(entity_kinds)} entities with their verbatim supporting quotes from this document."

    # Create simplified tool definition for single resource
    extraction_tool = ToolDefinition(
        name="extract_entities",
        description="Extract entities from a single scientific document with exact quotes",
        parameters_json_schema=EntityListOut.model_json_schema(),
    )

    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(system_prompt),
                UserPromptPart(
                    user_prompt
                    + "\n\nCall the extract_entities tool with the extracted entities."
                ),
            ]
        )
    ]

    # Try the extraction with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await model_request(
                model,
                messages,
                model_request_parameters=ModelRequestParameters(
                    function_tools=[extraction_tool],
                    allow_text_output=False,
                ),
            )

            # Get the tool call result
            tool_part = next(
                p
                for p in response.parts
                if getattr(p, "tool_name", None) == "extract_entities"
            )

            # Try to parse the result
            entity_list = EntityListOut(**tool_part.args)

            # Validate similar to the agent approach
            if not entity_list.entities and not entity_list.reasoning:
                error_msg = "No entities found and no reasoning provided. Please explain why no entities were found."
            else:
                # Validate entity structure
                for entity in entity_list.entities:
                    if not entity.name or not entity.kind:
                        error_msg = (
                            "All entities must have both name and kind specified."
                        )
                        break
                    if entity.kind not in entity_kinds:
                        error_msg = f"Entity kind '{entity.kind}' not in expected kinds {entity_kinds}"
                        break
                else:
                    # All validations passed - ensure quotes are attributed to the current resource
                    for entity in entity_list.entities:
                        # Since this is single-resource extraction, all quotes should be from this resource
                        all_quotes = []
                        # Collect all quotes from whatever structure the LLM provided
                        for quote_list in entity.quotes.values():
                            all_quotes.extend(quote_list)
                        # If no quotes found in the dict structure, keep original
                        if not all_quotes and isinstance(entity.quotes, dict):
                            # Extract from any key in the dict or use empty list
                            for quote_list in entity.quotes.values():
                                if quote_list:
                                    all_quotes.extend(quote_list)

                        # Assign all quotes to the current resource ID
                        entity.quotes = {resource.id.id: all_quotes}
                    return entity_list.entities

        except Exception as e:
            # Parse the validation error to provide helpful feedback
            error_msg = _create_helpful_error_message(e, entity_kinds)

        # If we're here, there was an error - retry with feedback
        if attempt < max_retries - 1:
            retry_prompt = f"""{error_msg}

Call the extract_entities tool again with the corrections."""
            messages.append(ModelRequest(parts=[UserPromptPart(retry_prompt)]))
        else:
            # Final attempt failed
            raise ValueError(
                f"Entity extraction failed after {max_retries} attempts: {error_msg}"
            )

    # If we get here, all attempts failed
    raise ValueError("Model did not call the expected extraction tool")


async def extract_entities_from_documents(
    documents: List[Resource],
    model: Model,
    entity_kinds: List[str],
) -> List[EntityOut]:
    """
    Extract entities from multiple documents by processing each individually.

    This eliminates attribution confusion by processing one resource at a time,
    then aggregating results.

    Args:
        documents: Source documents to extract from
        model: LLM model for extraction
        entity_kinds: Types of entities to extract

    Returns:
        List of extracted entities with quotes (aggregated from all resources)
    """
    all_entities = []

    # Process each resource individually to avoid attribution confusion
    for resource in documents:
        print(f"🔍 Extracting from resource: {resource.id.id}")

        # Extract entities from this single resource
        resource_entities = await extract_entities_from_single_resource(
            resource, model, entity_kinds
        )

        print(f"✅ Found {len(resource_entities)} entities in {resource.id.id}")
        all_entities.extend(resource_entities)

    # Aggregate entities with the same name/kind across resources
    aggregated_entities = aggregate_entities_across_resources(all_entities)

    print(f"🎯 Total: {len(aggregated_entities)} unique entities across all resources")
    return aggregated_entities


def aggregate_entities_across_resources(entities: List[EntityOut]) -> List[EntityOut]:
    """
    Aggregate entities with the same name/kind across different resources.

    Combines quotes from different resources for entities with the same identity.

    Args:
        entities: List of entities extracted from individual resources

    Returns:
        List of aggregated entities with quotes merged across resources
    """
    # Group entities by (name, kind) to identify same entities across resources
    entity_groups = {}

    for entity in entities:
        key = (entity.name.lower().strip(), entity.kind)
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(entity)

    aggregated = []
    for (name, kind), group in entity_groups.items():
        if not group:
            continue

        # Use the first entity as base and merge quotes from others
        base_entity = group[0]
        merged_quotes = dict(base_entity.quotes)
        merged_aliases = set(base_entity.aliases)

        # Merge quotes and aliases from all entities in group
        for entity in group[1:]:
            merged_aliases.update(entity.aliases)

            for resource_id, quote_list in entity.quotes.items():
                if resource_id in merged_quotes:
                    # Remove duplicates while preserving order
                    existing_quotes = set(merged_quotes[resource_id])
                    new_quotes = [q for q in quote_list if q not in existing_quotes]
                    merged_quotes[resource_id].extend(new_quotes)
                else:
                    merged_quotes[resource_id] = quote_list[:]

        # Create aggregated entity
        aggregated_entity = EntityOut(
            name=base_entity.name,  # Keep original casing from first occurrence
            kind=kind,
            aliases=sorted(list(merged_aliases)),
            quotes=merged_quotes,
        )
        aggregated.append(aggregated_entity)

    return aggregated


def _create_helpful_error_message(error: Exception, entity_kinds: List[str]) -> str:
    """
    Convert a pydantic validation error into specific, actionable guidance for the LLM.

    This follows the same pattern as ModelRetry messages in the agent-based approach,
    providing very specific instructions about what needs to be fixed.

    Args:
        error: The validation exception
        entity_kinds: Expected entity kinds for validation

    Returns:
        Specific, actionable error message with precise instructions
    """
    error_str = str(error)

    # Parse the validation error details for very specific guidance
    import re
    import json

    # Mirror ModelRetry message style - concise, direct, surgical corrections

    # Handle string where object expected (most common issue from user example)
    if "Input should be an object" in error_str:
        input_match = re.search(r'"input":\s*"([^"]*)"', error_str)
        if input_match and "Resource" in input_match.group(1):
            return "Replace string values in entities list with entity objects."
        return "All entities must be objects, not strings."

    # Missing entity_kinds field
    if "Field required" in error_str and "entity_kinds" in error_str:
        return f"Must include entity_kinds field: {json.dumps(entity_kinds)}"

    # Missing required fields in entities
    if "Field required" in error_str:
        if '"name"' in error_str:
            return "All entities must have both name and kind specified."
        elif '"kind"' in error_str:
            return "All entities must have both name and kind specified."
        elif '"aliases"' in error_str:
            return "All entities must have aliases field (use empty list [] if none)."
        elif '"quotes"' in error_str:
            return "All entities must have quotes field (use empty dict {} if none)."
        return "All entities must have both name and kind specified."

    # Type validation errors
    if "type=model_type" in error_str or "type=string_type" in error_str:
        if '"name"' in error_str:
            return "Entity names must be strings."
        elif '"kind"' in error_str:
            return f"Entity kinds must be strings from: {', '.join(entity_kinds)}"
        elif '"aliases"' in error_str:
            return "Entity aliases must be lists of strings."
        elif '"quotes"' in error_str:
            return "Entity quotes must be dictionaries."
        return "Check entity field types: name (string), kind (string), aliases (list), quotes (dict)."

    # Invalid entity kind
    if any(kind in error_str for kind in ["not in searched kinds", "not one of"]):
        return f"Entity kind must be one of: {', '.join(entity_kinds)}"

    # Fallback - keep it simple like ModelRetry
    return "All entities must have name, kind, aliases, and quotes fields with correct types."


def validate_entity_citations(
    entities: List[EntityOut], documents: List[Resource]
) -> ValidationResult:
    """
    Pure function: validate all entity quotes against source documents.

    Args:
        entities: List of extracted entities with quotes
        documents: Source documents for validation

    Returns:
        ValidationResult with valid entities and invalid quote details
    """
    valid_entities = []
    invalid_quotes = []
    total_quotes = 0
    valid_quotes = 0

    # Create document index for efficient lookup
    doc_index = {doc.id.id: doc for doc in documents}

    for entity in entities:
        valid_entity_quotes = {}
        entity_has_valid_quotes = False

        for resource_id, quote_list in entity.quotes.items():
            valid_resource_quotes = []

            for quote_text in quote_list:
                total_quotes += 1
                invalid_quote = validate_quote(
                    entity.name, quote_text, resource_id, entity.kind, doc_index
                )

                if invalid_quote is None:
                    # Quote is valid
                    valid_resource_quotes.append(quote_text)
                    valid_quotes += 1
                    entity_has_valid_quotes = True
                else:
                    # Quote is invalid
                    invalid_quotes.append(invalid_quote)

            if valid_resource_quotes:
                valid_entity_quotes[resource_id] = valid_resource_quotes

        # Only include entity if it has at least one valid quote
        if entity_has_valid_quotes:
            valid_entity = EntityOut(
                name=entity.name,
                kind=entity.kind,
                aliases=entity.aliases,
                quotes=valid_entity_quotes,
            )
            valid_entities.append(valid_entity)

    return ValidationResult(
        valid_entities=valid_entities,
        invalid_quotes=invalid_quotes,
        total_quotes=total_quotes,
        valid_quotes=valid_quotes,
    )


async def correct_invalid_citations(
    invalid_quotes: List[InvalidQuote],
    documents: List[Resource],
    model: Model,
) -> List[EntityOut]:
    """
    Correct invalid citations using targeted model calls.

    Args:
        invalid_quotes: List of quotes that failed validation
        documents: Source documents for correction
        model: LLM model for corrections

    Returns:
        List of entities with corrected quotes
    """
    if not invalid_quotes:
        return []

    # Group invalid quotes by entity for efficiency
    quotes_by_entity = {}
    for invalid_quote in invalid_quotes:
        key = (invalid_quote.entity_name, invalid_quote.resource_id)
        if key not in quotes_by_entity:
            quotes_by_entity[key] = []
        quotes_by_entity[key].append(invalid_quote)

    corrected_entities = []

    for (entity_name, resource_id), entity_invalid_quotes in quotes_by_entity.items():
        # Find the target resource
        target_resource = None
        for resource in documents:
            if resource.id.id == resource_id:
                target_resource = resource
                break

        if not target_resource:
            continue  # Skip if resource not found

        # Create correction request
        correction_requests = []
        entity_kind = (
            entity_invalid_quotes[0].entity_kind if entity_invalid_quotes else "gene"
        )

        for invalid_quote in entity_invalid_quotes:
            request = QuoteCorrectRequest(
                entity_name=entity_name,
                entity_kind=invalid_quote.entity_kind,
                invalid_quote=invalid_quote.quote_text,
                resource_id=resource_id,
                error_details=invalid_quote.diagnostics or invalid_quote.error_message,
                suggestion=invalid_quote.suggested_correction,
            )
            correction_requests.append(request)

        # Make targeted correction request
        corrected_quotes = await request_quote_corrections(
            correction_requests, target_resource, model
        )

        if corrected_quotes:
            # Create entity with corrected quotes
            entity_quotes = {resource_id: [q.corrected_quote for q in corrected_quotes]}
            corrected_entity = EntityOut(
                name=entity_name,
                kind=entity_kind,
                aliases=[],
                quotes=entity_quotes,
            )
            corrected_entities.append(corrected_entity)

    return corrected_entities


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def validate_quote(
    entity_name: str,
    quote_text: str,
    resource_id: str,
    entity_kind: str,
    doc_index: Dict[str, Resource],
) -> Optional[InvalidQuote]:
    """Validate quote and return InvalidQuote if failed, None if valid."""
    # Find the resource
    target_resource = doc_index.get(resource_id)
    if not target_resource:
        return InvalidQuote(
            entity_name=entity_name,
            quote_text=quote_text,
            resource_id=resource_id,
            error_message=f"Resource '{resource_id}' not found",
            entity_kind=entity_kind,
            diagnostics="The specified resource ID does not exist in the current batch.",
        )

    # Try to validate the quote directly
    try:
        target_resource.quote(quote_text)
        return None  # Valid quote
    except ValueError as e:
        # Quote failed - try simple corrections
        suggested_correction = None
        diagnostics = []

        # Check for simple fixes using existing helper functions
        recovery = find_longest_matching_subquote(quote_text, target_resource)
        if recovery and len(recovery.split()) / len(quote_text.split()) > 0.5:
            suggested_correction = extend_to_sentence_boundary(
                recovery, target_resource
            )
            diagnostics.append(
                "Possible paraphrasing detected - suggested verbatim quote"
            )

        return InvalidQuote(
            entity_name=entity_name,
            quote_text=quote_text,
            resource_id=resource_id,
            error_message=str(e),
            entity_kind=entity_kind,
            diagnostics="; ".join(diagnostics) if diagnostics else None,
            suggested_correction=suggested_correction,
        )


def extend_to_sentence_boundary(partial_quote: str, resource: Resource) -> str:
    """Extend a partial quote to natural sentence boundaries."""
    # Try extending backward and forward to find complete sentences
    words = partial_quote.strip().split()
    if len(words) < 3:
        return partial_quote

    # Check if this looks like a suffix match (doesn't start with capital)
    if not partial_quote[0].isupper():
        # Extend backward to find sentence start
        first_three = " ".join(words[:3])
        for resource_match in resource.text.split(first_three):
            if len(resource_match) > 0:
                # Look backwards for sentence start
                extended_start = resource_match.rfind(". ") + 2
                if extended_start > 1:  # Found sentence boundary
                    extended_text = (
                        resource_match[extended_start:]
                        + first_three
                        + partial_quote[len(first_three) :]
                    )
                    return extended_text.strip()

    return partial_quote


async def request_quote_corrections(
    requests: List[QuoteCorrectRequest],
    target_resource: Resource,
    model: Model,
) -> List[QuoteCorrectionOut]:
    """Make targeted correction request for specific quotes."""
    if not requests:
        return []

    # Build targeted correction prompt
    entity_name = requests[0].entity_name
    corrections_needed = []

    for req in requests:
        correction_text = f"""
**Invalid Quote:** "{req.invalid_quote}"
**Problem:** {req.error_details}"""
        if req.suggestion:
            correction_text += f"""
**Suggested Fix:** "{req.suggestion}"
**Task:** Provide the exact verbatim quote from the resource that mentions {entity_name}."""
        corrections_needed.append(correction_text)

    system_prompt = f"""You are correcting invalid quotes for the entity '{entity_name}'.

**CRITICAL REQUIREMENTS:**
1. Provide ONLY verbatim text from the resource
2. Do NOT modify, paraphrase, or rephrase any text
3. Quote must mention the entity '{entity_name}'
4. Use the suggested fix if provided, or find the correct quote

**Resource Content:**
{target_resource.text}"""

    # Create user prompt for correction
    user_prompt = f"""Please correct these invalid quotes:
{chr(10).join(corrections_needed)}

For each quote, provide the exact verbatim text from the resource."""

    try:
        # Create tool definition for structured output
        correction_tool = ToolDefinition(
            name="correct_quotes",
            description="Provide corrected verbatim quotes for the entity",
            parameters_json_schema=QuoteCorrectionsListOut.model_json_schema(),
        )

        messages = [
            ModelRequest(
                parts=[
                    SystemPromptPart(system_prompt),
                    UserPromptPart(
                        user_prompt
                        + "\n\nCall the correct_quotes tool with the corrected quotes."
                    ),
                ]
            )
        ]

        response = await model_request(
            model,
            messages,
            model_request_parameters=ModelRequestParameters(
                function_tools=[correction_tool],
                allow_text_output=False,  # Force structured tool output
            ),
        )

        # Get the tool call result
        try:
            tool_part = next(
                p
                for p in response.parts
                if getattr(p, "tool_name", None) == "correct_quotes"
            )
        except StopIteration:
            return []

        try:
            corrections_response = QuoteCorrectionsListOut(**tool_part.args)
            return corrections_response.corrections
        except Exception:
            return []

    except Exception:
        return []


# ============================================================================
# ASSOCIATION CREATION FUNCTIONS
# ============================================================================


def create_entity_associations(
    entities: List[EntityWithQuotes],
    entity_kinds: List[str],
    relationship_type: str = "interaction",
) -> List[EntityPairOut]:
    """Create associations between entities based on co-occurrence and context."""
    associations = []

    # Group entities by document using ResourceQuotes
    entities_by_doc = {}
    for entity in entities:
        # Get unique document IDs from the entity's quotes
        doc_ids = set()
        for quote in entity.quotes:
            doc_ids.add(quote.resource.id.id)

        for doc_id in doc_ids:
            entities_by_doc.setdefault(doc_id, []).append(entity)

    # Find associations within each document
    for doc_id, doc_entities in entities_by_doc.items():
        # Get the document from any entity's quotes
        document = None
        for entity in doc_entities:
            for quote in entity.quotes:
                if quote.resource.id.id == doc_id:
                    document = quote.resource
                    break
            if document:
                break

        if not document:
            continue

        # Check all entity pairs in this document
        for entity_a, entity_b in itertools.combinations(doc_entities, 2):
            # Skip if not a valid entity pair
            valid_pairs = get_valid_entity_pairs(entity_kinds)
            if (entity_a.kind, entity_b.kind) not in valid_pairs:
                continue

            # Find evidence and create association if found
            evidence = find_relationship_evidence(entity_a, entity_b, document)
            if evidence:
                association = EntityPairOut(
                    entity_a=entity_a,
                    entity_b=entity_b,
                    relationship=get_relationship_mapping(
                        entity_kinds, relationship_type
                    ).get((entity_a.kind, entity_b.kind), relationship_type),
                    confidence=calculate_relationship_confidence(evidence),
                    evidence_quotes=evidence,
                    reasoning=generate_association_reasoning(
                        entity_a, entity_b, evidence
                    ),
                )
                associations.append(association)

    return associations


def find_relationship_evidence(
    entity_a: EntityWithQuotes,
    entity_b: EntityWithQuotes,
    document: Resource,
) -> List[ResourceQuote]:
    """
    Find quotes that mention both entities as evidence for their relationship.

    Args:
        entity_a: First entity
        entity_b: Second entity
        document: Document to search in
        config: Configuration for proximity calculations

    Returns:
        List of ResourceQuote objects containing both entities
    """
    evidence_quotes = []

    # Get all quotes for both entities in this document
    quotes_a = [
        q.get_quote_text(1)
        for q in entity_a.quotes
        if q.resource.id.id == document.id.id
    ]
    quotes_b = [
        q.get_quote_text(1)
        for q in entity_b.quotes
        if q.resource.id.id == document.id.id
    ]

    # Look for quotes that mention both entities
    for quote_a in quotes_a:
        for quote_b in quotes_b:
            # Check if the quotes are in the same sentence or nearby
            proximity = calculate_quote_proximity(quote_a, quote_b, document)

            if proximity == "same_sentence":
                # Find the encompassing quote that includes both
                combined_quote = find_encompassing_quote(quote_a, quote_b, document)
                if combined_quote:
                    try:
                        resource_quote = document.quote(combined_quote)
                        evidence_quotes.append(resource_quote)
                    except ValueError:
                        # If combined quote fails, use individual quotes
                        try:
                            resource_quote_a = document.quote(quote_a)
                            evidence_quotes.append(resource_quote_a)
                        except ValueError:
                            pass
            elif proximity == "same_paragraph":
                # Include both quotes as separate evidence
                try:
                    resource_quote_a = document.quote(quote_a)
                    resource_quote_b = document.quote(quote_b)
                    evidence_quotes.extend([resource_quote_a, resource_quote_b])
                except ValueError:
                    pass

    # Remove duplicates based on quote text
    seen_quotes = set()
    unique_evidence = []
    for quote in evidence_quotes:
        quote_text = (
            quote.get_all_quote_texts()[0] if quote.get_all_quote_texts() else ""
        )
        if quote_text not in seen_quotes:
            seen_quotes.add(quote_text)
            unique_evidence.append(quote)

    return unique_evidence


def calculate_quote_proximity(quote_a: str, quote_b: str, document: Resource) -> str:
    """
    Calculate the proximity between two quotes in a document using configurable thresholds.

    Args:
        quote_a: First quote text
        quote_b: Second quote text
        document: Document containing both quotes
        config: Configuration with proximity thresholds

    Returns:
        "same_sentence", "same_paragraph", or "different_sections"
    """
    try:
        # Find positions of both quotes in the document
        text = document.text
        pos_a = text.find(quote_a)
        pos_b = text.find(quote_b)

        if pos_a == -1 or pos_b == -1:
            return "different_sections"

        # Check if they're in the same sentence
        if abs(pos_a - pos_b) < 100:  # Same sentence threshold
            return "same_sentence"

        # Check if they're in the same paragraph
        if abs(pos_a - pos_b) < 500:  # Same paragraph threshold
            return "same_paragraph"

        return "different_sections"

    except Exception:
        return "different_sections"


def find_encompassing_quote(
    quote_a: str, quote_b: str, document: Resource
) -> Optional[str]:
    """Find a quote that encompasses both individual quotes."""
    try:
        text = document.text
        pos_a = text.find(quote_a)
        pos_b = text.find(quote_b)

        if pos_a == -1 or pos_b == -1:
            return None

        # Find the sentence boundaries that contain both quotes
        start_pos = min(pos_a, pos_b)
        end_pos = max(pos_a + len(quote_a), pos_b + len(quote_b))

        # Extend to sentence boundaries
        sentence_start = text.rfind(". ", 0, start_pos) + 2
        if sentence_start < 2:
            sentence_start = text.rfind("\n", 0, start_pos) + 1

        sentence_end = text.find(".", end_pos)
        if sentence_end == -1:
            sentence_end = text.find("\n", end_pos)
        if sentence_end == -1:
            sentence_end = len(text)
        else:
            sentence_end += 1  # Include the period

        encompassing_quote = text[sentence_start:sentence_end].strip()

        # Verify it contains both original quotes
        if quote_a in encompassing_quote and quote_b in encompassing_quote:
            return encompassing_quote

        return None

    except Exception:
        return None


def calculate_relationship_confidence(
    evidence: List[ResourceQuote],
) -> Literal["high", "medium", "low"]:
    """
    Calculate confidence in a relationship based on evidence strength.

    Since LLMs already identified these quotes as relationship evidence,
    we trust their judgment. More evidence generally indicates higher confidence.

    Args:
        evidence: List of supporting quotes identified by LLM

    Returns:
        Confidence level based on amount of evidence
    """
    if not evidence:
        return "low"
    elif len(evidence) >= 3:
        return "high"  # Multiple strong evidence pieces
    elif len(evidence) >= 2:
        return "medium"  # Some supporting evidence
    else:
        return "low"  # Single piece of evidence


def merge_entity_quotes(
    valid_entities: List[EntityOut], corrected_entities: List[EntityOut]
) -> List[EntityOut]:
    """
    Merge valid and corrected entities, combining quotes for entities with same name.

    Args:
        valid_entities: Entities that passed validation
        corrected_entities: Entities with corrected quotes

    Returns:
        Merged list with no duplicate entities
    """
    # Use dict to track entities by (name, kind) tuple to handle merging
    entity_map = {}

    # Add valid entities first
    for entity in valid_entities:
        key = (entity.name, entity.kind)
        entity_map[key] = entity

    # Merge corrected entities, combining quotes if entity already exists
    for corrected in corrected_entities:
        key = (corrected.name, corrected.kind)
        if key in entity_map:
            # Entity exists, merge quotes
            existing = entity_map[key]
            merged_quotes = dict(existing.quotes)  # Copy existing quotes

            # Add corrected quotes, combining with existing ones
            for doc_id, quote_list in corrected.quotes.items():
                if doc_id in merged_quotes:
                    # Combine quotes, removing duplicates
                    existing_quotes = set(merged_quotes[doc_id])
                    new_quotes = [q for q in quote_list if q not in existing_quotes]
                    merged_quotes[doc_id].extend(new_quotes)
                else:
                    merged_quotes[doc_id] = quote_list[:]

            # Update entity with merged quotes
            entity_map[key] = EntityOut(
                name=existing.name,
                kind=existing.kind,
                aliases=list(
                    set(existing.aliases + corrected.aliases)
                ),  # Merge aliases too
                quotes=merged_quotes,
            )
        else:
            # New entity, add it
            entity_map[key] = corrected

    return list(entity_map.values())


def convert_to_entity_with_quotes(
    entity: EntityOut, doc_index: Dict[str, Resource]
) -> EntityWithQuotes:
    """Convert EntityOut to EntityWithQuotes with ResourceQuote objects."""
    resource_quotes = []

    for doc_id, quotes in entity.quotes.items():
        document = doc_index.get(doc_id)
        if not document:
            continue

        for quote_text in quotes:
            try:
                resource_quote = document.quote(quote_text)
                resource_quotes.append(resource_quote)
            except ValueError:
                # Skip invalid quotes
                continue

    return EntityWithQuotes(
        name=entity.name,
        kind=entity.kind,
        aliases=entity.aliases,
        quotes=resource_quotes,
        confidence=1.0,
    )


def generate_association_reasoning(
    entity_a: EntityWithQuotes,
    entity_b: EntityWithQuotes,
    evidence: List[ResourceQuote],
) -> str:
    """Generate reasoning text for why two entities are associated based on available evidence."""
    if not evidence:
        return f"{entity_a.name} and {entity_b.name} appear in the same document."

    evidence_texts = []
    for quote in evidence:
        quote_texts = quote.get_all_quote_texts()
        if quote_texts:
            evidence_texts.append(f'"{quote_texts[0]}"')

    return f"{entity_a.name} and {entity_b.name} are associated based on evidence: {'; '.join(evidence_texts)}"
