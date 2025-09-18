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


def find_longest_matching_prefix(quote_text: str, resource) -> Optional[str]:
    """
    Find the longest matching prefix by searching all occurrences of the first 3 words.

    Args:
        quote_text: Quote text to find prefix for
        resource: Resource to search within

    Returns:
        Longest matching prefix string or None if no match found
    """
    words = quote_text.strip().split()
    if len(words) < 3:
        return None

    # Search for all occurrences of the first 3 words
    first_three = " ".join(words[:3])
    longest_match = None

    try:
        # Find all locations where first 3 words appear
        first_three_quote = resource.quote(first_three)

        # For each location, try to expand as far as possible
        for start_pos, _ in first_three_quote.spans:
            # Try expanding word by word
            for end_word in range(3, len(words) + 1):
                test_prefix = " ".join(words[:end_word])
                try:
                    resource.quote(test_prefix)
                    # This prefix matches, continue expanding
                    longest_match = test_prefix
                except ValueError:
                    # This prefix doesn't match, stop expanding
                    break
    except ValueError:
        # First 3 words don't match anywhere
        return None

    return longest_match


def find_longest_matching_suffix(quote_text: str, resource) -> Optional[str]:
    """
    Find the longest matching suffix by searching all occurrences of the last 3 words.

    Args:
        quote_text: Quote text to find suffix for
        resource: Resource to search within

    Returns:
        Longest matching suffix string or None if no match found
    """
    words = quote_text.strip().split()
    if len(words) < 3:
        return None

    # Search for all occurrences of the last 3 words
    last_three = " ".join(words[-3:])
    longest_match = None

    try:
        # Find all locations where last 3 words appear
        last_three_quote = resource.quote(last_three)

        # For each location, try to expand backwards as far as possible
        for _, end_pos in last_three_quote.spans:
            # Try expanding word by word backwards
            for start_word in range(len(words) - 3, -1, -1):
                test_suffix = " ".join(words[start_word:])
                try:
                    resource.quote(test_suffix)
                    # This suffix matches, continue expanding
                    longest_match = test_suffix
                except ValueError:
                    # This suffix doesn't match, stop expanding
                    break
    except ValueError:
        # Last 3 words don't match anywhere
        return None

    return longest_match


def compute_bag_of_words_similarity(text1: str, text2: str) -> float:
    """
    Compute bag-of-words similarity between two texts.

    Uses Jaccard similarity coefficient based on word sets, which is
    robust to word order differences and handles quote corrections well.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Similarity score between 0.0 (no overlap) and 1.0 (identical words)
    """
    # Handle empty strings
    text1 = text1.strip() if text1 else ""
    text2 = text2.strip() if text2 else ""

    if not text1 and not text2:
        return 1.0  # Both empty = identical
    if not text1 or not text2:
        return 0.0  # One empty, one not = no similarity

    # Normalize and tokenize
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    # Jaccard similarity: |intersection| / |union|
    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def _build_individual_retry_message(
    entity_name: str,
    quote_text: str,
    split_info: Optional[dict],
    recovery_info: Optional[dict],
) -> str:
    """
    Build a targeted retry message for a specific quote error.

    Args:
        entity_name: Name of the entity with the failed quote
        quote_text: The original failed quote text
        split_info: Information about split quote detection
        recovery_info: Information about potential corrections

    Returns:
        Focused retry message for this specific quote error
    """
    message_parts = [
        f"Quote validation failed for entity '{entity_name}':",
        f'Failed Quote: "{quote_text}"',
        "",
    ]

    # Add split quote findings
    if split_info:
        message_parts.extend(
            [
                "❗ QUOTE APPEARS TO BE INCORRECTLY COMBINED:",
                f'  • Prefix: "{split_info["prefix"]}"',
                f'  • Suffix: "{split_info["suffix"]}"',
                '  This quote combines text from different parts without using "...".',
                "",
            ]
        )

    # Add recovery suggestions
    if recovery_info:
        message_parts.extend(
            [
                f"✅ POTENTIAL CORRECTION (partial match found):",
                f'  Matched part: "{recovery_info["matched_subquote"]}"',
                "  Suggested corrections:",
            ]
        )
        for suggestion in recovery_info["suggestions"]:
            message_parts.append(f'  • "{suggestion}"')
        message_parts.append("")

    # Add specific instructions for this quote
    message_parts.extend(
        [
            "INSTRUCTIONS FOR THIS QUOTE:",
            "• Use EXACT text - no paraphrasing or synonym substitution",
            "• Copy text character-for-character as it appears in the document",
        ]
    )

    if recovery_info:
        message_parts.append(
            "• If corrections are suggested above, use the exact suggested text"
        )

    if split_info:
        message_parts.append(
            "• For split quotes, use separate quotes for each part with correct text"
        )

    return "\n".join(message_parts)


def find_longest_matching_subquote(quote_text: str, resource) -> Optional[str]:
    """
    Find the longest matching subquote that includes either the start or end.

    Args:
        quote_text: Quote text to analyze
        resource: Resource to search within

    Returns:
        Longest matching subquote or None if neither prefix nor suffix match enough
    """
    prefix_match = find_longest_matching_prefix(quote_text, resource)
    suffix_match = find_longest_matching_suffix(quote_text, resource)

    # Return the longer of the two matches
    if not prefix_match and not suffix_match:
        return None
    elif not prefix_match:
        return suffix_match
    elif not suffix_match:
        return prefix_match
    else:
        # Return the longer match by word count
        if len(prefix_match.split()) >= len(suffix_match.split()):
            return prefix_match
        else:
            return suffix_match


def detect_split_quote(quote_text: str, resource) -> Optional[dict]:
    """
    Detect if a quote is incorrectly combined from different parts of the document.

    Uses binary search to find the longest matching prefix, then checks if the
    remaining suffix also matches elsewhere in the document.

    Args:
        quote_text: The quote that failed validation
        resource: The resource to search within

    Returns:
        Dictionary with prefix and suffix info if split detected, None otherwise
    """
    quote_text = quote_text.strip()
    words = quote_text.split()

    # Need at least 6 words to meaningfully split (min 3 words each part)
    if len(words) < 6:
        return None

    # Binary search for longest matching prefix
    left, right = 3, len(words) - 3  # Ensure both parts have at least 3 words
    longest_prefix = None
    longest_prefix_info = None

    while left <= right:
        mid = (left + right) // 2
        prefix = " ".join(words[:mid])

        try:
            prefix_quote = resource.quote(prefix)
            # Prefix matches - try to expand
            longest_prefix = prefix
            longest_prefix_info = prefix_quote
            left = mid + 1
        except ValueError:
            # Prefix doesn't match - try smaller
            right = mid - 1

    # If we found a matching prefix, check if suffix also matches
    if longest_prefix and longest_prefix_info:
        prefix_words = longest_prefix.split()
        suffix = " ".join(words[len(prefix_words) :])

        try:
            suffix_quote = resource.quote(suffix)

            # Check if prefix and suffix are contiguous in the document
            # If they're contiguous, this isn't a split quote - just overly aggressive matching
            for prefix_start, prefix_end in longest_prefix_info.spans:
                for suffix_start, suffix_end in suffix_quote.spans:
                    # Check if suffix immediately follows prefix (allowing for whitespace)
                    text_between = resource.text[prefix_end:suffix_start].strip()
                    if (
                        not text_between or len(text_between) < 10
                    ):  # Small gap = contiguous
                        return None  # Not a split quote, just adjacent text

            # Both parts match and are non-contiguous! This is a split quote
            return {
                "prefix": longest_prefix,
                "prefix_quote": longest_prefix_info,
                "suffix": suffix,
                "suffix_quote": suffix_quote,
            }
        except ValueError:
            # Suffix doesn't match - not a split quote
            pass

    return None


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
    entity_kinds: List[str],
    context: str,
    target_term: Optional[str] = None,
) -> Agent[ExtractionDeps, SimpleEntityListOut]:
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

    # Enhanced output validator that tests ResourceQuote creation
    @agent.output_validator
    def validate_resourcequote_creation(
        ctx: RunContext[ExtractionDeps], out: SimpleEntityListOut
    ) -> SimpleEntityListOut:
        """
        Validate ResourceQuote creation and provide detailed rich feedback for failures.

        This tests actual ResourceQuote creation using the resources from the context
        and displays detailed information about quote failures with Rich formatting.
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from datetime import datetime
        from .models import QuoteErrorRecord

        # Access resources from the context
        current_resources = ctx.deps.current_resources
        if not current_resources:
            return out  # Skip validation if no resources available

        console = Console()
        failed_quotes = []

        # Since we're processing one document at a time, use the first (only) resource
        resource = current_resources[0] if current_resources else None

        for entity in out.entities:
            entity_name = entity.name

            if resource:
                # Test ResourceQuote creation for each quote text
                for quote_text in entity.quotes:
                    quote_text = quote_text.strip()
                    try:
                        resource.quote(quote_text)
                        # Success - quote can be created
                    except ValueError as e:
                        # Failed - collect details for feedback
                        failure_info = {
                            "entity": entity_name,
                            "quote": quote_text,
                            "resource_id": resource.id.id,
                            "resource_url": resource.id.url,
                            "resource_title": resource.title,
                            "error": str(e),
                            "resource": resource,  # Keep reference for split detection
                        }
                        failed_quotes.append(failure_info)

        # If any quotes failed, display detailed information with Rich
        if failed_quotes:
            # Create main container
            console.print(
                f"\n[bold red]Failed Quotes ({len(failed_quotes)} total)[/bold red]"
            )

            # Process diagnostic information and create individual error records
            processed_failures = []
            retry_attempt = getattr(ctx, "_retry_count", 0) + 1

            for i, failure in enumerate(failed_quotes, 1):
                quote_text = failure["quote"].strip()

                # Single-document processing - no cross-resource checking needed
                quote_found_elsewhere = []

                # Check for split quotes
                split_info = detect_split_quote(quote_text, failure["resource"])

                # Check for quote recovery suggestions
                recovery_info = None
                if not quote_found_elsewhere and not split_info:
                    longest_subquote = find_longest_matching_subquote(
                        quote_text, failure["resource"]
                    )
                    if longest_subquote:
                        attempted_words = len(quote_text.split())
                        matched_words = len(longest_subquote.split())
                        match_percentage = (matched_words / attempted_words) * 100

                        if match_percentage > 50:
                            try:
                                actual_quote = failure["resource"].quote(
                                    longest_subquote
                                )
                                missing_words = attempted_words - matched_words

                                extended_suggestions = []

                                # Determine if we found a prefix or suffix match
                                quote_words = quote_text.split()
                                longest_words = longest_subquote.split()

                                # Calculate how much additional content we need to cover
                                missing_word_count = len(quote_words) - len(
                                    longest_words
                                )

                                # Check if match is at the beginning (prefix) or end (suffix)
                                # A prefix match means the longest_subquote starts at the beginning of quote_text
                                # A suffix match means the longest_subquote ends at the end of quote_text
                                is_prefix_match = (
                                    quote_text.startswith(longest_subquote)
                                    if longest_subquote
                                    else False
                                )
                                is_suffix_match = (
                                    quote_text.endswith(longest_subquote)
                                    if longest_subquote
                                    else False
                                )

                                for start_pos, end_pos in actual_quote.spans:
                                    if is_prefix_match:
                                        # We have a prefix match - extend FORWARD from the end to find natural completion
                                        extend_end = min(
                                            len(failure["resource"].text), end_pos + 500
                                        )  # Look forward up to 500 chars
                                        text_after = failure["resource"].text[
                                            end_pos:extend_end
                                        ]

                                        # Find sentence/phrase boundaries going forward
                                        sentence_ends = []
                                        for i, char in enumerate(text_after):
                                            if char in ".!?":
                                                # Found sentence end, this is a good stopping point
                                                sentence_ends.append(end_pos + i + 1)
                                            elif (
                                                i > 0
                                                and char.isupper()
                                                and text_after[i - 1] in " \n\t"
                                            ):
                                                # Capitalized word after whitespace (start of new sentence)
                                                sentence_ends.append(end_pos + i)

                                        # Try extending to natural sentence boundaries
                                        if sentence_ends:
                                            # Find the first ending point that gives us enough additional content
                                            for candidate_end in sentence_ends:
                                                suggested_text = (
                                                    failure["resource"]
                                                    .text[start_pos:candidate_end]
                                                    .strip()
                                                )
                                                if suggested_text:
                                                    suggested_words = (
                                                        suggested_text.split()
                                                    )
                                                    # Must have at least as many words as original quote
                                                    # and at least missing_word_count more than the partial match
                                                    if (
                                                        len(suggested_words)
                                                        >= len(quote_words)
                                                        and len(suggested_words)
                                                        >= len(longest_words)
                                                        + missing_word_count
                                                    ):
                                                        extended_suggestions.append(
                                                            suggested_text
                                                        )
                                                        break

                                        # Fallback: extend by a reasonable number of words, ensuring we cover the missing content
                                        if not extended_suggestions:
                                            words_after = text_after.split()
                                            # Calculate minimum extension needed: missing words + buffer
                                            min_extension = max(missing_word_count, 5)
                                            for word_count in [
                                                min_extension + 5,
                                                min_extension,
                                            ]:
                                                if len(words_after) >= word_count:
                                                    extended_text = " ".join(
                                                        words_after[:word_count]
                                                    )
                                                    suggested_text = (
                                                        failure["resource"].text[
                                                            start_pos:end_pos
                                                        ]
                                                        + " "
                                                        + extended_text
                                                    ).strip()
                                                    extended_suggestions.append(
                                                        suggested_text
                                                    )
                                                    break

                                    elif is_suffix_match:
                                        # We have a suffix match - extend backwards to find natural start
                                        extend_start = max(
                                            0, start_pos - 500
                                        )  # Look back up to 500 chars
                                        text_before = failure["resource"].text[
                                            extend_start:start_pos
                                        ]

                                        # Find sentence/phrase boundaries
                                        sentence_starts = []
                                        for i, char in enumerate(text_before):
                                            if char in ".!?":
                                                # Found sentence end, next non-whitespace is potential start
                                                remaining = text_before[
                                                    i + 1 :
                                                ].lstrip()
                                                if remaining and remaining[0].isupper():
                                                    sentence_starts.append(
                                                        extend_start
                                                        + i
                                                        + 1
                                                        + (
                                                            len(text_before[i + 1 :])
                                                            - len(remaining)
                                                        )
                                                    )
                                            elif i == 0 or (
                                                char.isupper()
                                                and text_before[i - 1] in " \n\t"
                                            ):
                                                # Capitalized word after whitespace
                                                sentence_starts.append(extend_start + i)

                                        # Try the most promising sentence start, ensuring adequate extension
                                        if sentence_starts:
                                            # Find the furthest start that gives us enough additional content
                                            for candidate_start in reversed(
                                                sorted(sentence_starts)
                                            ):
                                                suggested_text = (
                                                    failure["resource"]
                                                    .text[candidate_start:end_pos]
                                                    .strip()
                                                )
                                                if suggested_text:
                                                    suggested_words = (
                                                        suggested_text.split()
                                                    )
                                                    # Must have at least as many words as original quote
                                                    # and at least missing_word_count more than the partial match
                                                    if (
                                                        len(suggested_words)
                                                        >= len(quote_words)
                                                        and len(suggested_words)
                                                        >= len(longest_words)
                                                        + missing_word_count
                                                    ):
                                                        extended_suggestions.append(
                                                            suggested_text
                                                        )
                                                        break

                                    # Fallback: if no clear prefix/suffix, try both directions with adequate extension
                                    if not extended_suggestions:
                                        # Calculate minimum extension needed
                                        min_total_words = max(
                                            len(quote_words),
                                            len(longest_words) + missing_word_count,
                                        )

                                        # Try extending forward
                                        extension_chars = max(
                                            missing_word_count * 8, 200
                                        )  # Reasonable char estimate
                                        words_around = (
                                            failure["resource"]
                                            .text[start_pos : end_pos + extension_chars]
                                            .split()
                                        )
                                        if len(words_around) >= min_total_words:
                                            extended_forward = " ".join(
                                                words_around[:min_total_words]
                                            )
                                            extended_suggestions.append(
                                                extended_forward
                                            )

                                        # Try extending backward
                                        text_before_start = max(
                                            0, start_pos - extension_chars
                                        )
                                        words_around = (
                                            failure["resource"]
                                            .text[text_before_start:end_pos]
                                            .split()
                                        )
                                        if len(words_around) >= min_total_words:
                                            extended_backward = " ".join(
                                                words_around[-min_total_words:]
                                            )
                                            extended_suggestions.append(
                                                extended_backward
                                            )

                                if extended_suggestions:
                                    recovery_info = {
                                        "matched_subquote": longest_subquote,
                                        "match_percentage": match_percentage,
                                        "suggestions": extended_suggestions[:2],
                                    }
                            except Exception:
                                pass

                # Store processed information for both display and agent message
                processed_failure = {
                    "failure": failure,
                    "quote_text": quote_text,
                    "quote_found_elsewhere": quote_found_elsewhere,
                    "split_info": split_info,
                    "recovery_info": recovery_info,
                }
                processed_failures.append(processed_failure)

                # Create individual error record with targeted retry message
                # Determine error type based on the error message
                error_message = failure["error"].lower()
                if "not found" in error_message or "no exact match" in error_message:
                    error_type = "not_found"
                elif "split" in error_message or "combined" in error_message:
                    error_type = "split_quote"
                else:
                    error_type = "paraphrased"

                # Create error record with suggestions and individual retry message
                suggestions = []
                matched_percentage = None
                if recovery_info:
                    suggestions = recovery_info.get("suggestions", [])
                    matched_percentage = recovery_info.get("match_percentage")

                # Build individual retry message for this specific quote error
                individual_retry_message = _build_individual_retry_message(
                    failure["entity"], quote_text, split_info, recovery_info
                )

                error_record = QuoteErrorRecord(
                    entity_name=failure["entity"],
                    entity_kind=next(
                        (e.kind for e in out.entities if e.name == failure["entity"]),
                        "unknown",
                    ),
                    original_quote=quote_text,
                    error_type=error_type,
                    suggested_corrections=suggestions,
                    matched_percentage=matched_percentage,
                    retry_attempt=retry_attempt,
                    retry_message=individual_retry_message,
                )

                # Add to error log
                ctx.deps.quote_error_log.append(error_record)

            # Display Rich output for developer debugging
            for i, processed in enumerate(processed_failures, 1):
                failure = processed["failure"]
                quote_text = processed["quote_text"]
                quote_found_elsewhere = processed["quote_found_elsewhere"]
                split_info = processed["split_info"]
                recovery_info = processed["recovery_info"]

                # Create detailed panel for each failure
                failure_content = Text()

                failure_content.append(f"Entity: ", style="bold cyan")
                failure_content.append(f"{failure['entity']}\n", style="green")

                failure_content.append(f"Resource ID: ", style="bold cyan")
                failure_content.append(f"{failure['resource_id']}\n", style="white")

                failure_content.append(f"Resource URL: ", style="bold cyan")
                failure_content.append(f"{failure['resource_url']}\n", style="blue")

                failure_content.append(f"Resource Title: ", style="bold cyan")
                failure_content.append(
                    f"{failure['resource_title']}\n\n", style="white"
                )

                failure_content.append(f"Quote Text:\n", style="bold magenta")
                failure_content.append(f'"{quote_text}"')

                # Add split quote detection results
                if split_info:
                    failure_content.append(
                        f"\n\nQuote appears to be incorrectly combined:\n",
                        style="bold bright_yellow",
                    )

                    # Show prefix location
                    prefix_quote = split_info["prefix_quote"]
                    failure_content.append(f"• Prefix: ", style="bright_magenta")
                    failure_content.append(f'"{split_info["prefix"]}"')
                    prefix_spans = ", ".join(
                        f"{start}-{end}" for start, end in prefix_quote.spans
                    )
                    failure_content.append(
                        f" (chars {prefix_spans})\n",
                        style="dim white",
                    )

                    # Show suffix location
                    suffix_quote = split_info["suffix_quote"]
                    failure_content.append(f"• Suffix: ", style="bright_magenta")
                    failure_content.append(f'"{split_info["suffix"]}"')
                    suffix_spans = ", ".join(
                        f"{start}-{end}" for start, end in suffix_quote.spans
                    )
                    failure_content.append(
                        f" (chars {suffix_spans})\n",
                        style="dim white",
                    )

                # Add quote recovery suggestions
                if recovery_info:
                    failure_content.append(
                        f"\n\nPotential quote correction ({recovery_info['match_percentage']:.1f}% match):\n",
                        style="bold green",
                    )

                    failure_content.append(f"Matched part: ", style="green")
                    failure_content.append(
                        f'"{recovery_info["matched_subquote"]}"', style="dim green"
                    )
                    failure_content.append(
                        f"\n\nSuggested corrections:\n", style="green"
                    )

                    for j, suggestion in enumerate(recovery_info["suggestions"], 1):
                        failure_content.append(f"{j}. ", style="green")
                        failure_content.append(
                            f'"{suggestion}"\n', style="bright_green"
                        )

                panel_title = f"[bold red]Quote Failure #{i}[/bold red]"
                console.print(
                    Panel(failure_content, title=panel_title, border_style="red")
                )

            # Build combined message for the agent including all quote failures
            # Each quote now has its individual retry message stored in the error record
            combined_message_parts = [
                f"Quote validation failed for {len(failed_quotes)} quote(s). ",
                "Please fix ALL quotes using the specific guidance provided for each:",
                "",
            ]

            # Add summary of all failures for context
            for i, processed in enumerate(processed_failures, 1):
                failure = processed["failure"]
                quote_text = processed["quote_text"]
                recovery_info = processed["recovery_info"]
                split_info = processed["split_info"]

                combined_message_parts.append(
                    f"Quote #{i} (Entity: {failure['entity']}):"
                )
                combined_message_parts.append(f'  Failed: "{quote_text}"')

                if recovery_info:
                    suggestions = recovery_info["suggestions"]
                    if suggestions:
                        combined_message_parts.append(f'  → Use: "{suggestions[0]}"')
                elif split_info:
                    combined_message_parts.append("  → Split into separate quotes")
                else:
                    combined_message_parts.append("  → Find exact text in document")

                combined_message_parts.append("")

            combined_message_parts.extend(
                [
                    "GENERAL INSTRUCTIONS:",
                    "• Use EXACT text - no paraphrasing or synonym substitution",
                    "• Copy text character-for-character as it appears in the document",
                    "• For split quotes, create separate quote entries",
                ]
            )

            combined_message = "\n".join(combined_message_parts)
            raise ModelRetry(combined_message)

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

    # Enhanced validator to check ResourceQuote creation
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
                    "Please provide exact quotes from the documents."
                )

            # Validate the simple quote list structure
            if not isinstance(entity_quotes, list):
                raise ModelRetry(
                    f"Entity '{entity_name}' has invalid quote structure. "
                    "Quotes must be a list of quote strings."
                )

            # First, validate basic structure of all quotes
            for quote_text in entity_quotes:
                if not isinstance(quote_text, str):
                    raise ModelRetry(
                        f"Invalid quote type for entity '{entity_name}'. "
                        "All quotes must be strings."
                    )

                quote_text = quote_text.strip()
                if not quote_text:
                    raise ModelRetry(f"Empty quote text for entity '{entity_name}'")

            # Now check if at least ONE quote contains the entity name/alias
            entity_found_in_quotes = False
            normalized_entity_name = normalize_text_for_matching(entity.name)

            for quote_text in entity_quotes:
                quote_text = quote_text.strip()
                normalized_quote = normalize_text_for_matching(quote_text)

                # Check if entity name appears in quote (direct match)
                if normalized_entity_name in normalized_quote:
                    entity_found_in_quotes = True
                    break

                # Check if entity name appears in expanded shorthand forms
                quote_expansions = expand_scientific_shorthand(quote_text)
                for expanded_quote in quote_expansions:
                    expanded_normalized = normalize_text_for_matching(expanded_quote)
                    if normalized_entity_name in expanded_normalized:
                        entity_found_in_quotes = True
                        break

                if entity_found_in_quotes:
                    break

                # Check aliases if provided
                if entity.aliases:
                    for alias in entity.aliases:
                        normalized_alias = normalize_text_for_matching(alias)
                        # Direct alias match
                        if normalized_alias in normalized_quote:
                            entity_found_in_quotes = True
                            break
                        # Alias match in expanded forms
                        for expanded_quote in quote_expansions:
                            expanded_normalized = normalize_text_for_matching(
                                expanded_quote
                            )
                            if normalized_alias in expanded_normalized:
                                entity_found_in_quotes = True
                                break
                        if entity_found_in_quotes:
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
