"""Helper functions for the extraction pipeline.

Provides utilities for entity validation, proximal set identification,
text region construction, and pair key generation.
"""

import re
from collections import Counter

from interaction_finder.extraction.models import (
    EntityMention,
    EntityPairKey,
    ProximalEntitySet,
)
from interaction_finder.resources import (
    Resource,
    ResourceQuote,
    normalize_text_for_matching,
)


def build_permitted_pairs(entity_types: list[str]) -> dict[str, set[str]]:
    """Build mapping of which entity kinds can pair with which.

    The rule: a kind must appear at least twice in the input list to permit
    self-pairs (kind-kind). Cross-pairs (kindA-kindB) are permitted if both
    kinds appear at least once.

    Special case: if only one unique kind is provided, self-pairs are allowed
    regardless of count (otherwise no pairs would be possible).

    Parameters:
        entity_types: List of entity kinds (may contain duplicates)

    Returns:
        Dict mapping each kind to the set of kinds it can pair with

    Examples:
        >>> build_permitted_pairs(["gene", "disease"])
        {'gene': {'disease'}, 'disease': {'gene'}}

        >>> build_permitted_pairs(["gene", "gene", "disease"])
        {'gene': {'gene', 'disease'}, 'disease': {'gene'}}

        >>> build_permitted_pairs(["gene"])
        {'gene': {'gene'}}

        >>> build_permitted_pairs(["gene", "disease", "protein"])
        {'gene': {'disease', 'protein'}, 'disease': {'gene', 'protein'}, 'protein': {'gene', 'disease'}}
    """
    # Count occurrences of each kind
    counts = Counter(entity_types)
    unique_kinds = set(entity_types)

    # Build permitted pairs map
    permitted: dict[str, set[str]] = {}
    for kind in unique_kinds:
        # Start with all other kinds
        allowed = unique_kinds - {kind}
        # Add self if kind appears at least twice OR if it's the only kind
        if counts[kind] >= 2 or len(unique_kinds) == 1:
            allowed.add(kind)
        permitted[kind] = allowed

    return permitted


def strip_kind_annotation(entity_name: str) -> str:
    """Strip kind annotation from entity name if present.

    Removes trailing patterns like " (gene)", " (phenotype)", etc. that may
    have been incorrectly included by the LLM despite instructions.

    Parameters:
        entity_name: Entity name that may contain kind annotation

    Returns:
        Entity name with kind annotation removed

    Examples:
        >>> strip_kind_annotation("BRCA1 (gene)")
        'BRCA1'
        >>> strip_kind_annotation("Iron deficiency (phenotype)")
        'Iron deficiency'
        >>> strip_kind_annotation("BRCA1")
        'BRCA1'
    """
    # Match pattern: " (word)" at end of string
    return re.sub(r"\s+\([a-z_]+\)\s*$", "", entity_name).strip()


def normalize_for_comparison(text: str) -> str:
    """Normalize text for entity name comparison.

    Uses the same normalization as fuzzy quote matching to ensure consistent
    comparison behavior across the pipeline.

    Parameters:
        text: Text to normalize

    Returns:
        Normalized lowercase text
    """
    return normalize_text_for_matching(text)


def find_substring_entities(
    entities: dict[str, EntityMention],
) -> list[tuple[str, str]]:
    """Find entity pairs where normalized names are equal or one is a substring of another.

    Compares normalized lowercase versions of entity names to identify
    potential merge candidates (e.g., "BRCA" and "BRCA1", or "PAH" and "pah").

    Parameters:
        entities: Dict mapping canonical name to EntityMention

    Returns:
        List of (parent_name, child_name) tuples where:
        - For exact normalized matches: parent is the original (keeps first seen)
        - For substring matches: parent is the shorter/more general name, child is longer/more specific
    """
    candidates = []
    entity_names = list(entities.keys())

    # Compare each pair of entities
    for i, name1 in enumerate(entity_names):
        norm1 = normalize_for_comparison(name1)

        for name2 in entity_names[i + 1 :]:
            norm2 = normalize_for_comparison(name2)

            # Check if normalized forms are identical
            if norm1 == norm2:
                # Exact match after normalization → keep first, merge second
                candidates.append((name1, name2))
            # Check if either is a substring of the other (but not equal)
            elif norm1 in norm2:
                # name1 is substring of name2 → name1 is parent (general), name2 is child (specific)
                candidates.append((name1, name2))
            elif norm2 in norm1:
                # name2 is substring of name1 → name2 is parent (general), name1 is child (specific)
                candidates.append((name2, name1))

    return candidates


def identify_proximal_sets(
    entities: dict[str, EntityMention], threshold: int, resource: Resource
) -> list[ProximalEntitySet]:
    """Identify groups of entities found in close proximity using sliding window.

    Algorithm:
    1. Get chunk indices for each entity's quotes
    2. Sort quotes by (start_chunk, -length_in_chunks)
    3. Use sliding window with configurable threshold
    4. Group entities whose quotes appear within threshold chunks of each other

    Parameters:
        entities: Dict mapping canonical name to EntityMention
        threshold: Maximum chunk distance to consider entities proximal
        resource: Resource containing chunk boundary information

    Returns:
        List of ProximalEntitySet objects, each containing entities found near each other
    """
    if not entities:
        return []

    # Build list of (entity_name, quote, start_chunk, end_chunk)
    quote_info = []
    for entity_name, entity in entities.items():
        for quote in entity.quotes:
            # Get chunk indices for this quote
            chunk_indices = quote.chunk_indices
            if not chunk_indices:
                # Quote doesn't overlap any chunks, skip it
                continue

            start_chunk = min(chunk_indices)
            end_chunk = max(chunk_indices)
            quote_info.append((entity_name, quote, start_chunk, end_chunk))

    if not quote_info:
        return []

    # Sort by (start_chunk, -length) to process quotes in order
    quote_info.sort(key=lambda x: (x[2], -(x[3] - x[2])))

    # Sliding window algorithm
    proximal_sets = []
    current_entities: set[str] = set()
    current_quotes: dict[str, list[ResourceQuote]] = {}
    window_start = quote_info[0][2]
    window_end = quote_info[0][3]

    for entity_name, quote, start_chunk, end_chunk in quote_info:
        # Check if quote starts within current window + threshold
        if start_chunk <= window_end + threshold:
            # Add to current proximal set
            current_entities.add(entity_name)
            if entity_name not in current_quotes:
                current_quotes[entity_name] = []
            current_quotes[entity_name].append(quote)

            # Expand window to include this quote
            window_end = max(window_end, end_chunk)
        else:
            # Quote is too far away, finalize current set
            if len(current_entities) >= 2:  # Only keep sets with 2+ entities
                proximal_sets.append(
                    ProximalEntitySet(
                        entities=current_entities.copy(),
                        chunk_range=(window_start, window_end),
                        entity_quotes=current_quotes.copy(),
                    )
                )

            # Start new proximal set
            current_entities = {entity_name}
            current_quotes = {entity_name: [quote]}
            window_start = start_chunk
            window_end = end_chunk

    # Don't forget the last set
    if len(current_entities) >= 2:
        proximal_sets.append(
            ProximalEntitySet(
                entities=current_entities.copy(),
                chunk_range=(window_start, window_end),
                entity_quotes=current_quotes.copy(),
            )
        )

    return proximal_sets


def build_text_region(
    resource: Resource, chunk_start: int, chunk_end: int, padding: int
) -> str:
    """Build text region from chunk range with padding.

    Extracts all chunks from (chunk_start - padding) to (chunk_end + padding),
    concatenating them into a single text string.

    Parameters:
        resource: Resource containing the text and chunks
        chunk_start: Starting chunk index (inclusive)
        chunk_end: Ending chunk index (inclusive)
        padding: Number of chunks to add on each side

    Returns:
        Concatenated text from selected chunks
    """
    # Apply padding with bounds checking
    start_idx = max(0, chunk_start - padding)
    end_idx = min(len(resource.chunks) - 1, chunk_end + padding)

    # Extract chunks
    chunks = []
    for chunk_idx in range(start_idx, end_idx + 1):
        chunk_start_pos, chunk_end_pos = resource.chunks[chunk_idx]
        chunk_text = resource.text[chunk_start_pos:chunk_end_pos]
        chunks.append(chunk_text)

    # Concatenate chunks
    return "\n".join(chunks)


def collect_relevant_text_for_quotes(
    resource: Resource, quotes: list[ResourceQuote], padding: int
) -> str:
    """Build text region containing all provided quotes with padding.

    Identifies the chunk range spanning all quotes and returns the contiguous
    text from start to end (including padding).

    Parameters:
        resource: Resource containing the text and chunks
        quotes: List of quotes to cover
        padding: Number of chunks to add on each side

    Returns:
        Contiguous text covering all quotes with padding
    """
    if not quotes:
        return ""

    # Collect all chunk indices from all quotes
    all_chunks = set()
    for quote in quotes:
        all_chunks.update(quote.chunk_indices)

    if not all_chunks:
        return ""

    # Get range from min to max chunk with padding
    start_idx = max(0, min(all_chunks) - padding)
    end_idx = min(len(resource.chunks) - 1, max(all_chunks) + padding)

    # Extract all chunks in the range
    text_parts = []
    for chunk_idx in range(start_idx, end_idx + 1):
        chunk_start_pos, chunk_end_pos = resource.chunks[chunk_idx]
        chunk_text = resource.text[chunk_start_pos:chunk_end_pos]
        text_parts.append(chunk_text)

    return "\n".join(text_parts)


def make_entity_pair_key(
    entity1: EntityMention, entity2: EntityMention
) -> EntityPairKey:
    """Create consistent EntityPairKey for two entities.

    Orders entities lexicographically by their kinds to ensure consistent
    pairing (e.g., always gene-disease, not disease-gene).

    Parameters:
        entity1: First entity
        entity2: Second entity

    Returns:
        EntityPairKey with entities ordered by kind
    """
    # Order by kind lexicographically
    if entity1.kind < entity2.kind:
        return EntityPairKey(entity1_name=entity1.name, entity2_name=entity2.name)
    elif entity1.kind > entity2.kind:
        return EntityPairKey(entity1_name=entity2.name, entity2_name=entity1.name)
    else:
        # Same kind - order by name lexicographically
        if entity1.name < entity2.name:
            return EntityPairKey(entity1_name=entity1.name, entity2_name=entity2.name)
        else:
            return EntityPairKey(entity1_name=entity2.name, entity2_name=entity1.name)
