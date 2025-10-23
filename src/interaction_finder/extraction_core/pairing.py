"""
Entity pair formation via co-occurrence analysis.

Provides pure functional interface for generating entity pairs based on document
co-occurrence patterns using various strategies.
"""

from typing import Dict, List, Literal, Optional, Set, Tuple, cast
from itertools import combinations

# Import with TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction_graph_v2.models import EntityWithQuotes
    from ..extraction_graph_v3.models import PairCandidate
    from ..resources import Resource, ResourceQuote

from .models import CooccurrenceStrategy


def _check_same_chunk(quote_a: "ResourceQuote", quote_b: "ResourceQuote") -> bool:
    """
    Check if two quotes have overlapping chunk indices.

    Returns True if any occurrence of quote_a and quote_b share at least one
    chunk index, indicating they appear in the same document chunk.

    Args:
        quote_a: First entity quote
        quote_b: Second entity quote

    Returns:
        True if quotes share any chunk indices, False otherwise
    """
    # Get all chunk indices for both quotes
    chunks_a = set(quote_a.chunk_indices)
    chunks_b = set(quote_b.chunk_indices)

    # Check for intersection
    return bool(chunks_a & chunks_b)


def _check_adjacent_chunks(quote_a: "ResourceQuote", quote_b: "ResourceQuote") -> bool:
    """
    Check if two quotes appear in adjacent chunks.

    Returns True if any chunk indices differ by exactly 1, indicating the
    quotes appear in neighboring document sections.

    Args:
        quote_a: First entity quote
        quote_b: Second entity quote

    Returns:
        True if any chunks are adjacent (differ by 1), False otherwise
    """
    chunks_a = quote_a.chunk_indices
    chunks_b = quote_b.chunk_indices

    # Check all combinations for adjacency
    for chunk_a in chunks_a:
        for chunk_b in chunks_b:
            if abs(chunk_a - chunk_b) == 1:
                return True

    return False


def _get_shared_resources(
    entity_a: "EntityWithQuotes", entity_b: "EntityWithQuotes"
) -> List[str]:
    """
    Get list of resource IDs where both entities appear.

    Args:
        entity_a: First entity
        entity_b: Second entity

    Returns:
        List of resource ID strings shared by both entities
    """
    # Extract resource IDs from quotes
    resources_a = {quote.resource.id.id for quote in entity_a.quotes}
    resources_b = {quote.resource.id.id for quote in entity_b.quotes}

    # Return intersection as sorted list for deterministic output
    return sorted(resources_a & resources_b)


def _check_cooccurrence(
    entity_a: "EntityWithQuotes",
    entity_b: "EntityWithQuotes",
    resource_id: str,
    strategy: CooccurrenceStrategy,
    allowed_levels: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """
    Check if two entities co-occur in a resource based on strategy.

    Args:
        entity_a: First entity
        entity_b: Second entity
        resource_id: Resource ID to check
        strategy: Co-occurrence strategy to apply

    Returns:
        Tuple of (co_occurs, generation_strategy) where co_occurs is True if
        entities meet the strategy criteria, and generation_strategy indicates
        the specific level at which co-occurrence was found
    """
    allowed = allowed_levels or {"same_chunk", "adjacent_chunks", "document_level"}

    # Get quotes for this resource
    quotes_a = [q for q in entity_a.quotes if q.resource.id.id == resource_id]
    quotes_b = [q for q in entity_b.quotes if q.resource.id.id == resource_id]

    if not quotes_a or not quotes_b:
        return False, ""

    # Check based on strategy
    if strategy == CooccurrenceStrategy.SAME_CHUNK:
        # Only same-chunk co-occurrence
        if "same_chunk" not in allowed:
            return False, ""
        for qa in quotes_a:
            for qb in quotes_b:
                if _check_same_chunk(qa, qb):
                    return True, "same_chunk"
        return False, ""

    elif strategy == CooccurrenceStrategy.ADJACENT:
        # Same-chunk OR adjacent chunks
        if "same_chunk" in allowed:
            for qa in quotes_a:
                for qb in quotes_b:
                    if _check_same_chunk(qa, qb):
                        return True, "same_chunk"

        if "adjacent_chunks" in allowed:
            for qa in quotes_a:
                for qb in quotes_b:
                    if _check_adjacent_chunks(qa, qb):
                        return True, "adjacent_chunks"
        return False, ""

    elif strategy == CooccurrenceStrategy.DOCUMENT:
        # Any co-occurrence in document (already verified by shared resources)
        if "document_level" in allowed:
            return True, "document_level"
        return False, ""

    elif strategy == CooccurrenceStrategy.TIERED:
        # Try same_chunk first, then adjacent, then document
        if "same_chunk" in allowed:
            for qa in quotes_a:
                for qb in quotes_b:
                    if _check_same_chunk(qa, qb):
                        return True, "same_chunk"

        if "adjacent_chunks" in allowed:
            for qa in quotes_a:
                for qb in quotes_b:
                    if _check_adjacent_chunks(qa, qb):
                        return True, "adjacent_chunks"

        if "document_level" in allowed:
            return True, "document_level"

        return False, ""

    return False, ""


def find_cooccurring_pairs(
    entities: Dict[str, "EntityWithQuotes"],
    resources: List["Resource"],
    strategy: CooccurrenceStrategy,
    allowed_levels: Optional[Set[str]] = None,
    include_same_kind_pairs: bool = False,
) -> List["PairCandidate"]:
    """
    Generate entity pairs via co-occurrence analysis.

    Identifies entity pairs that appear together in documents based on the
    specified strategy. Each candidate includes co-occurrence count and
    shared resource tracking for provenance. Same-kind pairs (gene-gene,
    disease-disease) are excluded by default but can be included via configuration.

    Args:
        entities: Dictionary mapping entity names to EntityWithQuotes objects
        resources: List of Resource objects being analyzed
        strategy: Co-occurrence strategy (TIERED, SAME_CHUNK, ADJACENT, DOCUMENT)
        allowed_levels: Optional set of generation strategies ("same_chunk", "adjacent_chunks",
            "document_level") to permit when evaluating co-occurrence. Defaults to allowing all.
        include_same_kind_pairs: Include pairs where both entities share the same kind (default: False)

    Returns:
        List of PairCandidate objects with co-occurrence metadata

    Example:
        ```python
        # Find pairs using tiered strategy
        pairs = find_cooccurring_pairs(
            entities=merged_entities,
            resources=resource_list,
            strategy=CooccurrenceStrategy.TIERED
        )

        for pair in pairs:
            print(f"{pair.entity_a.name} <-> {pair.entity_b.name}")
            print(f"  Co-occurrences: {pair.co_occurrence_count}")
            print(f"  Strategy: {pair.generation_strategy}")
            print(f"  Shared resources: {len(pair.shared_resources)}")
        ```

    Notes:
        - TIERED strategy tries same_chunk → adjacent → document for each pair
        - Pairs are deduplicated (A-B same as B-A) using canonical ordering
        - Same-kind pairs can be included via include_same_kind_pairs=True
        - Empty inputs return empty list (no pairs generated)
    """
    # Import here to avoid circular dependency at module load time
    from ..extraction_graph_v3.models import PairCandidate

    # Handle empty inputs
    if not entities or not resources:
        return []

    candidates: List[PairCandidate] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    # Convert entities dict to list for combinations
    entity_list = list(entities.values())

    # Generate all possible entity pairs (combinations, not permutations)
    for entity_a, entity_b in combinations(entity_list, 2):
        # Filter out same-kind pairs
        if entity_a.kind == entity_b.kind and not include_same_kind_pairs:
            continue

        # Get canonical pair key (alphabetically sorted names)
        sorted_names = sorted([entity_a.name, entity_b.name])
        pair_key: Tuple[str, str] = (sorted_names[0], sorted_names[1])

        # Skip if already processed
        if pair_key in seen_pairs:
            continue

        # Find shared resources
        shared_resources = _get_shared_resources(entity_a, entity_b)

        if not shared_resources:
            continue

        # Check co-occurrence in each shared resource
        co_occurrence_count = 0
        generation_strategy = ""

        for resource_id in shared_resources:
            co_occurs, strat = _check_cooccurrence(
                entity_a, entity_b, resource_id, strategy, allowed_levels
            )
            if co_occurs:
                co_occurrence_count += 1
                # Use the most specific strategy found (prefer same_chunk over adjacent over document)
                if not generation_strategy:
                    generation_strategy = strat
                elif strat == "same_chunk" and generation_strategy != "same_chunk":
                    generation_strategy = strat
                elif (
                    strat == "adjacent_chunks"
                    and generation_strategy == "document_level"
                ):
                    generation_strategy = strat

        # Create pair candidate if co-occurrence found
        if co_occurrence_count > 0:
            # Ensure consistent ordering (alphabetically by name)
            if entity_a.name <= entity_b.name:
                pair_entity_a, pair_entity_b = entity_a, entity_b
            else:
                pair_entity_a, pair_entity_b = entity_b, entity_a

            candidate = PairCandidate(
                entity_a=pair_entity_a,
                entity_b=pair_entity_b,
                co_occurrence_count=co_occurrence_count,
                shared_resources=shared_resources,
                generation_strategy=cast(
                    Literal[
                        "same_chunk",
                        "adjacent_chunks",
                        "document_level",
                        "assessment_suggested",
                        "both",
                    ],
                    generation_strategy,
                ),
            )
            candidates.append(candidate)
            seen_pairs.add(pair_key)

    return candidates
