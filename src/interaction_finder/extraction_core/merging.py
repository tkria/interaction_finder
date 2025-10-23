"""
Entity merging and deduplication.

Provides pure functional interface for combining entities extracted from multiple
documents, handling aliases and quote consolidation.
"""

from typing import Dict, List

# Import with TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction_graph_v2.models import EntityWithQuotes


def merge_entities(
    entities: List["EntityWithQuotes"],
) -> Dict[str, "EntityWithQuotes"]:
    """
    Merge duplicate entities across resources by name.

    Deduplicates entities with the same name (case-sensitive) by combining their
    quotes and aliases. Useful for merging entities extracted from multiple documents
    into a single canonical entity per name.

    Args:
        entities: List of entities to merge (potentially from multiple resources)

    Returns:
        Dictionary mapping entity name to merged EntityWithQuotes. Each merged entity
        contains all quotes from all occurrences and deduplicated aliases.

    Note:
        - Entity name matching is **case-sensitive** (BRCA1 ≠ brca1)
        - First occurrence determines kind and initial confidence
        - Quotes from all occurrences are combined (order not guaranteed)
        - Aliases are deduplicated (converted to set, then back to list)
        - If entities with same name have different kinds, raises ValueError

    Example:
        ```python
        from interaction_finder.extraction_core import merge_entities
        from interaction_finder.extraction_graph_v2.models import EntityWithQuotes
        from interaction_finder.resources import Resource, ResourceId

        # Entities from two different resources
        resource1 = Resource(id=ResourceId('http://a.com', 1), title='A', text='BRCA1 is a gene.')
        resource2 = Resource(id=ResourceId('http://b.com', 2), title='B', text='BRCA1 encodes a protein.')

        quote1 = resource1.quote('BRCA1 is a gene', similarity_threshold=1.0)
        quote2 = resource2.quote('BRCA1 encodes', similarity_threshold=1.0)

        entity1 = EntityWithQuotes(name='BRCA1', kind='gene', quotes=[quote1])
        entity2 = EntityWithQuotes(name='BRCA1', kind='gene', quotes=[quote2], aliases=['FANCS'])

        # Merge entities by name
        merged = merge_entities([entity1, entity2])
        # Result: {'BRCA1': EntityWithQuotes(name='BRCA1', quotes=[quote1, quote2], aliases=['FANCS'])}
        ```
    """
    # Handle empty input
    if not entities:
        return {}

    # Accumulate merged entities keyed by name
    merged: Dict[str, "EntityWithQuotes"] = {}

    for entity in entities:
        if entity.name not in merged:
            # First occurrence: create a copy to avoid mutating input
            # Use model_copy() to create a shallow copy (quotes list is shared)
            # Then create a new quotes list to avoid mutation
            merged[entity.name] = entity.model_copy(
                update={"quotes": list(entity.quotes), "aliases": list(entity.aliases)}
            )
        else:
            # Duplicate name: merge quotes and aliases
            existing = merged[entity.name]

            # Validate that entities with same name have same kind
            if existing.kind != entity.kind:
                raise ValueError(
                    f"Cannot merge entities with same name '{entity.name}' but different kinds: "
                    f"'{existing.kind}' vs '{entity.kind}'"
                )

            # Extend quotes list with new entity's quotes
            existing.quotes.extend(entity.quotes)

            # Union aliases: convert to set, add new aliases, convert back to list
            existing_aliases_set = set(existing.aliases)
            existing_aliases_set.update(entity.aliases)
            existing.aliases = list(existing_aliases_set)

    return merged
