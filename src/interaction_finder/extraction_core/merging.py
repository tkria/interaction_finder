"""
Entity merging and deduplication.

Provides pure functional interface for combining entities extracted from multiple
documents, handling aliases and quote consolidation.
"""

from typing import List

# Import with TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction_graph_v2.models import EntityWithQuotes


def merge_entities(
    entities: List["EntityWithQuotes"],
    case_sensitive: bool = False,
) -> List["EntityWithQuotes"]:
    """
    Merge duplicate entities and consolidate their quotes.

    Combines entities with the same name (or aliases) across multiple documents,
    preserving all quotes and occurrence information. Handles case variations,
    aliases, and quote deduplication.

    Args:
        entities: List of entities to merge (can be from multiple documents)
        case_sensitive: Whether to treat entity names case-sensitively (default: False)

    Returns:
        List of merged entities with consolidated quotes and aliases

    Example:
        ```python
        # Entities from multiple documents
        doc1_entities = [
            EntityWithQuotes(name="BRCA1", kind="gene", quotes=[...]),
            EntityWithQuotes(name="TP53", kind="gene", quotes=[...])
        ]
        doc2_entities = [
            EntityWithQuotes(name="brca1", kind="gene", quotes=[...]),  # Case variant
            EntityWithQuotes(name="p53", kind="gene", aliases=["TP53"], quotes=[...])  # Alias
        ]

        merged = merge_entities(doc1_entities + doc2_entities)
        # Result: BRCA1 (merged cases), TP53 (merged with p53 alias)
        ```
    """
    raise NotImplementedError("merge_entities will be implemented in Task 03")
