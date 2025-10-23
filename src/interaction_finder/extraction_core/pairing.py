"""
Entity pair formation via co-occurrence analysis.

Provides pure functional interface for generating entity pairs based on document
co-occurrence patterns using various strategies.
"""

from typing import List

# Import with TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction_graph_v2.models import EntityWithQuotes
    from ..extraction_graph_v3.models import PairCandidate

from .models import CooccurrenceStrategy


def find_cooccurring_pairs(
    entities: List["EntityWithQuotes"],
    strategy: CooccurrenceStrategy,
    min_cooccurrences: int = 1,
) -> List["PairCandidate"]:
    """
    Generate entity pairs via co-occurrence analysis.

    Identifies entity pairs that appear together in documents based on the
    specified strategy. Each candidate includes co-occurrence count and
    shared resource tracking for provenance.

    Args:
        entities: List of entities to form pairs from
        strategy: Co-occurrence strategy (TIERED, SAME_CHUNK, ADJACENT, DOCUMENT)
        min_cooccurrences: Minimum co-occurrences required to form a pair (default: 1)

    Returns:
        List of PairCandidate objects with co-occurrence metadata

    Example:
        ```python
        # Find pairs using tiered strategy
        pairs = find_cooccurring_pairs(
            entities=merged_entities,
            strategy=CooccurrenceStrategy.TIERED,
            min_cooccurrences=2
        )

        for pair in pairs:
            print(f"{pair.entity_a.name} <-> {pair.entity_b.name}")
            print(f"  Co-occurrences: {pair.co_occurrence_count}")
            print(f"  Strategy: {pair.generation_strategy}")
            print(f"  Shared resources: {len(pair.shared_resources)}")
        ```

    Notes:
        - TIERED strategy tries same_chunk → adjacent → document, stopping early
        - Pairs are deduplicated (A-B same as B-A)
        - Only includes pairs meeting min_cooccurrences threshold
    """
    raise NotImplementedError("find_cooccurring_pairs will be implemented in Task 05")
