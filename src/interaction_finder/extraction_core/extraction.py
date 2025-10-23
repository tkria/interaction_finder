"""
Entity extraction from individual documents.

Provides pure functional interface for extracting named entities from scientific
documents with complete provenance tracking via ResourceQuotes.
"""

from typing import List, Union
from pydantic_ai.models import Model

# Import with TYPE_CHECKING to avoid circular imports at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..extraction_graph_v2.models import EntityWithQuotes
    from ..resources import Resource


async def extract_from_resource(
    resource: "Resource",
    entity_kinds: List[str],
    model: Union[Model, str],
    task_context: str,
) -> List["EntityWithQuotes"]:
    """
    Extract entities of specified kinds from a single resource.

    Processes a document to identify and extract biological entities (genes, proteins,
    diseases, etc.) with complete provenance tracking. Each extracted entity includes
    all its occurrences with exact character positions via ResourceQuotes.

    Args:
        resource: Document resource to extract from (with content in markdown format)
        entity_kinds: List of entity types to extract (e.g., ["gene", "disease"])
        model: LLM model for extraction (pydantic-ai Model or string like "openai:gpt-4o")
        task_context: Biological context to guide extraction (e.g., "breast cancer research")

    Returns:
        List of EntityWithQuotes, each with complete occurrence tracking

    Raises:
        ValueError: If resource has no content or entity_kinds is empty
        RuntimeError: If LLM extraction fails after retries

    Example:
        ```python
        entities = await extract_from_resource(
            resource=doc,
            entity_kinds=["gene", "protein"],
            model="openai:gpt-4o",
            task_context="cancer signaling pathways"
        )
        for entity in entities:
            print(f"{entity.name} ({entity.kind}): {entity.total_occurrences} occurrences")
        ```
    """
    raise NotImplementedError("extract_from_resource will be implemented in Task 02")
