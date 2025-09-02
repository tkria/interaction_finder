"""
State management for the extraction graph pipeline.

This module defines the shared, mutable state that tracks processing
throughout the graph execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import (
        IndividualEntityContext,
        IndividualEntityAssessment,
        EntityAggregationOut,
    )
    from ..resources import ResourcePool, Resource


@dataclass
class DocumentGroup:
    """
    A group of related documents with their chunks and metadata.

    This class maintains the mapping from URLs to their content chunks
    while preserving document attribution throughout processing.
    """

    urls: List[str]
    chunks: Dict[str, List[Dict[str, Any]]]  # url -> chunks with metadata
    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # url -> metadata

    def get_total_chunks(self) -> int:
        """Get total number of chunks across all documents in group."""
        return sum(len(chunks) for chunks in self.chunks.values())

    def get_chunk_text(self, url: str, chunk_index: int) -> str:
        """Get text content for a specific chunk."""
        if url not in self.chunks:
            raise ValueError(f"URL {url} not found in document group")

        chunks = self.chunks[url]
        if chunk_index >= len(chunks):
            raise ValueError(f"Chunk index {chunk_index} out of range for URL {url}")

        return chunks[chunk_index].get("text", "")

    def combine_all_text(self, max_chars: int = 50000) -> str:
        """
        Combine text from all chunks in the group with source attribution.

        Args:
            max_chars: Maximum characters to include

        Returns:
            Combined text with URL markers for source tracking
        """
        combined = []
        total_chars = 0

        for url in self.urls:
            if url not in self.chunks:
                continue

            # Add URL marker
            url_header = f"\n\n[Document: {url}]\n"
            if total_chars + len(url_header) > max_chars:
                break

            combined.append(url_header)
            total_chars += len(url_header)

            # Add chunks from this document
            for chunk in self.chunks[url]:
                chunk_text = chunk.get("text", "")
                if total_chars + len(chunk_text) > max_chars:
                    # Truncate the last chunk if needed
                    remaining = max_chars - total_chars
                    if remaining > 100:  # Only add if meaningful amount left
                        combined.append(chunk_text[:remaining] + "...")
                    break

                combined.append(chunk_text)
                total_chars += len(chunk_text)

            if total_chars >= max_chars:
                break

        return "".join(combined)

    def populate_resource_pool(
        self, resource_pool: "ResourcePool"
    ) -> Dict[str, "Resource"]:
        """
        Populate a ResourcePool with documents from this group.

        Args:
            resource_pool: ResourcePool to populate

        Returns:
            Dictionary mapping URLs to Resource objects for easy lookup
        """
        from ..resources import ResourcePool  # Import here to avoid circular imports

        resources = {}

        for url in self.urls:
            if url not in self.chunks:
                continue

            # Register the URL to get a ResourceId
            resource_id = resource_pool.register(url)

            # Combine all chunks for this document
            document_text = ""
            for chunk in self.chunks[url]:
                chunk_text = chunk.get("text", "")
                if document_text:
                    document_text += "\n\n"
                document_text += chunk_text

            # Get document title from metadata if available
            title = self.metadata.get(url, {}).get("title", f"Document from {url}")

            # Add content to the resource pool
            resource = resource_pool.add_content(resource_id, title, document_text)
            resources[url] = resource

        return resources


@dataclass
class EntityExtractionState:
    """
    Shared mutable state for the extraction graph pipeline.

    This tracks all processing state across nodes, following the
    pydantic-graph pattern of centralized run-time memory.
    """

    # Input data
    document_groups: List[DocumentGroup] = field(default_factory=list)
    current_group_index: int = 0

    # Entity configuration
    entity_kinds: List[str] = field(default_factory=list)
    relation_type: str = "interaction"

    # Extracted entities organized by kind
    extracted_entities: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Processing results
    individual_assessments: List[Dict[str, Any]] = field(default_factory=list)
    entity_pairs: List[Dict[str, Any]] = field(default_factory=list)
    aggregated_results: Dict[str, Any] = field(default_factory=dict)

    # Individual entity processing
    individual_entity_contexts: List["IndividualEntityContext"] = field(
        default_factory=list
    )
    individual_entity_assessments: List["IndividualEntityAssessment"] = field(
        default_factory=list
    )
    aggregation_result: Optional["EntityAggregationOut"] = None

    # Processing control for individual processing
    current_entity_batch: List[Dict[str, Any]] = field(default_factory=list)
    entities_processed: int = 0
    batch_size: int = 3

    # Processing metadata
    processing_notes: Dict[str, Any] = field(default_factory=dict)
    node_history: List[str] = field(default_factory=list)
    current_node: Optional[str] = None

    def __post_init__(self):
        """Initialize entity storage for all configured kinds."""
        for kind in self.entity_kinds:
            if kind not in self.extracted_entities:
                self.extracted_entities[kind] = []

    def get_current_group(self) -> Optional[DocumentGroup]:
        """Get the currently processing document group."""
        if 0 <= self.current_group_index < len(self.document_groups):
            return self.document_groups[self.current_group_index]
        return None

    def has_next_group(self) -> bool:
        """Check if there are more document groups to process."""
        return self.current_group_index + 1 < len(self.document_groups)

    def advance_to_next_group(self) -> bool:
        """
        Advance to the next document group.

        Returns:
            True if advanced successfully, False if no more groups
        """
        if self.has_next_group():
            self.current_group_index += 1
            # Clear per-group state
            self.extracted_entities = {kind: [] for kind in self.entity_kinds}
            self.individual_assessments = []
            # Clear individual processing state
            self.individual_entity_contexts = []
            self.individual_entity_assessments = []
            self.aggregation_result = None
            self.entities_processed = 0
            return True
        return False

    def add_entity(self, kind: str, entity_data: Dict[str, Any]) -> None:
        """Add an extracted entity of the specified kind."""
        if kind not in self.entity_kinds:
            raise ValueError(f"Unknown entity kind: {kind}")

        if kind not in self.extracted_entities:
            self.extracted_entities[kind] = []

        self.extracted_entities[kind].append(entity_data)

    def get_entity_count(self, kind: Optional[str] = None) -> int:
        """
        Get count of extracted entities.

        Args:
            kind: Specific entity kind, or None for total count

        Returns:
            Number of entities
        """
        if kind:
            return len(self.extracted_entities.get(kind, []))
        return sum(len(entities) for entities in self.extracted_entities.values())

    def log_node_entry(self, node_name: str) -> None:
        """Log entry into a graph node."""
        self.node_history.append(node_name)
        self.current_node = node_name

    def add_processing_note(self, key: str, value: Any) -> None:
        """Add a processing note for debugging/analysis."""
        self.processing_notes[key] = value

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of processing state for debugging."""
        current_group = self.get_current_group()

        return {
            "current_group_index": self.current_group_index,
            "total_groups": len(self.document_groups),
            "current_group_urls": current_group.urls if current_group else [],
            "current_group_chunks": current_group.get_total_chunks()
            if current_group
            else 0,
            "entity_kinds": self.entity_kinds,
            "entity_counts": {
                kind: len(entities)
                for kind, entities in self.extracted_entities.items()
            },
            "assessments": len(self.individual_assessments),
            "pairs": len(self.entity_pairs),
            "current_node": self.current_node,
            "node_history": self.node_history,
        }
