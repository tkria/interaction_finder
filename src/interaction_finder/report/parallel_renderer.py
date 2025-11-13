"""Parallel document rendering using multiprocessing.

Speeds up report generation by distributing document rendering across multiple
worker processes, avoiding Python's GIL limitations.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from multiprocessing import Process, Queue
from typing import Any

from interaction_finder.report.html_renderer import (
    DocumentAnnotator,
    MarkdownToHTMLRenderer,
)
from interaction_finder.resources import Resource, ResourceQuote


def _render_document_worker(
    input_queue: Queue,
    output_queue: Queue,
) -> None:
    """Worker function that renders documents from queue.

    Processes work items from input_queue, performs rendering, and sends
    results and progress updates to output_queue.

    Args:
        input_queue: Queue containing (resource, doc_quotes, doc_entities) tuples
        output_queue: Queue for sending ('progress', None) or ('result', data) messages
    """
    while True:
        # Get work item from queue
        item = input_queue.get()

        # Poison pill check - None signals worker to exit
        if item is None:
            break

        resource, doc_quotes, doc_entities = item

        try:
            # Render markdown to HTML
            renderer = MarkdownToHTMLRenderer(resource.text)
            renderer.render()

            # Annotate HTML with quotes and entities
            annotator = DocumentAnnotator(resource, renderer)
            prerendered = annotator.annotate(doc_quotes, doc_entities)

            # Extract metadata in JSON-serializable format
            metadata = {
                "id": resource.id.id,
                "url": resource.id.url,
                "title": resource.title or "Untitled",
                "quote_map": {
                    quote_id: {
                        "span_id": quote_meta.span_id,
                        "pair_indices": quote_meta.pair_indices,
                        "original_spans": quote_meta.original_spans,
                        "html_spans": quote_meta.html_spans,
                    }
                    for quote_id, quote_meta in prerendered.quote_map.items()
                },
                "entity_map": {
                    entity_id: {
                        "span_id": entity_meta.span_id,
                        "name": entity_meta.name,
                        "kind": entity_meta.kind,
                        "aliases": entity_meta.aliases,
                        "pair_indices": entity_meta.pair_indices,
                    }
                    for entity_id, entity_meta in prerendered.entity_map.items()
                },
            }

            # Send result back (progress will be updated when result is received)
            output_queue.put(("result", (resource.id.id, prerendered.html, metadata)))

        except Exception as e:
            # Send error back to main process
            output_queue.put(("error", (resource.id.id, e)))


def render_documents_parallel(
    docs_to_render: list[Resource],
    doc_to_quotes: dict[str, list[ResourceQuote]],
    doc_to_entities: dict[str, dict[int, dict[str, Any]]],
    progress_callback: Callable[[], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Render documents in parallel using multiprocessing.

    Args:
        docs_to_render: List of Resource objects to render
        doc_to_quotes: Mapping of doc_id -> list of ResourceQuote objects
        doc_to_entities: Mapping of doc_id -> {pair_idx: {entity1, entity2}}
        progress_callback: Optional callback function called after each document completes

    Returns:
        Tuple of (documents, document_html):
        - documents: Dict mapping doc_id -> metadata dict (JSON-serializable)
        - document_html: Dict mapping doc_id -> pre-rendered HTML string

    Raises:
        Exception: If any worker encounters an error during rendering
    """
    # Calculate worker count: cpu_count * 7/8 - 1, minimum 1
    cpu_count = os.cpu_count() or 1
    num_workers = max(1, int(cpu_count * 7 / 8 - 1))

    # Create queues for inter-process communication
    input_queue: Queue = Queue()
    output_queue: Queue = Queue()

    # Populate input queue with work items
    for resource in docs_to_render:
        doc_id = resource.id.id
        doc_quotes = doc_to_quotes.get(doc_id, [])
        doc_entities = doc_to_entities.get(doc_id, {})
        input_queue.put((resource, doc_quotes, doc_entities))

    # Add poison pills (one per worker) to signal completion
    for _ in range(num_workers):
        input_queue.put(None)

    # Start worker processes
    workers = []
    for _ in range(num_workers):
        worker = Process(
            target=_render_document_worker, args=(input_queue, output_queue)
        )
        worker.start()
        workers.append(worker)

    # Collect results from output queue
    documents = {}
    document_html = {}
    completed = 0
    errors = []

    while completed < len(docs_to_render):
        msg_type, msg_data = output_queue.get()

        if msg_type == "result":
            doc_id, html, metadata = msg_data
            document_html[doc_id] = html
            documents[doc_id] = metadata
            # Increment progress when result is received
            completed += 1
            if progress_callback:
                progress_callback()

        elif msg_type == "error":
            doc_id, error = msg_data
            errors.append((doc_id, error))
            completed += 1
            if progress_callback:
                progress_callback()

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    # Raise first error if any occurred
    if errors:
        doc_id, error = errors[0]
        raise RuntimeError(
            f"Document rendering failed for {doc_id}: {error}"
        ) from error

    return documents, document_html
