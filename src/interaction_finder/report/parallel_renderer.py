"""Parallel document rendering using multiprocessing.

Speeds up report generation by distributing document rendering across multiple
worker processes, avoiding Python's GIL limitations.
"""

from __future__ import annotations

import html as html_module
import logging
import os
from collections.abc import Callable
from multiprocessing import Process, Queue
from typing import Any

from interaction_finder.report.html_renderer import (
    DocumentAnnotator,
    DocumentQuoteEntry,
    MarkdownToHTMLRenderer,
)
from interaction_finder.resources import Resource

logger = logging.getLogger(__name__)


def _render_document_worker(
    input_queue: Queue,
    output_queue: Queue,
) -> None:
    """Worker function that renders documents from queue.

    Processes work items from input_queue, performs rendering, and sends
    results and progress updates to output_queue.

    Args:
        input_queue: Queue containing (doc_idx, resource, doc_quotes, doc_entities) tuples
        output_queue: Queue for sending ('progress', None) or ('result', data) messages
    """
    while True:
        # Get work item from queue
        item = input_queue.get()

        # Poison pill check - None signals worker to exit
        if item is None:
            break

        doc_idx, resource, doc_quotes, doc_entities = item

        try:
            # Render markdown to HTML
            renderer = MarkdownToHTMLRenderer(resource.text)
            renderer.render()

            # Annotate HTML with quotes and entities
            annotator = DocumentAnnotator(resource, renderer)
            prerendered = annotator.annotate(doc_idx, doc_quotes, doc_entities)

            # Send result back (just HTML, no metadata)
            output_queue.put(("result", (doc_idx, prerendered.html)))

        except Exception as e:
            # Send error back to main process
            output_queue.put(("error", (doc_idx, e)))


def render_documents_parallel(
    indexed_docs: list[tuple[int, Resource]],
    doc_to_quotes: dict[int, list[DocumentQuoteEntry]],
    doc_to_entities: dict[int, dict[int, dict[str, Any]]],
    progress_callback: Callable[[], None] | None = None,
) -> dict[int, str]:
    """Render documents in parallel using multiprocessing.

    Args:
        indexed_docs: List of (doc_idx, Resource) tuples
        doc_to_quotes: Mapping of doc_idx -> list of quote entries
        doc_to_entities: Mapping of doc_idx -> {pair_idx: {entity1, entity2}}
        progress_callback: Optional callback function called after each document completes

    Returns:
        Dict mapping doc_idx -> pre-rendered HTML string

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
    for doc_idx, resource in indexed_docs:
        doc_quotes = doc_to_quotes.get(doc_idx, [])
        doc_entities = doc_to_entities.get(doc_idx, {})
        input_queue.put((doc_idx, resource, doc_quotes, doc_entities))

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
    document_html = {}
    completed = 0
    errors = []

    while completed < len(indexed_docs):
        msg_type, msg_data = output_queue.get()

        if msg_type == "result":
            doc_idx, html = msg_data
            document_html[doc_idx] = html
            # Increment progress when result is received
            completed += 1
            if progress_callback:
                progress_callback()

        elif msg_type == "error":
            doc_idx, error = msg_data
            errors.append((doc_idx, error))
            completed += 1
            if progress_callback:
                progress_callback()

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    # Raise first error if any occurred
    if errors:
        doc_idx, error = errors[0]
        raise RuntimeError(
            f"Document rendering failed for doc index {doc_idx}: {error}"
        ) from error

    return document_html
