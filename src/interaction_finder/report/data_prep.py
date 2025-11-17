"""Data preparation for report generation.

Transforms ExtractionResult into report structure with minimal data-attributes.
No JSON generation - all data embedded in HTML structure.
"""

from collections import defaultdict
from typing import Any

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

from interaction_finder.checkpoint import PipelineCheckpoint
from interaction_finder.report.parallel_renderer import render_documents_parallel
from interaction_finder.report.reasoning_renderer import render_all_reasoning_templates


def _sort_assessments_by_date(
    assessments: list[dict[str, Any]],
    doc_idx_map: dict[str, int],
    indexed_docs: list[tuple[int, Any]],
) -> list[dict[str, Any]]:
    """Sort assessments by publication date (newest first), then quote count.

    Args:
        assessments: List of assessment dictionaries
        doc_idx_map: Mapping of resource_id -> doc_idx
        indexed_docs: List of (doc_idx, resource) tuples

    Returns:
        Sorted assessments list
    """
    # Build doc_idx -> resource map for quick lookup
    doc_resources = {idx: resource for idx, resource in indexed_docs}

    def sort_key(assess: dict[str, Any]) -> tuple:
        doc_idx = doc_idx_map.get(assess["resource_id"])
        if doc_idx is None:
            return ("", 0)

        resource = doc_resources.get(doc_idx)
        date = resource.publication_date if resource else ""
        quote_count = len(assess.get("quotes", []))
        # Negative for descending order (newest first, most quotes first)
        # Empty dates sort last
        return (date if date else "", -quote_count)

    return sorted(assessments, key=sort_key, reverse=True)


def prepare_report_data(
    checkpoint: PipelineCheckpoint,
    show_progress: bool = True,
    judgments_override: list | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[int, str],
    dict[str, dict[str, str]],
    list[tuple[int, Any]],
]:
    """Transform PipelineCheckpoint into report data structure.

    Args:
        checkpoint: Pipeline checkpoint containing extraction results
        show_progress: Show progress bars for rendering
        judgments_override: Optional filtered judgments list

    Returns:
        Tuple of (pairs, document_html, reasoning_templates, indexed_docs)
        - pairs: List of pair data dicts with doc indices (not IDs)
        - document_html: Mapping of doc_idx -> pre-rendered HTML string
        - reasoning_templates: Nested dict pair_idx -> template_type -> HTML
        - indexed_docs: List of (doc_idx, resource) tuples
    """
    if not checkpoint.extraction:
        raise ValueError("Checkpoint does not contain extraction results")

    # Use provided judgments list when filters were applied upstream
    judgments = (
        judgments_override
        if judgments_override is not None
        else checkpoint.extraction.judgments
    )

    # Group judgments by entity pair (to collapse different relationships)
    pair_groups: dict[tuple[str, str], list] = defaultdict(list)
    for judgment in judgments:
        # Create a sorted key for the entity pair
        entities_key = tuple(sorted([judgment.entity1.name, judgment.entity2.name]))
        pair_groups[entities_key].append(judgment)

    # Transform grouped judgments to compact representation
    pairs = []
    for group in pair_groups.values():
        # If multiple judgments for same entity pair, create variants
        if len(group) > 1:
            # Main judgment (first one)
            main_judgment = group[0]

            # Collect data from all judgments in group
            all_doc_ids = set()
            all_assessments = []
            total_quotes = 0

            for judgment in group:
                all_doc_ids.update(
                    assess.resource_id.id for assess in judgment.assessments
                )
                total_quotes += sum(
                    len(assess.quotes) for assess in judgment.assessments
                )
                all_assessments.extend(judgment.assessments)

            # Build relationship variants list
            variants = []
            for judgment in group:
                doc_count = len(
                    {assess.resource_id.id for assess in judgment.assessments}
                )
                quote_count = sum(len(assess.quotes) for assess in judgment.assessments)
                variants.append(
                    {
                        "relationship": judgment.relationship,
                        "confidence": judgment.confidence,
                        "accepted": judgment.accepted,
                        "reasoning": judgment.reasoning,
                        "doc_count": doc_count,
                        "quote_count": quote_count,
                    }
                )

            # Use first judgment's entities
            judgment = main_judgment
        else:
            judgment = group[0]
            variants = None
            all_doc_ids = {assess.resource_id.id for assess in judgment.assessments}
            total_quotes = sum(len(assess.quotes) for assess in judgment.assessments)
            all_assessments = judgment.assessments

        # Transform assessments for this pair
        assessments = []
        for assess in all_assessments:
            # Get resource information
            resource = checkpoint.resources.get(assess.resource_id.url)
            if resource is None:
                continue

            # Transform quotes with spans
            quote_data = [
                {
                    "text": q.query_text,
                    "spans": q.spans,
                    "fuzzy_corrected": q.fuzzy_corrected,
                }
                for q in assess.quotes
            ]

            assessments.append(
                {
                    "resource_id": assess.resource_id.id,
                    "title": resource.title or "Untitled",
                    "url": resource.id.url,
                    "confidence": assess.confidence,
                    "reasoning": assess.reasoning,
                    "relationship": assess.relationship,
                    "quotes": quote_data,
                }
            )

        pair_data = {
            "entity1": {
                "name": judgment.entity1.name,
                "kind": judgment.entity1.kind,
                "aliases": judgment.entity1.aliases,
            },
            "entity2": {
                "name": judgment.entity2.name,
                "kind": judgment.entity2.kind,
                "aliases": judgment.entity2.aliases,
            },
            "relationship": judgment.relationship,
            "confidence": judgment.confidence,
            "accepted": judgment.accepted,
            "reasoning": judgment.reasoning,
            "doc_count": len(all_doc_ids),
            "quote_count": total_quotes,
            "assessments": assessments,
        }

        # Add variants if present
        if variants:
            pair_data["variants"] = variants

        pairs.append(pair_data)

    # Sort pairs by: accepted status > confidence > doc count > lexicographic
    confidence_order = {"high": 0, "medium": 1, "low": 2}

    def pair_sort_key(pair):
        # Primary: accepted status (accepted first)
        # Secondary: confidence level (high > medium > low)
        # Tertiary: number of supporting documents (more is better, so negate)
        # Quaternary: entity names lexicographically
        return (
            not pair["accepted"],  # False (accepted) sorts before True (rejected)
            confidence_order.get(pair["confidence"], 3),
            -pair["doc_count"],  # Negate to sort descending
            pair["entity1"]["name"].lower(),
            pair["entity2"]["name"].lower(),
        )

    pairs.sort(key=pair_sort_key)

    # Sort assessments within each pair (do this before document rendering
    # since documents dict is needed for sorting)
    # Note: We'll do a second pass after documents are rendered
    # to ensure documents dict is available for accurate sorting

    # Create document index mapping: resource_id -> sequential doc_idx
    # First, collect all unique document IDs from assessments
    unique_doc_ids = set()
    for pair in pairs:
        for assess in pair["assessments"]:
            unique_doc_ids.add(assess["resource_id"])

    # Assign sequential indices to documents
    doc_idx_map: dict[str, int] = {}
    indexed_docs: list[tuple[int, Any]] = []

    for doc_idx, resource in enumerate(checkpoint.resources.resources):
        if resource.id.id in unique_doc_ids:
            doc_idx_map[resource.id.id] = doc_idx
            indexed_docs.append((doc_idx, resource))

    # Update assessments to use doc indices instead of IDs
    for pair in pairs:
        for assess in pair["assessments"]:
            assess["doc_idx"] = doc_idx_map[assess["resource_id"]]

    # Sort assessments within each pair by publication date
    for pair in pairs:
        pair["assessments"] = _sort_assessments_by_date(
            pair["assessments"], doc_idx_map, indexed_docs
        )

    # Build document rendering data structures
    # doc_idx -> [ResourceQuote objects]
    doc_to_quotes: dict[int, list[Any]] = defaultdict(list)
    # doc_idx -> {pair_idx: {entity1, entity2}}
    doc_to_entities: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)

    # Map from sorted (entity1, entity2) to pair_idx for tracking
    pair_idx_map = {}
    for pair_idx, pair in enumerate(pairs):
        e1, e2 = sorted([pair["entity1"]["name"], pair["entity2"]["name"]])
        pair_idx_map[(e1, e2)] = pair_idx

    # Go back to original judgments to get ResourceQuote objects
    for judgment in judgments:
        e1, e2 = sorted([judgment.entity1.name, judgment.entity2.name])
        pair_idx = pair_idx_map.get((e1, e2))

        if pair_idx is None:
            continue

        # Collect quotes and entities from each assessment
        for assess in judgment.assessments:
            doc_id = assess.resource_id.id
            doc_idx = doc_idx_map.get(doc_id)

            if doc_idx is None:
                continue

            # Store the actual ResourceQuote objects
            for quote in assess.quotes:
                if quote not in doc_to_quotes[doc_idx]:
                    doc_to_quotes[doc_idx].append(quote)

            # Store entities for this pair in this document
            if pair_idx not in doc_to_entities[doc_idx]:
                doc_to_entities[doc_idx][pair_idx] = {
                    "entity1": {
                        "name": judgment.entity1.name,
                        "kind": judgment.entity1.kind,
                        "aliases": judgment.entity1.aliases,
                    },
                    "entity2": {
                        "name": judgment.entity2.name,
                        "kind": judgment.entity2.kind,
                        "aliases": judgment.entity2.aliases,
                    },
                }

    # Pre-render documents in parallel using multiprocessing
    if show_progress and indexed_docs:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Rendering {len(indexed_docs)} documents...",
                total=len(indexed_docs),
            )

            def update_progress():
                progress.update(task, advance=1)

            document_html = render_documents_parallel(
                indexed_docs,
                doc_to_quotes,
                doc_to_entities,
                progress_callback=update_progress,
            )
    else:
        document_html = render_documents_parallel(
            indexed_docs,
            doc_to_quotes,
            doc_to_entities,
            progress_callback=None,
        )

    # Generate reasoning templates for all pairs
    reasoning_templates = render_all_reasoning_templates(pairs, doc_idx_map)

    return pairs, document_html, reasoning_templates, indexed_docs
