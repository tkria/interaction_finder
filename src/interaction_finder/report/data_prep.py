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
from interaction_finder.report.html_renderer import DocumentQuoteEntry
from interaction_finder.report.parallel_renderer import render_documents_parallel
from interaction_finder.report.reasoning_renderer import render_all_reasoning_templates

POLARITY_ORDER = ("supporting", "refuting", "neutral", "irrelevant")
CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _quote_key_for_id(
    spans: list[tuple[int, int]], text: str, fuzzy_corrected: bool
) -> tuple:
    """Generate lookup key for quote ID mapping.

    This key format is shared between data_prep and reasoning_renderer
    to ensure consistent quote ID lookups.
    """
    return (tuple(tuple(span) for span in spans), text, fuzzy_corrected)


def _pair_key(name_a: str, name_b: str) -> tuple[str, str]:
    """Create deterministic key for unordered entity names."""

    return tuple(sorted([name_a, name_b]))


def _iter_assessments_with_polarity(judgment):
    """Yield (assessment, polarity) tuples from a PairJudgment."""

    spread = judgment.spread
    for polarity in POLARITY_ORDER:
        for assessment in getattr(spread, polarity):
            yield assessment, polarity


def _build_pair_entry(judgment, resource_pool) -> dict[str, Any]:
    """Convert a PairJudgment into the lightweight dict used by the report."""

    doc_ids: set[str] = set()
    total_quotes = 0
    polarity_counts = {polarity: 0 for polarity in POLARITY_ORDER}
    polarity_best_conf = {polarity: None for polarity in POLARITY_ORDER}
    assessments: list[dict[str, Any]] = []

    for assessment, polarity in _iter_assessments_with_polarity(judgment):
        doc_ids.add(assessment.resource_id.id)
        total_quotes += len(assessment.quotes)
        polarity_counts[polarity] += 1
        confidence = assessment.confidence

        resource = resource_pool.get(assessment.resource_id)
        if resource is None:
            continue

        quote_data = [
            {
                "text": quote.query_text,
                "spans": quote.spans,
                "fuzzy_corrected": quote.fuzzy_corrected,
            }
            for quote in assessment.quotes
        ]

        assessments.append(
            {
                "resource_id": assessment.resource_id.id,
                "title": resource.title or "Untitled",
                "url": resource.id.url,
                "confidence": assessment.confidence,
                "reasoning": assessment.reasoning,
                "relationship": assessment.relationship,
                "quotes": quote_data,
                "polarity": polarity,
            }
        )

        if confidence:
            current = polarity_best_conf[polarity]
            current_rank = CONFIDENCE_ORDER.get(current, -1) if current else -1
            new_rank = CONFIDENCE_ORDER.get(confidence, -1)
            if new_rank > current_rank:
                polarity_best_conf[polarity] = confidence

    return {
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
        "doc_count": len(doc_ids),
        "quote_count": total_quotes,
        "assessments": assessments,
        "polarity_counts": polarity_counts,
        "polarity_summary": {
            polarity: {
                "count": polarity_counts[polarity],
                "confidence": polarity_best_conf[polarity],
            }
            for polarity in POLARITY_ORDER
        },
        "contentious": bool(
            polarity_counts["supporting"] and polarity_counts["refuting"]
        ),
    }


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


def _group_assessments_by_document(
    assessments: list[dict[str, Any]],
    doc_idx_map: dict[str, int],
    indexed_docs: list[tuple[int, Any]],
) -> list[dict[str, Any]]:
    """Group assessments by document, consolidating multiple extractions from same source.

    Args:
        assessments: Sorted list of assessment dictionaries (with doc_idx already set)
        doc_idx_map: Unused (kept for API compatibility)
        indexed_docs: Unused (kept for API compatibility)

    Returns:
        List of document groups, each with: doc_idx, assessments list, and aggregate metadata
    """
    # Group consecutive assessments by doc_idx (already sorted by date, so groups are together)
    # Use dict to handle non-consecutive same doc_idx (though shouldn't happen after sorting)
    groups_dict: dict[int, list[dict[str, Any]]] = {}
    for assess in assessments:
        doc_idx = assess["doc_idx"]
        groups_dict.setdefault(doc_idx, []).append(assess)

    # Build document groups in original order (order of first occurrence)
    groups = []
    seen = set()
    for assess in assessments:
        doc_idx = assess["doc_idx"]
        if doc_idx in seen:
            continue
        seen.add(doc_idx)

        doc_assessments = groups_dict[doc_idx]
        # Deduplicate quotes across assessments
        unique_quotes = {
            _quote_key_for_id(q["spans"], q["text"], q.get("fuzzy_corrected", False))
            for assess in doc_assessments
            for q in assess.get("quotes", [])
        }
        groups.append(
            {
                "doc_idx": doc_idx,
                "assessments": doc_assessments,
                "total_quotes": len(unique_quotes),
                "relationships": list(
                    dict.fromkeys(  # Preserve order, remove duplicates
                        a["relationship"]
                        for a in doc_assessments
                        if a.get("relationship")
                    )
                ),
            }
        )

    return groups


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

    pair_entries: list[tuple[tuple[str, str], Any, dict[str, Any]]] = []
    for judgment in judgments:
        key = _pair_key(judgment.entity1.name, judgment.entity2.name)
        pair_data = _build_pair_entry(judgment, checkpoint.resources)
        pair_entries.append((key, judgment, pair_data))

    pairs = [entry[2] for entry in pair_entries]

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

    pair_idx_map = {
        _pair_key(pair["entity1"]["name"], pair["entity2"]["name"]): idx
        for idx, pair in enumerate(pairs)
    }
    judgment_refs = [(pair_idx_map[key], judgment) for key, judgment, _ in pair_entries]

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

    # Group assessments by document for each pair
    # This consolidates multiple assessments from the same document
    for pair in pairs:
        pair["document_groups"] = _group_assessments_by_document(
            pair["assessments"], doc_idx_map, indexed_docs
        )

    # Build document rendering data structures
    # doc_idx -> [DocumentQuoteEntry]
    doc_to_quotes: dict[int, list[DocumentQuoteEntry]] = defaultdict(list)
    # Helpers for deduplicating quotes per document
    doc_quote_lookup: dict[int, dict[tuple, DocumentQuoteEntry]] = defaultdict(dict)
    # doc_idx -> {pair_idx: {entity1, entity2}}
    doc_to_entities: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    # (doc_idx, quote_key) -> quote_id for reasoning renderer lookup
    quote_id_map: dict[tuple[int, tuple], str] = {}

    for pair_idx, judgment in judgment_refs:
        for assessment, _ in _iter_assessments_with_polarity(judgment):
            doc_id = assessment.resource_id.id
            doc_idx = doc_idx_map.get(doc_id)
            if doc_idx is None:
                continue
            for quote in assessment.quotes:
                dedup_key = (
                    tuple((start, end) for start, end in quote.spans),
                    quote.query_text,
                    quote.is_disjoint,
                    bool(quote.fuzzy_corrected),
                    quote.original_query,
                )
                quote_lookup = doc_quote_lookup[doc_idx]
                entry = quote_lookup.get(dedup_key)
                if entry is None:
                    quote_idx = len(doc_to_quotes[doc_idx])
                    entry = DocumentQuoteEntry(quote=quote, pair_indices={pair_idx})
                    doc_to_quotes[doc_idx].append(entry)
                    quote_lookup[dedup_key] = entry
                    # Map quote key to assigned ID for reasoning renderer
                    id_key = _quote_key_for_id(
                        quote.spans, quote.query_text, bool(quote.fuzzy_corrected)
                    )
                    quote_id_map[(doc_idx, id_key)] = f"doc-{doc_idx}-quote-{quote_idx}"
                else:
                    entry.pair_indices.add(pair_idx)
            if pair_idx not in doc_to_entities[doc_idx]:
                pair_entities = pairs[pair_idx]
                doc_to_entities[doc_idx][pair_idx] = {
                    "entity1": pair_entities["entity1"],
                    "entity2": pair_entities["entity2"],
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
    reasoning_templates = render_all_reasoning_templates(pairs, quote_id_map)

    return pairs, document_html, reasoning_templates, indexed_docs
