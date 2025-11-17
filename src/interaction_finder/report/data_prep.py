"""Data preparation for report generation.

Transforms ExtractionResult into JSON-serializable report data structures
optimized for frontend rendering and filtering.
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


def prepare_report_data(
    checkpoint: PipelineCheckpoint,
    show_progress: bool = True,
    judgments_override: list | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Transform PipelineCheckpoint into report data structure.

    Args:
        checkpoint: Pipeline checkpoint containing extraction results

    Returns:
        Tuple of (json_data, document_html_map)
        - json_data: JSON-serializable dict with metadata for frontend
        - document_html_map: Mapping of doc_id -> pre-rendered HTML string
    """
    if not checkpoint.extraction:
        raise ValueError("Checkpoint does not contain extraction results")

    # Use provided judgments list when filters were applied upstream
    judgments = (
        judgments_override
        if judgments_override is not None
        else checkpoint.extraction.judgments
    )

    # Group judgments by entity pair (merge different relationships)
    pair_groups = defaultdict(list)
    for judgment in judgments:
        # Create canonical pair key (sorted entity names)
        e1, e2 = sorted([judgment.entity1.name, judgment.entity2.name])
        pair_key = (e1, e2)
        pair_groups[pair_key].append(judgment)

    # Extract entity statistics
    entity_kinds: dict[str, set[str]] = defaultdict(set)
    for judgment in judgments:
        entity_kinds[judgment.entity1.kind].add(judgment.entity1.name)
        entity_kinds[judgment.entity2.kind].add(judgment.entity2.name)

    entity_stats = {kind: len(names) for kind, names in entity_kinds.items()}

    # Confidence distribution
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    for judgment in judgments:
        confidence_counts[judgment.confidence] += 1

    # Build entity index: entity name -> list of pair indices
    entity_index: dict[str, list[int]] = defaultdict(list)
    for idx, judgment in enumerate(judgments):
        entity_index[judgment.entity1.name].append(idx)
        entity_index[judgment.entity2.name].append(idx)

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

    # Build document index with pre-rendered HTML
    # Collect document references directly from judgments
    doc_to_quotes: dict[str, list[Any]] = defaultdict(
        list
    )  # doc_id -> [ResourceQuote objects]
    doc_to_entities: dict[str, dict[int, dict[str, Any]]] = defaultdict(
        dict
    )  # doc_id -> {pair_idx: {entity1, entity2}}

    # Map from sorted (entity1, entity2) to pair_idx for tracking
    # Use sorted names to handle different orderings
    pair_idx_map = {}
    for pair_idx, pair in enumerate(pairs):
        # Create sorted key for matching
        e1, e2 = sorted([pair["entity1"]["name"], pair["entity2"]["name"]])
        pair_idx_map[(e1, e2)] = pair_idx

    # Go back to original judgments to get ResourceQuote objects
    for judgment in judgments:
        # Create sorted key to match against pair_idx_map
        e1, e2 = sorted([judgment.entity1.name, judgment.entity2.name])
        pair_idx = pair_idx_map.get((e1, e2))

        if pair_idx is None:
            continue

        # Collect quotes and entities from each assessment
        for assess in judgment.assessments:
            doc_id = assess.resource_id.id

            # Store the actual ResourceQuote objects
            for quote in assess.quotes:
                if quote not in doc_to_quotes[doc_id]:
                    doc_to_quotes[doc_id].append(quote)

            # Store entities for this pair in this document
            if pair_idx not in doc_to_entities[doc_id]:
                doc_to_entities[doc_id][pair_idx] = {
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
    # Count how many documents need rendering
    docs_to_render = [
        r for r in checkpoint.resources.resources if r.id.id in doc_to_quotes
    ]

    if show_progress and docs_to_render:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Rendering {len(docs_to_render)} documents...",
                total=len(docs_to_render),
            )

            def update_progress():
                progress.update(task, advance=1)

            documents, document_html = render_documents_parallel(
                docs_to_render,
                doc_to_quotes,
                doc_to_entities,
                progress_callback=update_progress,
            )
    else:
        documents, document_html = render_documents_parallel(
            docs_to_render,
            doc_to_quotes,
            doc_to_entities,
            progress_callback=None,
        )

    # Build comprehensive entity-to-pair mapping (including aliases)
    # Maps entity name/alias -> list of pair indices
    entity_to_pairs: dict[str, list[int]] = defaultdict(list)
    for pair_idx, pair in enumerate(pairs):
        # Add entity1
        entity_to_pairs[pair["entity1"]["name"].lower()].append(pair_idx)
        for alias in pair["entity1"]["aliases"]:
            entity_to_pairs[alias.lower()].append(pair_idx)

        # Add entity2
        entity_to_pairs[pair["entity2"]["name"].lower()].append(pair_idx)
        for alias in pair["entity2"]["aliases"]:
            entity_to_pairs[alias.lower()].append(pair_idx)

    json_data = {
        "metadata": {
            "topic": checkpoint.topic,
            "total_pairs": len(judgments),
            "accepted_pairs": sum(1 for j in judgments if j.accepted),
            "rejected_pairs": sum(1 for j in judgments if not j.accepted),
            "entity_stats": entity_stats,
            "confidence_counts": confidence_counts,
            "resource_count": checkpoint.extraction.metadata.resource_count,
        },
        "pairs": pairs,
        "documents": documents,
        "entity_index": {name: indices for name, indices in entity_index.items()},
        "entity_to_pairs": dict(entity_to_pairs),
    }

    return json_data, document_html
