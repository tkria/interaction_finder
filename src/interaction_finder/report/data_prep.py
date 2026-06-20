"""Data preparation for report generation.

Transforms ExtractionResult into report structure with minimal data-attributes.
No JSON generation - all data embedded in HTML structure.
"""

from collections import defaultdict
from datetime import date
from math import exp, sqrt
from typing import Any, Callable

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

from interaction_finder.checkpoint import PipelineCheckpoint
from interaction_finder.extraction.utils import (
    are_relationships_opposed,
    opposition_map_from_consolidations,
)
from interaction_finder.report.html_renderer import DocumentQuoteEntry
from interaction_finder.report.parallel_renderer import render_documents_parallel
from interaction_finder.report.reasoning_renderer import (
    render_all_reasoning_templates,
    _index_to_alpha_label,
)

POLARITY_ORDER = ("positive", "negative", "neutral", "irrelevant")


def _parse_pub_year(pub_date: str | None) -> int | None:
    """Extract the year from an ISO-format YYYY-MM-DD string, or None."""
    if not pub_date:
        return None
    try:
        return date.fromisoformat(pub_date[:10]).year
    except ValueError:
        return None


def _attach_rank_sum_score(pairs: list[dict[str, Any]]) -> None:
    """Compute the within-topic Borda rank-sum ranking score.

    Combines two features:
      R_pair: substantiated topic relevance. For each supporting assessment i,
              compute s_i = t_i · (1 − exp(−L_i / 200)) where t_i is the
              assessment's topic_relevance (1–5) and L_i is the total length
              of its supporting quotes in characters. Sort assessments by s_i
              descending and return the mean of the raw t_i of the top 3.
      A_pair: age-weighted sum over unique supporting documents of
              1 / sqrt(age + 1), where age is in years.

    Each feature is converted to a within-topic fractional rank (1 = highest)
    and the two ranks are summed. Lower is better. See the supplementary
    "Default Sort in the Interactive Report".
    """

    def fractional_ranks_descending(values: list[float]) -> list[float]:
        """Rank positions (1 = largest) with ties given the average rank."""
        order = sorted(range(len(values)), key=lambda i: -values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # +1 because ranks start at 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    r_pair_ranks = fractional_ranks_descending(
        [float(p["substantiated_relevance"]) for p in pairs]
    )
    age_w_ranks = fractional_ranks_descending([float(p["age_w"]) for p in pairs])
    for pair, r_rel, r_age in zip(pairs, r_pair_ranks, age_w_ranks):
        pair["rank_sum_score"] = r_rel + r_age


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


def _build_opposition_map(checkpoint: PipelineCheckpoint) -> dict[str, set[str]]:
    """Build opposition map from consolidated relationships.

    Falls back to empty map for old checkpoints without consolidated data.
    """
    consolidated = getattr(checkpoint.extraction, "consolidated", None)
    if not consolidated:
        return {}
    return opposition_map_from_consolidations(consolidated.relationships)


def _has_opposing_relationships(
    assessments_data: list[dict[str, Any]],
    opposition_map: dict[str, set[str]],
) -> bool:
    """Check if any assessments have opposing relationships.

    Args:
        assessments_data: List of assessment dicts with "relationship" keys
        opposition_map: Mapping of relationships to their semantic opposites

    Returns:
        True if any pair of relationships are opposed
    """
    if not opposition_map or len(assessments_data) < 2:
        return False
    from itertools import combinations

    relationships = [a["relationship"] for a in assessments_data]
    return any(
        are_relationships_opposed(rel1, rel2, opposition_map)
        for rel1, rel2 in combinations(relationships, 2)
    )


def _build_pair_entry(
    judgment, resource_pool, opposition_map: dict[str, set[str]]
) -> dict[str, Any]:
    """Convert a PairJudgment into the lightweight dict used by the report."""
    doc_ids: set[str] = set()
    # Publication year of each unique supporting document, if parseable.
    doc_pub_years: dict[str, int | None] = {}
    total_quotes = 0
    polarity_counts = {polarity: 0 for polarity in POLARITY_ORDER}
    polarity_best_level = {polarity: 0 for polarity in POLARITY_ORDER}
    evidence_levels: list[tuple[int, str]] = []  # (level, label) per assessment
    # Per-assessment (topic_relevance, total_quote_length_chars) tuples, used
    # to compute the substantiated-relevance ranking feature.
    assessment_rel_quote: list[tuple[int, int]] = []
    assessments: list[dict[str, Any]] = []
    for assessment, polarity in _iter_assessments_with_polarity(judgment):
        doc_ids.add(assessment.resource_id.id)
        assessment_quote_chars = sum(len(q.query_text) for q in assessment.quotes)
        total_quotes += len(assessment.quotes)
        polarity_counts[polarity] += 1
        evidence = assessment.evidence
        evidence_levels.append((evidence.overall, evidence.label))
        assessment_rel_quote.append(
            (int(assessment.topic_relevance), assessment_quote_chars)
        )
        resource = resource_pool.get(assessment.resource_id)
        if resource is None:
            continue
        # Cache publication year for the age-weighted ranking sum; fall back to
        # None if the date is missing or unparseable.
        if assessment.resource_id.id not in doc_pub_years:
            doc_pub_years[assessment.resource_id.id] = _parse_pub_year(
                resource.publication_date
            )
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
                "overall": evidence.overall,
                "label": evidence.label,
                "reasoning": assessment.reasoning,
                "relationship": assessment.relationship,
                "quotes": quote_data,
                "polarity": polarity,
            }
        )
        if evidence.overall > polarity_best_level[polarity]:
            polarity_best_level[polarity] = evidence.overall
    # Get judgment-level evidence
    evidence = judgment.evidence
    # Recency-weighted sum of supporting documents: one term per unique doc
    # with a parseable publication date, weight = 1/sqrt(age + 1), where age
    # is years between the doc's publication and report generation.
    current_year = date.today().year
    age_w = sum(
        1.0 / sqrt(max(0, current_year - y) + 1)
        for y in doc_pub_years.values()
        if y is not None
    )
    # Substantiated relevance: rank assessments by topic_relevance scaled by
    # quote-length saturation (τ = 200 chars), take the top 3, return the
    # mean of their raw topic_relevance values.
    if assessment_rel_quote:
        scored = [(t, t * (1.0 - exp(-L / 200.0))) for (t, L) in assessment_rel_quote]
        scored.sort(key=lambda x: -x[1])
        top = scored[: min(3, len(scored))]
        substantiated_relevance = sum(t for t, _ in top) / len(top)
    else:
        substantiated_relevance = 0.0
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
        "overall": evidence.overall,
        "label": evidence.label,
        "accepted": judgment.accepted,
        "reasoning": judgment.reasoning,
        "doc_count": len(doc_ids),
        "age_w": age_w,
        "substantiated_relevance": substantiated_relevance,
        "quote_count": total_quotes,
        "assessments": assessments,
        "polarity_counts": polarity_counts,
        "polarity_summary": {
            polarity: {
                "count": polarity_counts[polarity],
                "overall": polarity_best_level[polarity],
            }
            for polarity in POLARITY_ORDER
        },
        "evidence_levels": evidence_levels,
        "contentious": _has_opposing_relationships(assessments, opposition_map),
        "topic_relevance": judgment.topic_relevance,
        # Subject-trust gate (gate_review v3): None if the gate did not run or
        # the pair has no subject-kind entity; otherwise the taxonomic verdict.
        "subject_trust": (
            None
            if judgment.subject_trust is None
            else {
                "belongs": judgment.subject_trust.belongs,
                "reasoning": judgment.subject_trust.reasoning,
                "subject_name": judgment.subject_trust.subject_name,
            }
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
    progress_callback: "Callable[[int, int], None] | None" = None,
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
    # Build opposition map from consolidated relationships (or empty for old checkpoints)
    opposition_map = _build_opposition_map(checkpoint)
    pair_entries: list[tuple[tuple[str, str], Any, dict[str, Any]]] = []
    for judgment in judgments:
        key = _pair_key(judgment.entity1.name, judgment.entity2.name)
        pair_data = _build_pair_entry(judgment, checkpoint.resources, opposition_map)
        pair_entries.append((key, judgment, pair_data))

    pairs = [entry[2] for entry in pair_entries]
    _attach_rank_sum_score(pairs)

    # Sort pairs by: accepted status > rank-sum score > evidence level > lexicographic.
    # rank_sum_score combines within-topic ranks of pair_topic_rel and age_w
    # (lower is better); see supplementary "Default Sort in the Interactive Report"
    # for the empirical basis.
    def pair_sort_key(pair):
        return (
            not pair["accepted"],  # False (accepted) sorts before True (rejected)
            pair["rank_sum_score"],  # Ascending (lower rank-sum first)
            -pair["overall"],  # Negate to sort descending (9 first)
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

    # Assign alphabetic labels (A, B, C, ...) to documents based on display order
    # Labels are per-pair to match the visual accordion order
    pair_doc_labels: dict[int, dict[int, str]] = {}
    for pair_idx, pair in enumerate(pairs):
        doc_labels = {}
        for display_idx, doc_group in enumerate(pair["document_groups"]):
            doc_idx = doc_group["doc_idx"]
            doc_labels[doc_idx] = _index_to_alpha_label(display_idx)
        pair_doc_labels[pair_idx] = doc_labels

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

    # Pre-render documents in parallel using multiprocessing.
    total_docs = len(indexed_docs)
    if progress_callback is not None and indexed_docs:
        # Caller-supplied (done, total) sink, e.g. the web UI's progress table.
        # Report the total up front so the UI shows "0/N" immediately, before
        # the first document completes.
        progress_callback(0, total_docs)
        done = 0

        def update_progress():
            nonlocal done
            done += 1
            progress_callback(done, total_docs)

        document_html = render_documents_parallel(
            indexed_docs,
            doc_to_quotes,
            doc_to_entities,
            progress_callback=update_progress,
        )
    elif show_progress and indexed_docs:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Rendering {total_docs} documents...", total=total_docs
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

    # Generate reasoning templates for all pairs (with display-order labels)
    reasoning_templates = render_all_reasoning_templates(
        pairs,
        quote_id_map,
        doc_idx_map,
        pair_doc_labels,
        checkpoint.extraction.paper_quality,
    )

    return pairs, document_html, reasoning_templates, indexed_docs
