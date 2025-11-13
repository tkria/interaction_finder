"""Data preparation for report generation.

Transforms ExtractionResult into JSON-serializable report data structures
optimized for frontend rendering and filtering.
"""

from collections import defaultdict
from typing import Any

from interaction_finder.extraction.models import ExtractionResult


def prepare_report_data(
    result: ExtractionResult, include_rejected: bool = False
) -> dict[str, Any]:
    """Transform ExtractionResult into report data structure.

    Args:
        result: Extraction pipeline output
        include_rejected: Whether to include rejected pairs

    Returns:
        JSON-serializable dict with pre-computed structures for frontend
    """
    # Filter judgments based on accepted status
    judgments = (
        result.judgments
        if include_rejected
        else [j for j in result.judgments if j.accepted]
    )

    # Group judgments by entity pair (merge different relationships)
    from collections import defaultdict

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
            resource = result.resources.get(assess.resource_id.url)
            if resource is None:
                continue

            # Extract entity mentions with their quote positions
            entity1_mentions = _extract_entity_mentions(assess.entity1, assess.quotes)
            entity2_mentions = _extract_entity_mentions(assess.entity2, assess.quotes)

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
                    "entity1_mentions": entity1_mentions,
                    "entity2_mentions": entity2_mentions,
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

    # Build document index: resource_id -> document data
    documents = {}
    for resource in result.resources.resources:
        doc_id = resource.id.id
        documents[doc_id] = {
            "id": doc_id,
            "url": resource.id.url,
            "title": resource.title or "Untitled",
            "text": resource.text,
        }

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

    return {
        "metadata": {
            "topic": result.metadata.topic,
            "total_pairs": len(judgments),
            "accepted_pairs": sum(1 for j in judgments if j.accepted),
            "rejected_pairs": sum(1 for j in judgments if not j.accepted),
            "entity_stats": entity_stats,
            "confidence_counts": confidence_counts,
            "resource_count": result.metadata.resource_count,
        },
        "pairs": pairs,
        "documents": documents,
        "entity_index": {name: indices for name, indices in entity_index.items()},
        "entity_to_pairs": dict(entity_to_pairs),
    }


def _extract_entity_mentions(
    entity_mention,
    quotes: list,
) -> list[dict[str, Any]]:
    """Extract entity mention positions from quotes.

    Scans quote text for entity name and aliases, recording positions.

    Args:
        entity_mention: EntityMention from assessment
        quotes: ResourceQuote objects for the pair

    Returns:
        List of mention dicts with name, spans, and quote_idx
    """
    mentions = []
    search_terms = [entity_mention.name] + entity_mention.aliases

    # For each quote, find entity mentions
    for quote_idx, quote in enumerate(quotes):
        text = quote.query_text.lower()

        for term in search_terms:
            term_lower = term.lower()
            start = 0

            while True:
                pos = text.find(term_lower, start)
                if pos == -1:
                    break

                mentions.append(
                    {
                        "text": term,
                        "quote_idx": quote_idx,
                        "span": [pos, pos + len(term)],
                    }
                )
                start = pos + len(term)

    return mentions
