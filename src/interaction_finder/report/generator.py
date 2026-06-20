"""Main report generation interface.

Orchestrates data preparation, template rendering, and file writing
to produce self-contained reports from ExtractionResult data.
"""

from collections import Counter
from pathlib import Path
import sys
from typing import Any, Callable, Literal

from interaction_finder.checkpoint import ExtractionStageData, PipelineCheckpoint
from interaction_finder.extraction.models import ExtractionMetadata, PairJudgment
from interaction_finder.report.data_prep import prepare_report_data
from interaction_finder.report.template import render_template
from interaction_finder.resources import ResourcePool


def _normalize_filters(
    filter_spec: dict[str, str], known_entity_kinds: set[str] | None = None
) -> dict[str, Any]:
    """Normalize filter specifications into structured filter dict.

    Parameters:
        filter_spec: Raw filter key-value pairs from CLI
        known_entity_kinds: Optional set of valid entity kinds for validation

    Returns:
        Normalized filters with parsed values

    Supported filters:
        - accepted: yes/no/any
        - evidence: 1-9 or N+
        - <entity_kind>: entity name to match (e.g., gene:BRCA1)
    """
    normalized: dict[str, Any] = {}
    reserved_keys = {
        "accepted",
        "evidence",
        "on_topic",
        "on-topic",
        "on_subject",
        "on-subject",
        "subject_trust",
        "subject",
    }
    for key, raw_value in filter_spec.items():
        key_lower = key.lower()
        value = raw_value.strip()
        if key_lower in (
            "on_topic",
            "on-topic",
            "on_subject",
            "on-subject",
            "subject_trust",
            "subject",
        ):
            aliases = {
                "yes": "pass",
                "on": "pass",
                "pass": "pass",
                "no": "fail",
                "off": "fail",
                "fail": "fail",
                "unjudged": "unjudged",
                "na": "unjudged",
                "none": "unjudged",
                "any": "any",
            }
            val = aliases.get(value.lower())
            if val is None:
                raise ValueError(
                    "Invalid on_topic filter. Use 'yes', 'no', 'unjudged', or 'any'."
                )
            normalized["subject_trust"] = val
        elif key_lower == "accepted":
            val = value.lower()
            mapping = {
                "yes": "yes",
                "true": "yes",
                "no": "no",
                "false": "no",
                "any": "any",
            }
            if val not in mapping:
                raise ValueError(
                    "Invalid accepted filter value. Use 'yes', 'no', or 'any'."
                )
            normalized["accepted"] = mapping[val]
        elif key_lower == "evidence":
            normalized["evidence_min"] = _parse_evidence_filter(value)
        elif key_lower not in reserved_keys:
            # Treat as entity kind filter (e.g., gene:BRCA1)
            if known_entity_kinds and key_lower not in known_entity_kinds:
                valid_kinds = ", ".join(sorted(known_entity_kinds))
                raise ValueError(
                    f"Unknown entity kind '{key}'. Valid kinds: {valid_kinds}"
                )
            # Store entity filters as list of (kind, value) tuples
            if "entities" not in normalized:
                normalized["entities"] = []
            normalized["entities"].append((key_lower, value.lower()))
    return normalized


def _parse_evidence_filter(value: str) -> int:
    """Parse evidence filter value like '7', '7+', or 'any'.

    Returns minimum evidence level (1-9). 'any' returns 1.
    """
    val = value.strip().lower()
    if val == "any":
        return 1
    # Handle '7+' syntax (minimum level)
    if val.endswith("+"):
        val = val[:-1]
    try:
        level = int(val)
        if not 1 <= level <= 9:
            raise ValueError(
                "Invalid evidence filter. Use a number 1-9 (e.g., '7' or '7+')."
            )
        return level
    except ValueError:
        raise ValueError(
            "Invalid evidence filter. Use a number 1-9 (e.g., '7' or '7+')."
        )


def _collect_referenced_urls(judgments: list[PairJudgment]) -> set[str]:
    """Collect URLs of all resources referenced by judgments.

    A resource is referenced if any judgment has an assessment from that resource.
    """
    urls: set[str] = set()
    for judgment in judgments:
        for assessment in judgment.iter_assessments():
            urls.add(assessment.resource_id.url)
    return urls


def _recalculate_metadata(
    judgments: list[PairJudgment],
    resource_count: int,
    topic: str,
) -> ExtractionMetadata:
    """Recalculate metadata counts from filtered judgments.

    Some counts (merged entities, cache hits, proximal sets) cannot be
    reconstructed from filtered data and are set to 0.
    """
    # Count unique entities across all judgments
    entities: set[tuple[str, str]] = set()  # (kind, name)
    quotes_count = 0
    pairs_accepted = 0
    pairs_rejected = 0
    for judgment in judgments:
        entities.add((judgment.entity1.kind, judgment.entity1.name))
        entities.add((judgment.entity2.kind, judgment.entity2.name))
        if judgment.accepted:
            pairs_accepted += 1
        else:
            pairs_rejected += 1
        # Count quotes in all assessments
        for assessment in judgment.iter_assessments():
            quotes_count += len(assessment.quotes)
    return ExtractionMetadata(
        topic=topic,
        resource_count=resource_count,
        total_entities_found=len(entities),
        entities_after_validation=len(entities),
        entities_merged=0,
        merge_cache_hits=0,
        merge_cache_misses=0,
        proximal_sets_found=0,
        total_pairs_found=len(judgments),
        pairs_accepted=pairs_accepted,
        pairs_rejected=pairs_rejected,
        quotes_validated=quotes_count,
        quotes_failed=0,
    )


def _entity_matches(entity: Any, kind: str, value: str) -> bool:
    """Check if an entity matches the filter criteria.

    Parameters:
        entity: SimpleEntity with name, kind, aliases
        kind: Entity kind to match (case-insensitive)
        value: Canonical name to match (exact, case-insensitive)

    Returns:
        True if entity kind matches and canonical name equals the value
    """
    if entity.kind.lower() != kind:
        return False
    return value == entity.name.lower()


def _filter_judgments(
    judgments: list[PairJudgment],
    filters: dict[str, Any],
) -> list[PairJudgment]:
    """Filter judgments based on normalized filter criteria."""
    if not filters:
        return list(judgments)
    accepted_filter = filters.get("accepted")
    evidence_min: int | None = filters.get("evidence_min")
    entity_filters: list[tuple[str, str]] = filters.get("entities", [])
    subject_trust_filter = filters.get("subject_trust")
    filtered: list[PairJudgment] = []
    for judgment in judgments:
        if accepted_filter == "yes" and not judgment.accepted:
            continue
        if accepted_filter == "no" and judgment.accepted:
            continue
        if evidence_min and judgment.evidence.overall < evidence_min:
            continue
        # Subject-trust gate verdict (None == unjudged). Mirrors the report's
        # strict-by-default toggle: `pass` is the on-subject answer set.
        if subject_trust_filter and subject_trust_filter != "any":
            st = judgment.subject_trust
            if subject_trust_filter == "pass" and not (st and st.belongs):
                continue
            if subject_trust_filter == "fail" and not (st and not st.belongs):
                continue
            if subject_trust_filter == "unjudged" and st is not None:
                continue
        # Entity filters: pair must have at least one entity matching each filter
        if entity_filters:
            match = True
            for kind, value in entity_filters:
                if not (
                    _entity_matches(judgment.entity1, kind, value)
                    or _entity_matches(judgment.entity2, kind, value)
                ):
                    match = False
                    break
            if not match:
                continue
        filtered.append(judgment)
    return filtered


def filter_checkpoint(
    checkpoint: PipelineCheckpoint,
    filters: dict[str, Any],
) -> PipelineCheckpoint:
    """Return a filtered checkpoint as if non-matching data was never extracted.

    Filters are applied to judgments. The returned checkpoint has:
    - Filtered judgments
    - Pruned resource pool (only resources referenced by remaining judgments)
    - Updated paper_quality (only referenced documents)
    - Recalculated metadata counts

    Parameters:
        checkpoint: Source checkpoint with extraction data
        filters: Normalized filters from _normalize_filters()

    Returns:
        New PipelineCheckpoint representing the filtered state

    Raises:
        ValueError: If checkpoint has no extraction data
    """
    if not checkpoint.extraction:
        raise ValueError("Checkpoint does not contain extraction results")
    if not filters:
        return checkpoint
    # Filter judgments
    filtered_judgments = _filter_judgments(checkpoint.extraction.judgments, filters)
    # Collect referenced resources
    referenced_urls = _collect_referenced_urls(filtered_judgments)
    # Build pruned resource pool
    new_pool = ResourcePool()
    for url in referenced_urls:
        resource = checkpoint.resources.get(url)
        if resource:
            new_pool.resource_map[resource.id] = resource
    # Filter paper_quality to only referenced documents
    new_paper_quality = {
        url: quality
        for url, quality in checkpoint.extraction.paper_quality.items()
        if url in referenced_urls
    }
    # Recalculate metadata
    new_metadata = _recalculate_metadata(
        filtered_judgments, len(referenced_urls), checkpoint.topic
    )
    # Build new extraction stage data
    new_extraction = ExtractionStageData(
        target_entity_types=checkpoint.extraction.target_entity_types,
        permitted_pairs=checkpoint.extraction.permitted_pairs,
        subject_kind=checkpoint.extraction.subject_kind,
        subject_anchor=checkpoint.extraction.subject_anchor,
        judgments=filtered_judgments,
        metadata=new_metadata,
        consolidated=checkpoint.extraction.consolidated,
        paper_quality=new_paper_quality,
    )
    # Build new checkpoint (preserve other stages unchanged)
    return PipelineCheckpoint(
        topic=checkpoint.topic,
        resources=new_pool,
        keywords=checkpoint.keywords,
        search=checkpoint.search,
        extraction=new_extraction,
        created_by=checkpoint.created_by,
        usage=checkpoint.usage,
        config=checkpoint.config,
    )


def _collect_stats(checkpoint: PipelineCheckpoint) -> dict[str, Any]:
    """Collect statistics from checkpoint into a structured dict.

    Parameters:
        checkpoint: PipelineCheckpoint with stage data

    Returns:
        Dict with sections for each pipeline stage and their metrics
    """
    stats: dict[str, Any] = {"topic": checkpoint.topic}
    # Keywords stage
    if checkpoint.keywords:
        kw = checkpoint.keywords
        stats["keywords"] = {
            "Rounds completed": kw.rounds_completed,
            "Documents processed": kw.total_documents_processed,
            "Bridging terms found": len(kw.terms),
            "Resources added": len(kw.resource_urls),
        }
    # Search stage
    if checkpoint.search:
        search = checkpoint.search
        stats["search"] = {
            "Rounds completed": search.rounds_completed,
            "Queries executed": len(search.queries),
            "Results selected": len(search.results),
            "Keyphrases used": len(search.keyphrases),
        }
    # Extraction stage
    if checkpoint.extraction:
        ext = checkpoint.extraction
        meta = ext.metadata
        total_quotes = meta.quotes_validated + meta.quotes_failed
        stats["extraction"] = {
            "Documents processed": meta.resource_count,
            "Entities found": meta.total_entities_found,
            "Entities validated": f"{meta.entities_after_validation}/{meta.total_entities_found}",
            "Entities merged": meta.entities_merged,
            "Proximal sets found": meta.proximal_sets_found,
            "Quotes validated": f"{meta.quotes_validated}/{total_quotes}",
        }
        stats["pairs"] = {
            "Total pairs found": meta.total_pairs_found,
            "Accepted": meta.pairs_accepted,
            "Rejected": meta.pairs_rejected,
        }
        # Entity breakdown by kind (unique entities in accepted pairs)
        entities_by_kind: dict[str, set[str]] = {}
        for judgment in ext.judgments:
            if judgment.accepted:
                for entity in (judgment.entity1, judgment.entity2):
                    if entity.kind not in entities_by_kind:
                        entities_by_kind[entity.kind] = set()
                    entities_by_kind[entity.kind].add(entity.name)
        if entities_by_kind:
            stats["entities_by_kind"] = {
                kind: len(names) for kind, names in sorted(entities_by_kind.items())
            }
        # Relationship types
        rel_counts: Counter[str] = Counter()
        for judgment in ext.judgments:
            if judgment.accepted:
                rel_counts[judgment.relationship] += 1
        if rel_counts:
            stats["relationships"] = dict(rel_counts.most_common())
    # Resource pool summary
    stats["resources"] = {"Total in pool": len(checkpoint.resources.resource_map)}
    return stats


def _format_stats_plain(stats: dict[str, Any]) -> str:
    """Format collected stats as plain text.

    Parameters:
        stats: Dict from _collect_stats

    Returns:
        Plain text with section headers and indented key-value pairs
    """
    lines: list[str] = [f"Topic: {stats['topic']}", ""]
    section_names = {
        "keywords": "Keywords Stage",
        "search": "Search Stage",
        "extraction": "Extraction Stage",
        "pairs": "Pairs",
        "entities_by_kind": "Entities by Kind (in accepted pairs)",
        "relationships": "Relationship Types (accepted pairs)",
        "resources": "Resources",
    }
    for key, title in section_names.items():
        if key in stats:
            lines.append(title)
            for metric, value in stats[key].items():
                lines.append(f"  {metric}: {value}")
            lines.append("")
    # Remove trailing empty line
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _render_stats_rich(stats: dict[str, Any]) -> None:
    """Render collected stats to stdout using Rich formatting.

    Matches the styling of the progress tables shown during pipeline execution:
    category headers in bold cyan, indented metrics in cyan, values in bold yellow.

    Parameters:
        stats: Dict from _collect_stats
    """
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()
    # Topic header with separator (matching progress table style)
    topic_text = Text(stats["topic"], style="bold")
    console.print(topic_text)
    console.print(Text("─" * len(stats["topic"]), style="bold cyan"))
    # Build rows matching progress table format: (label, value)
    rows: list[tuple[str, str]] = []
    # Section configurations: (key, title)
    sections = [
        ("keywords", "Keywords Stage"),
        ("search", "Search Stage"),
        ("extraction", "Extraction Stage"),
        ("pairs", "Pairs"),
        ("entities_by_kind", "Entities by Kind"),
        ("relationships", "Relationship Types"),
        ("resources", "Resources"),
    ]
    for key, title in sections:
        if key not in stats:
            continue
        # Category header (bold cyan, no indent)
        rows.append((f"[bold cyan]{title}[/]", ""))
        # Metrics (cyan with 2-space indent, values in bold yellow)
        for metric, value in stats[key].items():
            rows.append((f"[cyan]  {metric}[/]", f"[bold yellow]{value}[/]"))
    # Render as grid table (matching progress table structure)
    table = Table.grid(padding=(0, 2))
    table.add_column()
    table.add_column(justify="right")
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)


def generate_report(
    checkpoint: PipelineCheckpoint,
    output: Path | str,
    title: str | None = None,
    format: Literal["html", "plain", "stats"] | str = "html",
    filters: dict[str, str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Generate report artifacts from extraction results.

    Args:
        checkpoint: PipelineCheckpoint containing extraction results
        output: Path or "-" (stdout) destination
        title: Report title (default: auto-generated from topic)
        format: "html" for interactive report, "plain" for tuples, "plain:kind"
            for unique entity listings, or "stats" for pipeline statistics
        filters: Optional mapping of filter keys to values

    Returns:
        Path to generated report file (Path("-") when writing to stdout)

    Raises:
        ValueError: If checkpoint is empty or invalid
        OSError: If output path cannot be written
    """
    normalized_format = format.lower()
    supported_formats = {"html", "plain", "stats"}
    is_plain_kind = normalized_format.startswith("plain:")
    if normalized_format not in supported_formats and not is_plain_kind:
        raise ValueError(
            f"Unsupported report format '{format}'. Supported: 'html', 'plain', 'plain:KIND', or 'stats'."
        )
    # Apply filters early (affects all formats including stats)
    filter_spec = dict(filters or {})
    if normalized_format not in {"html", "stats"} and "accepted" not in filter_spec:
        filter_spec["accepted"] = "yes"
    if filter_spec and checkpoint.extraction:
        known_kinds: set[str] = set()
        for judgment in checkpoint.extraction.judgments:
            known_kinds.add(judgment.entity1.kind.lower())
            known_kinds.add(judgment.entity2.kind.lower())
        normalized_filters = _normalize_filters(filter_spec, known_kinds)
        checkpoint = filter_checkpoint(checkpoint, normalized_filters)
        if not checkpoint.extraction or not checkpoint.extraction.judgments:
            raise ValueError("No judgments matched the provided filters")
    # Stats format
    if normalized_format == "stats":
        output_str = str(output)
        write_to_stdout = output_str == "-"
        stats = _collect_stats(checkpoint)
        if write_to_stdout and sys.stdout.isatty():
            _render_stats_rich(stats)
            return Path("-")
        stats_text = _format_stats_plain(stats)
        if write_to_stdout:
            sys.stdout.write(stats_text)
            if not stats_text.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
            return Path("-")
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(stats_text, encoding="utf-8")
        return output_path
    # All other formats require extraction results
    if not checkpoint.extraction:
        raise ValueError("Checkpoint does not contain extraction results")
    if not checkpoint.extraction.judgments:
        raise ValueError("Extraction results contain no judgments")
    # Output setup
    output_str = str(output)
    write_to_stdout = output_str == "-"
    output_path: Path | None = None
    if not write_to_stdout:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_output(text: str, ensure_trailing_newline: bool = False) -> Path:
        if write_to_stdout:
            sys.stdout.write(text)
            if ensure_trailing_newline and text and not text.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
            return Path("-")
        assert output_path is not None
        output_path.write_text(text, encoding="utf-8")
        return output_path

    judgments = checkpoint.extraction.judgments
    if normalized_format == "plain":
        lines = [
            f"{j.entity1.name}, {j.relationship}, {j.entity2.name}" for j in judgments
        ]
        return write_output("\n".join(lines), ensure_trailing_newline=True)
    if is_plain_kind:
        target_kind = normalized_format.split(":", 1)[1].strip()
        if not target_kind:
            raise ValueError("Entity kind must be specified for format 'plain:KIND'")
        seen_names: set[str] = set()
        lines: list[str] = []
        for judgment in judgments:
            for entity in (judgment.entity1, judgment.entity2):
                if entity.kind.lower() == target_kind and entity.name not in seen_names:
                    seen_names.add(entity.name)
                    lines.append(entity.name)
        return write_output("\n".join(lines), ensure_trailing_newline=True)
    # HTML generation path
    pairs, document_html, reasoning_templates, indexed_docs = prepare_report_data(
        checkpoint,
        show_progress=not write_to_stdout,
        progress_callback=progress_callback,
    )
    html = render_template(
        pairs,
        document_html,
        reasoning_templates,
        indexed_docs,
        checkpoint.topic,
        title=title,
        version=checkpoint.created_by,
    )
    return write_output(html)
