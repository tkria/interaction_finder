"""Main report generation interface.

Orchestrates data preparation, template rendering, and file writing
to produce self-contained reports from ExtractionResult data.
"""

from collections import Counter
from pathlib import Path
import sys
from typing import Any, Literal

from interaction_finder.checkpoint import PipelineCheckpoint
from interaction_finder.extraction.models import PairJudgment
from interaction_finder.report.data_prep import prepare_report_data
from interaction_finder.report.template import render_template


def _normalize_filters(filter_spec: dict[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, raw_value in filter_spec.items():
        key_lower = key.lower()
        value = raw_value.strip()
        if key_lower == "accepted":
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
        else:
            raise ValueError(
                f"Unsupported filter '{key}'. Supported: accepted, evidence."
            )
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


def _apply_filters(
    judgments: list[PairJudgment],
    filters: dict[str, Any],
) -> list[PairJudgment]:
    if not filters:
        return list(judgments)
    accepted_filter = filters.get("accepted")
    evidence_min: int | None = filters.get("evidence_min")
    filtered: list[PairJudgment] = []
    for judgment in judgments:
        if accepted_filter == "yes" and not judgment.accepted:
            continue
        if accepted_filter == "no" and judgment.accepted:
            continue
        if evidence_min and judgment.evidence.overall < evidence_min:
            continue
        filtered.append(judgment)
    return filtered


def generate_report(
    checkpoint: PipelineCheckpoint,
    output: Path | str,
    title: str | None = None,
    format: Literal["html", "plain", "stats"] | str = "html",
    filters: dict[str, str] | None = None,
) -> Path:
    """Generate report artifacts from extraction results.

    Args:
        checkpoint: PipelineCheckpoint containing extraction results
        output: Path or "-" (stdout) destination
        title: Report title (default: auto-generated from topic)
        format: "html" for interactive report, "plain" for tuples, "plain:kind"
            for unique entity listings, or "stats" for pipeline statistics
        filters: Optional mapping of filter keys to values (e.g., {"confidence": "high"})

    Returns:
        Path to generated report file (Path("-") when writing to stdout)

    Raises:
        ValueError: If checkpoint is empty or invalid
        OSError: If output path cannot be written
    """
    normalized_format = format.lower()
    # Stats format has different requirements - doesn't need extraction judgments
    if normalized_format == "stats":
        output_str = str(output)
        write_to_stdout = output_str == "-"
        stats = _collect_stats(checkpoint)
        # Use Rich formatting for TTY stdout, plain text otherwise
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
    supported_formats = {"html", "plain"}
    is_plain_kind = normalized_format.startswith("plain:")
    if normalized_format not in supported_formats and not is_plain_kind:
        raise ValueError(
            f"Unsupported report format '{format}'. Supported: 'html', 'plain', 'plain:KIND', or 'stats'."
        )

    filter_spec = dict(filters or {})
    if normalized_format != "html" and "accepted" not in filter_spec:
        filter_spec["accepted"] = "yes"

    normalized_filters = _normalize_filters(filter_spec)

    filtered_judgments = _apply_filters(
        checkpoint.extraction.judgments, normalized_filters
    )

    if not filtered_judgments:
        raise ValueError("No judgments matched the provided filters")

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

    if normalized_format == "plain":
        lines = [
            f"{judgment.entity1.name}, {judgment.relationship}, {judgment.entity2.name}"
            for judgment in filtered_judgments
        ]
        return write_output("\n".join(lines), ensure_trailing_newline=True)

    if is_plain_kind:
        target_kind = normalized_format.split(":", 1)[1].strip()
        if not target_kind:
            raise ValueError("Entity kind must be specified for format 'plain:KIND'")

        seen_names: set[str] = set()
        lines: list[str] = []
        for judgment in filtered_judgments:
            for entity in (judgment.entity1, judgment.entity2):
                if entity.kind.lower() == target_kind and entity.name not in seen_names:
                    seen_names.add(entity.name)
                    lines.append(entity.name)

        return write_output("\n".join(lines), ensure_trailing_newline=True)

    # HTML generation path
    pairs, document_html, reasoning_templates, indexed_docs = prepare_report_data(
        checkpoint,
        show_progress=not write_to_stdout,
        judgments_override=filtered_judgments,
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
