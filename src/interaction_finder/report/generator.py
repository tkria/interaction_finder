"""Main report generation interface.

Orchestrates data preparation, template rendering, and file writing
to produce self-contained reports from ExtractionResult data.
"""

from pathlib import Path
from typing import Any
import sys
from typing import Literal

from interaction_finder.checkpoint import PipelineCheckpoint
from interaction_finder.extraction.models import PairJudgment
from interaction_finder.report.data_prep import prepare_report_data
from interaction_finder.report.template import render_template

CONFIDENCE_LEVELS = ["low", "medium", "high"]


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
        elif key_lower == "confidence":
            normalized["confidence"] = _parse_confidence_filter(value)
        else:
            raise ValueError(
                f"Unsupported filter '{key}'. Supported: accepted, confidence."
            )
    return normalized


def _parse_confidence_filter(value: str) -> set[str]:
    val = value.strip().lower()
    if val == "any":
        return set(CONFIDENCE_LEVELS)

    if val.endswith("+"):
        base = val[:-1]
        if base not in CONFIDENCE_LEVELS:
            raise ValueError(
                "Invalid confidence filter. Use 'high', 'medium', 'low', or 'medium+'."
            )
        start_index = CONFIDENCE_LEVELS.index(base)
        return set(CONFIDENCE_LEVELS[start_index:])

    if val not in CONFIDENCE_LEVELS:
        raise ValueError(
            "Invalid confidence filter. Use 'high', 'medium', 'low', or 'medium+'."
        )
    return {val}


def _apply_filters(
    judgments: list[PairJudgment],
    filters: dict[str, Any],
) -> list[PairJudgment]:
    if not filters:
        return list(judgments)

    accepted_filter = filters.get("accepted")
    confidence_filter: set[str] | None = filters.get("confidence")

    filtered: list[PairJudgment] = []
    for judgment in judgments:
        if accepted_filter == "yes" and not judgment.accepted:
            continue
        if accepted_filter == "no" and judgment.accepted:
            continue

        if confidence_filter and judgment.confidence.lower() not in confidence_filter:
            continue

        filtered.append(judgment)

    return filtered


def generate_report(
    checkpoint: PipelineCheckpoint,
    output: Path | str,
    title: str | None = None,
    format: Literal["html", "plain"] | str = "html",
    filters: dict[str, str] | None = None,
) -> Path:
    """Generate report artifacts from extraction results.

    Args:
        checkpoint: PipelineCheckpoint containing extraction results
        output: Path or "-" (stdout) destination
        title: Report title (default: auto-generated from topic)
        format: "html" for interactive report, "plain" for tuples, or "plain:kind"
            for unique entity listings
        filters: Optional mapping of filter keys to values (e.g., {"confidence": "high"})

    Returns:
        Path to generated report file (Path("-") when writing to stdout)

    Raises:
        ValueError: If checkpoint is empty or invalid
        OSError: If output path cannot be written
    """
    if not checkpoint.extraction:
        raise ValueError("Checkpoint does not contain extraction results")

    if not checkpoint.extraction.judgments:
        raise ValueError("Extraction results contain no judgments")

    normalized_format = format.lower()
    supported_formats = {"html", "plain"}
    is_plain_kind = normalized_format.startswith("plain:")
    if normalized_format not in supported_formats and not is_plain_kind:
        raise ValueError(
            f"Unsupported report format '{format}'. Supported: 'html', 'plain', or 'plain:KIND'."
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
    pairs, document_html, reasoning_templates = prepare_report_data(
        checkpoint,
        show_progress=not write_to_stdout,
        judgments_override=filtered_judgments,
    )

    html = render_template(
        pairs, document_html, reasoning_templates, checkpoint.topic, title=title
    )
    return write_output(html)
