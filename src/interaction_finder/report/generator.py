"""Main report generation interface.

Orchestrates data preparation, template rendering, and file writing
to produce self-contained HTML reports from ExtractionResult data.
"""

from pathlib import Path

from interaction_finder.extraction.models import ExtractionResult
from interaction_finder.report.data_prep import prepare_report_data
from interaction_finder.report.template import render_template


def generate_report(
    result: ExtractionResult,
    output: Path,
    include_rejected: bool = False,
    title: str | None = None,
) -> Path:
    """Generate interactive HTML report from extraction results.

    Creates a self-contained HTML file with embedded data, CSS, and JavaScript
    that provides an interactive explorer for entity pairs with full provenance.

    Args:
        result: ExtractionResult from pipeline
        output: Path where HTML file should be written
        include_rejected: Whether to include rejected pairs (default: False)
        title: Report title (default: auto-generated from topic)

    Returns:
        Path to generated HTML file

    Raises:
        ValueError: If result is empty or invalid
        OSError: If output path cannot be written
    """
    # Validate input
    if not result.judgments:
        raise ValueError("ExtractionResult contains no judgments")

    # Prepare data
    data = prepare_report_data(result, include_rejected=include_rejected)

    # Generate title if not provided
    if title is None:
        topic = result.metadata.topic
        title = f"Extraction Report: {topic}"

    # Render HTML
    html = render_template(data, title=title)

    # Write to file
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return output_path
