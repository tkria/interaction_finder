"""
CLI interface for Interaction Finder using Typer.

This module provides command-line interface functionality for the interaction finder tool,
including a train sub-command for fetching URLs from training data files.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.traceback import Traceback
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .settings import IfetcherConfig
from .fetcher import PageFetcher

app = typer.Typer(
    name="interaction-finder",
    help="A tool for fetching and processing web content for interaction discovery.",
    rich_markup_mode="rich",
)

console = Console()


def group_errors(failed_pairs: List[tuple[str, Exception]]) -> Dict[str, List[str]]:
    """
    Group errors by normalized error message for better display.

    Args:
        failed_pairs: List of (url, exception) tuples

    Returns:
        Dictionary mapping error keys to lists of URLs with that error
    """
    error_groups = {}

    for url, exception in failed_pairs:
        # Normalize error message by removing URL-specific parts
        error_msg = str(exception)
        # Remove the specific URL from the error message for grouping
        normalized_msg = re.sub(r"https?://[^\s]+", "<URL>", error_msg)
        # Take just the first line for cleaner grouping
        normalized_msg = normalized_msg.split("\n")[0]
        error_key = f"{type(exception).__name__}: {normalized_msg}"

        if error_key not in error_groups:
            error_groups[error_key] = []
        error_groups[error_key].append(url)

    return error_groups


def display_error_summary(
    failed_pairs: List[tuple[str, Exception]], verbose: bool = False
):
    """
    Display errors in a user-friendly grouped format.

    Args:
        failed_pairs: List of (url, exception) tuples
        verbose: Whether to show detailed error information
    """
    if not failed_pairs:
        return

    console.print(f"\n[red]Failed URLs: {len(failed_pairs)}[/red]")

    if verbose:
        # Show pretty traceback for each failure
        for url, exc in failed_pairs:
            console.print(f"\n[red]Error for:[/red] {url}")
            tb = Traceback.from_exception(
                type(exc), exc, exc.__traceback__, show_locals=False
            )
            console.print(tb)
        return
    else:
        error_groups = group_errors(failed_pairs)

        for error_msg, urls in error_groups.items():
            if len(urls) == 1:
                console.print(f"  • {urls[0]}")
                console.print(f"    [dim]{error_msg}[/dim]")
            else:
                console.print(
                    f"  • {urls[0]} [dim](and {len(urls) - 1} more similar)[/dim]"
                )
                console.print(f"    [dim]{error_msg}[/dim]")
                if len(urls) <= 5:
                    for url in urls[1:]:
                        console.print(f"  • {url}")
                else:
                    for url in urls[1:3]:
                        console.print(f"  • {url}")
                    console.print(
                        f"    [dim]... and {len(urls) - 3} more with same error[/dim]"
                    )
            console.print()
        console.print("  (use --verbose to see full tracebacks)")


def separate_results_and_errors(
    results: List, urls: List[str]
) -> tuple[List, List[tuple[str, Exception]]]:
    """
    Separate successful results from exceptions with their corresponding URLs.

    Args:
        results: List containing mix of Exception objects and success data
        urls: List of URLs corresponding to results

    Returns:
        Tuple of (successful_results, failed_url_exception_pairs)
    """
    successful = []
    failed = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failed.append((urls[i], result))
        else:
            successful.append(result)

    return successful, failed


def parse_group_option(value: Optional[str]) -> tuple[bool, str]:
    """
    Parse the --group option value into grouping enabled flag and constraint.

    Handles forms:
    - None (not provided) -> (False, "count:3-8")
    - "true" or "" -> (True, "count:3-8")
    - "docs:3-5" or "count:3-5" -> (True, "count:3-5")
    - "words:500-800" -> (True, "words:500-800")

    Args:
        value: The option value from CLI

    Returns:
        Tuple of (grouping_enabled, constraint_string)
    """
    if value is None:
        return False, "count:3-8"

    if value == "" or value.lower() == "true":
        return True, "count:3-8"

    # Handle shorthand "docs:" -> "count:"
    if value.startswith("docs:"):
        value = value.replace("docs:", "count:", 1)

    # Validate constraint format
    import re

    if not re.match(r"(count|words):\d+-\d+", value):
        console.print(f"[red]Invalid group constraint format: {value}[/red]")
        console.print("[red]Expected format: 'count:min-max' or 'words:min-max'[/red]")
        raise typer.Exit(1)

    return True, value


def handle_operation_error(operation: str, error: Exception, context: str = "") -> None:
    """
    Handle errors from operations with consistent formatting and context.

    Args:
        operation: Description of what was being performed (e.g., "loading configuration")
        error: The exception that occurred
        context: Additional context about the operation
    """
    context_part = f" {context}" if context else ""
    console.print(f"[red]Error {operation}{context_part}: {error}[/red]")


def load_config(
    config_path: Optional[str] = None, mode: Optional[str] = None
) -> IfetcherConfig:
    """Load configuration from file or use defaults."""
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            handle_operation_error(
                "loading configuration", f"Configuration file {config_path} not found"
            )
            raise typer.Exit(1)
        return IfetcherConfig.from_path(config_file, mode=mode)
    else:
        # Try to find a config file in common locations
        for potential_config in [
            "config.toml",
            "interaction_finder.toml",
            ".interaction_finder.toml",
        ]:
            if Path(potential_config).exists():
                console.print(f"[dim]Using config file: {potential_config}[/dim]")
                return IfetcherConfig.from_path(potential_config, mode=mode)

        # Use default configuration
        console.print("[dim]Using default configuration[/dim]")
        return IfetcherConfig()


def extract_urls_from_jsonl(file_path: Path) -> tuple[List[str], int]:
    """
    Extract URLs from a JSONL training data file.

    Supports various JSONL formats:
    - {"url": "https://example.com", ...}
    - {"urls": ["https://example.com", ...], ...}
    - {"source": "https://example.com", ...}
    - {"link": "https://example.com", ...}
    - Any field containing URLs (detected by regex)

    Returns:
        tuple: (unique_urls, total_urls_found)
    """
    all_urls = []
    url_pattern = re.compile(r'https?://[^\s<>"]+')

    if not file_path.exists():
        handle_operation_error(
            "reading training data", f"Training data file {file_path} not found"
        )
        raise typer.Exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        console.print(
                            f"[yellow]Warning: Line {line_num} is not a JSON object, skipping[/yellow]"
                        )
                        continue

                    line_urls = (
                        set()
                    )  # Track URLs found in this line to avoid duplicates

                    # Extract URLs from explicit URL fields first (highest priority)
                    for field in ["url", "source", "link"]:
                        if field in data and isinstance(data[field], str):
                            if url_pattern.match(data[field]):
                                line_urls.add(data[field])

                    # Handle array fields for URLs
                    for field in ["urls", "sources", "links"]:
                        if field in data and isinstance(data[field], list):
                            for item in data[field]:
                                if isinstance(item, str) and url_pattern.match(item):
                                    line_urls.add(item)

                    # Only search for embedded URLs if no explicit URL fields were found
                    if not line_urls:
                        for key, value in data.items():
                            # Skip the fields we already checked
                            if key in [
                                "url",
                                "urls",
                                "source",
                                "sources",
                                "link",
                                "links",
                            ]:
                                continue

                            if isinstance(value, str):
                                found_urls = url_pattern.findall(value)
                                line_urls.update(found_urls)
                            elif isinstance(value, list):
                                for item in value:
                                    if isinstance(item, str):
                                        found_urls = url_pattern.findall(item)
                                        line_urls.update(found_urls)

                    all_urls.extend(line_urls)

                except json.JSONDecodeError as e:
                    console.print(
                        f"[yellow]Warning: Invalid JSON on line {line_num}: {e}[/yellow]"
                    )
                    continue

    except Exception as e:
        handle_operation_error("reading training data file", e)
        raise typer.Exit(1)

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls, len(all_urls)


@app.command()
def train(
    term: str = typer.Argument(..., help="Term to use for training data file lookup"),
    config: Optional[str] = typer.Option(
        None, "-c", "--config", help="Path to configuration file"
    ),
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    training_file: Optional[str] = typer.Option(
        None, "-f", "--file", help="Override training data file path"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show URLs that would be fetched without fetching them"
    ),
    fetch_only: bool = typer.Option(
        False,
        "--fetch-only",
        help="Only fetch and cache URLs from training data, skip further processing",
    ),
    max_concurrent: Optional[int] = typer.Option(
        None, "--max-concurrent", help="Maximum concurrent requests"
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Show verbose output and pretty tracebacks on errors",
    ),
    failfast: bool = typer.Option(
        False, "--failfast", help="Stop on first error while fetching"
    ),
    retry: bool = typer.Option(
        False, "--retry", help="Force retry of URLs previously marked as failed"
    ),
    group: Optional[str] = typer.Option(
        None,
        "--group",
        help="Group documents by similarity. Use --group for default (count:3-8), --group=count:3-5, --group=words:500-800, or --group=docs:3-5",
    ),
):
    """
    Train sub-command: fetch URLs from training data for prompt optimization.

    This command reads a JSONL training data file and fetches all URLs found within it.
    The training data file path is determined by the configuration and the provided term.

    Use --fetch-only to only fetch and cache the URLs without additional processing.

    Examples:
        interaction-finder train BRCA1
        interaction-finder train BRCA1 --config my_config.toml
        interaction-finder train BRCA1 --file custom_training.jsonl --dry-run
        interaction-finder train BRCA1 --fetch-only
        interaction-finder train BRCA1 --fetch-only --verbose
        interaction-finder train BRCA1 --group
        interaction-finder train BRCA1 --group=count:5-10
        interaction-finder train BRCA1 --group=words:1000-5000
    """
    console.print(f"[bold blue]Training mode for term: {term}[/bold blue]")

    # Load configuration
    try:
        cfg = load_config(config, mode)
    except Exception as e:
        handle_operation_error("loading configuration", e)
        raise typer.Exit(1)

    # Override max_concurrent if provided
    if max_concurrent is not None:
        cfg.tools.crawl4ai.max_concurrent = max_concurrent

    # Determine training data file path
    if training_file:
        training_data_path = Path(training_file)
    else:
        training_data_path = cfg.abspath(cfg.training_data, term=term)

    console.print(f"[dim]Training data file: {training_data_path}[/dim]")

    # Extract URLs from training data
    with console.status("[bold green]Reading training data..."):
        urls, total_entries = extract_urls_from_jsonl(training_data_path)

    if fetch_only:
        console.print(f"[dim]Fetch-only mode: will only fetch and cache URLs[/dim]")

    if not urls:
        console.print("[yellow]No URLs found in training data file[/yellow]")
        return

    # Parse the group option
    group_documents, group_constraint = parse_group_option(group)

    # Group documents if requested
    if group_documents and not fetch_only:
        try:
            console.print(f"[bold blue]Grouping {len(urls)} documents...[/bold blue]")
            # Run grouping in async context
            groups = asyncio.run(
                perform_document_grouping(urls, cfg, group_constraint, retry)
            )

            if groups:
                console.print(f"[green]✓ Created {len(groups)} document groups[/green]")
                total_docs = sum(len(group["documents"]) for group in groups)
                total_words = sum(group["total_words"] for group in groups)
                avg_cohesion = sum(group["cohesion_score"] for group in groups) / len(
                    groups
                )

                console.print(f"[cyan]- Total documents grouped: {total_docs}[/cyan]")
                console.print(
                    f"[cyan]- Total words across groups: {total_words:,}[/cyan]"
                )
                console.print(
                    f"[cyan]- Average group cohesion: {avg_cohesion:.3f}[/cyan]"
                )

                if verbose:
                    for i, group in enumerate(groups, 1):
                        console.print(
                            f"  Group {i}: {len(group['documents'])} docs, "
                            f"{group['total_words']:,} words, "
                            f"cohesion: {group['cohesion_score']:.3f}"
                        )
                        if len(group["documents"]) <= 3:
                            for doc in group["documents"]:
                                console.print(f"    • {doc}")
                        else:
                            for doc in group["documents"][:2]:
                                console.print(f"    • {doc}")
                            console.print(
                                f"    • ... and {len(group['documents']) - 2} more"
                            )

                # Note: For now, we continue with individual URL processing
                # Future enhancement could process groups together
                console.print(
                    "[dim]Continuing with individual document processing...[/dim]"
                )
            else:
                console.print(
                    "[yellow]No document groups created (all documents below threshold)[/yellow]"
                )

        except Exception as e:
            if verbose:
                import traceback

                console.print(f"[red]Error during document grouping: {e}[/red]")
                console.print("[red]" + traceback.format_exc() + "[/red]")
            else:
                console.print(f"[red]Error during document grouping: {e}[/red]")
            console.print(
                "[yellow]Continuing with individual document processing...[/yellow]"
            )
    elif group_documents and fetch_only:
        console.print("[yellow]Document grouping skipped in fetch-only mode[/yellow]")

    # Show URL count information
    unique_count = len(urls)
    if total_entries == unique_count:
        console.print(f"[green]Found {unique_count} unique URLs[/green]")
    else:
        console.print(
            f"[green]Found {unique_count} unique URLs from {total_entries} entries[/green]"
        )

    if verbose or dry_run:
        table = Table(title="URLs to fetch")
        table.add_column("Index", style="cyan", width=6)
        table.add_column("URL", style="blue")

        # Check cache status for each URL if in verbose mode
        if verbose and not dry_run:
            # We need to run an async function to check cache status
            cache_status = asyncio.run(check_urls_cache_status(urls, cfg))
            stats = cache_status.get("_stats", {})

            # Determine which columns to show based on what content exists
            show_html = (
                stats.get("html_count", 0) > 0 or stats.get("total_nothing", 0) > 0
            )
            show_pdf = stats.get("pdf_count", 0) > 0 or (
                stats.get("total_nothing", 0) > 0 and not show_html
            )
            show_md = stats.get("md_count", 0) > 0
            show_chunks = stats.get("chunks_count", 0) > 0
        elif verbose and dry_run:
            # In dry run mode, show all columns
            show_html = True
            show_pdf = True
            show_md = True
            show_chunks = True
        else:
            show_html = show_pdf = show_md = show_chunks = False

        if verbose:
            # Add cache status columns in verbose mode based on what's needed
            if show_html:
                table.add_column("HTML", style="dim", width=5, justify="center")
            if show_pdf:
                table.add_column("PDF", style="dim", width=5, justify="center")
            if show_md:
                table.add_column("MD", style="dim", width=5, justify="center")
            if show_chunks:
                table.add_column("Chunks", style="dim", width=6, justify="center")

        for i, url in enumerate(urls, 1):
            if verbose and not dry_run:
                status = cache_status.get(url, {})
                html_cached = status.get("html", False)
                pdf_cached = status.get("pdf", False)
                md_cached = status.get("markdown", False)
                chunks_cached = status.get("chunks", False)

                # Check if nothing is cached at all
                nothing_cached = not (
                    html_cached or pdf_cached or md_cached or chunks_cached
                )

                # Build row data based on which columns are shown
                row_data = [str(i), url]

                if show_html:
                    if nothing_cached:
                        html_count, pdf_count = (
                            stats.get("html_count", 0),
                            stats.get("pdf_count", 0),
                        )
                        most_common_is_html = html_count >= pdf_count
                        html_status = (
                            "[red bold]✗[/red bold]" if most_common_is_html else ""
                        )
                    else:
                        html_status = "[green]✓[/green]" if html_cached else ""
                    row_data.append(html_status)

                if show_pdf:
                    if nothing_cached:
                        html_count, pdf_count = (
                            stats.get("html_count", 0),
                            stats.get("pdf_count", 0),
                        )
                        most_common_is_html = html_count >= pdf_count
                        pdf_status = (
                            "[red bold]✗[/red bold]" if not most_common_is_html else ""
                        )
                    else:
                        pdf_status = "[green]✓[/green]" if pdf_cached else ""
                    row_data.append(pdf_status)

                if show_md:
                    md_status = "[green]✓[/green]" if md_cached else ""
                    row_data.append(md_status)

                if show_chunks:
                    chunks_status = "[green]✓[/green]" if chunks_cached else ""
                    row_data.append(chunks_status)

                table.add_row(*row_data)
            elif verbose and dry_run:
                # In dry run mode, show placeholders for all visible columns
                row_data = [str(i), url]
                if show_html:
                    row_data.append("[dim]?[/dim]")
                if show_pdf:
                    row_data.append("[dim]?[/dim]")
                if show_md:
                    row_data.append("[dim]?[/dim]")
                if show_chunks:
                    row_data.append("[dim]?[/dim]")
                table.add_row(*row_data)
            else:
                table.add_row(str(i), url)

        console.print(table)

    if dry_run:
        console.print("[yellow]Dry run mode - no URLs will be fetched[/yellow]")
        return

    # Fetch URLs
    asyncio.run(fetch_urls_async(urls, cfg, verbose, fetch_only, failfast, retry))


async def perform_document_grouping(
    urls: List[str],
    config: IfetcherConfig,
    constraint: str,
    retry: bool,
) -> List[Dict[str, Any]]:
    """
    Perform document grouping asynchronously.

    Args:
        urls: List of URLs to group
        config: Configuration for PageFetcher
        constraint: Grouping constraint string
        retry: Whether to retry failed URLs

    Returns:
        List of document groups
    """
    from .fetcher import PageFetcher

    fetcher = PageFetcher(config, show_status=False)
    with console.status("[bold blue]Computing document similarities..."):
        groups = await fetcher.get_groups(
            urls,
            constraint=constraint,
            prefetch=True,
            progress=False,  # Don't show nested progress bar
            retry=retry,
        )
    return groups


async def check_urls_cache_status(
    urls: List[str], config: IfetcherConfig
) -> Dict[str, Dict[str, bool]]:
    """
    Check cache status for multiple URLs.

    Args:
        urls: List of URLs to check
        config: Configuration for PageFetcher

    Returns:
        Dictionary mapping URLs to their cache status for each content type
    """
    from .fetcher import PageFetcher

    fetcher = PageFetcher(config, show_status=False)
    cache_status = {}
    html_count = 0
    pdf_count = 0
    md_count = 0
    chunks_count = 0
    total_nothing = 0

    for url in urls:
        status = {}
        try:
            status["html"] = await fetcher.cache.has_path(url, "html")
            status["pdf"] = await fetcher.cache.has_path(url, "pdf")
            status["markdown"] = await fetcher.cache.has_path(url, "markdown")
            status["chunks"] = await fetcher.cache.has_path(url, "chunks")

            # Count content types for statistics
            if status["html"]:
                html_count += 1
            if status["pdf"]:
                pdf_count += 1
            if status["markdown"]:
                md_count += 1
            if status["chunks"]:
                chunks_count += 1

            # Count URLs with nothing cached
            if not any(status.values()):
                total_nothing += 1
        except Exception:
            # If there's any error checking cache, assume nothing is cached
            status = {"html": False, "pdf": False, "markdown": False, "chunks": False}
            total_nothing += 1

        cache_status[url] = status

    # Add statistics for determining most common content type and column visibility
    cache_status["_stats"] = {
        "html_count": html_count,
        "pdf_count": pdf_count,
        "md_count": md_count,
        "chunks_count": chunks_count,
        "total_nothing": total_nothing,
    }

    return cache_status


async def fetch_urls_async(
    urls: List[str],
    config: IfetcherConfig,
    verbose: bool = False,
    fetch_only: bool = False,
    failfast: bool = False,
    retry: bool = False,
):
    """Fetch URLs asynchronously using PageFetcher."""
    # In CLI runs, always show status spinners for single-URL operations
    fetcher = PageFetcher(config, show_status=True, verbose=verbose)

    # Check which URLs are already cached
    already_cached = 0
    for url in urls:
        if await fetcher.cache.has_path(url, "chunks"):
            already_cached += 1

    urls_to_fetch = len(urls) - already_cached

    if already_cached > 0:
        if fetch_only:
            console.print(
                f"[bold green]Fetching and caching {urls_to_fetch} URLs...[/bold green]"
            )
        else:
            console.print(f"[bold green]Fetching {urls_to_fetch} URLs...[/bold green]")
    else:
        if fetch_only:
            console.print(
                f"[bold green]Fetching and caching {len(urls)} URLs...[/bold green]"
            )
        else:
            console.print(f"[bold green]Fetching {len(urls)} URLs...[/bold green]")

    try:
        if fetch_only:
            # Use the concurrent fetching method that returns error info
            results = await fetcher._fetch_multiple(
                urls, "chunks", progress=True, fail_fast=failfast, retry=retry
            )

            # Separate successful and failed results
            successful = []
            failed = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed.append((urls[i], result))
                else:
                    successful.append(result)

            total_successful = len(successful)
            newly_cached = total_successful - already_cached

            if already_cached > 0:
                console.print(f"[cyan]- Already cached: {already_cached}[/cyan]")
            console.print(f"[green]✓ Successfully cached: {newly_cached}[/green]")
            if failed:
                console.print(f"[red]✗ Failed to cache: {len(failed)}[/red]")
                display_error_summary(failed, verbose)
                if failfast:
                    raise typer.Exit(1)

            console.print("[dim]URLs have been fetched and cached for later use[/dim]")
        else:
            # Use the concurrent fetching method that returns error info
            results = await fetcher._fetch_multiple(
                urls, "chunks", progress=True, fail_fast=failfast, retry=retry
            )

            # Separate successful and failed results
            successful = []
            failed = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed.append((urls[i], result))
                else:
                    successful.append(result)

            total_successful = len(successful)
            newly_fetched = total_successful - already_cached

            if already_cached > 0:
                console.print(f"[cyan]- Already cached: {already_cached}[/cyan]")
            console.print(f"[green]✓ Successfully fetched: {newly_fetched}[/green]")
            if failed:
                console.print(f"[red]✗ Failed to fetch: {len(failed)}[/red]")
                display_error_summary(failed, verbose)
                if failfast:
                    raise typer.Exit(1)

            if verbose and successful:
                content_lengths = [len(str(result)) for result in successful]
                console.print(
                    f"[dim]Fetched content lengths: {content_lengths} characters[/dim]"
                )

    except Exception as e:
        handle_operation_error("during URL fetching", e)
        raise typer.Exit(1)


@app.command()
def config_info(
    config: Optional[str] = typer.Option(
        None, "-c", "--config", help="Path to configuration file"
    ),
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
):
    """Show current configuration information."""
    try:
        cfg = load_config(config, mode)

        table = Table(title="Configuration Information")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="blue")

        table.add_row("Cache Path", str(cfg.abspath(cfg.output.cache)))
        table.add_row("Training Data Template", cfg.training_data)
        table.add_row("Max Concurrent Requests", str(cfg.tools.crawl4ai.max_concurrent))
        table.add_row("Request Timeout", f"{cfg.tools.crawl4ai.timeout}s")
        table.add_row("Max Retries", str(cfg.tools.crawl4ai.max_retries))
        table.add_row(
            "Delay Between Requests", f"{cfg.tools.crawl4ai.delay_between_requests}s"
        )

        if cfg.modes:
            table.add_row("Available Modes", ", ".join(cfg.modes.keys()))

        console.print(table)

    except Exception as e:
        handle_operation_error("loading configuration", e)
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
