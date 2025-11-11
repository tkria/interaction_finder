"""
Fetch command implementation for interaction-finder CLI.

This module provides URL collection and validation logic for the fetch subcommand,
supporting both command-line argument input and file-based input. It also provides
fetch orchestration using PageFetcher for markdown and chunk content retrieval.
"""

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from interaction_finder.fetcher import PageFetcher, URLCache
from interaction_finder.settings import IfetcherConfig


def read_urls_from_file(path: Path) -> List[str]:
    """
    Read URLs from a text file, one URL per line.

    Empty lines and leading/trailing whitespace are stripped and ignored.
    Supports UTF-8 encoding for international URLs.

    Parameters:
        path: Path to the URL file

    Returns:
        List of URL strings (whitespace stripped, empty lines removed)

    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    # Check file existence and raise clear error
    if not path.exists():
        raise FileNotFoundError(f"URL file not found: {path}")

    # Read file with UTF-8 encoding
    with open(path, "r", encoding="utf-8") as f:
        # Strip whitespace and filter empty lines
        urls = []
        for line in f:
            stripped = line.strip()
            if stripped:  # Skip empty lines
                urls.append(stripped)

    return urls


def collect_urls(
    args_urls: Optional[List[str]], input_file: Optional[Path]
) -> List[str]:
    """
    Collect URLs from command-line arguments or file input.

    Enforces mutual exclusion: either provide URLs via args_urls OR via input_file,
    but not both. At least one source must be provided.

    Parameters:
        args_urls: List of URLs from command-line arguments (or None/empty)
        input_file: Path to file containing URLs (or None)

    Returns:
        List of URL strings ready for fetching

    Raises:
        ValueError: If both args_urls and input_file provided (mutual exclusion)
        ValueError: If no URLs provided (neither args nor file)
        FileNotFoundError: If input_file specified but does not exist
    """
    # Normalize args_urls to empty list if None
    if args_urls is None:
        args_urls = []

    # Check mutual exclusion: both sources provided
    has_args = bool(args_urls)
    has_file = input_file is not None

    if has_args and has_file:
        raise ValueError(
            "Cannot specify both command-line URLs and --input file. "
            "Please provide URLs via one method only."
        )

    # Check empty input: neither source provided
    if not has_args and not has_file:
        raise ValueError(
            "No URLs provided. Please specify URLs via command-line arguments "
            "or use --input to provide a file."
        )

    # Collect URLs from appropriate source
    if has_file:
        # Read from file (raises FileNotFoundError if missing)
        return read_urls_from_file(input_file)
    else:
        # Return args URLs directly
        return args_urls


@dataclass
class FetchResult:
    """Result of fetching a single URL with success/failure tracking."""

    url: str
    cache_path: Optional[Path]
    success: bool
    error: Optional[str]


async def fetch_urls_markdown(
    fetcher: PageFetcher, urls: List[str], verbose: bool
) -> List[FetchResult]:
    """
    Fetch markdown content for multiple URLs with per-URL success/failure tracking.

    Uses PageFetcher.fetch_documents for batch fetching with progress display.
    Retrieves cache paths using URLCache.get_file_path for successful fetches.

    Parameters:
        fetcher: PageFetcher instance configured with cache and client
        urls: List of URLs to fetch
        verbose: Show progress bar during fetching

    Returns:
        List of FetchResult objects, one per URL in input order

    Raises:
        No exceptions raised - failures recorded in FetchResult.error field
    """
    # Fetch all URLs with progress display controlled by verbose flag
    documents = await fetcher.fetch_documents(urls, progress=verbose)

    # Build results with cache paths for successful fetches
    results = []
    for url in urls:
        doc = documents.get(url)
        if doc is not None:
            # Success: document fetched
            # Retrieve cache path using URLCache for reliable path resolution
            cache_path = await fetcher.cache.get_file_path(url, "markdown")
            results.append(
                FetchResult(
                    url=url,
                    cache_path=cache_path,
                    success=True,
                    error=None,
                )
            )
        else:
            # Failure: document is None
            results.append(
                FetchResult(
                    url=url,
                    cache_path=None,
                    success=False,
                    error="Failed to fetch URL",
                )
            )

    return results


async def fetch_urls_chunks(
    fetcher: PageFetcher, urls: List[str], verbose: bool
) -> List[FetchResult]:
    """
    Fetch and chunk content for multiple URLs with per-URL success/failure tracking.

    Uses PageFetcher.get_chunks_with_embeddings for semantic chunking with embeddings.
    Retrieves cache paths using URLCache.get_file_path for successful fetches.

    Parameters:
        fetcher: PageFetcher instance configured with cache and client
        urls: List of URLs to fetch and chunk
        verbose: Show progress bar during fetching

    Returns:
        List of FetchResult objects, one per URL in input order

    Raises:
        No exceptions raised - failures recorded in FetchResult.error field
    """
    results = []

    for url in urls:
        try:
            # Attempt to fetch and chunk the URL
            _chunks = await fetcher.get_chunks_with_embeddings(
                url, progress=verbose, fail_fast=False
            )

            # Success: chunks returned
            # Retrieve cache path using URLCache
            cache_path = await fetcher.cache.get_file_path(url, "chunks")
            results.append(
                FetchResult(
                    url=url,
                    cache_path=cache_path,
                    success=True,
                    error=None,
                )
            )
        except Exception as e:
            # Failure: exception raised during chunking
            results.append(
                FetchResult(
                    url=url,
                    cache_path=None,
                    success=False,
                    error=f"Failed to chunk URL: {str(e)}",
                )
            )

    return results


async def run_fetch(
    config: IfetcherConfig, urls: List[str], chunk: bool, verbose: bool
) -> List[FetchResult]:
    """
    Orchestrate fetch operation for multiple URLs with markdown or chunk mode.

    Creates PageFetcher instance and routes to appropriate fetch function based
    on chunk flag. Handles both markdown content fetching and semantic chunking.

    Parameters:
        config: Configuration with cache directory and fetcher settings
        urls: List of URLs to fetch
        chunk: If True, perform chunking; if False, fetch markdown only
        verbose: Show progress bars and status messages

    Returns:
        List of FetchResult objects with success/failure tracking per URL

    Raises:
        No exceptions raised - all failures recorded in results
    """
    # Extract configuration values
    cache_dir = config.abspath(config.output.cache)
    timeout = config.tools.crawl4ai.timeout

    # Create PageFetcher with direct parameters
    fetcher = PageFetcher(
        cache_dir=cache_dir, timeout=timeout, show_status=verbose, verbose=verbose
    )

    # Route to appropriate fetch function based on mode
    if chunk:
        return await fetch_urls_chunks(fetcher, urls, verbose)
    else:
        return await fetch_urls_markdown(fetcher, urls, verbose)


def output_paths(results: List[FetchResult], console: Console) -> int:
    """
    Output cache paths for successful fetches to stdout (one per line).

    Prints absolute cache paths for all successful fetches to stdout, making output
    suitable for piping to other commands. Failed URLs are printed as warnings to
    stderr via Rich console.

    Parameters:
        results: List of fetch results with success/failure tracking
        console: Rich console for error/warning output (not stdout)

    Returns:
        Exit code: 0 if any fetch succeeded, 1 if all failed
    """
    # Separate successful and failed results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    # Print cache paths to stdout (one per line, absolute paths)
    for result in successful:
        if result.cache_path:
            # Use print() for stdout, not console (which may use stderr)
            print(str(result.cache_path.resolve()))

    # Print warnings for failed URLs to console (stderr)
    for result in failed:
        console.print(f"[yellow]Warning:[/yellow] Failed to fetch {result.url}")
        if result.error:
            console.print(f"  Reason: {result.error}")

    # Return exit code: 0 if any success, 1 if all failed
    if successful:
        return 0
    else:
        return 1


def output_content(results: List[FetchResult], console: Console) -> int:
    """
    Output markdown content for a single URL to stdout.

    Reads cached markdown content and prints to stdout for piping or viewing.
    Enforces single-URL constraint - raises error if multiple URLs provided.

    Parameters:
        results: List of fetch results (must contain exactly one URL)
        console: Rich console for error output

    Returns:
        Exit code: 0 if successful, 1 if failed

    Raises:
        ValueError: If results contains multiple URLs (content mode requires single URL)
    """
    # Validate single URL constraint
    if len(results) != 1:
        raise ValueError(
            f"Content output requires exactly one URL, got {len(results)}. "
            "Use --format paths for multiple URLs."
        )

    result = results[0]

    # Check if fetch was successful
    if not result.success:
        console.print(f"[red]Error:[/red] Failed to fetch {result.url}")
        if result.error:
            console.print(f"Reason: {result.error}")
        return 1

    # Read and output markdown content
    if result.cache_path and result.cache_path.exists():
        content = result.cache_path.read_text(encoding="utf-8")
        # Use print() for stdout, not console
        print(content, end="")  # No extra newline - preserve exact content
        return 0
    else:
        console.print(f"[red]Error:[/red] Cache file not found: {result.cache_path}")
        return 1


def print_fetch_errors(results: List[FetchResult], console: Console) -> None:
    """
    Display grouped error summary for failed fetches.

    Groups errors by type and displays them in a structured format using Rich.
    Shows individual URL failures with error reasons.

    Parameters:
        results: List of fetch results with failure tracking
        console: Rich console for formatted error output
    """
    # Filter failed results
    failed = [r for r in results if not r.success]

    if not failed:
        return  # No errors to display

    # Create error summary table
    table = Table(title="Fetch Errors", show_header=True, header_style="bold red")
    table.add_column("URL", style="cyan", no_wrap=False)
    table.add_column("Error", style="red")

    for result in failed:
        table.add_row(result.url, result.error or "Unknown error")

    console.print(table)


def print_summary(results: List[FetchResult], console: Console) -> None:
    """
    Display verbose summary of fetch operation results.

    Shows success/failure counts, lists failed URLs with reasons, and displays
    summary statistics in a Rich panel.

    Parameters:
        results: List of fetch results with success/failure tracking
        console: Rich console for formatted summary output
    """
    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful

    # Build summary text
    summary_lines = [
        f"Total URLs: {total}",
        f"Successful: {successful}",
        f"Failed: {failed}",
    ]

    # Add failed URL details if any
    if failed > 0:
        summary_lines.append("")
        summary_lines.append("[bold]Failed URLs:[/bold]")
        for result in results:
            if not result.success:
                summary_lines.append(f"  • {result.url}")
                if result.error:
                    summary_lines.append(f"    Reason: {result.error}")

    # Display summary in panel
    summary_text = "\n".join(summary_lines)
    panel = Panel(
        summary_text,
        title="[bold]Fetch Summary[/bold]",
        border_style="blue",
        expand=False,
    )
    console.print(panel)


def match_url_pattern(url: str, pattern: str) -> bool:
    """
    Check if URL matches a pattern using prefix, substring, or glob matching.

    Matching rules:
    - If pattern contains wildcards (*), use glob matching
    - Otherwise, use case-insensitive substring matching
    - "https://" prefix in pattern is optional (e.g., "pubmed" matches "https://pubmed.ncbi.nlm.nih.gov/...")

    Parameters:
        url: URL to check
        pattern: Pattern to match against (with optional https:// prefix)

    Returns:
        True if URL matches pattern, False otherwise
    """
    # Normalize both for case-insensitive comparison
    url_lower = url.lower()
    pattern_lower = pattern.lower()
    # Check for glob pattern (only * triggers glob, not ? since ? is common in URL query params)
    if "*" in pattern_lower:
        return fnmatch(url_lower, pattern_lower)
    # Substring matching (covers prefix matching as well)
    return pattern_lower in url_lower


async def clear_cache_urls(
    cache: URLCache, patterns: List[str], console: Console, dry_run: bool
) -> tuple[int, int]:
    """
    Clear cached URLs matching the provided patterns.

    If patterns is empty, clears ALL cached URLs (requires confirmation in caller).

    Parameters:
        cache: URLCache instance
        patterns: List of URL patterns to match (empty = clear all)
        console: Rich console for output
        dry_run: If True, show what would be deleted without deleting

    Returns:
        Tuple of (cleared_count, error_count)
    """
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
    )

    # List all cached URLs
    console.print("\n[dim]Loading cached URLs...[/dim]")
    all_urls = await cache.list_cached_urls()
    console.print(f"Total cached URLs: {len(all_urls)}")
    # Filter URLs based on patterns
    if not patterns:
        # No patterns = clear all
        matching_urls = all_urls
        console.print(
            f"\n[yellow]⚠ WARNING:[/yellow] Clearing ALL {len(all_urls)} cached URLs!"
        )
    else:
        # Match URLs against patterns
        matching_urls = []
        for url in all_urls:
            if any(match_url_pattern(url, pattern) for pattern in patterns):
                matching_urls.append(url)
        console.print(
            f"\n[bold]URLs matching patterns:[/bold] {len(matching_urls)} of {len(all_urls)}"
        )
    if not matching_urls:
        console.print("[green]No URLs to clear.[/green]")
        return 0, 0
    # Show sample of URLs to be deleted
    if len(matching_urls) <= 10:
        console.print("\n[dim]URLs to be cleared:[/dim]")
        for url in matching_urls:
            console.print(f"  • {url}")
    else:
        console.print(
            f"\n[dim]Sample of URLs to be cleared (showing first 10 of {len(matching_urls)}):[/dim]"
        )
        for url in matching_urls[:10]:
            console.print(f"  • {url}")
        console.print(f"  [dim]... and {len(matching_urls) - 10} more[/dim]")
    if dry_run:
        console.print(f"\n[yellow]Dry run mode:[/yellow] No files were deleted.")
        console.print(
            f"Run without --dry-run to actually clear {len(matching_urls)} URLs."
        )
        return 0, 0
    # Clear URLs with progress indication
    console.print(f"\n[bold]Clearing {len(matching_urls)} URLs...[/bold]")
    cleared_count = 0
    error_count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Clearing...", total=len(matching_urls))
        for url in matching_urls:
            try:
                await cache.clear_url(url)
                cleared_count += 1
            except Exception:
                error_count += 1
            progress.update(task, advance=1)
    return cleared_count, error_count
