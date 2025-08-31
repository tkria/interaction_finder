"""
CLI interface for Interaction Finder using Typer.

This module provides command-line interface functionality for the interaction finder tool,
including a train sub-command for fetching URLs from training data files.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.traceback import Traceback
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

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


def parse_config_override(override: str) -> tuple[str, Any]:
    """
    Parse a config override string into path and value.

    Args:
        override: String in format "path.to.key=value"

    Returns:
        Tuple of (dotted_path, parsed_value)

    Raises:
        ValueError: If override format is invalid
    """
    if "=" not in override:
        raise ValueError(
            f"Invalid override format: {override}. Expected 'path.to.key=value'"
        )

    path, value_str = override.split("=", 1)

    # Parse value with basic type inference
    value_str = value_str.strip()

    # Boolean values
    if value_str.lower() in ("true", "false"):
        value = value_str.lower() == "true"
    # Integer values
    elif value_str.isdigit():
        value = int(value_str)
    # Float values
    elif "." in value_str and value_str.replace(".", "").isdigit():
        value = float(value_str)
    # String values (remove quotes if present)
    else:
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            value = value_str[1:-1]
        else:
            value = value_str

    return path.strip(), value


def apply_config_overrides(config_data: dict, overrides: List[str]) -> dict:
    """
    Apply configuration overrides to config data.

    Args:
        config_data: Base configuration dictionary
        overrides: List of override strings

    Returns:
        Modified configuration dictionary
    """
    if not overrides:
        return config_data

    for override in overrides:
        try:
            path, value = parse_config_override(override)

            # Navigate to the nested dictionary location
            current = config_data
            path_parts = path.split(".")

            # Navigate to parent of target key
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the final value
            final_key = path_parts[-1]
            current[final_key] = value

            console.print(f"[dim]Override applied: {path} = {value}[/dim]")

        except Exception as e:
            console.print(f"[red]Error applying override '{override}': {e}[/red]")
            raise typer.Exit(1)

    return config_data


def load_config(
    config_path: Optional[str] = None,
    mode: Optional[str] = None,
    overrides: List[str] = None,
) -> IfetcherConfig:
    """Load configuration from file or use defaults."""
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            handle_operation_error(
                "loading configuration", f"Configuration file {config_path} not found"
            )
            raise typer.Exit(1)

        # Load TOML data and apply overrides before creating config
        import tomli

        config_data = tomli.loads(config_file.read_text("utf-8"))
        config_data = apply_config_overrides(config_data, overrides or [])

        # Apply mode overrides if specified
        if mode and "modes" in config_data and mode in config_data["modes"]:
            mode_overrides = config_data["modes"][mode]
            if isinstance(mode_overrides, dict):
                config_data = IfetcherConfig.apply_overrides(
                    config_data, mode_overrides
                )

        config = IfetcherConfig.model_validate(config_data)
        config._dir = config_file.parent
        return config
    else:
        # Try to find a config file in common locations
        for potential_config in [
            "config.toml",
            "interaction_finder.toml",
            ".interaction_finder.toml",
        ]:
            if Path(potential_config).exists():
                console.print(f"[dim]Using config file: {potential_config}[/dim]")

                # Load TOML data and apply overrides
                import tomli

                config_file = Path(potential_config)
                config_data = tomli.loads(config_file.read_text("utf-8"))
                config_data = apply_config_overrides(config_data, overrides or [])

                # Apply mode overrides if specified
                if mode and "modes" in config_data and mode in config_data["modes"]:
                    mode_overrides = config_data["modes"][mode]
                    if isinstance(mode_overrides, dict):
                        config_data = IfetcherConfig.apply_overrides(
                            config_data, mode_overrides
                        )

                config = IfetcherConfig.model_validate(config_data)
                config._dir = config_file.parent
                return config

        # Use default configuration with overrides
        console.print("[dim]Using default configuration[/dim]")
        config_data = {}
        config_data = apply_config_overrides(config_data, overrides or [])
        return IfetcherConfig.model_validate(config_data)


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


def extract_urls_from_plaintext(file_path: Path) -> tuple[List[str], int]:
    """
    Extract URLs from a plaintext file (one URL per line).

    Args:
        file_path: Path to plaintext file containing URLs

    Returns:
        tuple: (unique_urls, total_urls_found)
    """
    all_urls = []
    url_pattern = re.compile(r'https?://[^\s<>"]+')

    if not file_path.exists():
        handle_operation_error("reading URL file", f"URL file {file_path} not found")
        raise typer.Exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Check if line contains a URL
                if url_pattern.match(line):
                    all_urls.append(line)
                else:
                    console.print(
                        f"[yellow]Warning: Line {line_num} does not appear to be a valid URL, skipping: {line[:50]}...[/yellow]"
                    )

    except Exception as e:
        handle_operation_error("reading URL file", e)
        raise typer.Exit(1)

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls, len(all_urls)


def parse_input_source(
    input_source: str, config: IfetcherConfig, term: Optional[str] = None
) -> tuple[List[str], str]:
    """
    Parse input source to extract URLs and determine source type.

    Args:
        input_source: Either a direct URL or path to file containing URLs
        config: Configuration for path resolution
        term: Optional term for template resolution in file paths

    Returns:
        tuple: (urls_list, source_type) where source_type is 'url', 'jsonl', or 'plaintext'
    """
    url_pattern = re.compile(r'https?://[^\s<>"]+')

    # Check if input is a direct URL
    if url_pattern.match(input_source):
        return [input_source], "url"

    # Otherwise, treat as file path
    # Use config.abspath for consistent path resolution relative to config file
    if term:
        file_path = config.abspath(input_source, term=term)
    else:
        file_path = config.abspath(input_source)

    if not file_path.exists():
        handle_operation_error(
            "reading input file", f"Input file {file_path} not found"
        )
        raise typer.Exit(1)

    # Determine file type based on extension or content
    if file_path.suffix.lower() == ".jsonl" or file_path.name.endswith(".jsonl"):
        urls, total_entries = extract_urls_from_jsonl(file_path)
        return urls, "jsonl"
    else:
        # Try to detect JSONL by checking first non-empty line
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Try to parse as JSON
                        try:
                            json.loads(line)
                            # If successful, treat as JSONL
                            urls, total_entries = extract_urls_from_jsonl(file_path)
                            return urls, "jsonl"
                        except json.JSONDecodeError:
                            # Not JSON, treat as plaintext
                            break
        except Exception:
            pass

        # Default to plaintext
        urls, total_entries = extract_urls_from_plaintext(file_path)
        return urls, "plaintext"


def scan_available_terms(config: IfetcherConfig) -> List[str]:
    """
    Scan filesystem to find available terms based on training_data template.

    Similar to Julia's scanresultsdir, this extracts terms from file paths
    that match the training_data template pattern.

    Args:
        config: Configuration containing training_data template

    Returns:
        List of available term strings found in filesystem
    """
    import glob

    template = config.training_data
    if "{term}" not in template:
        return []

    # Convert template to glob pattern
    # e.g., "training_data/{term}.jsonl" -> "training_data/*.jsonl"
    glob_pattern = template.replace("{term}", "*")

    # Resolve pattern relative to config directory
    pattern_path = config.abspath(glob_pattern)

    terms = []
    for file_path in glob.glob(str(pattern_path)):
        # Extract term from file path by reversing the template
        relative_path = Path(os.path.relpath(file_path, config._dir or Path.cwd()))

        # Simple pattern matching for {term} extraction
        # This assumes template has {term} in filename, not nested in directories
        if template.count("/") == str(relative_path).count("/"):
            template_parts = template.split("/")
            path_parts = str(relative_path).split("/")

            for template_part, path_part in zip(template_parts, path_parts):
                if "{term}" in template_part:
                    # Extract term by removing the non-{term} parts
                    prefix = template_part.split("{term}")[0]
                    suffix = template_part.split("{term}")[1]

                    if path_part.startswith(prefix) and path_part.endswith(suffix):
                        term = path_part[
                            len(prefix) : len(path_part) - len(suffix)
                            if suffix
                            else len(path_part)
                        ]
                        if term:  # Only add non-empty terms
                            terms.append(term)

    return sorted(set(terms))  # Remove duplicates and sort


@app.command()
def terms(
    config: Optional[str] = typer.Option(
        None, "-c", "--config", help="Path to configuration file"
    ),
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
):
    """
    List available terms that have training data files.

    Scans the filesystem based on the training_data template to find
    all terms that have corresponding data files.

    Examples:
        interaction-finder terms
        interaction-finder terms --config my_config.toml
    """
    try:
        cfg = load_config(config, mode)
        available_terms = scan_available_terms(cfg)

        if not available_terms:
            console.print("[yellow]No terms found[/yellow]")
            console.print(f"[dim]Template: {cfg.training_data}[/dim]")
            console.print(
                f"[dim]Resolved to: {cfg.abspath(cfg.training_data.replace('{term}', '*'))}[/dim]"
            )
        else:
            console.print(
                f"[bold blue]Available terms ({len(available_terms)}):[/bold blue]"
            )
            for term in available_terms:
                file_path = cfg.abspath(cfg.training_data, term=term)
                console.print(f"  • {term} → {file_path}")

    except Exception as e:
        handle_operation_error("scanning for terms", e)
        raise typer.Exit(1)


async def check_cache_status(urls: List[str], cfg: IfetcherConfig) -> tuple[int, int]:
    """Check how many URLs are already cached vs need to be fetched."""
    fetcher = PageFetcher(cfg)
    cached_count = 0

    for url in urls:
        if await fetcher.is_cached(url):
            cached_count += 1

    return cached_count, len(urls) - cached_count


async def _run_dry_run_display(
    urls: List[str],
    source_type: str,
    source: str,
    cfg: IfetcherConfig,
    fetch_only: bool,
    term: Optional[str],
    output: Optional[str],
    mode: Optional[str],
    model: Optional[str],
    verbose: bool,
    parallelism: Optional[int],
) -> None:
    """Run the dry-run display with cache status checking."""
    # Calculate display path for source
    if source_type != "url":
        resolved_path = cfg.abspath(source, term=term) if term else cfg.abspath(source)
        display_path = os.path.relpath(resolved_path, Path.cwd())
    else:
        display_path = source

    # Calculate output path for display
    output_display = None
    if not fetch_only:
        if output is None:
            template_vars = {
                "term": term or "unknown",
                "mode": mode or "default",
                "model": model
                or str(cfg.agents.get("_", cfg.AgentSpec()).llm or "gpt-4o").replace(
                    ":", "_"
                ),
                "repeat": 1,
            }
            resolved_output = cfg.abspath(cfg.output.path + ".jsonl", **template_vars)
            output_display = os.path.relpath(resolved_output, Path.cwd())
        else:
            resolved_output = cfg.abspath(output)
            output_display = os.path.relpath(resolved_output, Path.cwd())

    # Check cache status for better dry-run information
    cached_count, uncached_count = await check_cache_status(urls, cfg)

    # Use the new organized display function
    print_dry_run_summary(
        urls=urls,
        source_type=source_type,
        display_path=display_path,
        cfg=cfg,
        fetch_only=fetch_only,
        cached_count=cached_count,
        term=term,
        output_path=output_display,
        verbose=verbose,
        parallelism=parallelism,
        model=model,
    )


# Styling helpers for dry-run display
def styled_key(text: str) -> str:
    """Style a key in key-value pairs."""
    return f"[dim]{text}[/dim]"


def styled_path(path: str, source: Optional[str] = None) -> str:
    """Style a file path with optional source indicator."""
    styled = f"[cyan]{path}[/cyan]"
    if source:
        styled += f" [dim][{source}][/dim]"
    return styled


def styled_count(number: int) -> str:
    """Style a count or number."""
    return f"[bright_white]{number}[/bright_white]"


def styled_percentage(text: str) -> str:
    """Style a percentage value."""
    return f"[yellow]{text}[/yellow]"


def styled_model(model: str) -> str:
    """Style a model name."""
    return f"[magenta]{model}[/magenta]"


def styled_config(value: str, source: Optional[str] = None) -> str:
    """Style a configuration value with optional source indicator."""
    styled = f"[green]{value}[/green]"
    if source:
        styled += f" [dim][{source}][/dim]"
    return styled


def print_dry_run_summary(
    urls: List[str],
    source_type: str,
    display_path: str,
    cfg: IfetcherConfig,
    fetch_only: bool,
    cached_count: Optional[int] = None,
    term: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = False,
    parallelism: Optional[int] = None,
    model: Optional[str] = None,
) -> None:
    """Print organized dry-run summary using Rich panels."""

    # Input Source Section
    source_table = Table(show_header=False, box=None, padding=(0, 1))
    source_table.add_row(styled_key("Source:"), styled_path(display_path, "config"))
    source_table.add_row(
        styled_key("Type:"),
        f"{source_type.upper()} file" if source_type != "url" else "Direct URL",
    )
    source_table.add_row(styled_key("URLs:"), styled_count(len(urls)))

    # Add cache status if available
    if cached_count is not None:
        cache_percentage = (cached_count / len(urls) * 100) if urls else 0
        uncached_count = len(urls) - cached_count
        cache_status = f"{styled_count(cached_count)}/{styled_count(len(urls))} cached {styled_percentage(f'({cache_percentage:.0f}%)')}"
        source_table.add_row(styled_key("Cache status:"), cache_status)
        if uncached_count > 0:
            source_table.add_row(
                styled_key("To fetch:"), f"{styled_count(uncached_count)} new URLs"
            )

    # Add URL preview inside panel if verbose
    if verbose and urls:
        source_table.add_row("", "")  # Empty row for spacing
        source_table.add_row("[bold]Preview:[/bold]", "")
        for i, url in enumerate(urls[:10], 1):
            source_table.add_row(f"  {i:2d}.", url)
        if len(urls) > 10:
            source_table.add_row("", f"... and {len(urls) - 10} more URLs")

    console.print(
        Panel(
            source_table,
            title="[bold blue]Input Source[/bold blue]",
            border_style="blue",
        )
    )

    if not fetch_only:
        # Configuration Section
        config_table = Table(show_header=False, box=None, padding=(0, 1))
        config_table.add_row(
            styled_key("Entity types:"),
            styled_config(", ".join(cfg.task.get_kind_names()), "config"),
        )
        config_table.add_row(
            styled_key("Relation:"), styled_config(cfg.task.relation, "config")
        )
        if cfg.task.context:
            config_table.add_row(
                styled_key("Context:"), styled_config(cfg.task.context, "config")
            )
        if term:
            config_table.add_row(styled_key("Term:"), styled_config(term, "cli"))

        # Show model information
        if model:
            model_display = model
            model_source = "cli"
        else:
            default_agent = cfg.agents.get("_", cfg.AgentSpec())
            model_display = default_agent.llm or "openai:gpt-4o"
            model_source = "config" if default_agent.llm else "default"
        config_table.add_row(
            styled_key("Model:"),
            styled_model(model_display) + f" [dim][{model_source}][/dim]",
        )

        console.print(
            Panel(
                config_table,
                title="[bold green]Extraction Configuration[/bold green]",
                border_style="green",
            )
        )

    # Processing Plan Section
    plan_table = Table(show_header=False, box=None, padding=(0, 1))
    mode_text = "Content fetching only" if fetch_only else "Full extraction pipeline"
    plan_table.add_row(styled_key("Mode:"), mode_text)

    if parallelism:
        plan_table.add_row(
            styled_key("Parallelism:"),
            f"{styled_count(parallelism)} concurrent requests [dim][cli][/dim]",
        )
    elif not fetch_only:
        plan_table.add_row(
            styled_key("Parallelism:"),
            f"{styled_count(cfg.tools.crawl4ai.max_concurrent)} concurrent requests [dim][default][/dim]",
        )

    if output_path:
        plan_table.add_row(styled_key("Output:"), styled_path(output_path, "config"))

    console.print(
        Panel(
            plan_table,
            title="[bold yellow]Processing Plan[/bold yellow]",
            border_style="yellow",
        )
    )

    # Show verbose hint if not in verbose mode
    if not verbose and urls:
        console.print(
            f"\n[dim]Use --verbose to see URL list inside Input Source panel[/dim]"
        )


@app.command()
def extract(
    term: Optional[str] = typer.Option(
        None,
        "--term",
        "-t",
        help="Query term for extraction context (e.g., BRCA1, diabetes)",
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        "-s",
        help="URL to process or path to file containing URLs (JSONL or plaintext). If not provided, uses training_data from config.",
    ),
    fetch_only: bool = typer.Option(
        False,
        "--fetch-only",
        help="Only fetch and cache content without running extraction pipeline",
    ),
    output: Optional[str] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file path (defaults to extraction_results.jsonl)",
    ),
    config: Optional[str] = typer.Option(
        None, "-c", "--config", help="Path to configuration file"
    ),
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override AI model (e.g., openai:gpt-4o)"
    ),
    parallelism: Optional[int] = typer.Option(
        None, "--parallelism", "-p", help="Maximum parallel operations"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be processed without executing"
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Show verbose output and pretty tracebacks on errors",
    ),
    failfast: bool = typer.Option(
        False, "--failfast", help="Stop on first error during processing"
    ),
    retry: bool = typer.Option(
        False, "--retry", help="Force retry of URLs previously marked as failed"
    ),
    overrides: List[str] = typer.Option(
        [],
        "-O",
        "--override",
        help="Override config values using dotted paths (e.g., -O agents.llm=openai:gpt-4)",
    ),
):
    """
    Extract entity interactions from web content.

    The primary workflow uses training data from configuration files. Custom sources
    can be specified with --source for processing arbitrary URLs or files.

    Source formats (when using --source):
    - Direct URL: https://pubmed.ncbi.nlm.nih.gov/123456
    - JSONL file: File with JSON objects containing URL fields
    - Plaintext file: One URL per line (# comments are ignored)

    Processing modes:
    - Default: Full pipeline including content fetching, document grouping, and entity extraction
    - --fetch-only: Only fetch and cache content for later use

    Examples:
        # Primary workflow: use training data from config
        interaction-finder extract -t BRCA1
        interaction-finder extract -t diabetes --fetch-only

        # Custom source with term context
        interaction-finder extract --source urls.txt -t "diabetes"
        interaction-finder extract -s https://pubmed.ncbi.nlm.nih.gov/123456 -t BRCA1

        # Template paths (resolved relative to config file)
        interaction-finder extract --source training_data/{term}.jsonl -t BRCA1

        # Full extraction with custom settings
        interaction-finder extract -t BRCA1 -o results.jsonl --model openai:gpt-4 -p 10

        # Preview processing
        interaction-finder extract -t diabetes --dry-run --verbose
    """
    # Load configuration first to handle case where no input_source is provided
    try:
        cfg = load_config(config, mode, overrides)

        # Update task context with term if provided
        if term and not cfg.task.context:
            cfg.task.context = f"Analysis focused on {term}"
        elif term and cfg.task.context and term not in cfg.task.context:
            cfg.task.context = f"{cfg.task.context} (term: {term})"

    except Exception as e:
        handle_operation_error("loading configuration", e)
        raise typer.Exit(1)

    # Handle case where no source is provided - use training_data from config
    if source is None:
        if not term:
            # Show available terms and require term selection
            available_terms = scan_available_terms(cfg)
            if available_terms:
                console.print(
                    "[yellow]No source provided. Available terms for training data:[/yellow]"
                )
                console.print(
                    f"[blue]Available terms:[/blue] {', '.join(available_terms[:10])}"
                )
                if len(available_terms) > 10:
                    console.print(
                        f"[dim]... and {len(available_terms) - 10} more (use 'interaction-finder terms' to see all)[/dim]"
                    )
                console.print(
                    f"[blue]Usage:[/blue] interaction-finder extract -t {available_terms[0]}"
                )
            else:
                console.print(
                    "[yellow]No source provided and no training data found.[/yellow]"
                )
                console.print(
                    "[blue]Usage:[/blue] interaction-finder extract --source <URL_or_file> -t <term>"
                )
            return
        else:
            # Use training_data template with the provided term
            source = cfg.training_data
            # For display purposes, resolve the template and make it relative to cwd
            resolved_path = (
                cfg.abspath(source, term=term) if term else cfg.abspath(source)
            )
            display_source = os.path.relpath(resolved_path, Path.cwd())

    # Parse input to get URLs
    try:
        urls, source_type = parse_input_source(source, cfg, term)

        # Set display_source for all paths if not already set
        if "display_source" not in locals():
            if source_type != "url":
                resolved_path = (
                    cfg.abspath(source, term=term) if term else cfg.abspath(source)
                )
                display_source = os.path.relpath(resolved_path, Path.cwd())
    except Exception as e:
        # If the error is due to template without term, provide helpful suggestion
        if "{term}" in source and not term:
            try:
                available_terms = scan_available_terms(cfg)

                console.print(
                    f"[yellow]Template path requires --term parameter: {source}[/yellow]"
                )
                if available_terms:
                    console.print(
                        f"[blue]Available terms:[/blue] {', '.join(available_terms[:10])}"
                    )
                    if len(available_terms) > 10:
                        console.print(
                            f"[dim]... and {len(available_terms) - 10} more (use 'interaction-finder terms' to see all)[/dim]"
                        )
                    console.print(
                        f"[blue]Example:[/blue] interaction-finder extract --source {source} -t {available_terms[0]}"
                    )
                else:
                    console.print(
                        "[dim]No training data files found. Use 'interaction-finder terms' to see available terms.[/dim]"
                    )
            except Exception:
                pass  # Fall back to original error

        handle_operation_error("parsing input source", e)
        raise typer.Exit(1)

    if not urls:
        console.print("[yellow]No URLs found in input source[/yellow]")
        return

    # Override max_concurrent if provided
    if parallelism is not None:
        cfg.tools.crawl4ai.max_concurrent = parallelism

    if dry_run:
        # Run async dry-run display
        asyncio.run(
            _run_dry_run_display(
                urls,
                source_type,
                source,
                cfg,
                fetch_only,
                term,
                output,
                mode,
                model,
                verbose,
                parallelism,
            )
        )
        return

    # Execute based on mode
    if fetch_only:
        # Use existing fetch_urls_async function
        asyncio.run(fetch_urls_async(urls, cfg, verbose, fetch_only, failfast, retry))
    else:
        # Run full extraction pipeline
        try:
            # Set up output path
            if output is None:
                # Use config's output.path template with available variables
                template_vars = {
                    "term": term or "unknown",
                    "mode": mode or "default",
                    "model": model
                    or str(
                        cfg.agents.get("_", cfg.AgentSpec()).llm or "gpt-4o"
                    ).replace(":", "_"),
                    "repeat": 1,  # Could be made configurable in future
                }
                output_path = cfg.abspath(cfg.output.path + ".jsonl", **template_vars)
            else:
                output_path = cfg.abspath(output)

            # Show extraction start summary
            summary_table = Table(show_header=False, box=None, padding=(0, 1))
            summary_table.add_row(styled_key("URLs:"), styled_count(len(urls)))
            summary_table.add_row(
                styled_key("Target:"),
                styled_config(
                    f"{', '.join(cfg.task.get_kind_names())} → {cfg.task.relation}"
                ),
            )
            summary_table.add_row(
                styled_key("Output:"),
                styled_path(os.path.relpath(output_path, Path.cwd())),
            )

            console.print(
                Panel(
                    summary_table,
                    title="[bold green]Starting Extraction Pipeline[/bold green]",
                    border_style="green",
                )
            )

            # Run extraction using extraction_graph
            from .extraction_graph import extract_from_urls
            from .fetcher import PageFetcher

            # Create PageFetcher with appropriate settings
            page_fetcher = PageFetcher(cfg, show_status=True, verbose=verbose)

            # Run extraction
            result = asyncio.run(
                _run_extraction_pipeline(
                    urls=urls, config=cfg, page_fetcher=page_fetcher, model=model
                )
            )

            # Save results
            from .extraction_graph import save_results

            save_results(result, output_path)

            # Display summary using existing function
            _display_extraction_summary(result)

        except Exception as e:
            if verbose:
                console.print(Traceback(show_locals=True))
            else:
                console.print(f"[bold red]Error:[/bold red] {e}", style="red")
            raise typer.Exit(1)


def _display_extraction_summary(result: Any):
    """Display extraction results summary."""

    # Create summary table with consistent styling
    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_row(
        styled_key("Document Groups:"), styled_count(result.total_groups)
    )
    summary_table.add_row(
        styled_key("Successful Groups:"), styled_count(result.successful_groups)
    )
    summary_table.add_row(
        styled_key("Total Entity Pairs:"), styled_count(result.total_pairs)
    )

    # Add entity counts by kind
    for kind, count in result.total_entities.items():
        summary_table.add_row(
            styled_key(f"{kind.title()} Entities:"), styled_count(count)
        )

    if result.errors:
        error_style = "[red]" + str(len(result.errors)) + "[/red]"
        summary_table.add_row(styled_key("Errors:"), error_style)

    console.print(
        Panel(
            summary_table,
            title="[bold green]Extraction Complete[/bold green]",
            border_style="green",
        )
    )

    # Show errors if any - grouped by type
    if result.errors:
        console.print()  # Add spacing
        error_groups = {}
        for error in result.errors:
            error_msg = error.get("error", "Unknown error")
            # Group similar errors together
            if error_msg in error_groups:
                error_groups[error_msg] += 1
            else:
                error_groups[error_msg] = 1

        error_table = Table(show_header=False, box=None, padding=(0, 1))
        for error_msg, count in list(error_groups.items())[
            :3
        ]:  # Show top 3 error types
            if count > 1:
                error_table.add_row(
                    f"[red]•[/red]", f"{error_msg} [dim]({count} occurrences)[/dim]"
                )
            else:
                error_table.add_row(f"[red]•[/red]", error_msg)

        if len(error_groups) > 3:
            total_remaining = sum(list(error_groups.values())[3:])
            error_table.add_row(
                f"[red]•[/red]",
                f"[dim]... and {len(error_groups) - 3} more error types ({total_remaining} total)[/dim]",
            )

        console.print(
            Panel(
                error_table,
                title="[bold red]Errors Encountered[/bold red]",
                border_style="red",
            )
        )


async def _run_extraction_pipeline(
    urls: List[str],
    config: IfetcherConfig,
    page_fetcher: "PageFetcher",
    model: Optional[str],
) -> Any:
    """Run the entity extraction pipeline."""
    from .extraction_graph import extract_from_urls

    # Run extraction
    result = await extract_from_urls(
        urls=urls, config=config, page_fetcher=page_fetcher, model=model
    )

    return result


async def perform_document_grouping(
    urls: List[str],
    config: IfetcherConfig,
    grouping_config: Any,  # Will be Workflow.Grouping instance
    retry: bool,
) -> List[Dict[str, Any]]:
    """
    Perform document grouping asynchronously.

    Args:
        urls: List of URLs to group
        config: Configuration for PageFetcher
        grouping_config: Grouping configuration object
        retry: Whether to retry failed URLs

    Returns:
        List of document groups
    """
    from .fetcher import PageFetcher

    fetcher = PageFetcher(config, show_status=True)
    groups = await fetcher.get_groups(
        urls,
        constraint_type=grouping_config.constraint_type,
        min_size=grouping_config.min_size,
        max_size=grouping_config.max_size,
        linkage_method=grouping_config.linkage_method,
        clustering_method=grouping_config.clustering_method,
        embedding_weights=grouping_config.embedding_weights,
        seeding_method=grouping_config.seeding_method,
        refinement_method=grouping_config.refinement_method,
        prefetch=True,
        progress=True,  # Enable progress display
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


# Create a subcommand group for config operations
config_app = typer.Typer(help="Configuration management commands")
app.add_typer(config_app, name="config")


@config_app.command("info")
def config_info(
    config: Optional[str] = typer.Option(
        None, "-c", "--config", help="Path to configuration file"
    ),
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    overrides: List[str] = typer.Option(
        [],
        "-O",
        help="Override config values using dotted paths (e.g., -O workflow.grouping.enabled=false)",
    ),
):
    """Show current configuration information."""
    try:
        cfg = load_config(config, mode, overrides)

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

        # Grouping configuration
        table.add_row(
            "Document Grouping",
            "Enabled" if cfg.workflow.grouping.enabled else "Disabled",
        )
        if cfg.workflow.grouping.enabled:
            table.add_row(
                "Grouping Constraint Type", cfg.workflow.grouping.constraint_type
            )
            table.add_row(
                "Grouping Size Range",
                f"{cfg.workflow.grouping.min_size}-{cfg.workflow.grouping.max_size}",
            )
            table.add_row(
                "Grouping Linkage Method", cfg.workflow.grouping.linkage_method
            )

        if cfg.modes:
            table.add_row("Available Modes", ", ".join(cfg.modes.keys()))

        console.print(table)

    except Exception as e:
        handle_operation_error("loading configuration", e)
        raise typer.Exit(1)


@config_app.command("edit")
def config_edit(
    config: Optional[str] = typer.Option(
        None, "-c", "--config", help="Path to configuration file to edit"
    ),
):
    """Open interactive configuration editor."""
    from .settings_editor import run_config_editor

    run_config_editor(config)


@config_app.command("validate")
def config_validate(
    config: Optional[str] = typer.Option(
        None, "-c", "--config", help="Path to configuration file to validate"
    ),
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to validate"
    ),
):
    """Validate configuration file without loading full settings."""
    try:
        cfg = load_config(config, mode)
        console.print(f"[green]✓ Configuration is valid[/green]")

        # Show some basic info
        config_path = Path(config) if config else Path("config.toml")
        if config_path.exists():
            console.print(f"[dim]Config file: {config_path}[/dim]")
            console.print(f"[dim]File size: {config_path.stat().st_size} bytes[/dim]")
        else:
            console.print(f"[dim]Using default configuration[/dim]")

        if mode:
            console.print(f"[dim]Mode: {mode}[/dim]")

    except Exception as e:
        console.print(f"[red]✗ Configuration validation failed:[/red]")
        console.print(f"[red]  {e}[/red]")
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
