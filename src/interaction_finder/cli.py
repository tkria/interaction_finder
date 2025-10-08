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
from typing import Optional, List, Dict, Any, NamedTuple

import typer
import click
from rich.console import Console
from rich.traceback import Traceback
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import logfire
except ImportError:
    # Create a no-op logfire if not available
    class _NoOpLogfire:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    logfire = _NoOpLogfire()
from rich.table import Table
from rich.panel import Panel

from .settings import IfetcherConfig, configure_logfire
from .fetcher import PageFetcher
from .search import SearchQuery, SearchCache
from .search.backends import PubMedBackend, PerplexicaBackend, OpenAISearchBackend
from .search.expansion import create_llm_expander, create_advanced_expander
from .search.diversification import create_diversified_queries, DiversifiedQuery
from .search.evaluation import (
    EvaluationRunner,
    EvaluationConfig,
    DEFAULT_BIOMEDICAL_QUERIES,
    DEFAULT_GENERAL_QUERIES,
)

app = typer.Typer(
    name="interaction-finder",
    help="A tool for fetching and processing web content for interaction discovery.",
    rich_markup_mode="rich",
    add_completion=False,  # Disable default completion flags
)

console = Console()

# Initialize configuration models after all imports
# Import SearchConfig and make it available for forward references
from .search.config import SearchConfig
from . import settings
import sys

# Make SearchConfig available for forward references in settings
setattr(settings, "SearchConfig", SearchConfig)
IfetcherConfig.model_rebuild()


class GlobalOptions(NamedTuple):
    """Common global options for all commands."""

    config: typer.Option = typer.Option(
        None, "-c", "--config", help="Path to configuration file"
    )
    verbose: typer.Option = typer.Option(
        False, "-v", "--verbose", help="Show verbose output"
    )
    overrides: typer.Option = typer.Option(
        [],
        "-O",
        "--override",
        help="Override config values using dotted paths (e.g., -O tools.search.backend=pubmed)",
    )


GLOBAL_OPTIONS = GlobalOptions()


# Global options that apply to all subcommands
@app.callback()
def main(
    config: Optional[str] = GLOBAL_OPTIONS.config,
    verbose: bool = GLOBAL_OPTIONS.verbose,
    overrides: List[str] = GLOBAL_OPTIONS.overrides,
):
    """
    A tool for fetching and processing web content for interaction discovery.

    Global options like --config, --verbose, and --override can be used with any subcommand.
    """
    # Store options in the context - they will be used as fallbacks if not provided locally
    ctx = click.get_current_context()
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose
    ctx.obj["overrides"] = overrides


def get_options_with_fallback(
    config: Optional[str] = None,
    verbose: Optional[bool] = None,
    overrides: Optional[List[str]] = None,
):
    """
    Get options, using global values as fallback for None/empty local values.

    Args:
        config: Local config value
        verbose: Local verbose value
        overrides: Local overrides value

    Returns:
        Tuple of (effective_config, effective_verbose, effective_overrides)
    """
    ctx = click.get_current_context()
    if not ctx.obj:
        return config, verbose or False, overrides or []

    # Use local values if provided, otherwise fall back to global
    effective_config = config if config is not None else ctx.obj.get("config")
    effective_verbose = (
        verbose if verbose is not None else ctx.obj.get("verbose", False)
    )
    effective_overrides = overrides if overrides else ctx.obj.get("overrides", [])

    return effective_config, effective_verbose, effective_overrides


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
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    config: Optional[str] = GLOBAL_OPTIONS.config,
    verbose: bool = GLOBAL_OPTIONS.verbose,
    overrides: List[str] = GLOBAL_OPTIONS.overrides,
):
    """
    List available terms that have training data files.

    Scans the filesystem based on the training_data template to find
    all terms that have corresponding data files.

    Examples:
        interaction-finder terms
        interaction-finder terms --config my_config.toml
    """
    # Get effective options with global fallback
    effective_config, effective_verbose, effective_overrides = (
        get_options_with_fallback(config, verbose, overrides)
    )

    try:
        cfg = load_config(effective_config, mode, effective_overrides)
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


@app.command()
def search(
    query: str = typer.Argument(help="Search query for finding relevant papers"),
    backend: Optional[str] = typer.Option(
        None,
        "-b",
        "--backend",
        help="Search backend to use (alias for -O tools.search.backend=VALUE)",
    ),
    max_results: Optional[int] = typer.Option(
        None, "-n", "--max-results", help="Maximum number of results to return"
    ),
    output_format: str = typer.Option(
        "table", "-f", "--format", help="Output format: table, json, jsonl, urls, csv"
    ),
    save_results: Optional[str] = typer.Option(
        None, "-o", "--output", help="Save results to file"
    ),
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    config: Optional[str] = GLOBAL_OPTIONS.config,
    verbose: bool = GLOBAL_OPTIONS.verbose,
    overrides: List[str] = GLOBAL_OPTIONS.overrides,
):
    """
    Search for academic papers and documents.

    This command searches academic databases for papers related to your query.
    Results can be displayed in different formats or saved for later use.

    Examples:
        interaction-finder search "BRCA1 mutations"
        interaction-finder search "p53 interactions" --backend pubmed --max-results 50
        interaction-finder search "diabetes" --format json -O tools.search.backend=openai_search
        interaction-finder search "TNF alpha" --output results.json
    """
    # Get effective options with global fallback
    effective_config, effective_verbose, effective_overrides = (
        get_options_with_fallback(config, verbose, overrides)
    )

    # Handle --backend as a config override
    final_overrides = list(effective_overrides)
    if backend is not None:
        final_overrides.append(f"tools.search.backend={backend}")

    asyncio.run(
        _run_search_command(
            query=query,
            max_results=max_results,
            output_format=output_format,
            save_results=save_results,
            config=effective_config,
            mode=mode,
            overrides=final_overrides,
            verbose=effective_verbose,
        )
    )


async def _apply_query_expansion(
    query: str, search_query: SearchQuery, cfg: IfetcherConfig, verbose: bool = False
) -> None:
    """Apply query expansion to search query if enabled in configuration."""
    if not cfg.tools.search.expansion.enabled:
        return

    if verbose:
        console.print(f"[dim]Expanding query: '{query}'[/dim]")

    # Advanced expansion is now always used when expansion is enabled
    use_advanced = cfg.tools.search.expansion.enabled

    if use_advanced:
        # Use advanced expansion method (parameters now in main expansion config)
        expander = create_advanced_expander(
            model_name=cfg.tools.search.expansion.model_name,
            max_terms=cfg.tools.search.expansion.max_expansion_terms,
            per_category_cap=3,  # sensible default
            mmr_lambda=0.7,  # sensible default
            temperature=cfg.tools.search.expansion.temperature,
            deterministic_seed=42,  # sensible default
        )

        expansion_context = {
            "max_terms": cfg.tools.search.expansion.max_expansion_terms,
            "per_category_cap": 3,  # sensible default
            "mmr_lambda": 0.7,  # sensible default
            "min_expansion_terms": cfg.tools.search.expansion.min_expansion_terms,
            "min_confidence": cfg.tools.search.expansion.min_confidence,
            "domain": "biomedical research",
        }

        expansion_result = await expander.expand_query(query, expansion_context)

        # Convert enhanced result to standard format for compatibility
        standard_result = expansion_result.to_expanded_query()
        search_query.expanded_terms = [
            term.term for term in standard_result.expanded_terms
        ]

        if verbose:
            if expansion_result.expansion_terms:
                console.print(
                    f"[dim]Intent: {expansion_result.intent_card.split('1)')[1].split('2)')[0].strip()}[/dim]"
                )
                console.print(
                    f"[dim]Expanded terms ({len(expansion_result.expansion_terms)}): "
                    f"{', '.join(term.term for term in expansion_result.expansion_terms)}[/dim]"
                )
                if expansion_result.hyde_text:
                    console.print(
                        f"[dim]HyDE surrogate generated ({len(expansion_result.hyde_text)} chars)[/dim]"
                    )
            else:
                console.print("[dim]No expansion terms generated[/dim]")

    else:
        # Use standard LLM expansion method (using consolidated configs)
        expander = create_llm_expander(
            model_name=cfg.tools.search.expansion.model_name,
            max_terms=cfg.tools.search.expansion.max_expansion_terms,
            min_confidence=cfg.tools.search.expansion.min_confidence,
            expansion_types=["synonyms", "related", "abbreviations"],
            domain_context="biomedical research",
            timeout_seconds=30,
        )

        expansion_context = {
            "max_expansion_terms": cfg.tools.search.expansion.max_expansion_terms,
            "min_expansion_terms": cfg.tools.search.expansion.min_expansion_terms,
            "min_confidence": cfg.tools.search.expansion.min_confidence,
        }
        # Add LLM-specific context
        expansion_context.update(llm_config)
        expansion_result = await expander.expand_query(query, expansion_context)
        search_query.expanded_terms = [
            term.term for term in expansion_result.expanded_terms
        ]

        if verbose and expansion_result.expanded_terms:
            console.print(
                f"[dim]Expanded terms: {', '.join(term.term for term in expansion_result.expanded_terms)}[/dim]"
            )


async def _run_search_for_extract(
    query: str,
    backend: Optional[str],
    max_results: Optional[int],
    cfg: IfetcherConfig,
    term: Optional[str],
    verbose: bool,
):
    """Run search specifically for integration with extract command."""
    # Use configured backend if none specified via --backend
    if backend is None:
        backend = cfg.tools.search.backend

    # Create search query with query expansion enabled by default for extract
    search_query = SearchQuery(
        query=query,
        max_results=max_results or 50,  # Default to 50 for extract
    )

    # Apply query expansion automatically
    await _apply_query_expansion(query, search_query, cfg, verbose)

    # Create and run search backend
    backend_config = cfg.tools.search.get_backend_config(backend)

    if backend == "pubmed":
        search_backend = PubMedBackend(backend_config)
    elif backend == "perplexica":
        search_backend = PerplexicaBackend(backend_config)
    elif backend == "openai_search":
        search_backend = OpenAISearchBackend(backend_config)
    else:
        available_backends = cfg.tools.search.get_available_backends()
        console.print(f"[red]Backend '{backend}' is not implemented[/red]")
        console.print(
            f"[blue]Available backends:[/blue] {', '.join(available_backends)}"
        )
        raise typer.Exit(1)

    # Display search request details in verbose mode
    if verbose:
        _display_search_request(search_query, backend)

    try:
        async with search_backend:
            results = await search_backend.search(search_query)
            return results
    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        if verbose:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


async def _run_search_command(
    query: str,
    max_results: Optional[int] = None,
    output_format: str = "table",
    save_results: Optional[str] = None,
    config: Optional[str] = None,
    mode: Optional[str] = None,
    overrides: List[str] = None,
    verbose: bool = False,
) -> None:
    """Run the search command asynchronously."""
    try:
        # Load configuration with overrides
        cfg = load_config(config, mode, overrides or [])

        # Get backend from configuration (may be overridden by --backend)
        backend = cfg.tools.search.backend

        # Set up search cache if enabled
        cache = None
        if cfg.tools.search.cache.enabled:
            cache_dir = cfg.abspath(cfg.output.cache) / "search"
            cache = SearchCache(cache_dir, cfg.tools.search.cache.ttl_hours)

        # Determine search strategy
        search_strategy = getattr(
            cfg.tools.search.expansion, "search_strategy", "single_query"
        )

        use_multi_query = (
            cfg.tools.search.expansion.enabled and search_strategy == "multi_query"
        )

        use_review_informed = (
            cfg.tools.search.expansion.enabled and search_strategy == "review_informed"
        )

        use_pubmed_mesh = (
            cfg.tools.search.expansion.enabled and search_strategy == "pubmed_mesh"
        )

        # Check if MeSH expansion should be used in other strategies
        use_mesh_expansion = (
            cfg.tools.search.expansion.enabled
            and cfg.tools.search.expansion.use_mesh_expansion
            and backend == "pubmed"
            and not use_pubmed_mesh  # Don't double-apply
        )

        if use_mesh_expansion and verbose:
            console.print("[dim]MeSH expansion enabled for PubMed queries[/dim]")

        try:
            if use_pubmed_mesh:
                # Use PubMed MeSH strategy: LLM decomposition + MeSH expansion per component
                if verbose:
                    console.print(
                        "[dim]Using PubMed MeSH strategy: LLM decomposition + MeSH expansion[/dim]"
                    )

                # Force PubMed backend
                if backend != "pubmed":
                    if verbose:
                        console.print(
                            "[yellow]Switching to PubMed backend for MeSH strategy[/yellow]"
                        )
                    backend = "pubmed"

                target_results = max_results or cfg.tools.search.max_results

                results = await _perform_pubmed_mesh_strategy(
                    original_query=query,
                    cfg=cfg,
                    max_results=target_results,
                    verbose=verbose,
                    cache=cache,
                )
            elif use_review_informed:
                # Use review-informed search strategy
                if verbose:
                    console.print(
                        "[dim]Using review-informed search strategy for expert-guided coverage[/dim]"
                    )

                # Pass the full max_results - review-informed search will handle per-query limits internally
                target_results = max_results or cfg.tools.search.max_results

                results = await _perform_review_informed_search(
                    original_query=query,
                    backend=backend,
                    cfg=cfg,
                    max_results_per_query=target_results,
                    verbose=verbose,
                    cache=cache,
                )
            elif use_multi_query:
                # Use multi-query deep search
                if verbose:
                    console.print(
                        "[dim]Using multi-query search strategy for deep coverage[/dim]"
                    )

                max_per_query = (
                    max_results or cfg.tools.search.max_results
                ) // cfg.tools.search.expansion.target_searches
                max_per_query = max(max_per_query, 10)  # Minimum 10 results per query

                results = await _perform_multi_query_search(
                    original_query=query,
                    backend=backend,
                    cfg=cfg,
                    max_results_per_query=max_per_query,
                    verbose=verbose,
                    cache=cache,
                )
            else:
                # Use single-query approach (original behavior)
                search_query = SearchQuery(
                    query=query, max_results=max_results or cfg.tools.search.max_results
                )

                # Apply traditional query expansion
                await _apply_query_expansion(query, search_query, cfg, verbose)

                # Check cache first
                if cache:
                    cached_results = await cache.get(search_query, backend)
                    if cached_results:
                        if verbose:
                            console.print(
                                f"[dim]Found cached results for '{query}' with {backend}[/dim]"
                            )
                        _display_search_results(
                            cached_results, output_format, verbose, save_results
                        )
                        return

                # Perform single search
                results = await _perform_single_search(
                    search_query, backend, cfg, cache, verbose
                )

                # Cache results if cache is enabled
                if cache:
                    await cache.set(search_query, backend, results)

            # Display results
            _display_search_results(results, output_format, verbose, save_results)

        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")
            if verbose:
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise typer.Exit(1)

    except Exception as e:
        handle_operation_error("during search", e)
        raise typer.Exit(1)


async def _perform_multi_query_search(
    original_query: str,
    backend: str,
    cfg: IfetcherConfig,
    max_results_per_query: int = 20,
    verbose: bool = False,
    cache: Optional[SearchCache] = None,
) -> "SearchResults":
    """Perform multi-query deep search using advanced expansion."""
    from .search.base import SearchResults, SearchResult

    # Step 1: Get advanced expansion
    if not cfg.tools.search.expansion.enabled:
        # Fallback to single query if expansion disabled
        search_query = SearchQuery(
            query=original_query, max_results=max_results_per_query
        )
        return await _perform_single_search(search_query, backend, cfg, cache, verbose)

    # Use advanced expansion (parameters now in main expansion config)
    expander = create_advanced_expander(
        model_name=cfg.tools.search.expansion.model_name,
        max_terms=cfg.tools.search.expansion.max_expansion_terms,
        per_category_cap=3,  # sensible default
        mmr_lambda=0.7,  # sensible default
        temperature=cfg.tools.search.expansion.temperature,
        deterministic_seed=42,  # sensible default
    )

    expansion_context = {
        "max_terms": cfg.tools.search.expansion.max_expansion_terms,
        "per_category_cap": 3,  # sensible default
        "mmr_lambda": 0.7,  # sensible default
        "min_expansion_terms": cfg.tools.search.expansion.min_expansion_terms,
        "min_confidence": cfg.tools.search.expansion.min_confidence,
        "domain": "biomedical research",
    }

    if verbose:
        console.print(f"[dim]Generating diverse queries for: '{original_query}'[/dim]")

    expansion_result = await expander.expand_query(original_query, expansion_context)

    # Step 2: Generate diversified queries
    target_searches = cfg.tools.search.expansion.target_searches
    diversified_queries = await create_diversified_queries(
        expansion_result, target_queries=target_searches, context=expansion_context
    )

    if verbose:
        console.print(
            f"[dim]Generated {len(diversified_queries)} diverse search queries[/dim]"
        )
        for i, div_query in enumerate(diversified_queries, 1):
            console.print(
                f"[dim]{i}. {div_query.focus}: {div_query.query[:60]}{'...' if len(div_query.query) > 60 else ''}[/dim]"
            )

    # Step 3: Execute all searches
    backend_config = cfg.tools.search.get_backend_config(backend)

    if backend == "pubmed":
        search_backend = PubMedBackend(backend_config)
    elif backend == "perplexica":
        search_backend = PerplexicaBackend(backend_config)
    elif backend == "openai_search":
        search_backend = OpenAISearchBackend(backend_config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    all_results = []
    successful_queries = 0

    async with search_backend:
        for i, div_query in enumerate(diversified_queries, 1):
            try:
                search_query = SearchQuery(
                    query=div_query.query,
                    max_results=max_results_per_query,
                )

                # Check cache first
                cached_result = None
                if cache:
                    cached_result = await cache.get(search_query, backend)

                if cached_result:
                    results = cached_result
                    if verbose:
                        console.print(
                            f"[dim]Query {i}/{len(diversified_queries)} (cached): {div_query.focus}[/dim]"
                        )
                else:
                    results = await search_backend.search(search_query)
                    if cache:
                        await cache.set(search_query, backend, results)
                    if verbose:
                        console.print(
                            f"[dim]Query {i}/{len(diversified_queries)}: {div_query.focus} -> {len(results.results)} results[/dim]"
                        )

                # Add metadata to results indicating the query strategy
                for result in results.results:
                    result.metadata = result.metadata or {}
                    result.metadata["query_focus"] = div_query.focus
                    result.metadata["query_rationale"] = div_query.rationale
                    result.metadata["query_weight"] = div_query.weight

                all_results.extend(results.results)
                successful_queries += 1

            except Exception as e:
                if verbose:
                    console.print(
                        f"[yellow]Query {i} failed ({div_query.focus}): {e}[/yellow]"
                    )
                continue

    # Step 4: Combine and deduplicate results
    unique_results = {}
    for result in all_results:
        # Use URL or title as deduplication key
        key = result.url or result.title
        if key not in unique_results:
            unique_results[key] = result
        else:
            # Merge metadata from multiple query strategies
            existing = unique_results[key]
            if existing.metadata and result.metadata:
                existing_focuses = existing.metadata.get("query_focus", "")
                new_focus = result.metadata.get("query_focus", "")
                if new_focus and new_focus not in existing_focuses:
                    existing.metadata["query_focus"] = (
                        f"{existing_focuses}, {new_focus}"
                    )

    final_results = list(unique_results.values())

    # Create combined SearchResults
    combined_query = SearchQuery(query=original_query, max_results=len(final_results))
    combined_results = SearchResults(
        query=combined_query,
        results=final_results,
        total_found=len(final_results),
        search_time=0.0,  # Would need to track timing
        backend=backend,
    )

    if verbose:
        console.print(
            f"[green]Multi-query search completed: {successful_queries}/{len(diversified_queries)} queries successful[/green]"
        )
        console.print(
            f"[green]Total results: {len(all_results)} raw -> {len(final_results)} unique[/green]"
        )

    return combined_results


async def _perform_review_informed_search(
    original_query: str,
    backend: str,
    cfg: IfetcherConfig,
    max_results_per_query: int = 20,
    verbose: bool = False,
    cache: Optional[SearchCache] = None,
) -> "SearchResults":
    """Perform review-informed search using review papers to guide query generation."""
    from .search.review_informed import create_review_informed_search
    from .fetcher import PageFetcher

    # Get simplified review-informed configuration (only the few specific settings)
    review_config = cfg.tools.search.expansion.review_informed.model_dump()

    # Build LLM configuration from consolidated expansion config
    llm_config = {
        "model_name": cfg.tools.search.expansion.model_name,
        "temperature": cfg.tools.search.expansion.temperature,
        "max_terms": cfg.tools.search.expansion.target_searches,
        "min_confidence": cfg.tools.search.expansion.min_confidence,
        "domain_context": "biomedical research",
        "timeout_seconds": 30,
    }

    # Create PageFetcher for content retrieval
    fetcher = PageFetcher(cfg, show_status=verbose, verbose=verbose)

    # Get search backend
    from .search.backends.pubmed import PubMedBackend
    from .search.backends.openai_search import OpenAISearchBackend
    from .search.backends.perplexica import PerplexicaBackend

    backend_config = cfg.tools.search.get_backend_config(backend)

    if backend == "pubmed":
        search_backend = PubMedBackend(backend_config)
    elif backend == "openai_search":
        search_backend = OpenAISearchBackend(backend_config)
    elif backend == "perplexica":
        search_backend = PerplexicaBackend(backend_config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Get agent specification for LLM configuration
    agent_spec = cfg.agents.get("_")  # Default agent spec

    try:
        # Perform review-informed search
        results = await create_review_informed_search(
            original_query=original_query,
            backend=search_backend,
            fetcher=fetcher,
            review_config=review_config,
            llm_config=llm_config,
            agent_spec=agent_spec,
            max_results_per_query=max_results_per_query,
            verbose=verbose,
        )

        return results

    except Exception as e:
        if verbose:
            console.print(f"[red]Review-informed search failed: {e}[/red]")
        # Fallback to single search
        search_query = SearchQuery(
            query=original_query, max_results=max_results_per_query
        )
        return await _perform_single_search(search_query, backend, cfg, cache, verbose)

    finally:
        # Clean up resources
        await search_backend.close()


async def _apply_mesh_expansion_to_query(
    query: str,
    cfg: IfetcherConfig,
    verbose: bool = False,
) -> str:
    """Apply PubMed MeSH co-occurrence expansion to a query.

    Returns the expanded query string, or original if expansion fails/disabled.
    """
    from .search.expansion import create_pubmed_mesh_expander

    # Get PubMed MeSH expansion config
    mesh_config = cfg.tools.search.expansion.pubmed_mesh.model_dump()

    # Get PubMed backend config for credentials
    pubmed_config = cfg.tools.search.pubmed.model_dump()

    # Merge NCBI credentials
    mesh_config["email"] = pubmed_config.get("email")
    mesh_config["api_key"] = pubmed_config.get("api_key")

    try:
        # Create MeSH expander
        expander = create_pubmed_mesh_expander(**mesh_config)

        # Expand the query
        expanded_query = await expander.expand_query(query)

        if verbose and expanded_query.expanded_terms:
            query_preview = query[:50] + "..." if len(query) > 50 else query
            console.print(f"[dim]  → MeSH expansion for '{query_preview}':[/dim]")
            for term in expanded_query.expanded_terms:
                console.print(f"[dim]     • {term.term}[/dim]")

        # Build expanded search query
        if expanded_query.expanded_terms:
            # Combine original query with MeSH terms using OR
            mesh_terms = " OR ".join(
                f'"{term.term}"' for term in expanded_query.expanded_terms
            )
            full_query = f"({query}) OR ({mesh_terms})"
        else:
            full_query = query

        await expander.close()
        return full_query

    except Exception as e:
        if verbose:
            console.print(f"[yellow]MeSH expansion failed: {e}[/yellow]")
        return query


async def _perform_pubmed_mesh_strategy(
    original_query: str,
    cfg: IfetcherConfig,
    max_results: int = 100,
    verbose: bool = False,
    cache: Optional[SearchCache] = None,
) -> "SearchResults":
    """
    Perform PubMed MeSH strategy: LLM decomposition + MeSH expansion per component.

    Workflow:
    1. Use LLM to decompose query into 3-5 focused components
    2. Apply MeSH expansion to each component
    3. Search PubMed with each expanded query
    4. Merge results with frequency tracking
    """
    from .search.backends.pubmed import PubMedBackend
    from .search.base import SearchQuery, SearchResults
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent
    from collections import Counter

    class QueryComponent(BaseModel):
        """A focused component of the original query."""

        component: str = Field(description="Focused search component")
        rationale: str = Field(description="Why this component is important")

    class QueryDecomposition(BaseModel):
        """Decomposition of a complex query into key concepts."""

        components: list[QueryComponent] = Field(
            description="2-4 key concepts extracted from the query",
            min_length=2,
            max_length=4,
        )

    # Step 1: Decompose query with LLM
    if verbose:
        console.print("[dim]📋 Decomposing query into focused components...[/dim]")

    agent = Agent(
        cfg.tools.search.expansion.model_name,
        output_type=QueryDecomposition,
        instructions="""
        You are a biomedical search expert. Extract the KEY CONCEPTS from the user's query.
        DO NOT invent new concepts or add domain knowledge - only extract what's explicitly present.

        Rules:
        - Identify the main entities/concepts mentioned in the query
        - Keep the original terminology (don't paraphrase or elaborate)
        - Separate different concepts (e.g., separate "genes" from "disease name")
        - Return 2-4 components (usually 2-3 is sufficient)
        - Each component should be a distinct concept that can be independently expanded with MeSH terms

        Examples:

        Query: "Genes associated with Pulmonary Arterial Hypertension"
        Components:
        1. "Genes" (genetic/molecular concept)
        2. "Pulmonary Arterial Hypertension" (disease concept)

        Query: "BRCA1 mutations in breast cancer treatment response"
        Components:
        1. "BRCA1" (specific gene)
        2. "breast cancer" (disease)
        3. "treatment response" (outcome)

        Query: "microRNA regulation of inflammation"
        Components:
        1. "microRNA" (molecular entity)
        2. "inflammation" (biological process)

        Query: "diabetes complications"
        Components:
        1. "diabetes" (disease)
        2. "complications" (disease outcomes)
        """,
    )

    try:
        decomp_result = await agent.run(f"Decompose this query: {original_query}")
        decomposition = decomp_result.output

        if verbose:
            console.print(
                f"[dim]Found {len(decomposition.components)} components:[/dim]"
            )
            for i, comp in enumerate(decomposition.components, 1):
                console.print(f"  [dim]{i}. {comp.component}[/dim]")

    except Exception as e:
        if verbose:
            console.print(
                f"[yellow]Decomposition failed: {e}, using original query[/yellow]"
            )
        decomposition = QueryDecomposition(
            components=[QueryComponent(component=original_query, rationale="Fallback")]
        )

    # Step 2: Apply MeSH expansion to each component
    if verbose:
        console.print(
            f"[dim]🔬 Applying MeSH expansion to {len(decomposition.components)} components...[/dim]"
        )

    expanded_components = []
    for i, comp in enumerate(decomposition.components, 1):
        if verbose:
            console.print(f"[dim]Component {i}: {comp.component}[/dim]")

        expanded_query = await _apply_mesh_expansion_to_query(
            comp.component, cfg, verbose
        )
        expanded_components.append(expanded_query)

    # Step 3: Combine all expanded components with AND
    if verbose:
        console.print(
            f"[dim]🔗 Combining {len(expanded_components)} expanded components with AND...[/dim]"
        )

    # Join all expanded components with AND
    combined_query = " AND ".join(f"({comp})" for comp in expanded_components)

    if verbose:
        # Show a preview of the combined query
        preview = (
            combined_query
            if len(combined_query) <= 200
            else combined_query[:200] + "..."
        )
        console.print(f"[dim]Combined query: {preview}[/dim]")

    # Step 4: Single search with combined query
    pubmed_config = cfg.tools.search.pubmed.model_dump()
    backend = PubMedBackend(pubmed_config)
    search_query = SearchQuery(query=combined_query, max_results=max_results)

    try:
        results = await backend.search(search_query)
        await backend.close()

        if verbose:
            console.print(f"[dim]✓ Found {len(results.results)} results[/dim]")

        return results

    except Exception as e:
        await backend.close()
        if verbose:
            console.print(f"[red]Search failed: {e}[/red]")
        # Fallback to original query
        backend = PubMedBackend(pubmed_config)
        search_query = SearchQuery(query=original_query, max_results=max_results)
        results = await backend.search(search_query)
        await backend.close()
        return results


async def _perform_single_search(
    search_query: SearchQuery,
    backend: str,
    cfg: IfetcherConfig,
    cache: Optional[SearchCache] = None,
    verbose: bool = False,
) -> "SearchResults":
    """Perform a single search query (original behavior)."""
    backend_config = cfg.tools.search.get_backend_config(backend)

    if backend == "pubmed":
        search_backend = PubMedBackend(backend_config)
    elif backend == "perplexica":
        search_backend = PerplexicaBackend(backend_config)
    elif backend == "openai_search":
        search_backend = OpenAISearchBackend(backend_config)
    else:
        available_backends = cfg.tools.search.get_available_backends()
        raise ValueError(
            f"Backend '{backend}' is not implemented. Available: {available_backends}"
        )

    if verbose:
        _display_search_request(search_query, backend)

    async with search_backend:
        return await search_backend.search(search_query)


def _display_search_request(search_query: SearchQuery, backend: str) -> None:
    """Display pretty search request details."""

    # Create search details table
    details_table = Table(show_header=False, show_edge=False, pad_edge=False)
    details_table.add_column("key", style="bold cyan", no_wrap=True, width=12)
    details_table.add_column("value", style="white")

    # Add basic search info
    details_table.add_row("Backend:", f"[yellow]{backend}[/yellow]")
    details_table.add_row("Query:", f'"{search_query.query}"')
    details_table.add_row("Max Results:", str(search_query.max_results))

    # Add expanded terms if any
    if search_query.expanded_terms:
        expanded_text = Text()
        for i, term in enumerate(search_query.expanded_terms):
            if i > 0:
                expanded_text.append(", ")
            expanded_text.append(f'"{term}"', style="green")
        details_table.add_row("Expanded:", expanded_text)

    # Add filters if any
    if search_query.filters:
        filter_items = []
        for key, value in search_query.filters.items():
            filter_items.append(f"{key}={value}")
        details_table.add_row("Filters:", ", ".join(filter_items))

    # Create panel with search details
    panel = Panel(
        details_table,
        title="🔍 Search Request",
        title_align="left",
        border_style="blue",
        expand=False,
    )

    console.print(panel)


def _display_search_summary(results) -> None:
    """Display pretty search results summary."""

    # Create summary table
    summary_table = Table(show_header=False, show_edge=False, pad_edge=False)
    summary_table.add_column("key", style="bold magenta", no_wrap=True, width=12)
    summary_table.add_column("value", style="white")

    # Add summary info
    summary_table.add_row("Results:", f"[green]{results.result_count}[/green]")
    if results.total_found and results.total_found != results.result_count:
        summary_table.add_row("Total Found:", f"[yellow]{results.total_found}[/yellow]")

    if results.search_time:
        summary_table.add_row("Time:", f"[cyan]{results.search_time:.2f}s[/cyan]")

    summary_table.add_row("Backend:", f"[yellow]{results.backend}[/yellow]")

    # Add domain diversity
    if hasattr(results, "unique_domains"):
        domains = results.unique_domains
        if len(domains) > 5:
            domain_text = f"{len(domains)} unique domains"
        else:
            domain_text = ", ".join(domains[:5])
            if len(domains) > 5:
                domain_text += f" (+{len(domains) - 5} more)"
        summary_table.add_row("Domains:", domain_text)

    # Create panel with summary
    panel = Panel(
        summary_table,
        title="📊 Search Results",
        title_align="left",
        border_style="green",
        expand=False,
    )

    console.print(panel)


def _display_search_results(
    results, output_format: str, verbose: bool, save_results: Optional[str] = None
) -> None:
    """Display search results in the specified format."""

    # If saving to file, show summary but skip console output of results
    if save_results:
        if verbose:
            _display_search_summary(results)
        _save_search_results(results, save_results, output_format)
        return

    # Otherwise, display to console as usual
    # Show summary first in verbose mode
    if verbose:
        _display_search_summary(results)
    if output_format == "table":
        _display_results_table(results, verbose)
    elif output_format == "json":
        _display_results_json(results)
    elif output_format == "urls":
        _display_results_urls(results)
    elif output_format == "csv":
        _display_results_csv(results)
    elif output_format == "jsonl":
        _display_results_jsonl(results)
    else:
        console.print(f"[red]Unknown output format: {output_format}[/red]")
        return


def _display_results_table(results, verbose: bool = False) -> None:
    """Display search results as a Rich table."""
    # Create summary panel
    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_row(styled_key("Backend:"), styled_config(results.backend))
    summary_table.add_row(styled_key("Query:"), styled_path(results.query.query))

    if results.query.expanded_terms:
        summary_table.add_row(
            styled_key("Expanded:"),
            styled_config(f"+{len(results.query.expanded_terms)} terms"),
        )

    summary_table.add_row(styled_key("Results:"), styled_count(results.result_count))

    if results.total_found and results.total_found > results.result_count:
        summary_table.add_row(
            styled_key("Total Available:"), styled_count(results.total_found)
        )

    if results.search_time:
        summary_table.add_row(
            styled_key("Search Time:"), styled_config(f"{results.search_time:.2f}s")
        )

    console.print(
        Panel(
            summary_table,
            title="[bold blue]Search Results[/bold blue]",
            border_style="blue",
        )
    )

    if not results.results:
        console.print("[yellow]No results found[/yellow]")
        return

    # Create results table
    results_table = Table(show_header=True, header_style="bold blue")
    results_table.add_column("Title", style="white", max_width=70)
    results_table.add_column("Backend", style="dim white", max_width=15)
    results_table.add_column("Domain", style="cyan", max_width=20)

    if verbose:
        results_table.add_column(
            "Relevance", style="yellow", justify="center", max_width=10
        )

    for result in results.results:
        domain = (
            result.domain
            if hasattr(result, "domain")
            else result.url.split("/")[2]
            if "/" in result.url
            else result.url
        )

        row = [
            result.title[:67] + "..." if len(result.title) > 70 else result.title,
            result.backend,
            domain,
        ]

        if verbose and result.relevance_score:
            row.append(f"{result.relevance_score:.2f}")
        elif verbose:
            row.append("[dim]N/A[/dim]")

        results_table.add_row(*row)

    console.print(results_table)

    # Show URLs if verbose
    if verbose:
        console.print(f"\n[dim]URLs:[/dim]")
        for i, result in enumerate(results.results, 1):
            console.print(f"  {i:2d}. {result.url}")


def _display_results_json(results) -> None:
    """Display search results as JSON."""
    import json

    results_dict = results.model_dump()
    console.print(json.dumps(results_dict, indent=2, default=str))


def _display_results_urls(results) -> None:
    """Display only the URLs from search results."""
    for result in results.results:
        console.print(result.url)


def _display_results_csv(results) -> None:
    """Display search results as CSV."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(
        ["title", "url", "backend", "snippet", "relevance_score", "frequency"]
    )

    # Write data
    for result in results.results:
        writer.writerow(
            [
                result.title,
                result.url,
                result.backend,
                result.snippet or "",
                result.relevance_score or "",
                result.metadata.get("frequency", 1),
            ]
        )

    console.print(output.getvalue().rstrip())


def _display_results_jsonl(results) -> None:
    """Display search results as JSONL format compatible with extract command."""
    import json

    for result in results.results:
        # Only include fields that contain actual data
        record = {
            "url": result.url,
            "title": result.title,
            "backend": result.backend,
        }

        # Include snippet if it exists
        if result.snippet:
            record["snippet"] = result.snippet

        # Include relevance score for ranking
        if result.relevance_score is not None:
            record["relevance_score"] = result.relevance_score

        # Include frequency as top-level field
        if "frequency" in result.metadata:
            record["frequency"] = result.metadata["frequency"]

        console.print(json.dumps(record, default=str))


def _save_search_results(results, filename: str, format_type: str) -> None:
    """Save search results to file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            if format_type == "json":
                import json

                json.dump(results.model_dump(), f, indent=2, default=str)
            elif format_type == "csv":
                import csv

                writer = csv.writer(f)
                writer.writerow(
                    [
                        "title",
                        "url",
                        "backend",
                        "snippet",
                        "relevance_score",
                        "frequency",
                    ]
                )

                for result in results.results:
                    writer.writerow(
                        [
                            result.title,
                            result.url,
                            result.backend,
                            result.snippet or "",
                            result.relevance_score or "",
                            result.metadata.get("frequency", 1),
                        ]
                    )
            elif format_type == "jsonl":
                import json

                for result in results.results:
                    record = {
                        "url": result.url,
                        "title": result.title,
                        "snippet": result.snippet,
                        "backend": result.backend,
                        "relevance_score": result.relevance_score,
                    }

                    # Include frequency as top-level field if present
                    if "frequency" in result.metadata:
                        record["frequency"] = result.metadata["frequency"]

                    f.write(json.dumps(record, default=str) + "\n")
            elif format_type == "urls":
                for result in results.results:
                    f.write(f"{result.url}\n")
            else:
                # Default to JSON
                import json

                json.dump(results.model_dump(), f, indent=2, default=str)

        console.print(f"[green]Results saved to {filename}[/green]")

    except Exception as e:
        console.print(f"[red]Failed to save results: {e}[/red]")


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


def _parse_known_resources_jsonl(jsonl_path: Path) -> List["KnownResource"]:
    """
    Parse JSONL file containing known resources.

    Parameters:
        jsonl_path: Path - Path to JSONL file

    Returns:
        List[KnownResource] - Parsed resources

    Raises:
        ValueError: If file is malformed or missing required fields
        FileNotFoundError: If file doesn't exist
    """
    import json
    from interaction_finder.search.reverse.models import (
        KnownResource,
        ResourceParseError,
    )

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    resources = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                entry = json.loads(line)

                # Extract required and optional fields
                url = entry.get("url")
                if not url:
                    raise ResourceParseError(
                        f"Missing required field 'url' at line {line_num}",
                        context={"line": line_num, "entry": entry},
                    )

                pmid = entry.get("pmid")

                # Extract hint fields (all fields except pmid/url)
                hint_fields = {
                    k: v for k, v in entry.items() if k not in ["pmid", "url"]
                }

                # Create KnownResource
                resource = KnownResource(
                    pmid=pmid,
                    url=url,
                    hint_fields=hint_fields,
                )
                resources.append(resource)

            except json.JSONDecodeError as e:
                raise ResourceParseError(
                    f"Invalid JSON at line {line_num}: {e}",
                    context={"line": line_num},
                )
            except Exception as e:
                raise ResourceParseError(
                    f"Failed to parse resource at line {line_num}: {e}",
                    context={
                        "line": line_num,
                        "entry": entry if "entry" in locals() else None,
                    },
                )

    if not resources:
        raise ValueError(f"No valid resources found in {jsonl_path}")

    return resources


def _write_reverse_search_output(
    session: "ReverseSearchSession",
    output_path: Path,
) -> None:
    """
    Write reverse search results to JSONL file.

    Format: One JSON object per line
    - Per-query results: query, query_index, resources_found, new_finds, coverage, time
    - Final summary: total_queries, final_coverage, found/unfound counts, stopping_reason

    Parameters:
        session: ReverseSearchSession - Completed session
        output_path: Path - Output file path
    """
    import json

    with open(output_path, "w", encoding="utf-8") as f:
        # Write per-query results
        for result in session.query_results:
            query_entry = {
                "query": result.query,
                "query_index": result.query_index,
                "resources_found": [r.canonical_url for r in result.resources_found],
                "new_finds": result.new_finds,
                "cumulative_coverage": result.cumulative_coverage,
                "search_time": result.search_time,
                "backend": result.backend,
            }
            f.write(json.dumps(query_entry, ensure_ascii=False) + "\n")

        # Write final summary
        summary = {
            "summary": True,
            "total_queries": session.total_queries,
            "final_coverage": session.final_coverage,
            "total_resources": len(session.target_resources),
            "found_resources": session.found_count,
            "unfound_resources": [r.canonical_url for r in session.unfound_resources],
            "total_time": session.total_time,
            "stopping_reason": session.stopping_reason,
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


async def _run_reverse_search_command(
    target_resources: List["KnownResource"],
    backend_name: str,
    cfg: IfetcherConfig,
    output_path: Path,
    verbose: bool,
    dry_run: bool,
    console: Console,
) -> None:
    """
    Execute reverse search command asynchronously.

    Parameters:
        target_resources: List[KnownResource] - Resources to find
        backend_name: str - Search backend name
        cfg: IfetcherConfig - Configuration
        output_path: Path - Output file path
        verbose: bool - Show progress
        dry_run: bool - Preview only, don't execute
        console: Console - Rich console for output
    """
    from interaction_finder.search.reverse import ReverseSearcher
    from interaction_finder.search.cache import SearchCache
    from interaction_finder.fetcher import PageFetcher

    # Instantiate backend (reuse pattern from search command)
    backend_config = cfg.tools.search.get_backend_config(backend_name)

    if backend_name == "pubmed":
        search_backend = PubMedBackend(backend_config)
    elif backend_name == "perplexica":
        search_backend = PerplexicaBackend(backend_config)
    elif backend_name == "openai_search":
        search_backend = OpenAISearchBackend(backend_config)
    else:
        available = cfg.tools.search.get_available_backends()
        console.print(f"[red]Backend '{backend_name}' not available[/red]")
        console.print(f"[blue]Available backends:[/blue] {', '.join(available)}")
        raise typer.Exit(1)

    # Create cache with custom TTL for reverse search
    cache_dir = Path(cfg.output.cache) / "search"
    cache = SearchCache(
        cache_dir=cache_dir,
        ttl_hours=cfg.tools.search.reverse.cache_ttl_days * 24,  # Convert days to hours
    )

    # Create PageFetcher
    fetcher = PageFetcher(cfg, show_status=verbose)

    # Create ReverseSearcher
    reverse_config = cfg.tools.search.reverse
    searcher = ReverseSearcher(reverse_config, search_backend, cache, fetcher)

    if dry_run:
        # Preview mode: generate queries but don't execute
        console.print("[yellow]Dry run mode: Previewing query generation[/yellow]")
        try:
            queries = await searcher.query_generator.generate_initial_queries(
                target_resources
            )
            console.print(
                f"\n[green]Would generate {len(queries)} initial queries:[/green]"
            )
            for i, query in enumerate(queries, start=1):
                console.print(f"  {i}. {query}")
            console.print(f"\n[blue]Dry run complete. No queries executed.[/blue]")
        except Exception as e:
            console.print(f"[red]Query generation failed: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Execute reverse search
        async with search_backend:
            try:
                session = await searcher.search(target_resources, verbose=verbose)

                # Write output
                _write_reverse_search_output(session, output_path)
                console.print(f"\n[green]Results written to:[/green] {output_path}")

                # Print summary
                console.print(f"\n[bold]Summary:[/bold]")
                console.print(f"  Total queries: {session.total_queries}")
                console.print(
                    f"  Found: {session.found_count} / {len(target_resources)} ({session.coverage_pct:.1f}%)"
                )
                console.print(f"  Stopping reason: {session.stopping_reason}")
                console.print(f"  Total time: {session.total_time:.1f}s")

            except Exception as e:
                console.print(f"[red]Reverse search failed: {e}[/red]")
                if verbose:
                    import traceback

                    console.print(traceback.format_exc())
                raise typer.Exit(1)


@app.command()
def reverse_search(
    known: Path = typer.Option(
        ...,
        "--known",
        "-k",
        help="Path to JSONL file with known resources (required: url; optional: pmid, hint fields)",
    ),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="Search backend to use (overrides config). Options: pubmed, perplexica, openai_search",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSONL file path (default: <known_file>_results.jsonl)",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file",
    ),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        "-m",
        help="Configuration mode to use",
    ),
    overrides: Optional[List[str]] = typer.Option(
        None,
        "-O",
        help="Config overrides in key=value format (e.g., -O tools.search.reverse.coverage_target=0.90)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed progress and execution information",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview query generation without executing searches",
    ),
):
    """
    Reverse search: Generate queries to find known resources.

    Given a JSONL file of known resources (PMIDs/URLs), generates and executes
    search queries to locate those resources in literature databases, tracking
    coverage until 95% found or 3 consecutive queries yield no new discoveries.

    Example JSONL format:
        {"pmid": "12345678", "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"}
        {"url": "https://example.com/paper", "celltype": "langerhans cell", "marker": "CD1A"}

    Examples:
        # Basic usage with default backend (PubMed)
        interaction-finder reverse-search --known resources.jsonl

        # Use Perplexica backend with verbose output
        interaction-finder reverse-search -k resources.jsonl -b perplexica -v

        # Override coverage target via config override
        interaction-finder reverse-search -k resources.jsonl -O tools.search.reverse.coverage_target=0.90

        # Dry run to preview query generation
        interaction-finder reverse-search -k resources.jsonl --dry-run -v

        # Custom output file
        interaction-finder reverse-search -k resources.jsonl -o results.jsonl
    """
    # Get effective options with global fallback
    effective_config, effective_verbose, effective_overrides = (
        get_options_with_fallback(config_path, verbose, overrides)
    )

    try:
        # Load config
        cfg = load_config(effective_config, mode, effective_overrides)

        # Parse known resources from JSONL
        try:
            target_resources = _parse_known_resources_jsonl(known)
            console.print(
                f"[green]Loaded {len(target_resources)} target resources[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error parsing JSONL: {e}[/red]")
            raise typer.Exit(1)

        # Determine backend
        backend_name = backend or cfg.tools.search.reverse.search_backend
        console.print(f"[blue]Using backend:[/blue] {backend_name}")

        # Determine output path
        output_path = output or known.with_stem(known.stem + "_results").with_suffix(
            ".jsonl"
        )

        # Run async reverse search
        asyncio.run(
            _run_reverse_search_command(
                target_resources=target_resources,
                backend_name=backend_name,
                cfg=cfg,
                output_path=output_path,
                verbose=effective_verbose,
                dry_run=dry_run,
                console=console,
            )
        )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if effective_verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1)


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
    search_query: Optional[str] = typer.Option(
        None,
        "--search-query",
        help="Search for papers using this query instead of using a source file. Use with --search-backend to choose backend.",
    ),
    search_backend: Optional[str] = typer.Option(
        None,
        "--search-backend",
        help="Search backend to use with --search-query (pubmed, perplexica)",
    ),
    search_max_results: Optional[int] = typer.Option(
        None,
        "--search-max-results",
        help="Maximum search results to process (default: 50)",
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
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override AI model (e.g., openai:gpt-4o)"
    ),
    parallelism: Optional[int] = typer.Option(
        None, "--parallelism", "-p", help="Maximum parallel operations"
    ),
    ver: int = typer.Option(
        3, "--ver", help="Pipeline version (1, 2, or 3). Default is 3."
    ),
    checkpoint: Optional[str] = typer.Option(
        None, "--checkpoint", help="Resume from checkpoint (V3 only)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be processed without executing"
    ),
    failfast: bool = typer.Option(
        False, "--failfast", help="Stop on first error during processing"
    ),
    retry: bool = typer.Option(
        False, "--retry", help="Force retry of URLs previously marked as failed"
    ),
    config: Optional[str] = GLOBAL_OPTIONS.config,
    verbose: bool = GLOBAL_OPTIONS.verbose,
    overrides: List[str] = GLOBAL_OPTIONS.overrides,
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
    # Validate version parameter
    if ver not in [1, 2, 3]:
        console.print(
            f"[bold red]Error:[/bold red] Invalid version: {ver}. Must be 1, 2, or 3."
        )
        raise typer.Exit(1)

    # Validate checkpoint is only used with V3
    if checkpoint and ver != 3:
        console.print(
            f"[bold red]Error:[/bold red] --checkpoint is only supported with --ver 3"
        )
        raise typer.Exit(1)

    # Get effective options with global fallback
    effective_config, effective_verbose, effective_overrides = (
        get_options_with_fallback(config, verbose, overrides)
    )

    # Load configuration first to handle case where no input_source is provided
    try:
        cfg = load_config(effective_config, mode, effective_overrides)

        # Update task context with term if provided
        if term and not cfg.task.context:
            cfg.task.context = f"Analysis focused on {term}"
        elif term and cfg.task.context and term not in cfg.task.context:
            cfg.task.context = f"{cfg.task.context} (term: {term})"

    except Exception as e:
        handle_operation_error("loading configuration", e)
        raise typer.Exit(1)

    # Handle search query option first
    if search_query is not None:
        if effective_verbose:
            console.print(f"[dim]Using search query: '{search_query}'[/dim]")

        # Run search to get URLs
        search_results = asyncio.run(
            _run_search_for_extract(
                search_query,
                search_backend,
                search_max_results,
                cfg,
                term,
                effective_verbose,
            )
        )

        urls = [result.url for result in search_results.results]
        source_type = "search"

        if effective_verbose:
            console.print(f"[dim]Found {len(urls)} URLs from search[/dim]")

    # Handle case where no source is provided - use training_data from config
    elif source is None:
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
                # Get model name and strip provider prefix
                full_model = model or str(
                    cfg.agents.get("_", cfg.AgentSpec()).llm or "gpt-4o"
                )
                clean_model = (
                    full_model.split(":", 1)[-1] if ":" in full_model else full_model
                )

                template_vars = {
                    "term": term or "unknown",
                    "mode": mode or "default",
                    "model": clean_model,
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

            # Log extraction start
            logfire.info(
                f"Starting extraction pipeline: {len(urls)} URLs, target: {cfg.task.get_kind_names()} → {cfg.task.relation}"
            )

            # Import extraction function based on version
            if ver == 3:
                from .extraction_graph_v3.run import (
                    extract_from_urls_v3 as extract_from_urls,
                )

                # NOTE: V3 currently uses default settings from ExtractionDepsV3.from_config()
                # Configuration via [extraction.v3] config section is planned but not yet implemented
                # All parallelism, co-occurrence, and caching settings use sensible defaults
                logfire.info("Using extraction pipeline v3")
            elif ver == 2:
                from .extraction_graph_v2.run import (
                    extract_from_urls_v2 as extract_from_urls,
                )

                logfire.info("Using extraction pipeline v2")
            elif ver == 1:
                from .extraction_graph.run import (
                    extract_from_urls as extract_from_urls,
                )

                logfire.info("Using extraction pipeline v1")
            else:
                # Should never reach here due to validation
                raise ValueError(f"Invalid version: {ver}")
            from .fetcher import PageFetcher

            # Create PageFetcher with appropriate settings
            page_fetcher = PageFetcher(cfg, show_status=True, verbose=verbose)

            import time

            pipeline_start_time = time.time()

            # Run extraction
            logfire.info(
                f"Starting extraction pipeline with parallelism={parallelism or cfg.tools.crawl4ai.max_concurrent}"
            )

            # Create incremental save callback if output path is provided
            save_callback = None
            processed_term = term or "unknown"
            processed_mode = mode or "default"
            if output_path and processed_term and processed_mode:

                def incremental_save(intermediate_result):
                    try:
                        # Import version-specific save function
                        if ver == 3:
                            from .extraction_graph_v3.run import save_results_v3

                            save_results_v3(
                                intermediate_result,
                                output_path,
                                term=processed_term,
                                mode=processed_mode,
                                repeat=1,
                                entity_kinds=cfg.task.get_kind_names(),
                            )
                        elif ver == 2:
                            from .extraction_graph_v2.run import save_results_v2

                            save_results_v2(
                                intermediate_result,
                                output_path,
                                term=processed_term,
                                mode=processed_mode,
                                repeat=1,
                                entity_kinds=cfg.task.get_kind_names(),
                            )
                        # V1 doesn't support incremental saves
                        if verbose:
                            pairs_count = len(intermediate_result.entity_pairs)
                            print(f"  Saved intermediate results: {pairs_count} pairs")
                    except Exception as e:
                        if verbose:
                            print(
                                f"  Warning: Failed to save intermediate results: {e}"
                            )

            # Run extraction with selected version
            # Prepare version-specific parameters
            checkpoint_path_obj = Path(checkpoint) if checkpoint else None

            # Call with version-specific parameters
            if ver == 1:
                # V1: Only base parameters (no target_term, output_dir, checkpoint_path)
                result = asyncio.run(
                    extract_from_urls(
                        urls=urls,
                        config=cfg,
                        page_fetcher=page_fetcher,
                        model=model,
                        verbose=verbose,
                        save_callback=save_callback,
                        parallelism=parallelism,
                    )
                )
            elif ver == 2:
                # V2: Base + target_term, output_dir (no checkpoint_path)
                result = asyncio.run(
                    extract_from_urls(
                        urls=urls,
                        config=cfg,
                        page_fetcher=page_fetcher,
                        model=model,
                        verbose=verbose,
                        save_callback=save_callback,
                        parallelism=parallelism,
                        target_term=term,
                        output_dir=output_path.with_suffix(
                            ""
                        ),  # Remove .jsonl, use directory
                    )
                )
            elif ver == 3:
                # V3: All parameters including checkpoint_path
                result = asyncio.run(
                    extract_from_urls(
                        urls=urls,
                        config=cfg,
                        page_fetcher=page_fetcher,
                        model=model,
                        verbose=verbose,
                        save_callback=save_callback,
                        parallelism=parallelism,
                        target_term=term,
                        output_dir=output_path.with_suffix(
                            ""
                        ),  # Remove .jsonl, use directory
                        checkpoint_path=checkpoint_path_obj,
                    )
                )
            else:
                # Should never reach here due to validation
                raise ValueError(f"Invalid version: {ver}")

            pipeline_time = time.time() - pipeline_start_time

            # Log comprehensive results
            logfire.info(f"Extraction pipeline completed in {pipeline_time:.1f}s")
            logfire.info(
                f"Results: {result.total_pairs} pairs from {result.successful_groups}/{result.total_groups} groups"
            )
            logfire.info(f"Entities found: {result.total_entities}")

            # Save results using version-specific format
            if ver == 3:
                from .extraction_graph_v3.run import save_results_v3 as save_results
            elif ver == 2:
                from .extraction_graph_v2.run import save_results_v2 as save_results
            elif ver == 1:
                from .extraction_graph.run import save_results as save_results

            save_results(
                result,
                output_path,
                term=term or "unknown",
                mode=mode or "basic",
                repeat=1,  # Could be made configurable in future
                entity_kinds=cfg.task.get_kind_names(),
            )

            logfire.info(
                f"Results saved to: {os.path.relpath(output_path.parent, Path.cwd())}"
            )

            # Display summary using existing function
            _display_extraction_summary(result)

        except Exception as e:
            logfire.error(f"Extraction pipeline failed: {str(e)}")
            if verbose:
                console.print(Traceback(show_locals=True))
            else:
                console.print(f"Error: {e}", style="bold red")
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
    verbose: bool = False,
    output_path: Optional[Path] = None,
    term: Optional[str] = None,
    mode: Optional[str] = None,
    parallelism: Optional[int] = None,
) -> Any:
    """Run the entity extraction pipeline."""
    from .extraction_graph import extract_from_urls, save_results

    # Create incremental save callback if output path is provided
    save_callback = None
    # Use processed values (with defaults) instead of original CLI params
    processed_term = term or "unknown"
    processed_mode = mode or "default"
    if output_path and processed_term and processed_mode:

        def incremental_save(intermediate_result):
            try:
                save_results(
                    intermediate_result,
                    output_path,
                    term=processed_term,
                    mode=processed_mode,
                    repeat=1,
                    entity_kinds=config.task.get_kind_names(),
                )
                if verbose:
                    pairs_count = len(intermediate_result.entity_pairs)
                    groups_processed = len(intermediate_result.group_summaries)
                    print(
                        f"  Saved intermediate results: {pairs_count} pairs from {groups_processed} groups"
                    )
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to save intermediate results: {e}")

        save_callback = incremental_save

    # Run extraction
    result = await extract_from_urls(
        urls=urls,
        config=config,
        page_fetcher=page_fetcher,
        model=model,
        verbose=verbose,
        save_callback=save_callback,
        parallelism=parallelism,
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
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    config: Optional[str] = GLOBAL_OPTIONS.config,
    verbose: bool = GLOBAL_OPTIONS.verbose,
    overrides: List[str] = GLOBAL_OPTIONS.overrides,
):
    """Show current configuration information."""
    # Get effective options with global fallback
    effective_config, effective_verbose, effective_overrides = (
        get_options_with_fallback(config, verbose, overrides)
    )
    try:
        cfg = load_config(effective_config, mode, effective_overrides)

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
    config: Optional[str] = GLOBAL_OPTIONS.config,
    verbose: bool = GLOBAL_OPTIONS.verbose,
    overrides: List[str] = GLOBAL_OPTIONS.overrides,
):
    """Open interactive configuration editor."""
    from .settings_editor import run_config_editor

    # Get effective options with global fallback
    effective_config, effective_verbose, effective_overrides = (
        get_options_with_fallback(config, verbose, overrides)
    )
    run_config_editor(effective_config)


@config_app.command("validate")
def config_validate(
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to validate"
    ),
    config: Optional[str] = GLOBAL_OPTIONS.config,
    verbose: bool = GLOBAL_OPTIONS.verbose,
    overrides: List[str] = GLOBAL_OPTIONS.overrides,
):
    """Validate configuration file without loading full settings."""
    # Get effective options with global fallback
    effective_config, effective_verbose, effective_overrides = (
        get_options_with_fallback(config, verbose, overrides)
    )
    try:
        cfg = load_config(effective_config, mode, effective_overrides)
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


@app.command()
def evaluate(
    backends: str = typer.Argument(
        ...,
        help="Backend names to evaluate, comma-separated (e.g., 'pubmed,perplexica')",
    ),
    query_set: str = typer.Option(
        "biomedical",
        "--query-set",
        "-q",
        help="Query set to use: 'biomedical', 'general', or path to custom file",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for evaluation results (JSON format)",
    ),
    max_results: int = typer.Option(
        20, "--max-results", help="Maximum results to evaluate per query"
    ),
    timeout: int = typer.Option(
        60, "--timeout", help="Timeout for search operations in seconds"
    ),
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    config: Optional[str] = GLOBAL_OPTIONS.config,
    verbose: bool = GLOBAL_OPTIONS.verbose,
    overrides: List[str] = GLOBAL_OPTIONS.overrides,
) -> None:
    """
    Evaluate and compare search backends.

    This command runs comparative evaluations of different search backends
    using predefined or custom query sets. It measures performance metrics
    like search time, result count, and similarity between backends.

    Examples:
        interaction-finder evaluate pubmed
        interaction-finder evaluate pubmed,perplexica --query-set biomedical
        interaction-finder evaluate pubmed,perplexica --output evaluation_results.json
    """
    # Get effective options with global fallback
    effective_config, effective_verbose, effective_overrides = (
        get_options_with_fallback(config, verbose, overrides)
    )
    asyncio.run(
        _run_evaluate_command(
            backends=backends,
            query_set=query_set,
            output=output,
            max_results=max_results,
            timeout=timeout,
            config=effective_config,
            mode=mode,
            verbose=effective_verbose,
            overrides=effective_overrides,
        )
    )


async def _run_evaluate_command(
    backends: str,
    query_set: str,
    output: Optional[str],
    max_results: int,
    timeout: int,
    config: Optional[str],
    mode: Optional[str],
    verbose: bool,
    overrides: List[str],
) -> None:
    """Run evaluation command asynchronously."""
    try:
        # Load configuration
        cfg = load_config(config, mode, overrides)

        # Parse backend list
        backend_names = [name.strip() for name in backends.split(",")]

        # Validate backends
        for backend_name in backend_names:
            if backend_name not in cfg.tools.search.enabled_backends:
                console.print(f"[red]Backend '{backend_name}' is not enabled[/red]")
                console.print(
                    f"[blue]Enabled backends:[/blue] {', '.join(cfg.tools.search.enabled_backends)}"
                )
                raise typer.Exit(1)

        # Get query set
        if query_set == "biomedical":
            queries = DEFAULT_BIOMEDICAL_QUERIES
        elif query_set == "general":
            queries = DEFAULT_GENERAL_QUERIES
        elif Path(query_set).exists():
            # Load custom query set from file
            with open(query_set, "r") as f:
                queries = [line.strip() for line in f if line.strip()]
        else:
            console.print(
                f"[red]Unknown query set or file not found: {query_set}[/red]"
            )
            console.print(
                "[blue]Available query sets:[/blue] biomedical, general, or path to file"
            )
            raise typer.Exit(1)

        console.print(
            f"[blue]Evaluating {len(backend_names)} backend(s) with {len(queries)} queries[/blue]"
        )

        # Set up evaluation configuration
        eval_config = EvaluationConfig(
            max_results_to_evaluate=max_results,
            timeout_seconds=timeout,
        )

        runner = EvaluationRunner(eval_config)

        if len(backend_names) == 1:
            # Single backend evaluation
            backend_name = backend_names[0]
            backend_config = cfg.tools.search.get_backend_config(backend_name)

            results = await runner.run_backend_evaluation(
                queries=queries,
                backend_name=backend_name,
                backend_config=backend_config,
                output_file=output,
            )

            runner.display_evaluation_results(results)

        else:
            # Comparative evaluation
            backend_configs = {}
            for backend_name in backend_names:
                backend_configs[backend_name] = cfg.tools.search.get_backend_config(
                    backend_name
                )

            results = await runner.run_comparison_evaluation(
                queries=queries,
                backend_configs=backend_configs,
                output_file=output,
            )

            runner.display_evaluation_results(results)

    except Exception as e:
        handle_operation_error("during evaluation", e)


@app.command()
def cache(
    action: str = typer.Argument(
        ...,
        help="Action to perform: 'stats', 'clear', 'clear-expired'",
    ),
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    config: Optional[str] = GLOBAL_OPTIONS.config,
    verbose: bool = GLOBAL_OPTIONS.verbose,
    overrides: List[str] = GLOBAL_OPTIONS.overrides,
) -> None:
    """
    Manage search result cache.

    Actions:
    - stats: Show cache statistics
    - clear: Clear all cached results
    - clear-expired: Remove only expired cache entries

    Examples:
        interaction-finder cache stats
        interaction-finder cache clear
        interaction-finder cache clear-expired
    """
    # Get effective options with global fallback
    effective_config, effective_verbose, effective_overrides = (
        get_options_with_fallback(config, verbose, overrides)
    )
    asyncio.run(
        _run_cache_command(
            action, effective_config, mode, effective_verbose, effective_overrides
        )
    )


async def _run_cache_command(
    action: str,
    config: Optional[str],
    mode: Optional[str],
    verbose: bool,
    overrides: List[str],
) -> None:
    """Run cache management command asynchronously."""
    try:
        # Load configuration
        cfg = load_config(config, mode, overrides)

        # Set up cache
        cache_dir = cfg.abspath(cfg.output.cache) / "search"
        cache = SearchCache(cache_dir, cfg.tools.search.cache.ttl_hours)

        if action == "stats":
            stats = await cache.get_stats()
            console.print("[blue]Search Cache Statistics[/blue]")
            console.print(f"Total entries: {stats['total_entries']}")
            console.print(f"Valid entries: {stats['valid_entries']}")
            console.print(f"Expired entries: {stats['expired_entries']}")
            console.print(
                f"Cache size: {stats['cache_size_bytes'] / (1024 * 1024):.2f} MB"
            )

        elif action == "clear":
            removed = await cache.clear_all()
            if removed > 0:
                console.print(f"[green]Cleared {removed} cache entries[/green]")
            else:
                console.print("[dim]Cache was already empty[/dim]")

        elif action == "clear-expired":
            removed = await cache.clear_expired()
            if removed > 0:
                console.print(f"[green]Removed {removed} expired cache entries[/green]")
            else:
                console.print("[dim]No expired entries found[/dim]")

        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("[blue]Available actions:[/blue] stats, clear, clear-expired")
            raise typer.Exit(1)

    except Exception as e:
        handle_operation_error("during cache management", e)


# Hidden completion subcommands
@app.command(name="install-completion", hidden=True)
def install_completion():
    """Install shell completion for the current shell."""
    console.print("[blue]Shell completion setup:[/blue]")
    console.print("")
    console.print("[green]For Bash, add this to ~/.bashrc:[/green]")
    console.print(
        'eval "$(_INTERACTION_FINDER_COMPLETE=bash_source interaction-finder)"'
    )
    console.print("")
    console.print("[green]For Zsh, add this to ~/.zshrc:[/green]")
    console.print(
        'eval "$(_INTERACTION_FINDER_COMPLETE=zsh_source interaction-finder)"'
    )
    console.print("")
    console.print(
        "[green]For Fish, add this to ~/.config/fish/completions/interaction-finder.fish:[/green]"
    )
    console.print(
        "eval (env _INTERACTION_FINDER_COMPLETE=fish_source interaction-finder)"
    )


@app.command(name="show-completion", hidden=True)
def show_completion(
    shell: str = typer.Option("bash", help="Shell type: bash, zsh, fish"),
):
    """Show shell completion script for the current shell."""
    import os

    # Set the completion environment variable and call the app
    env_var = f"_INTERACTION_FINDER_COMPLETE"
    shell_source = f"{shell}_source"

    console.print(f"# Completion script for {shell}")
    console.print(
        f'# Run: eval "$(_INTERACTION_FINDER_COMPLETE={shell_source} interaction-finder)"'
    )
    console.print("")
    console.print(f"export {env_var}={shell_source}")
    console.print(
        "# Note: This requires the Click completion system to be properly set up"
    )


def main():
    """Main entry point for the CLI."""
    # Configure logfire early if available
    configure_logfire()
    logfire.info("Starting interaction-finder CLI")

    app()


if __name__ == "__main__":
    main()
