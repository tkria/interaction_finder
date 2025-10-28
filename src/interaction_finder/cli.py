"""
CLI interface for Interaction Finder using Typer.

Minimal implementation providing configuration management and basic commands.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional, List, Any

import typer
import click
from rich.console import Console
from rich.table import Table

from .settings import IfetcherConfig
from .term_parser import parse_term_line
from . import cli_fetch

app = typer.Typer(
    name="interaction-finder",
    help="A tool for fetching and processing web content for interaction discovery.",
    rich_markup_mode="rich",
    add_completion=False,
)

console = Console()


# Global options that apply to all subcommands
@app.callback()
def global_options(
    config: Optional[str] = typer.Option(
        None, "-c", "--config", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show verbose output"),
    overrides: List[str] = typer.Option(
        [],
        "-O",
        "--override",
        help="Override config values using dotted paths (e.g., -O output.cache=my_cache)",
    ),
):
    """
    A tool for fetching and processing web content for interaction discovery.

    Global options like --config, --verbose, and --override can be used with any subcommand.
    """
    # Store options in the context for use by subcommands
    ctx = click.get_current_context()
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose
    ctx.obj["overrides"] = overrides


def get_options_with_fallback(
    config: Optional[str] = None,
    verbose: Optional[bool] = None,
    overrides: Optional[List[str]] = None,
) -> tuple[Optional[str], bool, List[str]]:
    """
    Get options, using global values as fallback for None/empty local values.

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


def parse_config_override(override: str) -> tuple[str, Any]:
    """
    Parse a configuration override string in the format 'key.path=value'.

    Args:
        override: String in format 'key.path=value'

    Returns:
        Tuple of (key_path, value)
    """
    if "=" not in override:
        raise ValueError(
            f"Invalid override format: '{override}'. Expected 'key.path=value'"
        )
    key, value = override.split("=", 1)
    key = key.strip()
    value = value.strip()
    # Try to parse as JSON for complex types, fall back to string
    try:
        parsed_value = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        parsed_value = value
    return key, parsed_value


def apply_config_overrides(config_data: dict, overrides: List[str]) -> dict:
    """Apply command-line overrides to configuration data."""
    override_dict = {}
    for override_str in overrides:
        key, value = parse_config_override(override_str)
        override_dict[key] = value
    return IfetcherConfig.apply_overrides(config_data, override_dict)


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    mode: Optional[str] = None,
) -> IfetcherConfig:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Path to config file (searches standard locations if None)
        overrides: List of override strings
        mode: Configuration mode to apply

    Returns:
        Loaded IfetcherConfig instance
    """
    # Search for config in standard locations if not specified
    if config_path is None:
        search_paths = [
            Path("config.toml"),
            Path("interaction_finder.toml"),
            Path(".interaction_finder.toml"),
        ]
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break
    # Load config or create default
    if config_path and Path(config_path).exists():
        override_dict = {}
        if overrides:
            for override_str in overrides:
                key, value = parse_config_override(override_str)
                override_dict[key] = value
        config = IfetcherConfig.from_path(
            config_path, overrides=override_dict, mode=mode
        )
    else:
        # No config file found, use defaults
        config = IfetcherConfig()
        if overrides:
            # Apply overrides to default config
            config_dict = config.model_dump()
            for override_str in overrides:
                key, value = parse_config_override(override_str)
                config_dict = IfetcherConfig.apply_overrides(config_dict, {key: value})
            config = IfetcherConfig.model_validate(config_dict)
    return config


def scan_available_terms(config: IfetcherConfig) -> List[str]:
    """
    Scan for available term files in the training data directory.

    Args:
        config: Configuration with training_data path template

    Returns:
        List of available term names
    """
    # Get training data path template
    template = config.training_data
    # Extract directory and pattern
    template_path = Path(template)
    if "{term}" not in template:
        return []
    # Get the directory to scan (resolve parent path without formatting)
    parent_template = str(template_path.parent)
    if "{term}" in parent_template:
        # Can't scan if term is in the directory path
        return []
    directory = config.abspath(parent_template)
    if not directory.exists():
        return []
    # Build glob pattern
    pattern = template_path.name.replace("{term}", "*")
    terms = []
    for file_path in directory.glob(pattern):
        # Extract term from filename
        name = file_path.name
        prefix = template_path.name.split("{term}")[0]
        suffix = (
            template_path.name.split("{term}")[1]
            if "{term}" in template_path.name
            else ""
        )
        if name.startswith(prefix) and name.endswith(suffix):
            term = name[len(prefix) : -len(suffix) if suffix else None]
            terms.append(term)
    return sorted(terms)


@app.command()
def terms(
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    config: Optional[str] = typer.Option(None, "-c", "--config", help="Config file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    overrides: List[str] = typer.Option(
        [], "-O", "--override", help="Config overrides"
    ),
):
    """
    List available terms found in training data directory.
    """
    config_path, verbose, overrides = get_options_with_fallback(
        config, verbose, overrides
    )
    try:
        cfg = load_config(config_path, overrides, mode)
        available_terms = scan_available_terms(cfg)
        if not available_terms:
            console.print("[yellow]No terms found in training data directory[/yellow]")
            # Show the directory we searched (without the {term} placeholder)
            template_path = Path(cfg.training_data)
            search_dir = cfg.abspath(str(template_path.parent))
            console.print(f"Searched in: {search_dir}")
            console.print(f"Pattern: {template_path.name}")
            return
        # Display terms in a table
        table = Table(
            title="Available Terms", show_header=True, header_style="bold magenta"
        )
        table.add_column("Term", style="cyan")
        table.add_column("File Path", style="dim")
        for term in available_terms:
            file_path = cfg.abspath(cfg.training_data, term=term)
            table.add_row(term, str(file_path))
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def config(
    action: str = typer.Argument(
        help="Action to perform: 'info' to show config, 'validate' to check validity"
    ),
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    config_path: Optional[str] = typer.Option(
        None, "-c", "--config", help="Config file"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    overrides: List[str] = typer.Option(
        [], "-O", "--override", help="Config overrides"
    ),
):
    """
    Manage configuration: show info or validate config file.

    Actions:
      info      - Display current configuration
      validate  - Validate configuration file
    """
    config_path, verbose, overrides = get_options_with_fallback(
        config_path, verbose, overrides
    )
    try:
        if action == "info":
            cfg = load_config(config_path, overrides, mode)
            # Display configuration info
            table = Table(
                title="Configuration", show_header=True, header_style="bold magenta"
            )
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="white")
            # Show key settings
            table.add_row("Config Dir", str(cfg._dir) if cfg._dir else "None")
            table.add_row("Cache Path", str(cfg.abspath(cfg.output.cache)))
            table.add_row("Output Path", cfg.output.path)
            table.add_row("Training Data", cfg.training_data)
            # Show agent configs
            if cfg.agents:
                for agent_name, agent_spec in cfg.agents.items():
                    if agent_spec.llm:
                        table.add_row(f"Agent '{agent_name}' LLM", agent_spec.llm)
            console.print(table)
        elif action == "validate":
            cfg = load_config(config_path, overrides, mode)
            console.print("[green]✓[/green] Configuration is valid")
            if verbose:
                console.print(cfg.model_dump_json(indent=2))
        else:
            console.print(f"[red]Unknown action:[/red] {action}")
            console.print("Valid actions: info, validate")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def fetch(
    urls: List[str] = typer.Argument(
        default=[],
        help="URLs to fetch (can specify multiple)",
    ),
    input_file: Optional[Path] = typer.Option(
        None,
        "-i",
        "--input",
        help="Read URLs from file (one per line)",
    ),
    format: str = typer.Option(
        "paths",
        "--format",
        help="Output format: 'paths' (cache file paths) or 'content' (markdown content)",
    ),
    chunk: bool = typer.Option(
        False,
        "--chunk/--no-chunk",
        help="Perform semantic chunking and cache chunks",
    ),
    mode: Optional[str] = typer.Option(
        None, "-m", "--mode", help="Configuration mode to use"
    ),
    config: Optional[str] = typer.Option(None, "-c", "--config", help="Config file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
    overrides: List[str] = typer.Option(
        [], "-O", "--override", help="Config overrides"
    ),
):
    """
    Fetch web content and cache it using PageFetcher.

    Downloads HTML/PDF content, converts to markdown, and caches locally.
    Useful for prefetching content, debugging cache behavior, and scripting workflows.

    \b
    Examples:
      # Fetch single URL and show cache path
      interaction-finder fetch https://example.com

      # Fetch multiple URLs
      interaction-finder fetch https://example.com https://example.org

      # Fetch from file
      interaction-finder fetch --input urls.txt

      # Show content instead of path
      interaction-finder fetch https://example.com --format content

      # Perform semantic chunking
      interaction-finder fetch https://example.com --chunk

      # With config override
      interaction-finder fetch https://example.com -O output.cache=custom_cache/
    """
    # Get effective options with fallback to global options
    config_path, verbose, overrides = get_options_with_fallback(
        config, verbose, overrides
    )

    try:
        # Load configuration
        cfg = load_config(config_path, overrides, mode)

        # Collect URLs from args or file input
        collected_urls = cli_fetch.collect_urls(urls, input_file)

        # Validate format option
        if format not in ("paths", "content"):
            console.print(
                f"[red]Error:[/red] Invalid format '{format}'. "
                "Must be 'paths' or 'content'"
            )
            raise typer.Exit(1)

        # Validate content format constraint (single URL only)
        if format == "content" and len(collected_urls) != 1:
            console.print(
                f"[red]Error:[/red] Content format requires exactly one URL, "
                f"got {len(collected_urls)}. Use --format paths for multiple URLs."
            )
            raise typer.Exit(1)

        # Define async implementation
        async def fetch_impl():
            return await cli_fetch.run_fetch(
                config=cfg,
                urls=collected_urls,
                chunk=chunk,
                verbose=verbose,
            )

        # Run fetch operation
        results = asyncio.run(fetch_impl())

        # Output results based on format
        if format == "paths":
            exit_code = cli_fetch.output_paths(results, console)
        else:  # format == "content"
            exit_code = cli_fetch.output_content(results, console)

        # Print summary if verbose
        if verbose:
            cli_fetch.print_summary(results, console)

        # Exit with appropriate code
        if exit_code != 0:
            raise typer.Exit(exit_code)

    except ValueError as e:
        # Handle validation errors (mutual exclusion, empty input, etc.)
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except FileNotFoundError as e:
        # Handle file not found errors
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        # Handle unexpected errors
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
