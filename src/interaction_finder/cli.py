"""
CLI interface for Interaction Finder using Typer.

Minimal implementation providing configuration management and basic commands.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import click
import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.text import Text

from . import cli_fetch
from .logging import configure_logging, dump_log, error_log_path, setup_log_output
from .settings import IfetcherConfig, sanitize_topic_for_filename
from .version import (
    check_checkpoint_version,
    get_version_string,
    parse_version_string,
)

# Disable tokenizer parallelism before any imports that might load HuggingFace models.
# This prevents "The current process just got forked" warnings when sentence-transformers
# or similar libraries are used alongside multiprocessing/threading.
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _format_json_schema_type(prop: dict) -> str:
    """Format a JSON Schema property type as a readable string."""
    if "anyOf" in prop:
        types = []
        for option in prop["anyOf"]:
            # Skip null type - in TOML, optional fields are simply omitted
            if option.get("type") == "null":
                continue
            elif "$ref" in option:
                types.append(option["$ref"].split("/")[-1])
            else:
                types.append(option.get("type", "?"))
        return " | ".join(types) if types else "any"
    if "$ref" in prop:
        return prop["$ref"].split("/")[-1]
    if "type" in prop:
        t = prop["type"]
        if t == "array" and "items" in prop:
            item_type = _format_json_schema_type(prop["items"])
            return f"list[{item_type}]"
        return t
    return "any"


def _format_constraints(prop: dict) -> str | None:
    """Extract constraint information from JSON Schema property."""
    constraints = []
    if "minimum" in prop:
        constraints.append(f"≥{prop['minimum']}")
    if "maximum" in prop:
        constraints.append(f"≤{prop['maximum']}")
    if "minLength" in prop:
        constraints.append(f"min_length={prop['minLength']}")
    if "maxLength" in prop:
        constraints.append(f"max_length={prop['maxLength']}")
    # Check inside anyOf for constraints
    for option in prop.get("anyOf", []):
        if "minimum" in option:
            constraints.append(f"≥{option['minimum']}")
        if "maximum" in option:
            constraints.append(f"≤{option['maximum']}")
    return ", ".join(constraints) if constraints else None


def _add_schema_to_tree(
    tree: Tree,
    schema: dict,
    defs: dict,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 10,
) -> None:
    """Recursively add JSON Schema properties to a Rich tree."""
    if depth > max_depth:
        return
    properties = schema.get("properties", {})
    for name, prop in properties.items():
        toml_key = f"{prefix}.{name}" if prefix else name
        # Resolve $ref if present
        if "$ref" in prop:
            ref_name = prop["$ref"].split("/")[-1]
            prop = defs.get(ref_name, prop)
        # Check for nested object (either direct or via anyOf)
        nested_schema = None
        if prop.get("type") == "object" and "properties" in prop:
            nested_schema = prop
        elif "anyOf" in prop:
            for option in prop["anyOf"]:
                if "$ref" in option:
                    ref_name = option["$ref"].split("/")[-1]
                    ref_schema = defs.get(ref_name, {})
                    if ref_schema.get("type") == "object" or "properties" in ref_schema:
                        nested_schema = ref_schema
                        break
        if nested_schema and "properties" in nested_schema:
            # This is a section with nested properties
            desc = nested_schema.get("description", prop.get("description", ""))
            section_text = Text()
            section_text.append(f"[{toml_key}]", style="bold cyan")
            if desc:
                section_text.append(f"  {desc}", style="dim")
            branch = tree.add(section_text)
            _add_schema_to_tree(branch, nested_schema, defs, toml_key, depth + 1)
        else:
            # This is a leaf field
            type_str = _format_json_schema_type(prop)
            default = prop.get("default", "(required)")
            desc = prop.get("description", "")
            constraints = _format_constraints(prop)
            # Build the display line
            field_text = Text()
            field_text.append(name, style="green")
            field_text.append(f" : {type_str}", style="yellow")
            if default != "(required)":
                default_str = (
                    f'"{default}"' if isinstance(default, str) else str(default)
                )
                field_text.append(f" = {default_str}", style="blue")
            if constraints:
                field_text.append(f" ({constraints})", style="magenta")
            if desc:
                field_text.append(f"\n    {desc}", style="dim")
            tree.add(field_text)


def render_config_help() -> Tree:
    """Render configuration schema as a Rich tree for display."""
    schema = IfetcherConfig.model_json_schema()
    defs = schema.get("$defs", {})
    tree = Tree(Text("interaction-finder configuration", style="bold"))
    _add_schema_to_tree(tree, schema, defs)
    return tree


app = typer.Typer(
    name="interaction-finder",
    help="""\
Automated extraction of biological relationships from papers.

\b
Configuration:
  Uses config.toml in current directory if present. Override any
  setting with -O key.path=value. Run 'config schema' to see all options.

\b
Example:
  interaction-finder extract "genes associated with hypotension" \\
    -e gene -e phenotype -o hypotension.json
  interaction-finder report hypotension.json
""",
    rich_markup_mode="rich",
    add_completion=False,
)

console = Console()


# Common parameter factories to reduce boilerplate
# These are functions that return fresh Option instances (avoiding mutable default issues)
def config_option():
    return typer.Option(None, "-c", "--config", help="Config file")


def mode_option():
    return typer.Option(None, "-m", "--mode", help="Configuration mode")


def verbose_option():
    return typer.Option(False, "-v", "--verbose", help="Verbose output")


def overrides_option():
    return typer.Option([], "-O", "--override", help="Config overrides")


def log_option():
    return typer.Option(None, "--log", help="Log output file")


def loglevel_option():
    return typer.Option("INFO", "--loglevel", help="Log level for file output")


# Global options that apply to all subcommands
@app.callback()
def global_options(
    config: Optional[str] = config_option(),
    mode: Optional[str] = mode_option(),
    verbose: bool = verbose_option(),
    overrides: List[str] = overrides_option(),
    log: Optional[Path] = log_option(),
    loglevel: str = loglevel_option(),
):
    """Global options that can be used with any subcommand."""
    # Configure logging at startup
    setup_log_output(log, getattr(logging, loglevel.upper(), logging.INFO))
    configure_logging(verbose=verbose)
    # Store options in the context for use by subcommands
    ctx = click.get_current_context()
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["mode"] = mode
    ctx.obj["verbose"] = verbose
    ctx.obj["overrides"] = overrides


def get_options_with_fallback(
    config: Optional[str] = None,
    mode: Optional[str] = None,
    verbose: Optional[bool] = None,
    overrides: Optional[List[str]] = None,
) -> tuple[Optional[str], Optional[str], bool, List[str]]:
    """
    Get options, using global values as fallback for None/empty local values.

    Returns:
        Tuple of (effective_config, effective_mode, effective_verbose, effective_overrides)
    """
    ctx = click.get_current_context()
    if not ctx.obj:
        return config, mode, verbose or False, overrides or []
    # Use local values if provided, otherwise fall back to global
    effective_config = config if config is not None else ctx.obj.get("config")
    effective_mode = mode if mode is not None else ctx.obj.get("mode")
    effective_verbose = (
        verbose if verbose is not None else ctx.obj.get("verbose", False)
    )
    effective_overrides = overrides if overrides else ctx.obj.get("overrides", [])
    return effective_config, effective_mode, effective_verbose, effective_overrides


def _dump_logs_on_error(output_path: Path | None) -> None:
    """Dump buffered logs to error log file."""
    try:
        written = dump_log(error_log_path(output_path))
        if written:
            console.print(f"[dim]Log written to: {written}[/dim]")
    except Exception:
        pass  # Don't mask original error


def _handle_keyboard_interrupt(output_path: Path | None) -> None:
    """Handle Ctrl+C: dump logs and exit."""
    _dump_logs_on_error(output_path)
    console.print("\n[yellow]Interrupted[/yellow]")
    raise typer.Exit(130)


def _handle_exception(e: Exception, output_path: Path | None, verbose: bool) -> None:
    """Handle exception: dump logs, display error, and exit."""
    _dump_logs_on_error(output_path)
    console.print(f"\n[red]Error:[/red] {e}")
    if verbose:
        console.print_exception()
    raise typer.Exit(1)


def _check_and_backup_checkpoint(path: Path, checkpoint_version: str | None) -> None:
    """Check checkpoint version and create backup if needed."""

    def colorize_version(version_str: str, ver_color: str) -> str:
        count, semver, hash_ = parse_version_string(version_str)
        semver_str = f"{semver[0]}.{semver[1]}.{semver[2]}"
        parts = [
            f"[blue]{count}[/blue]" if count else "",
            f"v[{ver_color}]{semver_str}[/{ver_color}]",
        ]
        if hash_:
            parts.append(f"[dim]#{hash_}[/dim]")
        return "".join(parts)

    is_older, is_breaking = check_checkpoint_version(checkpoint_version)
    current = get_version_string()
    old_fmt = (
        colorize_version(checkpoint_version, "yellow")
        if checkpoint_version
        else "[yellow]unknown[/yellow]"
    )
    new_fmt = colorize_version(current, "green")
    if is_breaking:
        console.print(
            f"[yellow]⚠ Checkpoint was created with {old_fmt} "
            f"but current version is {new_fmt} (breaking change)[/yellow]",
            highlight=False,
        )
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(path.read_text())
        console.print(f"[dim]Created backup: {backup_path}[/dim]")
    elif is_older:
        console.print(
            f"[dim]ℹ Checkpoint was created with older version {old_fmt} "
            f"(current: {new_fmt})[/dim]",
            highlight=False,
        )


def load_checkpoint_or_create(checkpoint_or_topic: str) -> tuple[Any, str]:
    """Load checkpoint from file or create empty checkpoint from topic string.

    Parameters:
        checkpoint_or_topic: File path or topic string

    Returns:
        Tuple of (checkpoint, topic)
    """
    from interaction_finder.checkpoint import PipelineCheckpoint
    from interaction_finder.upgrade import create_empty_checkpoint

    # Check if input is an existing file
    path = Path(checkpoint_or_topic)
    if path.exists() and path.is_file():
        checkpoint = PipelineCheckpoint.model_validate_json(path.read_text())
        # Check version and backup if needed
        _check_and_backup_checkpoint(path, checkpoint.created_by)
        return checkpoint, checkpoint.topic

    # Treat as topic string
    return create_empty_checkpoint(checkpoint_or_topic), checkpoint_or_topic


def create_search_backend(backend_name: str, config: IfetcherConfig) -> Any:
    """Create search backend instance from name.

    Parameters:
        backend_name: Backend identifier (pubmed, perplexica, openai)
        config: Configuration for backend-specific settings

    Returns:
        SearchBackend instance
    """
    from interaction_finder.search.backends.openai import OpenAIBackend
    from interaction_finder.search.backends.perplexica import PerplexicaBackend
    from interaction_finder.search.backends.pubmed import PubMedBackend

    search_config = config.tools.search
    timeout = search_config.timeout
    if backend_name == "perplexica":
        perplexica = search_config.perplexica

        def model_dict(spec):
            return {"providerId": spec.provider_id, "key": spec.key} if spec else None

        return PerplexicaBackend(
            config={
                "timeout": timeout,
                "base_url": perplexica.base_url,
                "sources": perplexica.sources,
                "optimization_mode": perplexica.optimization_mode,
                "chat_model": model_dict(perplexica.chat_model),
                "embedding_model": model_dict(perplexica.embedding_model),
            }
        )
    elif backend_name == "pubmed":
        pubmed = search_config.pubmed
        return PubMedBackend(
            config={
                "timeout": timeout,
                "email": pubmed.email,
                "api_key": pubmed.api_key,
                "rate_limit": pubmed.rate_limit,
                "use_mesh": pubmed.use_mesh,
            }
        )
    elif backend_name == "openai":
        openai = search_config.openai
        return OpenAIBackend(
            config={
                "timeout": timeout,
                "api_key": openai.api_key,
                "base_url": openai.base_url,
                "model": openai.model,
            }
        )
    else:
        valid = "pubmed, perplexica, openai"
        raise ValueError(f"Unknown backend '{backend_name}'. Valid: {valid}")


def _parse_filter_options(filter_args: List[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    allowed_keys = {"accepted", "confidence"}

    for raw in filter_args:
        if ":" in raw:
            key, value = raw.split(":", 1)
        elif "=" in raw:
            key, value = raw.split("=", 1)
        else:
            raise typer.BadParameter(
                f"Invalid filter '{raw}'. Use key:value syntax, e.g., --filter accepted:yes",
                param_hint="--filter",
            )

        key = key.strip().lower()
        value = value.strip()

        if not key or key not in allowed_keys:
            raise typer.BadParameter(
                f"Unsupported filter '{key}'. Supported keys: accepted, confidence.",
                param_hint="--filter",
            )
        if not value:
            raise typer.BadParameter(
                f"Filter '{key}' requires a value.",
                param_hint="--filter",
            )

        parsed[key] = value

    return parsed


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


def default_output_path(cfg: IfetcherConfig, topic: str) -> Path:
    """Derive default output path from config template and topic string."""
    sanitized = sanitize_topic_for_filename(topic)
    return cfg.abspath(cfg.output.path.format(topic=sanitized))


@app.command()
def config(
    action: str = typer.Argument(help="Action: 'schema', 'info', or 'validate'"),
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
    Manage configuration: show schema docs, current config, or validate file.

    Actions:
      schema    - Display configuration schema with all options and defaults
      info      - Display current configuration values
      validate  - Validate configuration file syntax
    """
    config_path, mode, verbose, overrides = get_options_with_fallback(
        config_path, mode, verbose, overrides
    )
    try:
        if action == "schema":
            # Display configuration schema documentation
            console.print()
            console.print("[bold]Configuration Schema[/bold]")
            console.print("Override any value with: [cyan]-O key.path=value[/cyan]")
            console.print()
            tree = render_config_help()
            console.print(tree)
        elif action == "info":
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
            table.add_row("Output Template", cfg.output.path)
            # Show agent configs (handle nested structure)
            if cfg.agents:
                for key, value in cfg.agents.items():
                    if hasattr(value, "llm") and value.llm:
                        table.add_row(f"Agent '{key}' LLM", value.llm)
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if hasattr(subvalue, "llm") and subvalue.llm:
                                table.add_row(
                                    f"Agent '{key}.{subkey}' LLM", subvalue.llm
                                )
            console.print(table)
        elif action == "validate":
            cfg = load_config(config_path, overrides, mode)
            console.print("[green]✓[/green] Configuration is valid")
            if verbose:
                console.print(cfg.model_dump_json(indent=2))
        else:
            console.print(f"[red]Unknown action:[/red] {action}")
            console.print("Valid actions: schema, info, validate")
            raise typer.Exit(1)
    except KeyboardInterrupt:
        _handle_keyboard_interrupt(None)
    except Exception as e:
        _handle_exception(e, None, verbose)


@app.command()
def fetch(
    urls: List[str] = typer.Argument(
        default=[],
        help="URLs or files to fetch (files are expanded to their URLs)",
    ),
    retry_failed: bool = typer.Option(
        False,
        "-r",
        "--retry-failed",
        help="Clear cached failures before fetching",
    ),
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Clear matching URLs from cache instead of fetching",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be cleared without actually clearing (use with --clear-cache)",
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
    config: Optional[str] = config_option(),
    mode: Optional[str] = mode_option(),
    verbose: bool = verbose_option(),
    overrides: List[str] = overrides_option(),
):
    """
    Fetch web content and cache it using PageFetcher, or clear cache entries.

    Downloads HTML/PDF content, converts to markdown, and caches locally.
    Arguments can be URLs directly, or paths to files containing URLs.
    File paths are auto-detected (checkpoint JSON or text file with one URL per line).

    \b
    Examples:
      # Fetch single URL and show cache path
      interaction-finder fetch https://example.com

      # Fetch multiple URLs
      interaction-finder fetch https://example.com https://example.org

      # Fetch URLs from checkpoint file
      interaction-finder fetch research.json

      # Fetch URLs from text file
      interaction-finder fetch urls.txt

      # Mix files and URLs
      interaction-finder fetch research.json https://extra-url.com

      # Retry previously failed URLs
      interaction-finder fetch research.json --retry-failed

      # Show content instead of path
      interaction-finder fetch https://example.com --format content

      # Perform semantic chunking
      interaction-finder fetch https://example.com --chunk

      # Clear all PubMed URLs from cache
      interaction-finder fetch pubmed --clear-cache

      # Clear specific URLs (with or without https://)
      interaction-finder fetch "pmc.ncbi.nlm.nih.gov" --clear-cache

      # Preview what would be cleared
      interaction-finder fetch pubmed --clear-cache --dry-run

      # Clear ALL cached URLs (dangerous!)
      interaction-finder fetch --clear-cache

      # With config override
      interaction-finder fetch https://example.com -O output.cache=custom_cache/
    """
    # Get effective options with fallback to global options
    config_path, mode, verbose, overrides = get_options_with_fallback(
        config, mode, verbose, overrides
    )

    try:
        # Load configuration
        cfg = load_config(config_path, overrides, mode)
        # Handle cache clearing mode
        if clear_cache:
            from interaction_finder.fetcher import URLCache

            cache_dir = cfg.abspath(cfg.output.cache)
            cache = URLCache(cache_dir)
            console.print(f"\n[bold]Cache directory:[/bold] {cache_dir}")
            # Collect URL patterns (or empty for clear-all)
            try:
                patterns = cli_fetch.resolve_urls(urls) if urls else []
            except ValueError:
                # No URLs provided = clear all
                patterns = []
            # Confirm clear-all operation
            if not patterns and not dry_run:
                confirm = typer.confirm(
                    "\n⚠️  Are you ABSOLUTELY sure you want to delete ALL cached URLs? This cannot be undone!"
                )
                if not confirm:
                    console.print("[yellow]Cancelled.[/yellow]")
                    raise typer.Exit(0)

            # Run cache clearing
            async def clear_impl():
                return await cli_fetch.clear_cache_urls(
                    cache, patterns, console, dry_run
                )

            cleared, errors = asyncio.run(clear_impl())
            # Report results
            if not dry_run:
                console.print(f"\n[green]✓[/green] Successfully cleared {cleared} URLs")
                if errors > 0:
                    console.print(f"[yellow]⚠[/yellow] {errors} errors occurred")
            return
        # Validate dry_run only used with clear_cache
        if dry_run and not clear_cache:
            console.print(
                "[red]Error:[/red] --dry-run can only be used with --clear-cache"
            )
            raise typer.Exit(1)
        # Resolve URLs from args (files are expanded to their URLs)
        collected_urls = cli_fetch.resolve_urls(urls)
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
            from interaction_finder.fetcher import URLCache

            # Handle retry-failed: clear failure markers before fetching
            if retry_failed:
                cache_dir = cfg.abspath(cfg.output.cache)
                cache = URLCache(cache_dir)
                cleared = await cli_fetch.clear_failed_urls(cache, collected_urls)
                if cleared > 0:
                    console.print(f"[dim]Cleared {cleared} cached failure(s)[/dim]")
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
    except KeyboardInterrupt:
        _handle_keyboard_interrupt(None)
    except Exception as e:
        _handle_exception(e, None, verbose)


@app.command()
def keywords(
    topic: str = typer.Argument(help="Research topic to find bridging terms for"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output file for results (JSON)"
    ),
    backend: Optional[str] = typer.Option(
        None,
        "-b",
        "--backend",
        help="Search backend to use (pubmed, perplexica, openai)",
    ),
    max_rounds: Optional[int] = typer.Option(
        None, "--max-rounds", help="Override maximum search rounds"
    ),
    force: bool = typer.Option(
        False, "--force", help="Replace existing results if present"
    ),
    config: Optional[str] = config_option(),
    mode: Optional[str] = mode_option(),
    verbose: bool = verbose_option(),
    overrides: List[str] = overrides_option(),
):
    """
    Extract bridging terms for a research topic by analyzing review articles.

    This command performs iterative search, fetch, and keyword extraction to
    discover "bridging terms" - related concepts that can improve literature
    search coverage but don't appear in the original topic name.

    Example:
        interaction-finder keywords "pulmonary arterial hypertension"

        interaction-finder keywords "machine learning" -o keywords.json
    """
    try:
        # Get effective options (local flags override global)
        config_path, mode, verbose, overrides = get_options_with_fallback(
            config, mode, verbose, overrides
        )
        # Load config
        cfg = load_config(config_path, overrides, mode)
        if not output:
            output = default_output_path(cfg, topic)
        # Check if output file exists and has keywords results
        if output.exists() and not force:
            from pydantic import ValidationError
            from interaction_finder.checkpoint import PipelineCheckpoint

            try:
                existing_checkpoint = PipelineCheckpoint.model_validate_json(
                    output.read_text()
                )
                if existing_checkpoint.keywords is not None:
                    console.print(
                        "[yellow]⚠ Keywords results already exist in output file. Use --force to replace.[/yellow]"
                    )
                    raise typer.Exit(1)
            except (json.JSONDecodeError, ValidationError):
                # File exists but is not a valid checkpoint, proceed with warning
                pass
        # Apply CLI overrides
        if max_rounds is not None:
            cfg.stage.keywords.max_rounds = max_rounds

        # Create search backend
        backend_name = backend if backend else cfg.stage.keywords.search_backend
        search_backend = create_search_backend(backend_name, cfg)

        # Import keywords pipeline
        from interaction_finder.keywords import run_keyword_research
        from interaction_finder.keywords.progress import create_keywords_progress
        from interaction_finder.fetcher import ensure_playwright_installed

        # Ensure Playwright is installed before showing progress table
        asyncio.run(ensure_playwright_installed())
        # Run keywords stage with progress display
        console.print(f"[bold]Extracting bridging terms for:[/bold] {topic}\n")
        progress_counter = create_keywords_progress()
        with progress_counter:
            result_checkpoint = asyncio.run(
                run_keyword_research(
                    topic,
                    cfg,
                    search_backend=search_backend,
                    verbose=False,
                    progress=progress_counter,
                )
            )

        console.print()

        # Extract keywords data for display
        keywords_data = result_checkpoint.keywords

        # Display results
        console.print(
            f"\n[bold green]✓ Found {len(keywords_data.terms)} bridging terms[/bold green]"
        )
        console.print(f"Documents processed: {keywords_data.total_documents_processed}")
        console.print(f"Rounds completed: {keywords_data.rounds_completed}")
        console.print(f"\nCoverage: {keywords_data.coverage_assessment}\n")

        # Print terms with similarity scores
        if keywords_data.terms:
            console.print("[bold]Bridging Terms:[/bold]")
            for term, score in zip(keywords_data.terms, keywords_data.scores):
                console.print(f" [dim]{score:5.2f}[/dim] • {term}")
        else:
            console.print("[yellow]No bridging terms found[/yellow]")

        # Save checkpoint
        output.write_text(result_checkpoint.model_dump_json(indent=2))
        console.print(f"\n[dim]Saved to {output}[/dim]")

    except KeyboardInterrupt:
        _handle_keyboard_interrupt(output)
    except Exception as e:
        _handle_exception(e, output, verbose)


@app.command()
def search(
    checkpoint_or_topic: str = typer.Argument(
        help="Checkpoint file path OR research topic string"
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output file for checkpoint (JSON)"
    ),
    backend: Optional[str] = typer.Option(
        None,
        "-b",
        "--backend",
        help="Search backend to use (pubmed, perplexica, openai)",
    ),
    max_rounds: Optional[int] = typer.Option(
        None, "--max-rounds", help="Override maximum search rounds"
    ),
    fetch: bool = typer.Option(
        False, "--fetch", help="Fetch and cache content for all selected results"
    ),
    force: bool = typer.Option(
        False, "--force", help="Replace existing results if present"
    ),
    config: Optional[str] = config_option(),
    mode: Optional[str] = mode_option(),
    verbose: bool = verbose_option(),
    overrides: List[str] = overrides_option(),
):
    """
    Execute search with automatic stage progression.

    Accepts either a checkpoint file OR a topic string. Missing stages
    (keywords) are run automatically.

    Example:
        interaction-finder search "cancer genomics" -o results.json

        interaction-finder search keywords.json -o results.json

        interaction-finder search "cancer" -b perplexica --fetch -o out.json
    """
    try:
        from interaction_finder.upgrade import ensure_search

        # Load config and parse input
        config_path, mode, verbose, overrides = get_options_with_fallback(
            config, mode, verbose, overrides
        )
        cfg = load_config(config_path, overrides, mode)
        checkpoint, topic = load_checkpoint_or_create(checkpoint_or_topic)
        input_is_file = (
            Path(checkpoint_or_topic).exists() and Path(checkpoint_or_topic).is_file()
        )
        if not output and not input_is_file:
            output = default_output_path(cfg, topic)
        # Check if search results already exist
        if checkpoint.search is not None and not force:
            console.print(
                "[yellow]⚠ Search results already exist. Use --force to replace.[/yellow]"
            )
            raise typer.Exit(1)
        # Apply CLI overrides
        if max_rounds is not None:
            cfg.stage.search.max_rounds = max_rounds

        # Create search backend
        backend_name = backend if backend else cfg.stage.search.search_backend
        search_backend = create_search_backend(backend_name, cfg)

        # Show PubMed API key warning if applicable
        if (
            backend_name == "pubmed"
            and hasattr(search_backend, "should_show_api_key_warning")
            and search_backend.should_show_api_key_warning()
        ):
            console.print(
                "[yellow]Note:[/yellow] PubMed API key not configured. "
                "Using default rate limit of 3 req/sec.\n"
                "With an API key, you can increase to 10 req/sec. "
                "Get your free key at: https://www.ncbi.nlm.nih.gov/account/settings/\n"
            )

        # Pre-load reranker if enabled
        reranker = None
        if cfg.stage.search.rerank_top_k > 0:
            from interaction_finder.widesearch.reranker import Reranker

            reranker = Reranker(
                model_name=cfg.stage.search.reranker_model,
                device=cfg.stage.search.reranker_device,
            )
            _ = reranker._get_model()

        # Determine checkpoint path (for saving after each stage)
        if output:
            checkpoint_path = str(output)
        elif input_is_file:
            checkpoint_path = checkpoint_or_topic
        else:
            checkpoint_path = None

        # Ensure Playwright is installed before showing progress tables
        from interaction_finder.fetcher import ensure_playwright_installed

        asyncio.run(ensure_playwright_installed())
        # Run pipeline (ensure_search manages progress internally for each stage)
        checkpoint = asyncio.run(
            ensure_search(
                checkpoint,
                search_backend,
                cfg,
                console=console,
                checkpoint_path=checkpoint_path,
                force=force,
            )
        )

        # Fetch content if requested
        if fetch:
            from interaction_finder.widesearch import fetch_and_populate_results

            results_count = len(checkpoint.search.results) if checkpoint.search else 0
            console.print()
            console.print(
                f"[bold]Fetching content for {results_count} results...[/bold]"
            )
            fetch_stats = asyncio.run(fetch_and_populate_results(checkpoint, cfg))
            console.print(
                f"[green]✓[/green] Fetched {fetch_stats['fetched']}/{fetch_stats['total']} "
                f"({fetch_stats['cached']} cached, {fetch_stats['failed']} failed)"
            )

            # Save checkpoint again after fetching (only if we have a path)
            if checkpoint_path:
                Path(checkpoint_path).write_text(checkpoint.model_dump_json(indent=2))
                console.print(f"[dim]Saved checkpoint to {checkpoint_path}[/dim]")

    except KeyboardInterrupt:
        _handle_keyboard_interrupt(output)
    except Exception as e:
        _handle_exception(e, output, verbose)


@app.command()
def extract(
    checkpoint_or_topic: str = typer.Argument(
        help="Checkpoint file path OR research topic string"
    ),
    entity_types: List[str] = typer.Option(
        ...,
        "--entity-type",
        "-e",
        help="Entity types to extract (can specify multiple)",
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output file for extraction results (JSON)"
    ),
    backend: Optional[str] = typer.Option(
        None,
        "-b",
        "--backend",
        help="Search backend (pubmed, perplexica, openai) if search stage needed",
    ),
    force: bool = typer.Option(
        False, "--force", help="Replace existing results if present"
    ),
    config: Optional[str] = config_option(),
    mode: Optional[str] = mode_option(),
    verbose: bool = verbose_option(),
    overrides: List[str] = overrides_option(),
):
    """
    Extract entity-entity associations with automatic stage progression.

    Accepts either a checkpoint file OR a topic string. Missing stages
    (keywords, search) are run automatically.

    Example:
        interaction-finder extract "cancer genomics" -e gene -e disease -o results.json

        interaction-finder extract keywords.json -e gene -e disease -o results.json

        interaction-finder extract search.json -e gene -e protein -o results.json
    """
    try:
        from interaction_finder.upgrade import ensure_extraction

        # Load config and parse input
        config_path, mode, verbose, overrides = get_options_with_fallback(
            config, mode, verbose, overrides
        )
        cfg = load_config(config_path, overrides, mode)
        checkpoint, topic = load_checkpoint_or_create(checkpoint_or_topic)
        input_is_file = (
            Path(checkpoint_or_topic).exists() and Path(checkpoint_or_topic).is_file()
        )
        if not output and not input_is_file:
            output = default_output_path(cfg, topic)
        # Block if extraction already complete (incomplete extractions can resume)
        if (
            checkpoint.extraction is not None
            and checkpoint.extraction.metadata.is_complete
            and not force
        ):
            console.print(
                "[yellow]⚠ Extraction already complete. Use --force to replace.[/yellow]"
            )
            raise typer.Exit(1)
        # Create search backend (needed if search stage must run)
        backend_name = backend if backend else cfg.stage.search.search_backend
        search_backend = create_search_backend(backend_name, cfg)

        # Determine checkpoint path (for saving after each stage)
        if output:
            checkpoint_path = str(output)
        elif input_is_file:
            checkpoint_path = checkpoint_or_topic
        else:
            checkpoint_path = None

        # Run pipeline (ensure_extraction manages progress internally for each stage)
        checkpoint = asyncio.run(
            ensure_extraction(
                checkpoint,
                entity_types,
                search_backend,
                cfg,
                console=console,
                checkpoint_path=checkpoint_path,
                force=force,
            )
        )

        console.print()

        # Show sample of accepted pairs
        accepted_judgments = [j for j in checkpoint.extraction.judgments if j.accepted]
        if accepted_judgments:
            console.print("\n[bold]Sample accepted pairs:[/bold]")
            for judgment in accepted_judgments[:5]:
                console.print(
                    f"  • {judgment.entity1.name} ({judgment.entity1.kind}) "
                    f"[dim]{judgment.relationship}[/dim] "
                    f"{judgment.entity2.name} ({judgment.entity2.kind})"
                )
                total_quotes = sum(len(a.quotes) for a in judgment.assessments)
                console.print(
                    f"    Evidence: {total_quotes} quotes, "
                    f"{len(judgment.assessments)} assessments, "
                    f"quality: {judgment.evidence.label} ({judgment.evidence.overall}/9)"
                )
            if len(accepted_judgments) > 5:
                console.print(f"  ... and {len(accepted_judgments) - 5} more")

    except KeyboardInterrupt:
        _handle_keyboard_interrupt(output)
    except Exception as e:
        _handle_exception(e, output, verbose)


@app.command()
def report(
    extraction_file: Path = typer.Argument(help="Path to extraction results JSON file"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output file path (defaults based on format)"
    ),
    title: Optional[str] = typer.Option(
        None, "-t", "--title", help="Custom report title"
    ),
    format: str = typer.Option(
        "html",
        "-f",
        "--format",
        help="Output format ('html', 'plain', or 'plain:KIND' for entity lists)",
        show_default=True,
    ),
    filters: List[str] = typer.Option(
        [],
        "--filter",
        help="Filter pairs (key:value). Supported keys: accepted, confidence",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """
    Generate report output from extraction results.

    HTML format creates a self-contained interactive explorer with provenance
    tracking. Plain format emits simple "entity, relationship, entity" tuples
    (one per line). Use "plain:KIND" to emit unique entity names of a specific
    kind (e.g., genes, diseases) one per line. Apply filters with --filter
    (e.g., --filter confidence:high, --filter accepted:any).

    Example:
        interaction-finder report results.json -o report.html

        interaction-finder report pah-results.json -o pah-report.html --title "PAH Report"
    """
    output_is_stdout = output is not None and str(output) == "-"
    log_console = console if not output_is_stdout else Console(stderr=True)

    try:
        # Validate extraction file exists
        if not extraction_file.exists():
            raise FileNotFoundError(f"Extraction file not found: {extraction_file}")

        # Load extraction checkpoint

        from interaction_finder.checkpoint import PipelineCheckpoint

        log_console.print(f"Loading {extraction_file}...")
        checkpoint = PipelineCheckpoint.model_validate_json(extraction_file.read_text())
        # Check version and backup if needed
        _check_and_backup_checkpoint(extraction_file, checkpoint.created_by)
        # Ensure extraction stage is present
        if not checkpoint.extraction:
            raise ValueError("Checkpoint does not contain extraction results")
        # Display summary
        meta = checkpoint.extraction.metadata
        log_console.print(f"  Topic: {checkpoint.topic}")
        log_console.print(
            f"  Pairs: {meta.pairs_accepted} accepted, {meta.pairs_rejected} rejected | Documents: {meta.resource_count}"
        )

        normalized_format = format.lower()
        supported_formats = {"html", "plain"}
        plain_kind = normalized_format.startswith("plain:")
        if normalized_format not in supported_formats and not plain_kind:
            raise typer.BadParameter(
                "Unsupported format '{format}'. Supported formats: 'html', 'plain', or 'plain:KIND'.".format(
                    format=format
                ),
                param_hint="--format",
            )

        parsed_filters = _parse_filter_options(filters)
        if (
            normalized_format != "html"
            and "accepted" not in parsed_filters
            and not plain_kind
        ):
            parsed_filters["accepted"] = "yes"
        elif plain_kind and "accepted" not in parsed_filters:
            parsed_filters["accepted"] = "yes"

        # Generate output path if not specified
        if output is None:
            suffix = ".html" if normalized_format == "html" else ".txt"
            output = extraction_file.with_suffix(suffix)
        else:
            output_is_stdout = str(output) == "-"
            if output_is_stdout:
                log_console = Console(stderr=True)

        # Generate report
        from interaction_finder.report import generate_report

        output_path = generate_report(
            checkpoint=checkpoint,
            output=output,
            title=title,
            format=normalized_format,
            filters=parsed_filters if parsed_filters else None,
        )
        destination_label = "stdout" if str(output_path) == "-" else str(output_path)
        log_console.print(f"[green]✓[/green] Report written to {destination_label}")

    except KeyboardInterrupt:
        _handle_keyboard_interrupt(None)
    except Exception as e:
        _handle_exception(e, None, verbose)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
