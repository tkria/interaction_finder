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
    Fetch web content and cache it using PageFetcher, or clear cache entries.

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

      # Clear all PubMed URLs from cache
      interaction-finder fetch pubmed --clear-cache

      # Clear specific URLs (with or without https://)
      interaction-finder fetch "pmc.ncbi.nlm.nih.gov" --clear-cache

      # Clear URLs from file
      interaction-finder fetch --input pubmed-urls.txt --clear-cache

      # Preview what would be cleared
      interaction-finder fetch pubmed --clear-cache --dry-run

      # Clear ALL cached URLs (dangerous!)
      interaction-finder fetch --clear-cache

      # Glob pattern matching
      interaction-finder fetch "https://pubmed.ncbi.nlm.nih.gov/123*" --clear-cache

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
        # Handle cache clearing mode
        if clear_cache:
            from interaction_finder.fetcher import URLCache

            cache_dir = cfg.abspath(cfg.output.cache)
            cache = URLCache(cache_dir)
            console.print(f"\n[bold]Cache directory:[/bold] {cache_dir}")
            # Collect URL patterns (or empty for clear-all)
            try:
                patterns = (
                    cli_fetch.collect_urls(urls, input_file)
                    if (urls or input_file)
                    else []
                )
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


@app.command()
def keywords(
    topic: str = typer.Argument(help="Research topic to find bridging terms for"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output file for results (JSON)"
    ),
    max_rounds: Optional[int] = typer.Option(
        None, "--max-rounds", help="Override maximum search rounds"
    ),
    backend: Optional[str] = typer.Option(
        None, "-b", "--backend", help="Search backend to use"
    ),
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
        # Load configuration
        config_path = None  # Use default config
        cfg = IfetcherConfig.from_path(config_path) if config_path else IfetcherConfig()

        # Apply overrides if provided
        if max_rounds:
            cfg.tools.keywords.max_rounds = max_rounds
        if backend:
            cfg.tools.keywords.search_backend = backend

        # Show PubMed API key warning if using PubMed backend
        # Note: Keywords module currently hardcodes PubMedBackend
        backend_name = cfg.tools.keywords.search_backend
        if backend_name == "pubmed":
            # Import backend to check for API key
            from interaction_finder.search.backends.pubmed import PubMedBackend

            test_backend = PubMedBackend(config={"timeout": cfg.tools.search.timeout})
            if test_backend.should_show_api_key_warning():
                console.print(
                    "[yellow]Note:[/yellow] PubMed API key not configured. "
                    "Using default rate limit of 3 req/sec.\n"
                    "With an API key, you can increase to 10 req/sec. "
                    "Get your free key at: https://www.ncbi.nlm.nih.gov/account/settings/\n"
                )

        # Import here to avoid slow imports at CLI startup
        from interaction_finder.keywords import run_keyword_research

        # Run keyword research
        console.print(f"[bold]Extracting bridging terms for:[/bold] {topic}\n")
        result = asyncio.run(run_keyword_research(topic, cfg, verbose=True))

        # Display results
        console.print(
            f"\n[bold green]✓ Found {len(result.terms)} bridging terms[/bold green]"
        )
        console.print(f"Documents processed: {result.total_documents_processed}")
        console.print(f"Rounds completed: {result.rounds_completed}")
        console.print(f"\nCoverage: {result.coverage_assessment}\n")

        # Print terms with similarity scores
        if result.terms:
            console.print("[bold]Bridging Terms:[/bold]")
            for term, score in zip(result.terms, result.scores):
                console.print(f" [dim]{score:5.2f}[/dim] • {term}")
        else:
            console.print("[yellow]No bridging terms found[/yellow]")

        # Save to file if requested
        if output:
            output_data = result.model_dump(mode="json")
            output.write_text(json.dumps(output_data, indent=2))
            console.print(f"\n[dim]Saved to {output}[/dim]")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        console.print_exception()
        raise typer.Exit(1)


@app.command()
def widesearch(
    keywords_file: Path = typer.Argument(
        help="Path to keywords JSON file from keywords command"
    ),
    topic: Optional[str] = typer.Argument(
        None,
        help="Research topic being investigated (inferred from keywords file if not provided)",
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
    Execute widesearch using keywords from keywords command.

    Performs iterative query expansion and search using provided keyphrases,
    saving complete checkpoint (results, queries, resource pool) to JSON.
    The checkpoint format enables downstream processing and search resumption.

    Topic is inferred from the keywords file's run_metadata if not explicitly provided.

    Example:
        interaction-finder widesearch keywords.json -o results.json

        interaction-finder widesearch keywords.json "cancer" -b perplexica -o out.json

        interaction-finder widesearch keywords.json --fetch -o results.json
    """
    try:
        # Get effective options
        config_path, verbose, overrides = get_options_with_fallback(
            config, verbose, overrides
        )
        # Load config
        cfg = load_config(config_path, overrides, mode)
        # Validate keywords file exists
        if not keywords_file.exists():
            raise FileNotFoundError(f"Keywords file not found: {keywords_file}")
        # Parse checkpoint JSON
        from interaction_finder.checkpoint import PipelineCheckpoint
        from pydantic import ValidationError

        input_checkpoint = PipelineCheckpoint.model_validate_json(
            keywords_file.read_text()
        )

        # Infer topic from checkpoint if not provided
        if topic is None:
            topic = input_checkpoint.topic
            console.print(f"[dim]Inferred topic from checkpoint:[/dim] {topic}")
        # Apply CLI overrides to config
        if max_rounds is not None:
            cfg.tools.widesearch.max_rounds = max_rounds
        # Determine backend
        backend_name = backend if backend else cfg.tools.widesearch.search_backend
        # Import backends and create mapping
        from interaction_finder.search.backends.openai import OpenAIBackend
        from interaction_finder.search.backends.perplexica import PerplexicaBackend
        from interaction_finder.search.backends.pubmed import PubMedBackend

        BACKENDS = {
            "pubmed": PubMedBackend,
            "perplexica": PerplexicaBackend,
            "openai": OpenAIBackend,
        }
        # Instantiate backend
        backend_class = BACKENDS.get(backend_name)
        if not backend_class:
            valid = ", ".join(BACKENDS.keys())
            raise ValueError(f"Unknown backend '{backend_name}'. Valid: {valid}")
        search_backend = backend_class(config={"timeout": cfg.tools.search.timeout})
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
        # Import widesearch entrypoint
        from interaction_finder.widesearch import run_widesearch_with_checkpoint

        # Display start message
        console.print(f"[bold]Running widesearch for:[/bold] {topic}")
        # Show keyphrases info if keywords stage present
        if input_checkpoint.keywords:
            console.print(
                f"Using {len(input_checkpoint.keywords.terms)} keyphrases from {keywords_file.name}"
            )
        # Show resource pool info if present
        resource_count = len(input_checkpoint.resources.resource_map)
        if resource_count > 0:
            console.print(f"Starting with {resource_count} existing resources in pool")
        console.print()
        # Pre-load reranker if enabled to avoid debug messages during progress display
        reranker = None
        if cfg.tools.widesearch.rerank_top_k > 0:
            from interaction_finder.widesearch.reranker import Reranker

            reranker = Reranker(
                model_name=cfg.tools.widesearch.reranker_model,
                device=cfg.tools.widesearch.reranker_device,
            )
            # Trigger model loading before progress display starts
            _ = reranker._get_model()
        # Create progress display
        from interaction_finder.widesearch.progress import WidesearchProgress

        progress_counter = WidesearchProgress()
        # Run widesearch with checkpoint, passing the entire checkpoint
        with progress_counter:
            checkpoint = asyncio.run(
                run_widesearch_with_checkpoint(
                    input_checkpoint=input_checkpoint,
                    search_backend=search_backend,
                    config=cfg,
                    max_rounds=max_rounds,
                    reranker=reranker,
                    progress=progress_counter,
                )
            )
        # Progress display already shows completion summary, just add newline
        console.print()
        # Fetch content if requested
        if fetch:
            from interaction_finder.widesearch import fetch_and_populate_results

            results_count = len(checkpoint.search.results) if checkpoint.search else 0
            console.print(
                f"\n[bold]Fetching content for {results_count} results...[/bold]"
            )
            fetch_stats = asyncio.run(fetch_and_populate_results(checkpoint, cfg))
            console.print(
                f"[green]✓[/green] Fetched {fetch_stats['fetched']}/{fetch_stats['total']} "
                f"({fetch_stats['cached']} cached, {fetch_stats['failed']} failed)"
            )
        # Save to file if requested
        if output:
            checkpoint_data = checkpoint.model_dump(mode="json")
            output.write_text(json.dumps(checkpoint_data, indent=2))
            console.print(f"\n[dim]Saved checkpoint to {output}[/dim]")

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Invalid keywords file:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def extract(
    checkpoint_file: Path = typer.Argument(
        help="Path to widesearch checkpoint JSON file"
    ),
    topic: Optional[str] = typer.Argument(
        None,
        help="Research topic for extraction context (inferred from checkpoint file if not provided)",
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
    Extract entity-entity associations from widesearch results.

    Takes a widesearch checkpoint JSON file and runs the extraction pipeline
    to identify and validate associations between entities with comprehensive
    quote-level provenance tracking. Automatically fetches missing content.

    Topic is inferred from the checkpoint file's run_metadata if not explicitly provided.

    Example:
        interaction-finder extract searches.json -e gene -e disease -o results.json

        interaction-finder extract pah-searches.json "PAH genetics" -e gene -e protein -o pah-pairs.json
    """
    try:
        # Get effective options
        config_path, verbose, overrides = get_options_with_fallback(
            config, verbose, overrides
        )
        # Load config
        cfg = load_config(config_path, overrides, mode)
        # Validate checkpoint file exists
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        # Load and validate checkpoint
        from interaction_finder.checkpoint import PipelineCheckpoint
        from pydantic import ValidationError

        input_checkpoint = PipelineCheckpoint.model_validate_json(
            checkpoint_file.read_text()
        )

        # Infer topic from checkpoint file if not provided
        if topic is None:
            topic = input_checkpoint.topic
            console.print(f"[dim]Inferred topic from checkpoint:[/dim] {topic}")

        # Display checkpoint info
        console.print(f"\n[bold]Loading checkpoint:[/bold] {checkpoint_file.name}")
        console.print(f"Topic: {input_checkpoint.topic}")
        if input_checkpoint.search:
            console.print(f"Results: {len(input_checkpoint.search.results)}")
            console.print(f"Queries executed: {len(input_checkpoint.search.queries)}")
        console.print(
            f"Resources in pool: {len(input_checkpoint.resources.resource_map)}"
        )
        # Check if content needs to be fetched
        resources_with_content = sum(
            1 for r in input_checkpoint.resources.resource_map.values() if r is not None
        )
        resources_without_content = (
            len(input_checkpoint.resources.resource_map) - resources_with_content
        )
        if resources_without_content > 0:
            from interaction_finder.widesearch import fetch_and_populate_results

            console.print(
                f"\n[bold]Fetching content for {resources_without_content} resources...[/bold]"
            )
            fetch_stats = asyncio.run(fetch_and_populate_results(input_checkpoint, cfg))
            console.print(
                f"[green]✓[/green] Fetched {fetch_stats['fetched']}/{fetch_stats['total']} "
                f"({fetch_stats['cached']} cached, {fetch_stats['failed']} failed)"
            )
            resources_with_content = sum(
                1
                for r in input_checkpoint.resources.resource_map.values()
                if r is not None
            )
        # Run extraction
        console.print(
            f"\n[bold]Running extraction:[/bold] {topic} (types: {', '.join(entity_types)})"
        )
        console.print(f"Processing {resources_with_content} resources with content\n")
        from interaction_finder.extraction import run_extraction
        from interaction_finder.extraction.progress import ExtractionProgress

        # Create progress display
        progress = ExtractionProgress()

        # Run extraction with config and progress
        with progress:
            result_checkpoint = asyncio.run(
                run_extraction(
                    input_checkpoint=input_checkpoint,
                    target_entity_types=entity_types,
                    config=cfg,
                    progress=progress,
                )
            )
        # Progress display already shows completion summary
        console.print()
        # Show sample of accepted pairs
        accepted_judgments = [
            j for j in result_checkpoint.extraction.judgments if j.accepted
        ]
        if accepted_judgments:
            console.print("\n[bold]Sample accepted pairs:[/bold]")
            for judgment in accepted_judgments[:5]:
                console.print(
                    f"  • {judgment.entity1.name} ({judgment.entity1.kind}) "
                    f"[dim]{judgment.relationship}[/dim] "
                    f"{judgment.entity2.name} ({judgment.entity2.kind})"
                )
                # Count total quotes across all assessments
                total_quotes = sum(len(a.quotes) for a in judgment.assessments)
                console.print(
                    f"    Evidence: {total_quotes} quotes, "
                    f"{len(judgment.assessments)} assessments, "
                    f"confidence: {judgment.confidence}"
                )
            if len(accepted_judgments) > 5:
                console.print(f"  ... and {len(accepted_judgments) - 5} more")
        # Save to file if requested
        if output:
            output_data = result_checkpoint.model_dump(mode="json")
            output.write_text(json.dumps(output_data, indent=2))
            console.print(f"\n[dim]Saved results to {output}[/dim]")
        else:
            console.print("\n[dim]Use -o/--output to save results to a file[/dim]")

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except (json.JSONDecodeError, ValidationError) as e:
        console.print(f"[red]Invalid checkpoint file:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


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
        from pydantic import ValidationError

        log_console.print(
            f"\n[bold]Loading extraction results:[/bold] {extraction_file.name}"
        )
        checkpoint = PipelineCheckpoint.model_validate_json(extraction_file.read_text())

        # Ensure extraction stage is present
        if not checkpoint.extraction:
            raise ValueError("Checkpoint does not contain extraction results")

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

        # Display summary
        log_console.print(f"Topic: {checkpoint.topic}")
        log_console.print(
            f"Total pairs: {checkpoint.extraction.metadata.total_pairs_found}"
        )
        log_console.print(f"Accepted: {checkpoint.extraction.metadata.pairs_accepted}")
        log_console.print(f"Rejected: {checkpoint.extraction.metadata.pairs_rejected}")
        log_console.print(f"Resources: {checkpoint.extraction.metadata.resource_count}")

        # Generate output path if not specified
        if output is None:
            suffix = ".html" if normalized_format == "html" else ".txt"
            output = extraction_file.with_suffix(suffix)
        else:
            output_is_stdout = str(output) == "-"
            if output_is_stdout:
                log_console = Console(stderr=True)

        # Generate report
        log_console.print(f"\n[bold]Generating report ({normalized_format})...[/bold]")
        from interaction_finder.report import generate_report

        output_path = generate_report(
            checkpoint=checkpoint,
            output=output,
            title=title,
            format=normalized_format,
            filters=parsed_filters if parsed_filters else None,
        )

        destination_label = "stdout" if str(output_path) == "-" else str(output_path)
        log_console.print(f"[green]✓[/green] Report generated: {destination_label}")

        # Show statistics (all pairs are included)
        log_console.print(f"\n[bold]Report contains:[/bold]")
        log_console.print(
            f"  • {checkpoint.extraction.metadata.total_pairs_found} pairs ({checkpoint.extraction.metadata.pairs_accepted} accepted, {checkpoint.extraction.metadata.pairs_rejected} rejected)"
        )
        log_console.print(
            f"  • {checkpoint.extraction.metadata.resource_count} documents"
        )
        log_console.print(f"  • Interactive filtering and search")
        log_console.print(f"  • Full provenance tracking")

    except FileNotFoundError as e:
        log_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except (json.JSONDecodeError, ValidationError) as e:
        log_console.print(f"[red]Invalid extraction file:[/red] {e}")
        if verbose:
            log_console.print_exception()
        raise typer.Exit(1)
    except ValueError as e:
        log_console.print(f"[red]Error:[/red] {e}")
        if verbose:
            log_console.print_exception()
        raise typer.Exit(1)
    except Exception as e:
        log_console.print(f"\n[red]Error:[/red] {e}")
        if verbose:
            log_console.print_exception()
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
