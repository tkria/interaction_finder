"""Stage upgrade chain for progressive checkpoint completion.

Provides idempotent `ensure_*()` functions that check prerequisites and run
missing stages. Each ensure function chains to its prerequisite automatically.

Structure:
    ensure_keywords()    - Ensure keywords complete (none → keywords)
    ensure_search()      - Ensure search complete (→ keywords → search)
    ensure_extraction()  - Ensure extraction complete (→ keywords → search → extraction)

Properties:
- Idempotent: Safe to call multiple times
- Composable: Chain prerequisites automatically
- Minimal: Direct calls to underlying pipeline functions
"""

from typing import Any, Literal

from rich.console import Console

from interaction_finder.checkpoint import PipelineCheckpoint
from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchBackend
from interaction_finder.settings import IfetcherConfig

# Type aliases for clarity
StageLevel = Literal["none", "keywords", "search", "extraction"]


def checkpoint_stage(checkpoint: PipelineCheckpoint) -> StageLevel:
    """Determine the current stage level of a checkpoint.

    Parameters:
        checkpoint: PipelineCheckpoint — checkpoint to inspect

    Returns:
        StageLevel — highest completed stage ("none", "keywords", "search", "extraction")
    """
    if checkpoint.extraction is not None:
        return "extraction"
    elif checkpoint.search is not None:
        return "search"
    elif checkpoint.keywords is not None:
        return "keywords"
    else:
        return "none"


async def ensure_keywords(
    checkpoint: PipelineCheckpoint,
    config: IfetcherConfig,
    search_backend: SearchBackend | None = None,
    console: Console | None = None,
    checkpoint_path: str | None = None,
) -> PipelineCheckpoint:
    """Ensure keywords stage complete. Idempotent.

    Parameters:
        checkpoint: Checkpoint at any stage
        config: Configuration
        search_backend: Search backend for keyword extraction (defaults to PubMed)
        console: Optional console for status messages
        checkpoint_path: Optional path to save checkpoint after completion

    Returns:
        Checkpoint with at least keywords stage
    """
    # Run keywords stage if not already complete
    if checkpoint.keywords is None:
        # Print stage start message
        if console:
            console.print(
                f"[bold]Running keywords stage for:[/bold] {checkpoint.topic}\n"
            )

        from interaction_finder.keywords import run_keyword_research
        from interaction_finder.keywords.progress import KeywordsProgress

        keywords_progress = KeywordsProgress()
        with keywords_progress:
            checkpoint = await run_keyword_research(
                topic=checkpoint.topic,
                config=config,
                search_backend=search_backend,
                verbose=False,
                progress=keywords_progress,
            )

        # Save checkpoint if path provided (only after running the stage)
        if checkpoint_path:
            from pathlib import Path

            Path(checkpoint_path).write_text(checkpoint.model_dump_json(indent=2))
            if console:
                console.print(f"[dim]Saved checkpoint to {checkpoint_path}[/dim]\n")

    return checkpoint


async def ensure_search(
    checkpoint: PipelineCheckpoint,
    search_backend: SearchBackend,
    config: IfetcherConfig,
    keywords_backend: SearchBackend | None = None,
    console: Console | None = None,
    checkpoint_path: str | None = None,
) -> PipelineCheckpoint:
    """Ensure search stage complete. Idempotent. Runs keywords if needed.

    Parameters:
        checkpoint: Checkpoint at any stage
        search_backend: Search backend for widesearch
        config: Configuration
        keywords_backend: Search backend for keywords stage (defaults to same as search_backend)
        console: Optional console for status messages
        checkpoint_path: Optional path to save checkpoint after completion

    Returns:
        Checkpoint with at least search stage
    """
    # Run search stage if not already complete
    if checkpoint.search is None:
        # Ensure keywords first (use keywords_backend or fall back to search_backend)
        kw_backend = (
            keywords_backend if keywords_backend is not None else search_backend
        )
        checkpoint = await ensure_keywords(
            checkpoint,
            config,
            search_backend=kw_backend,
            console=console,
            checkpoint_path=checkpoint_path,
        )

        # Print stage start message
        if console:
            console.print(
                f"[bold]Running widesearch stage for:[/bold] {checkpoint.topic}\n"
            )

        # Run search with dedicated widesearch progress counter
        from interaction_finder.widesearch import run_widesearch_with_checkpoint
        from interaction_finder.widesearch.progress import WidesearchProgress

        widesearch_progress = WidesearchProgress()
        with widesearch_progress:
            checkpoint = await run_widesearch_with_checkpoint(
                input_checkpoint=checkpoint,
                search_backend=search_backend,
                config=config,
                progress=widesearch_progress,
            )

        # Save checkpoint if path provided (only after running the stage)
        if checkpoint_path:
            from pathlib import Path

            Path(checkpoint_path).write_text(checkpoint.model_dump_json(indent=2))
            if console:
                console.print(f"[dim]Saved checkpoint to {checkpoint_path}[/dim]\n")

    return checkpoint


async def ensure_extraction(
    checkpoint: PipelineCheckpoint,
    target_entity_types: list[str],
    search_backend: SearchBackend,
    config: IfetcherConfig,
    console: Console | None = None,
    checkpoint_path: str | None = None,
) -> PipelineCheckpoint:
    """Ensure extraction stage complete. Idempotent. Runs all prerequisites if needed.

    Parameters:
        checkpoint: Checkpoint at any stage
        target_entity_types: Entity types to extract
        search_backend: Search backend (if search needed)
        config: Configuration
        console: Optional console for status messages
        checkpoint_path: Optional path to save checkpoint after completion

    Returns:
        Checkpoint with extraction stage
    """
    # Run extraction stage if not already complete
    if checkpoint.extraction is None:
        # Ensure search first (which ensures keywords)
        checkpoint = await ensure_search(
            checkpoint,
            search_backend,
            config,
            console=console,
            checkpoint_path=checkpoint_path,
        )

        # Print stage start message
        if console:
            console.print(
                f"[bold]Running extraction stage for:[/bold] {checkpoint.topic} "
                f"(types: {', '.join(target_entity_types)})\n"
            )

        # Fetch content and run extraction with dedicated extraction progress counter
        from interaction_finder.extraction import run_extraction
        from interaction_finder.extraction.progress import ExtractionProgress
        from interaction_finder.widesearch import fetch_and_populate_results

        await fetch_and_populate_results(checkpoint, config)

        extraction_progress = ExtractionProgress()
        with extraction_progress:
            checkpoint = await run_extraction(
                input_checkpoint=checkpoint,
                target_entity_types=target_entity_types,
                config=config,
                progress=extraction_progress,
            )

        # Save checkpoint if path provided (only after running the stage)
        if checkpoint_path:
            from pathlib import Path

            Path(checkpoint_path).write_text(checkpoint.model_dump_json(indent=2))
            if console:
                console.print(f"[dim]Saved checkpoint to {checkpoint_path}[/dim]\n")

    return checkpoint


def create_empty_checkpoint(topic: str) -> PipelineCheckpoint:
    """Create a minimal checkpoint with just topic and empty resources.

    This is the starting point for a full pipeline run from scratch.

    Parameters:
        topic: Research topic to investigate

    Returns:
        Empty checkpoint at 'none' stage

    Example:
        >>> checkpoint = create_empty_checkpoint("cancer genomics")
        >>> result = await upgrade_checkpoint(checkpoint, "extraction", ...)
    """
    return PipelineCheckpoint(
        topic=topic,
        resources=ResourcePool(),
    )
