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
    progress: Any | None = None,
) -> PipelineCheckpoint:
    """Ensure keywords stage complete. Idempotent.

    Parameters:
        checkpoint: Checkpoint at any stage
        config: Configuration
        search_backend: Search backend for keyword extraction (defaults to PubMed)
        progress: Optional progress counter for live display (currently unused in keywords)

    Returns:
        Checkpoint with at least keywords stage
    """
    if checkpoint.keywords is not None:
        return checkpoint

    from interaction_finder.keywords import run_keyword_research

    # Note: run_keyword_research doesn't yet support progress parameter
    return await run_keyword_research(
        topic=checkpoint.topic,
        config=config,
        search_backend=search_backend,
        verbose=False,
    )


async def ensure_search(
    checkpoint: PipelineCheckpoint,
    search_backend: SearchBackend,
    config: IfetcherConfig,
    keywords_backend: SearchBackend | None = None,
    progress: Any | None = None,
) -> PipelineCheckpoint:
    """Ensure search stage complete. Idempotent. Runs keywords if needed.

    Parameters:
        checkpoint: Checkpoint at any stage
        search_backend: Search backend for widesearch
        config: Configuration
        keywords_backend: Search backend for keywords stage (defaults to same as search_backend)
        progress: Optional progress counter for live display

    Returns:
        Checkpoint with at least search stage
    """
    if checkpoint.search is not None:
        return checkpoint

    # Ensure keywords first (use keywords_backend or fall back to search_backend)
    kw_backend = keywords_backend if keywords_backend is not None else search_backend
    checkpoint = await ensure_keywords(
        checkpoint, config, search_backend=kw_backend, progress=progress
    )

    # Run search
    from interaction_finder.widesearch import run_widesearch_with_checkpoint

    return await run_widesearch_with_checkpoint(
        input_checkpoint=checkpoint,
        search_backend=search_backend,
        config=config,
        progress=progress,
    )


async def ensure_extraction(
    checkpoint: PipelineCheckpoint,
    target_entity_types: list[str],
    search_backend: SearchBackend,
    config: IfetcherConfig,
) -> PipelineCheckpoint:
    """Ensure extraction stage complete. Idempotent. Runs all prerequisites if needed.

    Parameters:
        checkpoint: Checkpoint at any stage
        target_entity_types: Entity types to extract
        search_backend: Search backend (if search needed)
        config: Configuration

    Returns:
        Checkpoint with extraction stage
    """
    if checkpoint.extraction is not None:
        return checkpoint

    # Ensure search first (which ensures keywords)
    checkpoint = await ensure_search(checkpoint, search_backend, config)

    # Fetch content and run extraction
    from interaction_finder.extraction import run_extraction
    from interaction_finder.widesearch import fetch_and_populate_results

    await fetch_and_populate_results(checkpoint, config)

    return await run_extraction(
        input_checkpoint=checkpoint,
        target_entity_types=target_entity_types,
        config=config,
    )


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
