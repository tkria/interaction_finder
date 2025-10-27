# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**interaction-finder** is a Python tool for automated extraction of biological interactions (gene-disease, protein-protein, cell-biomarker) from scientific literature using AI-driven graph workflows. Built for research benchmarking of biomedical relation extraction approaches with comprehensive provenance tracking and resource management.

## Development Commands

**Virtual Environment**: This project uses `uv` for dependency management:
```bash
# Run code with dependencies
uv run python -m interaction_finder.cli

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_fetcher.py

# Run tests with verbose output
uv run pytest -v

# Install dependencies after changes
uv sync
```

**CLI Usage**:
```bash
# Primary workflow: extract interactions from training data
uv run interaction-finder extract -t BRCA1

# Extract from custom source
uv run interaction-finder extract --source urls.txt -t diabetes

# Fetch content only (no extraction)
uv run interaction-finder extract -t BRCA1 --fetch-only

# Show available terms
uv run interaction-finder terms

# Configuration management
uv run interaction-finder config info
uv run interaction-finder config edit
uv run interaction-finder config validate

# Dry run to preview processing
uv run interaction-finder extract -t BRCA1 --dry-run --verbose

# Reverse search with investigation logging
uv run interaction-finder reverse-search --known resources.jsonl --investigation-log investigation.jsonl
```

**Investigation Logging for Reverse Search**:

Investigation logging provides detailed insights into the reverse search pipeline, recording every stage of query generation, search execution, and resource matching. This is invaluable for debugging search strategies, analyzing coverage patterns, and understanding why certain resources were found or missed.

Enable investigation logging with the `--investigation-log` flag:
```bash
# Basic usage
uv run interaction-finder reverse-search --known resources.jsonl --investigation-log investigation.jsonl

# With verbose output to see warnings
uv run interaction-finder reverse-search --known resources.jsonl --investigation-log investigation.jsonl -v
```

**Log Format**: JSON Lines (one JSON object per line), with each entry containing:
- `session_id`: Unique identifier linking all entries from a single search session
- `stage`: Pipeline stage (session_start, content_fetch, query_generation, search_execution, matching, session_end)
- `timestamp`: ISO 8601 timestamp
- Stage-specific data (queries, results, match details, etc.)

**Analysis Examples** (using jq):
```bash
# View all generated queries
jq -r 'select(.stage == "query_generation") | .final_query' investigation.jsonl

# Count matches by method
jq 'select(.stage == "matching") | .match_details[].match_method' investigation.jsonl | sort | uniq -c

# Find resources that weren't matched
jq 'select(.stage == "matching") | .match_details[] | select(.matched == false) | .resource.url' investigation.jsonl

# View session summary
jq 'select(.stage == "session_end")' investigation.jsonl

# Check query coverage evolution
jq 'select(.stage == "query_generation") | {query_index, cumulative_coverage}' investigation.jsonl

# Analyze why specific resource was found
jq 'select(.stage == "matching") | .match_details[] | select(.resource.pmid == "12345678")' investigation.jsonl
```

**Log Stages**:
- `session_start`: Initial configuration, target resources, backend selection
- `content_fetch`: Resource content retrieval timing and statistics
- `query_generation`: Generated query, contributing resources, coverage progress
- `search_execution`: Query execution timing, result counts, backend response
- `matching`: Resource matching attempts, methods tried, final outcomes
- `session_end`: Final metrics, coverage, stopping reason, total time

**Error Handling**: Investigation logging failures are non-fatal—if logging encounters errors (disk full, permissions, etc.), the search continues and warnings are printed (with `-v` flag). This ensures logging never blocks your actual search work flow.

## Architecture Overview

### Core Components

**Package Structure**: The main package is at `src/interaction_finder/` with key modules:
- `models.py` - Core `Term` data model with name, kind, and attributes
- `settings.py` - TOML-based configuration with Pydantic validation  
- `term_parser.py` - Parser for `"term # &attr=value"` syntax
- `agents.py` - Legacy AI agents (deprecated in favor of extraction graphs)
- `cli.py` - Typer-based CLI with extract, terms, and config subcommands
- `resources.py` - Resource management with comprehensive provenance tracking
- `settings_editor.py` - Interactive configuration editor
- `fetcher.py` - High-level PageFetcher and URLCache for web content

**Extraction Graph Packages**: AI-driven extraction workflows:
- `extraction_graph/` - Original graph-based extraction system
- `extraction_graph_v2/` - Simplified and optimized extraction pipeline
  - `run.py` - Main pipeline orchestration and URL processing
  - `nodes.py` - Individual processing nodes (ExtractEntities, AssessIndividually, etc.)
  - `state.py` - Shared state management across pipeline stages
  - `models.py` - Pydantic models for extraction results and provenance
  - `agents.py` - Pydantic-AI agents for LLM interactions
  - `deps.py` - Dependency injection for external services

### Key Design Patterns

**Graph-Based Processing**: Extraction workflows built as directed graphs with nodes for entity extraction, assessment, and pair formation.

**Resource Pool Management**: Centralized document management with `ResourcePool` for tracking content and provenance.

**Async-First Processing**: Built for async/await to handle concurrent LLM calls and web fetching efficiently.

**File-Based Caching**: URLCache uses base36 hashed filenames with multiple content types per URL (`.html`, `.pdf`, `.md`, `.chunks`, `.doi`, `.redir`).

**Content Standardization**: All inputs converted to Markdown before LLM processing for consistency.

**Flexible Configuration**: TOML files with dotted-key overrides and relative path resolution, plus interactive config editor.

**Pydantic Boundaries**: Type safety and validation at all system boundaries (config, models, results).

**Quote-Level Provenance**: Every extracted entity and pair includes supporting quotes from source documents.

## Configuration System

Configuration loaded from TOML files with Pydantic validation:

```python
# Load config
config = IfetcherConfig.from_path("config.toml")
path = config.abspath("training_data/{term}.jsonl", term="BRCA1")
```

**Configuration locations checked automatically**:
1. `config.toml`
2. `interaction_finder.toml` 
3. `.interaction_finder.toml`

**Configuration modes and overrides**:
```bash
# Use specific mode from config file
uv run interaction-finder extract -t BRCA1 -m development

# Override specific settings
uv run interaction-finder extract -t BRCA1 -O agents.llm=openai:gpt-4o

# Interactive configuration editor
uv run interaction-finder config edit
```

## Testing Strategy

**Test Structure**:
- `tests/test_*.py` - Core module tests (agents, models, settings, term_parser, resources)
- `tests/fetcher/` - Comprehensive PageFetcher and URLCache testing
  - `test_page_fetcher.py`, `test_cache.py` - Core functionality with mock HTTP servers
  - `test_failure_*.py` - Error handling and retry mechanisms
  - `test_regression.py` - Comprehensive regression tests for refactoring
- `tests/extraction_v2/` - Extraction graph pipeline testing
  - `test_run.py`, `test_nodes.py` - Pipeline orchestration and individual nodes
  - `test_integration.py` - End-to-end extraction workflows
  - `test_agents.py`, `test_models.py` - Pydantic-AI agents and data models

**Testing Approach**: Mock HTTP servers for web fetching tests, async test patterns, pytest fixtures for temporary directories, comprehensive integration tests for extraction pipelines.

## Development Patterns

**Import Structure**: Main exports via `__init__.py`:
```python
from interaction_finder import PageFetcher, URLCache, IfetcherConfig, extraction_graph
```

**Error Handling**: Custom error types with rich display methods, complete error context including operation, entities, and remediation suggestions.

**CLI Error Display**: Grouped error reporting with --verbose flag for full tracebacks, fail-fast options for development.

**Cache Management**: Automatic cache status checking, retry mechanisms for failed URLs, concurrent fetching with Rich progress displays.

**Resource Management**: Comprehensive resource tracking with `ResourcePool`, `ResourceId`, and `ResourceQuote` for complete provenance chains.

**Pipeline Orchestration**: Extraction graphs with state management, dependency injection, and incremental result saving.

## Extension Points

- **New Extraction Nodes**: Add nodes to `extraction_graph_v2/nodes.py` following existing patterns
- **Content Sources**: Extend URLCache file extensions or add crawl4ai configurations  
- **CLI Commands**: Add new subcommands to `cli.py` using Typer
- **Chunking Strategies**: Customize chonkie RecursiveChunker parameters in PageFetcher
- **Progress Displays**: Customize Rich progress bars for different use cases
- **Agent Models**: Configure different LLM providers in `extraction_graph_v2/agents.py`
- **Resource Types**: Extend `ResourcePool` for new content types and provenance tracking

## Current Architecture Focus

**Primary Pipeline**: The system now uses `extraction_graph_v2` as the main extraction pipeline, featuring:
- Simplified single-group processing
- Three-phase workflow: entity extraction → assessment → pair formation
- Comprehensive quote-level provenance tracking
- Pydantic-AI integration for structured LLM interactions
- Rich CLI with dry-run previews and incremental progress saves

**Development Priority**: The v2 extraction graph represents the current architectural direction, while v1 extraction graph and legacy agents are maintained for compatibility.

The architecture prioritizes research iteration speed through comprehensive caching, async concurrency, configuration flexibility, and complete provenance tracking for expensive LLM operations.
