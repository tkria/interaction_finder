# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**interaction-finder** is a Python tool for automated discovery of biological associations from scientific literature using AI-driven pipelines. The system implements a multi-stage workflow: keyword extraction → wide search → entity extraction, with comprehensive provenance tracking, semantic clustering, and flexible search backends.

## Development Commands

**Virtual Environment**: This project uses `uv` for dependency management:
```bash
# Run CLI with dependencies
uv run interaction-finder --help

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/extraction/test_graph.py

# Run tests with verbose output and exclude slow tests
uv run pytest -v -m "not slow"

# Run tests excluding integration tests
uv run pytest -m "not integration"

# Type checking (if configured)
uv run pyright

# Install dependencies after changes
uv sync
```

**Primary Workflow**: Three-stage pipeline with automatic progression:

```bash
# Full pipeline from topic string (no intermediate files needed)
uv run interaction-finder extract "pulmonary arterial hypertension" -e gene -e disease -o results.json

# Traditional staged workflow with in-place checkpoint updates
# Stage 1: Extract bridging terms (keywords) from review articles
uv run interaction-finder keywords "pulmonary arterial hypertension" -o research.json

# Stage 2: Search (updates research.json in place with search results)
uv run interaction-finder search research.json

# Stage 3: Extract entity associations (updates research.json in place with extraction results)
uv run interaction-finder extract research.json -e gene -e disease

# Alternative: Start from any stage with topic strings
uv run interaction-finder search "pulmonary arterial hypertension" -o searches.json
uv run interaction-finder extract "PAH genetics" -e gene -e disease -o results.json

# Alternative: Create separate output files instead of updating in place
uv run interaction-finder search keywords.json -o searches.json
uv run interaction-finder extract searches.json -e gene -e disease -o results.json
```

**Checkpoint File Behavior**:
- When input is a checkpoint file and no `-o` specified: **updates the input file in place**
- When input is a checkpoint file and `-o` specified: **writes to the output file**, leaves input unchanged
- When input is a topic string: **requires `-o` to save results**

**Entity Kind Pair Filtering**:
The extraction stage filters pairs based on entity kinds. By default, specifying two different kinds (e.g., `-e gene -e disease`) only permits cross-kind pairs (gene-disease). To allow same-kind pairs (gene-gene, disease-disease), repeat the kind flag:

```bash
# Only gene-disease pairs allowed (no gene-gene or disease-disease)
uv run interaction-finder extract "topic" -e gene -e disease -o results.json

# Allow gene-gene AND gene-disease pairs (no disease-disease)
uv run interaction-finder extract "topic" -e gene -e gene -e disease -o results.json

# Allow all combinations: gene-gene, gene-disease, disease-disease
uv run interaction-finder extract "topic" -e gene -e gene -e disease -e disease -o results.json

# Single kind always allows self-pairs
uv run interaction-finder extract "topic" -e gene -o results.json
```

**Agent Configuration**: Configure LLM models for different pipeline stages:

```bash
# Override agent models via CLI
uv run interaction-finder extract "topic" -e gene \
  -O agents.extraction.judge.llm=openai:gpt-4o

# Use configuration modes
uv run interaction-finder search "topic" -m development
```

**Additional Commands**:
```bash
# Fetch and cache web content
uv run interaction-finder fetch https://example.com
uv run interaction-finder fetch --input urls.txt --chunk

# Configuration management
uv run interaction-finder config info
uv run interaction-finder config validate

# Override configuration values and specify backend
uv run interaction-finder extract "topic" -e gene -b perplexica -O agents.llm=openai:gpt-4o
```

**Search Backend Options**:
The system supports multiple search backends configured via `--backend` flag:
- `pubmed` - NCBI PubMed search (biomedical literature)
- `perplexica` - Local Perplexica instance (web search)
- `openai` - OpenAI web search API

## Architecture Overview

### Core Package Structure

```
src/interaction_finder/
├── models.py           # Core Term data model
├── settings.py         # TOML-based configuration with Pydantic validation
├── term_parser.py      # Parser for "term # &attr=value" syntax
├── resources.py        # Resource management with comprehensive provenance tracking
├── cli.py              # Typer-based CLI interface
├── cli_fetch.py        # Fetch command implementation
├── fetcher/            # Web content fetching and caching
│   ├── page_fetcher.py        # High-level async web content fetcher
│   ├── cache.py               # File-based URL caching system
│   ├── web_client.py          # HTTP client with session management
│   ├── content_processor.py   # Content conversion and chunking
│   └── batch_operations.py    # Concurrent URL processing
├── keywords/           # Bridging terms extraction pipeline
│   ├── extractors/            # Keyword extraction algorithms (YAKE, RAKE, TF-IDF)
│   ├── graph.py               # Pydantic graph workflow
│   ├── nodes.py               # Individual processing nodes
│   ├── agents.py              # LLM agents for keyword assessment
│   ├── reranker.py            # Semantic reranking
│   └── run.py                 # Main entrypoint
├── widesearch/         # Query expansion and literature discovery
│   ├── graph.py               # Pydantic graph workflow
│   ├── nodes.py               # Query generation and reflection nodes
│   ├── reranker.py            # Semantic result reranking
│   ├── progress.py            # Live progress display
│   └── run.py                 # Main entrypoint with checkpoint support
├── extraction/         # Entity-entity association extraction
│   ├── core/                  # Core extraction logic (entities, pairs, quotes)
│   ├── graph.py               # Pydantic graph workflow
│   ├── nodes.py               # Extraction, assessment, and judgment nodes
│   ├── agents.py              # LLM agents for extraction
│   └── run.py                 # Main entrypoint
└── search/             # Search backends abstraction
    ├── models.py              # SearchBackend, SearchQuery, SearchResult
    └── backends/              # PubMed, Perplexica, OpenAI implementations
```

### Three-Stage Pipeline

**Stage 1: Keyword Extraction** (`keywords/`)
- Searches for review articles on a topic
- Extracts candidate keywords using multiple algorithms (YAKE, RAKE, TF-IDF)
- Uses LLM to assess and filter keywords for relevance
- Outputs: List of "bridging terms" + ResourcePool with fetched content

**Stage 2: Wide Search** (`widesearch/`)
- Takes bridging terms and generates diverse search queries
- Executes searches across selected backend (PubMed/Perplexica/OpenAI)
- Uses LLM reflection to determine coverage and stop condition
- Optional semantic reranking for result quality
- Outputs: WidesearchCheckpoint (queries, results, resource pool)

**Stage 3: Entity Extraction** (`extraction/`)
- Fetches full text for search results
- Extracts entities of specified types from documents
- Identifies entity pairs with relationship evidence
- Assesses each pair with multi-stage LLM evaluation
- Validates quotes and provenance chains
- Outputs: ExtractionResult (accepted/rejected pairs with full provenance)

### Key Design Patterns

**Pydantic AI Graphs**: All three stages implemented as Pydantic AI graphs with nodes for LLM calls, data processing, and control flow.

**Resource Pool Pattern**: Centralized document management with `ResourcePool` tracking all fetched content, metadata, and provenance across pipeline stages.

**Checkpoint-Based Workflow**: Each stage outputs a complete checkpoint (JSON) enabling resumption, inspection, and stage-by-stage processing.

**File-Based Caching**: URLCache uses base36 hashed filenames with multiple content types per URL (`.html`, `.pdf`, `.md`, `.chunks`, `.doi`, `.redir`).

**Content Standardization**: All inputs converted to Markdown before LLM processing for consistency. PDFs handled via crawl4ai.

**Async-First Processing**: Built for async/await to handle concurrent LLM calls and web fetching efficiently.

**Flexible Configuration**: TOML files with dotted-key overrides (`-O agents.llm=...`) and mode support (`-m development`).

**Quote-Level Provenance**: Every extracted entity and pair includes supporting quotes from source documents with validation.

## Configuration System

Configuration loaded from TOML files with Pydantic validation:

```python
from interaction_finder import IfetcherConfig

# Load config from standard locations or specified path
config = IfetcherConfig.from_path("config.toml")
config = IfetcherConfig.from_path("config.toml", mode="development")

# Apply overrides programmatically
overrides = {"agents.llm": "openai:gpt-4o"}
config = IfetcherConfig.from_path("config.toml", overrides=overrides)

# Resolve paths relative to config file
path = config.abspath("output/{name}.json", name="results")
```

**Configuration locations checked automatically** (in order):
1. `config.toml`
2. `interaction_finder.toml`
3. `.interaction_finder.toml`

**Key configuration sections**:
- `stage.keywords` - Keywords stage parameters (max rounds, search backend)
- `stage.search` - Search stage parameters (max rounds, reranking, backends)
- `stage.extraction` - Extraction stage parameters (concurrency, clustering)
- `tools.search` - Search backend configuration (PubMed, Perplexica, OpenAI)
- `tools.keywords` - Keyword extraction algorithm configs (RAKE, YAKE, TF-IDF, KeyBERT)
- `agents.*` - **Multi-tier LLM agent configuration** (see below)
- `output.*` - Cache paths and output locations

### Multi-Tier Agent Configuration

Agents use hierarchical configuration with fallback: `agents.module.agent` → `agents.module._` → `agents._` → code default

```toml
# config.toml example
[agents._]                      # Global default
llm = "openai:gpt-4o-mini"

[agents.extraction.judge]      # Individual override
llm = "openai:gpt-4o"
```

**Agent names**: keywords (`query_expander`, `result_selector`, `keyword_evaluator`, `document_summarizer`, `reflector`), search (`goal_planner`, `query_generator`, `result_selector`, `reflector`), extraction (`entity`, `entity_merger`, `proximal_pair`, `pair_judge`, `cross_judge`)

**For developers** - define agents with `agent_getter()`:
```python
from interaction_finder.agent_config import agent_getter

get_entity_agent = agent_getter(
    "extraction", "entity", EntityExtractionOut, Deps,
    """You are an expert...""",
)
# Usage: agent = get_entity_agent(ctx.deps.config)
```

## Testing Strategy

**Test Structure**:
- `tests/test_*.py` - Core module tests (models, settings, term_parser, resources)
- `tests/fetcher/` - Comprehensive PageFetcher and URLCache testing
  - `test_page_fetcher.py`, `test_cache.py` - Core functionality with mock HTTP servers
  - `test_failure_*.py` - Error handling and retry mechanisms
  - `test_regression.py` - Comprehensive regression tests for refactoring
- `tests/keywords/` - Keyword extraction pipeline testing
  - `test_graph.py`, `test_nodes.py` - Pipeline orchestration and nodes
  - `test_agents.py` - LLM agents for keyword assessment
  - `extractors/` - Tests for YAKE, RAKE, TF-IDF extractors
- `tests/widesearch/` - Wide search pipeline testing
  - `test_run.py`, `test_integration.py` - End-to-end workflows
  - `test_reranker.py` - Semantic reranking
  - `test_progress.py` - Progress display components
- `tests/extraction/` - Entity extraction pipeline testing
  - `test_graph.py`, `test_nodes.py` - Pipeline orchestration
  - `test_integration.py` - End-to-end extraction workflows
  - `test_agents.py`, `test_models.py` - LLM agents and data models
- `tests/search/backends/` - Search backend implementations

**Testing Approach**: Mock HTTP servers for web fetching, async test patterns with pytest-asyncio, fixtures for temporary directories, comprehensive integration tests with dirty-equals for flexible assertions.

By default, only tests in files marked as changed by git are run. Other tests are skipped. To override this behaviour set `TEST_ALL=1`.

**Test Markers**:
```bash
# Skip slow tests (large LLM calls, network operations)
pytest -m "not slow"

# Skip integration tests (end-to-end workflows)
pytest -m "not integration"
```

## Development Patterns

**Import Structure**: Main exports via `__init__.py` for clean API:
```python
from interaction_finder import PageFetcher, URLCache, IfetcherConfig
from interaction_finder.keywords import run_keyword_research
from interaction_finder.widesearch import run_widesearch_with_checkpoint
from interaction_finder.extraction import run_extraction
```

**Error Handling**: Custom error types with rich display methods, complete error context including operation, entities, and remediation suggestions.

**CLI Error Display**: Grouped error reporting with `--verbose` flag for full tracebacks.

**Cache Management**: Automatic cache status checking, retry mechanisms for failed URLs, concurrent fetching with Rich progress displays.

**Resource Pool Management**: All pipeline stages share a `ResourcePool` that accumulates content, tracks provenance, and enables quote validation.

**Graph-Based Pipelines**: Each major workflow (keywords, widesearch, extraction) implemented as a Pydantic AI graph with explicit state management and dependency injection.

**Progress Displays**: Live progress counters using Rich for all long-running operations with granular status updates.

## Pipeline Data Flow

**Keyword Extraction → Wide Search → Entity Extraction**:

1. **Keyword Extraction Output** (`BridgingTermsOut`):
   - `terms: List[str]` - Bridging terms found
   - `scores: List[float]` - Relevance scores
   - `resources: ResourcePool` - Review articles processed

2. **Wide Search Output** (`WidesearchCheckpoint`):
   - `topic: str` - Research topic
   - `queries: List[str]` - All queries executed
   - `results: List[SearchResult]` - Unique search results
   - `resources: ResourcePool` - Accumulated from keywords + new results
   - `rounds_completed: int` - Number of search rounds

3. **Entity Extraction Output** (`ExtractionResult`):
   - `accepted_pairs: List[PairWithProvenance]` - Validated associations
   - `rejected_pairs: List[PairWithProvenance]` - Rejected associations
   - `metadata: ExtractionMetadata` - Summary statistics

**Checkpoint Files**: All intermediate outputs are JSON files enabling:
- Resumption of failed runs
- Stage-by-stage processing
- Manual inspection and debugging
- External analysis (jq, pandas, etc.)

## Extension Points

**New Search Backends**: Implement `SearchBackend` abstract class in `search/backends/`:
```python
class MyBackend(SearchBackend):
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        # Implementation
```

**New Keyword Extractors**: Implement `KeywordExtractor` interface in `keywords/extractors/`:
```python
class MyExtractor(KeywordExtractor):
    def extract_keywords(self, text: str, n: int = 30) -> List[ScoredKeyword]:
        # Implementation
```

**Custom Graph Nodes**: Add new nodes to pipeline graphs following existing patterns in `*/nodes.py`.

**Custom LLM Agents**: Define new Pydantic AI agents in `*/agents.py` for specialized tasks.

**Content Processors**: Extend `ContentProcessor` in `fetcher/content_processor.py` for new content types.

**CLI Commands**: Add new Typer commands to `cli.py` following existing command patterns.

**Configuration Extensions**: Add new sections to TOML config and update `settings.py` Pydantic models.

## Key Dependencies

- **pydantic / pydantic-ai** - Data validation, LLM agents, graph workflows
- **crawl4ai** - Advanced web scraping with PDF support
- **chonkie** - Semantic text chunking
- **httpx** - Async HTTP client
- **typer / rich** - CLI framework and terminal formatting
- **sentence-transformers** - Semantic similarity and reranking
- **yake / rake-nltk / scikit-learn** - Keyword extraction algorithms
- **tomli / tomli-w** - TOML configuration parsing

## Current Development Focus

The system is in active development with all three pipeline stages functional:

**Working**:
- Complete three-stage pipeline (keywords → widesearch → extraction)
- Multiple search backends (PubMed, Perplexica, OpenAI)
- Comprehensive web content fetching and caching
- Semantic reranking and clustering
- Quote-level provenance tracking
- Checkpoint-based workflow with resumption
- Rich CLI with progress displays
- Extensive test coverage

**Architecture Direction**: The system prioritizes research iteration speed through comprehensive caching, checkpoint-based workflows, async concurrency, and complete provenance tracking for expensive LLM operations. Each pipeline stage can be run independently or as part of the full workflow.

<!-- Added by the TI research software framework adoption. Review and edit. -->

## Project purpose

> Automated discovery of biological associations from scientific literature using AI-driven pipelines

Owner: Timothy Chapman. Stage: exploratory. Describe intended users and non-goals here.

## Scientific constraints

- Do not invent assumptions, labels, thresholds, metrics, dataset splits, or expected results.
- Ask for clarification or mark unresolved scientific decisions explicitly.
- Do not silently change filters, exclusions, splits, prompts, models, reference versions, evaluation procedures, or interpretation rules.
- Preserve links between results, code versions, configuration, and data provenance.
- Treat passing tests as necessary evidence, not proof that the scientific design is correct.

## Development workflow

Before implementing a change: state the intended behaviour, identify the scientific assumptions and invariants involved, define acceptance criteria and at least one failure case, and make the smallest reasonable change.

After implementing a change: inspect the complete diff, run the relevant tests or checks, inspect representative outputs, update documentation and provenance when behaviour changes, and commit a meaningful working checkpoint.

## Privacy and security

- Never commit credentials, tokens, private keys, identifying information, or restricted raw data.
- Use `.env.example` for variable names and placeholders only.
- Do not send restricted data to unapproved models, services, tools, or providers.
- Request human approval before expensive, destructive, modifying, or consequential operations.

## Commands

- Install: `uv sync`
- Framework check: `python scripts/framework_check.py`
- Test: `uv run pytest` (set `TEST_ALL=1` to run the full suite)
- Run main example: `uv run interaction-finder extract "pulmonary arterial hypertension" -e gene -e disease -o results.json`

## Definition of done

A scoped task is complete only when the requested behaviour is implemented, the relevant tests or validation checks pass, representative outputs have been inspected, documentation and provenance are updated where needed, no secrets or restricted data were introduced, and the complete AI-generated diff has been reviewed by a researcher.
