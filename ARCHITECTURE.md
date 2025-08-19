# Interaction Finder Architecture

## Purpose

Automated extraction of biological interactions (gene-disease, ligand-receptor, cell-biomarker) from scientific literature using LLMs. The goal is to benchmark different LLM-based approaches for biomedical relation extraction.

## Core Architecture

```
src/interaction_finder/
├── models.py          # Term(name, kind, attributes) - core data model
├── settings.py        # TOML-based configuration with validation
├── fetcher.py         # URLCache, PageFetcher - web content fetching/caching
├── term_parser.py     # Parse "gene # &kind=gene &disease=cancer" format
└── agents.py          # AI agents for extraction (gene-disease only)
```

## Key Components

### Term Model
```python
class Term(BaseModel):
    name: str                    # "BRCA1"
    kind: Optional[str]          # "gene" 
    attributes: dict[str, str]   # {"disease": "cancer"}
```

### Configuration System
- TOML files with Pydantic validation
- Agent configs (LLM models, prompts, retries)
- Tool configs (web scrapers, ontologies)
- Path templates with `config.abspath(path, **kwargs)` resolution

### URLCache & PageFetcher
**URLCache**: Low-level file-based caching system
- Methods: `get_html()`, `set_html()`, `get_pdf()`, `set_pdf()`, `get_markdown()`, `set_markdown()`
- Cache management: `has_url()`, `has_path()`, `clear_url()`, `list_cached_urls()`
- Base36 hashed filenames for URL collision avoidance
- Supports multiple content types per URL with separate file extensions
- Automatic redirect handling with `.redir` sidecar files for final URL tracking

**PageFetcher**: High-level async web content fetcher with intelligent caching
- Single URL methods: `get_html()`, `get_pdf()`, `get_markdown()`, `get_raw()`
- Batch URL support: All methods accept `Union[str, List[str]]` for concurrent fetching
- Automatic PDF vs HTML detection based on URL extension and headers
- Rich progress displays with granular crawl4ai status updates
- DOI extraction from HTML using XPath selectors (citation_doi, publication_doi)
- Reference section removal from markdown content
- Configurable concurrency limits and progress display options

### AI Agents
- Currently: gene-disease extraction only
- Uses pydantic-ai framework
- 50k character limit per paper
- Converts results to Term objects

### Term Parser
- Flexible attribute syntax: `termname # &attr1=value, &attr2=multi word value`
- Handles comments, empty lines, malformed input
- Special `kind` attribute becomes Term.kind

## Current Status

**Working**: Configuration, caching, term parsing, gene-disease extraction, comprehensive tests
**Missing**: Other extraction types, researcher modes, benchmarking framework, CLI

## Dependencies

- **pydantic/pydantic-ai**: Data models and AI agents
- **crawl4ai/httpx**: Web scraping and fetching  
- **tomli**: TOML configuration parsing

## Usage Patterns

**Configuration**:
```python
config = IfetcherConfig.from_path("config.toml")
config.abspath("training_data/{term}.jsonl", term="BRCA1")
```

**Content Fetching**:
```python
fetcher = PageFetcher(config, show_status=True)

# Single URL fetching
markdown = await fetcher.get_markdown("https://paper.url")  # Auto-cached
html = await fetcher.get_html("https://example.com")
pdf_content = await fetcher.get_pdf("https://paper.pdf")

# Batch URL fetching with automatic concurrency
urls = ["https://paper1.url", "https://paper2.url"] 
markdowns = await fetcher.get_markdown(urls)  # Returns list of content

# Standalone batch functions with progress bars
results = await fetch_urls_with_progress(urls, config, "markdown")
results = await fetch_urls_concurrent_with_progress(urls, config, "html", max_concurrent=10)
```

**Term Processing**:
```python
terms = parse_terms_from_lines(["BRCA1 # &kind=gene &disease=cancer"])
```

**AI Extraction**:
```python
result = extract_genes_for_disease(markdown, "breast cancer", config)
terms = create_gene_extraction_terms(result)
```

## Entry Points & CLI

**Console Script**: `interaction-finder` (defined in pyproject.toml)
- Maps to `interaction_finder:main()` function
- Currently basic placeholder implementation

**Module Functions**: Available for direct import
- `fetch_urls_with_progress()` - Sequential batch fetching with progress bars
- `fetch_urls_concurrent_with_progress()` - Concurrent batch fetching with semaphore limiting
- `PageFetcher` class for programmatic use
- `URLCache` class for low-level cache operations

## Key Design Choices

**Async-First Processing**: Built for async/await patterns to handle concurrent LLM calls and web fetching efficiently

**File-Based Caching**: Persistent disk cache with base36 hash-based naming for collision avoidance and cross-session reliability

**Content Standardization**: All inputs (HTML/PDF/etc.) converted to Markdown before LLM processing for consistency

**Intelligent Content Detection**: Automatic PDF vs HTML handling based on URL extensions and response headers

**Rich Progress Integration**: Granular status updates showing crawl4ai stages, domain processing, and content sizes

**Flexible Batch Processing**: Single interface supports both individual and batch URL processing with automatic concurrency

**DOI Metadata Extraction**: Automatic DOI detection and caching from HTML meta tags for academic paper tracking

**Factory Pattern for Agents**: Agent creation via factory functions rather than inheritance for flexible configuration

**Pydantic Validation Boundaries**: Type safety and validation at all system boundaries (config, models, results)

**Dotted-Key Configuration**: Override system using `"agents.llm"` syntax for flexible deployment configurations

**Relative Path Strategy**: All paths relative to config file location for portable deployments

## Extension Points

1. **New Agent Types**: Follow `create_gene_disease_agent()` pattern in `agents.py`
2. **Researcher Modes**: Add modes to `valid_modes` in settings validation 
3. **Content Sources**: Extend URLCache for new content types or add new file extensions
4. **Parsers**: Add new term formats in `term_parser.py`
5. **Tools**: Configure external tools via TOML sections
6. **Fetching Strategies**: Add new crawl4ai configurations or extraction strategies in PageFetcher
7. **Progress Displays**: Customize Rich progress bars for different use cases
8. **CLI Commands**: Extend the main() function or add new console script entry points

The architecture prioritizes research iteration speed with expensive LLM operations through comprehensive caching, async concurrency, and configuration flexibility.
