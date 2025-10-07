# Interaction Finder Architecture

## Purpose

Automated extraction of biological interactions (gene-disease, ligand-receptor, cell-biomarker) from scientific literature using LLMs. The goal is to benchmark different LLM-based approaches for biomedical relation extraction.

## Core Architecture

```
src/interaction_finder/
├── models.py          # Term(name, kind, attributes) - core data model
├── settings.py        # TOML-based configuration with validation
├── term_parser.py     # Parse "gene # &kind=gene &disease=cancer" format
├── agents.py          # AI agents for extraction (gene-disease only)
├── fetcher/           # Modular web content fetching package
│   ├── page_fetcher.py      # High-level async web content fetcher
│   ├── cache.py             # File-based URL caching system
│   ├── web_client.py        # HTTP client with session management
│   ├── content_processor.py # Content conversion and chunking
│   ├── batch_operations.py  # Concurrent URL processing
│   ├── progress_display.py  # Rich progress bars integration
│   ├── document_grouper.py  # Document clustering orchestrator
│   ├── document_embedding.py # Document embedding strategies
│   └── document_clustering.py # Advanced clustering algorithms
└── extraction_graph/ # Entity extraction pipeline
    ├── run.py         # High-level extraction API
    ├── graph.py       # LangGraph-based extraction workflow
    ├── nodes.py       # Individual processing nodes
    ├── state.py       # Extraction state management
    ├── models.py      # Result data models
    └── deps.py        # Dependency injection
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
- Methods: `get_html()`, `set_html()`, `get_pdf()`, `set_pdf()`, `get_markdown()`, `set_markdown()`, `get_chunks()`, `set_chunks()`
- Cache management: `has_url()`, `has_path()`, `clear_url()`, `list_cached_urls()`
- Base36 hashed filenames for URL collision avoidance
- Supports multiple content types per URL with separate file extensions (`.html`, `.pdf`, `.md`, `.json`, `.doi`, `.redir`)
- Automatic redirect handling with `.redir` sidecar files for final URL tracking

**PageFetcher**: High-level async web content fetcher with intelligent caching
- Single URL methods: `get_html()`, `get_pdf()`, `get_markdown()`, `get_chunks()`, `get_raw()`
- Batch URL support: All methods accept `Union[str, List[str]]` for concurrent fetching
- Automatic PDF vs HTML detection based on URL extension and headers
- Rich progress displays with granular crawl4ai status updates
- DOI extraction from HTML using XPath selectors (citation_doi, publication_doi)
- Reference section removal from markdown content
- **Text chunking**: Automatic chunking of markdown content using chonkie RecursiveChunker
- **Document grouping**: Advanced clustering capabilities with multiple algorithms and embedding strategies
- Configurable concurrency limits and progress display options

### AI Agents
- Currently: gene-disease extraction only
- Uses pydantic-ai framework
- 50k character limit per paper
- Converts results to Term objects

### Document Clustering System
**DocumentEmbedder**: Strategies for converting chunk embeddings into document-level representations
- `SimpleAverageEmbedder`: Uniform averaging of chunk embeddings
- `IDFEmbedder`: Corpus-aware IDF-like weighting with 5-step pipeline:
  1. Compute chunk specificity using cross-document similarities
  2. Convert to weights with monotone sharpening (configurable power)
  3. Weighted average with small unweighted blend for stability
  4. Remove PC1 component to eliminate corpus-common background
  5. Ensure unit norm for cosine similarity compatibility

**DocumentClusterer**: Multiple clustering algorithms with size constraints
- `AgglomerativeClusterer`: Standard linkage-based clustering with comprehensive metrics
- `SizeAnnealedAgglomerativeClusterer`: Parameter-free clustering with 4-gate acceptance system:
  - Size cap enforcement for balanced groups
  - Objective safety check against baseline similarity floor (1.0×S̄)
  - Cross-pair threshold validation for merge quality
  - Incremental heap updates and similarity caching for performance
- `SpectralClusterer`: Eigenvalue-based clustering with size constraint enforcement
- `HybridClusterer`: Spectral seeding + agglomerative refinement for quality optimization
- `RandomClusterer`: Random baseline for comparison

**Comprehensive Metrics**: 5 high-leverage clustering evaluation metrics
- `pair_weighted_cohesion`: Within-cluster similarity weighted by cluster pairs
- `contrast`: Within-cluster mean minus corpus average similarity
- `robust_cohesion`: 10th percentile within-cluster similarities (outlier-resistant)
- `leakage`: Maximum between-cluster similarity with top-3 pairs (quality check)
- `silhouette`: Standard silhouette coefficient adapted for cosine distance

**Performance Optimizations**: Vectorized operations using NumPy
- Advanced indexing with `np.ix_` for submatrix operations
- Broadcasting for batch distance computations
- Incremental similarity matrix updates during clustering
- 4.48× performance improvement over naive implementations

### Term Parser
- Flexible attribute syntax: `termname # &attr1=value, &attr2=multi word value`
- Handles comments, empty lines, malformed input
- Special `kind` attribute becomes Term.kind

## Current Status

**Working**:
- Configuration system with TOML validation
- Comprehensive web content fetching and caching
- Advanced document clustering with 5 algorithms and 2 embedding strategies
- Comprehensive clustering metrics (cohesion, contrast, robustness, leakage, silhouette)
- Rich progress displays with completion summaries
- Entity extraction pipeline with document grouping
- Gene-disease extraction, term parsing
- Extensive test coverage

**Missing**: Other extraction types (ligand-receptor, cell-biomarker), researcher modes, benchmarking framework

## Dependencies

- **pydantic/pydantic-ai**: Data models and AI agents
- **crawl4ai/httpx**: Web scraping and fetching
- **chonkie**: Text chunking with RecursiveChunker
- **numpy**: Vectorized operations for clustering algorithms
- **scikit-learn**: Spectral clustering and silhouette metrics
- **rich**: Progress bars and console formatting
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

# Text chunking - returns simple list of strings
chunks = await fetcher.get_chunks("https://paper.url")  # ["chunk1", "chunk2", ...]

# Batch URL fetching with automatic concurrency
urls = ["https://paper1.url", "https://paper2.url"]
markdowns = await fetcher.get_markdown(urls)  # Returns list of content
all_chunks = await fetcher.get_chunks(urls)    # Returns list of lists [["chunk1", "chunk2"], ["chunk3"]]

# Standalone batch functions with progress bars (return content or exceptions)
results = await fetch_urls_with_progress(urls, config, "markdown")
results = await fetch_urls_concurrent_with_progress(urls, config, "chunks", max_concurrent=10)

# Document grouping with clustering
groups = await fetcher.get_groups(
    urls,
    constraint_type="count",
    min_size=2, max_size=8,
    clustering_method="agglomerative",
    embedding_weights="idf",
    linkage_method="average"
)
```

**Term Processing**:
```python
terms = parse_terms_from_lines(["BRCA1 # &kind=gene &disease=cancer"])
```

**AI Extraction**:
```python
result = extract_genes_for_disease(markdown, "breast cancer", config)
terms = create_gene_extraction_terms(result)

# Entity extraction pipeline with document grouping
from interaction_finder.extraction_graph.run import extract_from_urls
result = await extract_from_urls(urls, config)
print(f"Extracted {result.total_pairs} entity pairs from {result.successful_groups} groups")
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

**Automatic Text Chunking**: Built-in chunking capabilities using chonkie RecursiveChunker for preparing content for LLM processing

**Factory Pattern for Agents**: Agent creation via factory functions rather than inheritance for flexible configuration

**Modular Clustering Architecture**: Separate abstractions for embeddings and clustering with pluggable algorithms

**Mathematically Rigorous Metrics**: 5 complementary metrics providing comprehensive clustering quality assessment

**Performance-First Clustering**: Vectorized NumPy operations and incremental updates for large-scale document processing

**Parameter-Free Clustering**: Size-annealed agglomerative method eliminates manual parameter tuning

**Pydantic Validation Boundaries**: Type safety and validation at all system boundaries (config, models, results)

**Dotted-Key Configuration**: Override system using `"agents.llm"` syntax for flexible deployment configurations

**Relative Path Strategy**: All paths relative to config file location for portable deployments

## Extension Points

1. **New Agent Types**: Follow `create_gene_disease_agent()` pattern in `agents.py`
2. **Clustering Algorithms**: Extend `DocumentClusterer` ABC or add new algorithms to `document_clustering.py`
3. **Embedding Strategies**: Implement `DocumentEmbedder` interface for new embedding approaches
4. **Clustering Metrics**: Add new evaluation metrics to `compute_comprehensive_metrics()`
5. **Content Sources**: Extend URLCache for new content types or add new file extensions
6. **Parsers**: Add new term formats in `term_parser.py`
7. **Extraction Nodes**: Add new processing nodes to the extraction graph pipeline
8. **Chunking Strategies**: Customize chunking parameters or add new chunkers beyond RecursiveChunker
9. **Progress Displays**: Customize Rich progress bars for different use cases
10. **CLI Commands**: Add new Typer commands to the CLI interface

The architecture prioritizes research iteration speed with expensive LLM operations through comprehensive caching, async concurrency, and configuration flexibility.
