"""Web client for fetching HTML and PDF content."""

from typing import Dict, List
from rich.console import Console

# Web client constants
STEALTH_RETRY_THRESHOLD = 3000  # characters of raw markdown
CHUNK_SIZE_TOKENS = 4096
CHUNK_MIN_SENTENCES = 2
CHUNK_SKIP_WINDOW = 1
CHUNK_SIMILARITY_THRESHOLD = 0.5

# PubMed full-text link following configuration
PUBMED_MAX_CONCURRENT_LINKS = 3
PUBMED_CONTENT_IMPROVEMENT_THRESHOLD = 0.5  # 50% improvement required

# PubMed metadata headings to exclude from superset comparison
# These are PubMed-specific UI elements, not part of the actual article
PUBMED_METADATA_HEADINGS = {
    "figures",
    "conflict of interest statement",
    "references",
    "similar articles",
    "cited by",
    "publication types",
    "mesh terms",
    "substances",
    "grant support",
    "supplementary material",
    "related information",
}

# DOI extraction schema for XPath-based extraction from academic publishers
DOI_EXTRACTION_SCHEMA = {
    "name": "DOI extractor (XPath)",
    "baseSelector": "/html",
    "fields": [
        {
            "name": "doi_meta_cite",
            "selector": "//meta[@name='citation_doi']",
            "type": "attribute",
            "attribute": "content",
        },
        {
            "name": "doi_meta_pub",
            "selector": "//meta[@name='publication_doi']",
            "type": "attribute",
            "attribute": "content",
        },
        {
            "name": "doi_dc",
            "selector": "//meta[translate(@name,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='dc.identifier' and contains(@content, 'doi.org/')]",
            "type": "attribute",
            "attribute": "content",
        },
        {
            "name": "doi_dc_doi",
            "selector": "//meta[translate(@name,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='dc.identifier.doi']",
            "type": "attribute",
            "attribute": "content",
        },
        {
            "name": "doi_canonical",
            "selector": "//link[@rel='canonical' and contains(@href, 'doi.org/')]",
            "type": "attribute",
            "attribute": "href",
        },
    ],
}

# PubMed full-text link extraction schema
PUBMED_FULLTEXT_LINKS_SCHEMA = {
    "name": "PubMed full-text links extractor",
    "baseSelector": "/html",
    "fields": [
        {
            "name": "fulltext_links",
            "selector": "//div[contains(@class, 'full-text-links-list')]//a",
            "type": "attribute",
            "attribute": "href",
        }
    ],
}

# Full text expansion JavaScript for dynamic content loading
CLICK_AND_MONITOR_JS = r"""
(() => {
  if (!window.__c4ai_mon) {
    const mon = window.__c4ai_mon = {
      inflight: 0,
      last: Date.now(),
      started: false,
      mutations: 0,
      baseline: (document.body.innerText || '').length,
    };
    const bump = () => { mon.last = Date.now(); };

    const origFetch = window.fetch;
    if (origFetch) {
      window.fetch = (...args) => {
        mon.inflight++; bump();
        return origFetch(...args)
          .finally(() => { mon.inflight--; bump(); });
      };
    }

    const XS = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send = function(...args) {
      mon.inflight++; bump();
      this.addEventListener('loadend', () => { mon.inflight--; bump(); }, { once: true });
      return XS.apply(this, args);
    };

    const mo = new MutationObserver(muts => {
      mon.mutations += muts.length; bump();
      for (const m of muts) {
        m.addedNodes && m.addedNodes.forEach(n => {
          const t = n.tagName && n.tagName.toLowerCase();
          if (t === 'img' || t === 'iframe' || t === 'video' || t === 'audio') {
            n.addEventListener('load', bump, true);
            n.addEventListener('error', bump, true);
          }
        });
      }
    });
    mo.observe(document, { subtree: true, childList: true, characterData: true, attributes: true });

    window.addEventListener('load', bump, true);
    window.addEventListener('error', bump, true);
  }

  const mon = window.__c4ai_mon;
  const LABEL = /full\s*text/i;

  const candidates = [
    ...document.querySelectorAll('h1,h2,h3,h4,h5,h6')
  ].
    filter(h => LABEL.test((h.innerText || '').trim())).
    flatMap(h => [h, ...h.querySelectorAll('button,[role="button"],a')]).
    concat(
      [...document.querySelectorAll('button,[role="button"],a,[aria-label]')]
        .filter(el => LABEL.test((el.innerText || el.getAttribute('aria-label') || '').trim()))
    );

  const seen = new Set(); const uniq = [];
  for (const el of candidates) { if (el && !seen.has(el)) { seen.add(el); uniq.push(el); } }

  const isClosed = (el) => el.getAttribute && el.hasAttribute('aria-expanded')
    ? el.getAttribute('aria-expanded') !== 'true'
    : true;

  const click = (el) => {
    el.scrollIntoView({ block: 'center' });
    el.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    el.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    el.dispatchEvent(new PointerEvent('pointerup', { bubbles: true }));
    el.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
    el.dispatchEvent(new KeyboardEvent('keyup',   { key: 'Enter', code: 'Enter', bubbles: true }));
  };

  let didAny = false;
  for (const el of uniq) {
    const target = (/^h[1-6]$/i.test(el.tagName) ? (el.querySelector('button,[role="button"],a') || el) : el);
    if (target && isClosed(target) && getComputedStyle(target).display !== 'none') {
      click(target); didAny = true;
    }
  }
  if (didAny) mon.started = true;
})();
"""

WAIT_FOR_READY_JS = (
    "js:() => {"
    "  const m = window.__c4ai_mon; if (!m) return true;"
    "  const idle = m.inflight === 0 && (Date.now() - m.last) > 1000;"
    "  if (!m.started) return idle;"
    "  const grew = (document.body.innerText || '').length > m.baseline + 50;"
    "  return idle && (m.mutations > 0 || grew);"
    "}"
)

# Lazy import cache variables
_crawl4ai_html_imports = None
_crawl4ai_pdf_imports = None
_quiet_logger = None
_chunker = None


def _get_crawl4ai_imports():
    """Lazy import crawl4ai modules with simple caching."""
    global _crawl4ai_html_imports
    if _crawl4ai_html_imports is None:
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
            from crawl4ai.extraction_strategy import JsonXPathExtractionStrategy
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
            from crawl4ai.async_logger import AsyncLoggerBase

            _crawl4ai_html_imports = (
                AsyncWebCrawler,
                CrawlerRunConfig,
                JsonXPathExtractionStrategy,
                AsyncLoggerBase,
                DefaultMarkdownGenerator,
            )
        except ImportError:
            raise RuntimeError(
                "crawl4ai package is required for fetching. Install with: pip install crawl4ai"
            )
    return _crawl4ai_html_imports


def _get_crawl4ai_pdf_imports():
    """Lazy import crawl4ai PDF modules with simple caching."""
    global _crawl4ai_pdf_imports
    if _crawl4ai_pdf_imports is None:
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
            from crawl4ai.processors.pdf import (
                PDFCrawlerStrategy,
                PDFContentScrapingStrategy,
            )
            from crawl4ai.async_logger import AsyncLoggerBase

            _crawl4ai_pdf_imports = (
                AsyncWebCrawler,
                CrawlerRunConfig,
                PDFCrawlerStrategy,
                PDFContentScrapingStrategy,
                AsyncLoggerBase,
            )
        except ImportError:
            raise RuntimeError(
                "crawl4ai package is required for fetching. Install with: pip install crawl4ai"
            )
    return _crawl4ai_pdf_imports


def _get_browser_config(*, enable_stealth: bool = False, headless: bool = True):
    """Create and return a BrowserConfig instance with desired options."""
    try:
        from crawl4ai import BrowserConfig
    except ImportError:
        raise RuntimeError(
            "crawl4ai package is required for fetching. Install with: pip install crawl4ai"
        )
    return BrowserConfig(enable_stealth=enable_stealth, headless=headless)


def _get_chunker():
    """Lazy import and initialize chonkie chunker with simple caching."""
    global _chunker
    if _chunker is None:
        from chonkie import SemanticChunker

        _chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",  # Default model
            threshold=CHUNK_SIMILARITY_THRESHOLD,  # Similarity threshold (0-1)
            chunk_size=CHUNK_SIZE_TOKENS,  # Maximum tokens per chunk
            min_sentences=CHUNK_MIN_SENTENCES,  # Initial sentences per chunk
            similarity_window=CHUNK_SKIP_WINDOW,  # Number of sentences to consider for similarity (renamed from skip_window)
        )
    return _chunker


def _get_quiet_logger():
    """Get a quiet logger for crawl4ai operations."""
    global _quiet_logger
    if _quiet_logger is None:
        try:
            from crawl4ai.async_logger import AsyncLoggerBase

            class QuietLogger(AsyncLoggerBase):
                async def alog(self, level, message, tag="", **kwargs):
                    pass  # Suppress all logging

                # Implement all required abstract methods
                def debug(self, message, tag="", **kwargs):
                    pass

                def info(self, message, tag="", **kwargs):
                    pass

                def success(self, message, tag="", **kwargs):
                    pass

                def warning(self, message, tag="", **kwargs):
                    pass

                def error(self, message, tag="", **kwargs):
                    pass

                def error_status(self, url, error, tag="ERROR", url_length=100):
                    pass

                def url_status(
                    self, url, success=True, timing=0, tag="FETCH", url_length=100
                ):
                    pass

            _quiet_logger = QuietLogger()
        except ImportError:
            _quiet_logger = None
    return _quiet_logger


def _get_granular_logger(status_display=None):
    """Get a granular logger that updates status display."""
    try:
        from crawl4ai.async_logger import AsyncLoggerBase

        class GranularLogger(AsyncLoggerBase):
            def __init__(self, status_display=None):
                self.status_display = status_display

            async def alog(self, level, message, tag="", **kwargs):
                if self.status_display and level in ("INFO", "SUCCESS", "WARNING"):
                    if "Crawling" in message:
                        self.status_display.update(f"[yellow]{message}[/yellow]")
                    elif "Processing" in message:
                        self.status_display.update(f"[blue]{message}[/blue]")
                    elif "Complete" in message or "Success" in message:
                        self.status_display.update(f"[green]{message}[/green]")

            def error_status(
                self, url: str, error: str, tag: str = "ERROR", url_length: int = 100
            ):
                pass

            async def astatus_update(
                self, url: str, tag: str = "FETCH", url_length: int = 100
            ) -> None:
                if self.status_display:
                    domain = (
                        url.split("//")[1].split("/")[0] if "//" in url else url[:20]
                    )
                    self.status_display.update(f"[cyan]Fetching[/cyan] {domain}")

            # Implement all required abstract methods
            def debug(self, message, tag="", **kwargs):
                pass

            def info(self, message, tag="", **kwargs):
                pass

            def success(self, message, tag="", **kwargs):
                pass

            def warning(self, message, tag="", **kwargs):
                pass

            def error(self, message, tag="", **kwargs):
                pass

            def url_status(
                self, url, success=True, timing=0, tag="FETCH", url_length=100
            ):
                pass

        return GranularLogger(status_display)
    except ImportError:
        return None


class WebClient:
    """Low-level web client for fetching HTML and PDF content."""

    def __init__(self, timeout: int = 30, verbose: bool = False):
        """
        Initialize web client.

        Parameters:
            timeout: Request timeout in seconds (default: 30)
            verbose: Enable verbose logging (default: False)
        """
        self.timeout = timeout
        self.verbose = verbose
        self._debug_console = Console(stderr=True) if verbose else None
        self._crawler_configs = self._build_crawler_configs()

    def _build_crawler_configs(self) -> Dict[str, any]:
        """Pre-build crawler configurations to avoid repeated construction."""
        timeout_ms = self.timeout * 1000

        return {
            "simple_html": self._build_html_crawler_config(timeout_ms, with_js=False),
            "stealth_html": self._build_html_crawler_config(timeout_ms, with_js=True),
            "simple_browser": _get_browser_config(),
            "stealth_browser": _get_browser_config(enable_stealth=True),
            "pdf_config": self._build_pdf_crawler_config(timeout_ms),
        }

    def _build_html_crawler_config(self, timeout_ms: int, *, with_js: bool) -> any:
        """Factory for CrawlerRunConfig for HTML pages, sharing common options."""
        (
            AsyncWebCrawler,
            CrawlerRunConfig,
            JsonXPathExtractionStrategy,
            AsyncLoggerBase,
            DefaultMarkdownGenerator,
        ) = _get_crawl4ai_imports()

        md_gen = DefaultMarkdownGenerator(
            content_source="cleaned_html",
            options={
                "ignore_links": True,
                "ignore_images": True,
                "escape_html": True,
                "skip_internal_links": True,
            },
        )

        extra_kwargs = {}
        if with_js:
            extra_kwargs.update(
                {
                    "js_code": [CLICK_AND_MONITOR_JS],
                    "wait_for": WAIT_FOR_READY_JS,
                    "delay_before_return_html": 3.0,
                }
            )

        return CrawlerRunConfig(
            extraction_strategy=JsonXPathExtractionStrategy(
                DOI_EXTRACTION_SCHEMA, verbose=False
            ),
            markdown_generator=md_gen,
            page_timeout=timeout_ms,
            delay_before_return_html=extra_kwargs.get("delay_before_return_html", 0.5),
            word_count_threshold=10,
            excluded_tags=["nav", "footer", "aside", "form", "dialog"],
            excluded_selector="[role=dialog], .footer, .reference-citations",
            **{
                k: v for k, v in extra_kwargs.items() if k != "delay_before_return_html"
            },
        )

    def _build_pdf_crawler_config(self, timeout_ms: int) -> any:
        """Build crawler configuration for PDF files."""
        (
            AsyncWebCrawler,
            CrawlerRunConfig,
            PDFCrawlerStrategy,
            PDFContentScrapingStrategy,
            AsyncLoggerBase,
        ) = _get_crawl4ai_pdf_imports()

        pdf_crawler_cfg = PDFCrawlerStrategy()
        pdf_scraping_cfg = PDFContentScrapingStrategy()

        return CrawlerRunConfig(
            scraping_strategy=pdf_scraping_cfg,
            page_timeout=timeout_ms,
            word_count_threshold=10,
        )

    async def fetch_html(self, url: str, retry: bool = False) -> Dict[str, str]:
        """Fetch HTML with automatic retry escalation and PubMed full-text following."""
        # First, fetch the initial content
        result = await self._fetch_with_retry_escalation(
            url, self._fetch_html_simple, self._fetch_html_stealth, retry
        )

        # Check if this is a PubMed URL and try to find better full-text
        if self._is_pubmed_url(url):
            # Extract full-text links from already-fetched HTML
            fulltext_links = self._extract_pubmed_fulltext_links(
                result["raw_content"], url
            )
            if fulltext_links:
                better_result = await self._try_pubmed_fulltext_links(
                    result, fulltext_links
                )
                if better_result:
                    return better_result

        return result

    async def fetch_pdf(self, url: str, retry: bool = False) -> Dict[str, str]:
        """Fetch PDF content with format detection."""
        if not self._is_pdf_url(url):
            raise ValueError(f"URL does not appear to be a PDF: {url}")
        return await self._fetch_pdf_content(url)

    async def fetch_markdown(self, url: str, retry: bool = False) -> str:
        """Fetch content and return processed markdown."""
        if self._is_pdf_url(url):
            result = await self.fetch_pdf(url, retry=retry)
        else:
            result = await self.fetch_html(url, retry=retry)

        from .content_processor import ContentProcessor

        processor = ContentProcessor()
        return processor.refine_article(result["markdown_content"])

    def create_chunks(
        self, markdown_content: str | List[str]
    ) -> List[dict] | List[List[dict]]:
        """
        Chunk markdown content and return list of chunk objects with embeddings.

        Supports both single document and batch processing for efficiency.

        Parameters:
            markdown_content: Single markdown string or list of markdown strings

        Returns:
            Single list of chunks (if input is str) or list of chunk lists (if input is List[str])
        """
        chunker = _get_chunker()

        # Detect batch vs single mode
        is_batch = isinstance(markdown_content, list)

        if is_batch:
            # Batch mode - use chunk_batch() for efficient processing
            batch_chunk_objects = chunker.chunk_batch(markdown_content)
            return [
                self._process_chunk_objects(chunks) for chunks in batch_chunk_objects
            ]
        else:
            # Single mode - process one document
            chunk_objects = chunker(markdown_content)
            return self._process_chunk_objects(chunk_objects)

    def _process_chunk_objects(self, chunk_objects) -> List[dict]:
        """
        Process chonkie chunk objects into our chunk format with weighted embeddings.

        Computes chunk embeddings as weighted average of sentence embeddings,
        with position-based and length-based weighting.
        """
        import numpy as np

        chunks_data = []
        for chunk_obj in chunk_objects:
            # Trim leading newlines and trailing whitespace
            cleaned_text = chunk_obj.text.lstrip("\n").rstrip()

            chunk_dict = {
                "text": cleaned_text,
                "wordcount": len(cleaned_text.split()),
                "embedding": None,
            }

            # Extract embedding from chonkie chunk object
            # Compute chunk embedding as weighted average of sentence embeddings
            if hasattr(chunk_obj, "sentences") and chunk_obj.sentences:
                sentence_embeddings = []
                sentence_weights = []

                # Get text lengths for length-based weighting
                sentence_lengths = [
                    len(sentence.text) for sentence in chunk_obj.sentences
                ]
                max_length = max(sentence_lengths) if sentence_lengths else 1

                for i, sentence in enumerate(chunk_obj.sentences):
                    if (
                        hasattr(sentence, "embedding")
                        and sentence.embedding is not None
                    ):
                        sentence_embeddings.append(np.array(sentence.embedding))

                        # Position weight: higher for first/last 20% of sentences
                        num_sentences = len(chunk_obj.sentences)
                        position_cutoff = max(1, int(0.2 * num_sentences))
                        if i < position_cutoff or i >= num_sentences - position_cutoff:
                            position_weight = 1.5  # Important positions
                        else:
                            position_weight = 1.0  # Middle sentences

                        # Length weight: normalize by max length
                        length_weight = sentence_lengths[i] / max_length

                        # Combined weight
                        total_weight = position_weight * length_weight
                        sentence_weights.append(total_weight)

                if sentence_embeddings:
                    # Weighted average of sentence embeddings
                    sentence_weights = np.array(sentence_weights)
                    weights_normalized = sentence_weights / sentence_weights.sum()

                    chunk_embedding = np.average(
                        sentence_embeddings, axis=0, weights=weights_normalized
                    )
                    chunk_dict["embedding"] = chunk_embedding.tolist()

            chunks_data.append(chunk_dict)

        return chunks_data

    async def _fetch_with_retry_escalation(
        self, url: str, simple_fetcher, stealth_fetcher, force_retry: bool = False
    ):
        """Higher-order function for retry escalation pattern."""
        try:
            # if self.verbose:
            #     self._debug_console.print(
            #         f"\r[blue]Fetching {url} (simple mode)[/blue]"
            #     )
            result = await simple_fetcher(url)

            if force_retry or self._should_retry_with_stealth(result):
                if self.verbose:
                    self._debug_console.print(
                        f"\r[yellow]Retrying {url} with stealth mode[/yellow]"
                    )
                stealth_result = await stealth_fetcher(url)
                if self.verbose:
                    still_needs_retry = self._should_retry_with_stealth(stealth_result)
                    status = (
                        "still triggers retry"
                        if still_needs_retry
                        else "retry conditions resolved"
                    )
                    self._debug_console.print(
                        f"\r[cyan]Stealth result for {url}: {status}[/cyan]"
                    )
                return stealth_result
            else:
                pass
                # if self.verbose:
                #     self._debug_console.print(
                #         f"\r[green]Simple fetch successful for {url}[/green]"
                #     )
            return result
        except Exception as e:
            if force_retry:
                if self.verbose:
                    self._debug_console.print(
                        f"\r[red]Simple fetch failed for {url}, trying stealth: {e}[/red]"
                    )
                return await stealth_fetcher(url)
            raise

    async def _fetch_html_simple(self, url: str) -> Dict[str, str]:
        """Fetch HTML using simple configuration."""
        return await self._fetch_html_with_config(url, "simple_html", "simple_browser")

    async def _fetch_html_stealth(self, url: str) -> Dict[str, str]:
        """Fetch HTML using stealth configuration with JS execution."""
        return await self._fetch_html_with_config(
            url, "stealth_html", "stealth_browser"
        )

    async def _fetch_html_with_config(
        self, url: str, config_key: str, browser_key: str
    ) -> Dict[str, str]:
        """Fetch HTML using specified configuration."""
        (
            AsyncWebCrawler,
            CrawlerRunConfig,
            JsonXPathExtractionStrategy,
            AsyncLoggerBase,
            DefaultMarkdownGenerator,
        ) = _get_crawl4ai_imports()

        crawler_config = self._crawler_configs[config_key]
        browser_config = self._crawler_configs[browser_key]
        logger = _get_quiet_logger() if not self.verbose else _get_granular_logger()

        async with AsyncWebCrawler(config=browser_config, logger=logger) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)
            if not result.success:
                raise RuntimeError(
                    f"{result.status_code} error fetching {url}: {result.error_message}"
                )

            return {
                "raw_content": result.html,
                "markdown_content": str(result.markdown),
                "final_url": self._extract_final_url(result, url),
                "doi": self._extract_doi(result),
            }

    async def _fetch_pdf_content(self, url: str) -> Dict[str, str]:
        """Fetch PDF content using crawl4ai."""
        (
            AsyncWebCrawler,
            CrawlerRunConfig,
            PDFCrawlerStrategy,
            PDFContentScrapingStrategy,
            AsyncLoggerBase,
        ) = _get_crawl4ai_pdf_imports()

        pdf_config = self._crawler_configs["pdf_config"]
        browser_config = self._crawler_configs["simple_browser"]
        logger = _get_quiet_logger() if not self.verbose else _get_granular_logger()

        async with AsyncWebCrawler(config=browser_config, logger=logger) as crawler:
            result = await crawler.arun(url=url, config=pdf_config)
            if not result.success:
                raise RuntimeError(
                    f"{result.status_code} error fetching PDF {url}: {result.error_message}"
                )

            return {
                "raw_content": result.extracted_content or result.html,
                "markdown_content": str(result.markdown),
                "final_url": self._extract_final_url(result, url),
                "doi": "",  # PDFs don't have DOI extraction
            }

    def _extract_final_url(self, result, original_url: str) -> str:
        """Extract the final URL after any redirects."""
        final_url = original_url

        # Check crawl4ai result structure for redirects
        if hasattr(result, "_results") and result._results:
            first_result = result._results[0]
            if hasattr(first_result, "redirected_url") and first_result.redirected_url:
                final_url = first_result.redirected_url
            elif hasattr(first_result, "url"):
                final_url = first_result.url

        if final_url == original_url and hasattr(result, "url"):
            final_url = result.url

        return final_url

    def _extract_doi(self, result) -> str:
        """Extract DOI from crawl4ai extracted content."""
        import json

        doi = ""
        try:
            if hasattr(result, "extracted_content") and result.extracted_content:
                # Try to parse as JSON string (crawl4ai format)
                if isinstance(result.extracted_content, str):
                    extracted_data = json.loads(result.extracted_content)
                else:
                    # If it's already parsed, use directly
                    extracted_data = result.extracted_content

                # Handle both list and dict formats
                if isinstance(extracted_data, list):
                    items_to_check = extracted_data
                elif isinstance(extracted_data, dict):
                    items_to_check = [extracted_data]
                else:
                    return ""

                for doi_data in items_to_check:
                    for field_name in [
                        "doi_meta_pub",
                        "doi_meta_cite",
                        "doi_dc_doi",
                        "doi_dc",
                    ]:
                        if field_name in doi_data and doi_data[field_name]:
                            doi = doi_data[field_name]
                            if isinstance(doi, list) and doi:
                                doi = doi[0]
                            break

                    if (
                        not doi
                        and "doi_canonical" in doi_data
                        and doi_data["doi_canonical"]
                    ):
                        canonical_url = doi_data["doi_canonical"]
                        if isinstance(canonical_url, list) and canonical_url:
                            canonical_url = canonical_url[0]
                        if "doi.org/" in canonical_url:
                            doi = canonical_url.split("doi.org/")[1]

                    if doi:
                        break
        except (json.JSONDecodeError, AttributeError, KeyError, IndexError, TypeError):
            pass
        return doi

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file based on extension."""
        from urllib.parse import urlparse

        return urlparse(url).path.lower().endswith(".pdf")

    def _is_pubmed_url(self, url: str) -> bool:
        """Check if URL is a PubMed article page."""
        import re

        return bool(re.match(r"https://pubmed\.ncbi\.nlm\.nih\.gov/\d+/?$", url))

    def _extract_pubmed_fulltext_links(self, html: str, base_url: str) -> List[str]:
        """
        Extract full-text links from PubMed HTML.

        Parameters:
            html: Raw HTML content from PubMed page
            base_url: PubMed URL for resolving relative links

        Returns:
            List of absolute full-text link URLs
        """
        from lxml import etree
        from io import StringIO
        from urllib.parse import urljoin

        try:
            parser = etree.HTMLParser()
            tree = etree.parse(StringIO(html), parser)
            # Extract all hrefs from links in the full-text-links-list div
            links = tree.xpath(
                "//div[contains(@class, 'full-text-links-list')]//a/@href"
            )
            # Convert to absolute URLs and deduplicate
            return list(
                dict.fromkeys(urljoin(base_url, link) for link in links if link)
            )
        except Exception:
            return []

    def _is_content_superset(self, candidate_md: str, baseline_md: str) -> bool:
        """
        Check if candidate is a strict superset of baseline content.

        Criteria:
        - Markdown length >= 1.5x baseline (50% improvement threshold)
        - All baseline headings present in candidate
        - Total heading content length >= baseline

        Parameters:
            candidate_md: Candidate markdown content
            baseline_md: Baseline markdown content

        Returns:
            True if candidate is a strict superset, False otherwise
        """
        # Criterion 1: Length check (50% improvement)
        if len(candidate_md) < len(baseline_md) * (
            1 + PUBMED_CONTENT_IMPROVEMENT_THRESHOLD
        ):
            return False

        # Extract headings from both documents
        from .content_processor import ContentProcessor

        processor = ContentProcessor()
        baseline_headings = processor._extract_and_classify_headings(baseline_md)
        candidate_headings = processor._extract_and_classify_headings(candidate_md)

        # Criterion 2: All baseline headings must be present in candidate
        # (excluding PubMed-specific metadata headings)
        baseline_heading_texts = {h.text.lower().strip() for h in baseline_headings}
        candidate_heading_texts = {h.text.lower().strip() for h in candidate_headings}

        # Filter out PubMed metadata headings from baseline
        baseline_content_headings = baseline_heading_texts - PUBMED_METADATA_HEADINGS

        if not baseline_content_headings.issubset(candidate_heading_texts):
            return False

        # Criterion 3: Total heading content length comparison
        # Extract content under each heading
        def get_heading_content_length(markdown: str, headings) -> int:
            """Calculate total length of content under headings."""
            total_length = 0
            for i, heading in enumerate(headings):
                # Find content between this heading and next heading (or end)
                start = heading.end_pos
                if i + 1 < len(headings):
                    end = headings[i + 1].start_pos
                else:
                    end = len(markdown)
                content = markdown[start:end].strip()
                total_length += len(content)
            return total_length

        baseline_content_length = get_heading_content_length(
            baseline_md, baseline_headings
        )
        candidate_content_length = get_heading_content_length(
            candidate_md, candidate_headings
        )

        if candidate_content_length < baseline_content_length:
            return False

        return True

    async def _try_pubmed_fulltext_links(
        self,
        pubmed_result: Dict[str, str],
        fulltext_links: List[str],
    ) -> Dict[str, str] | None:
        """
        Try fetching full-text links to find better content.

        Parameters:
            pubmed_result: Result dict from PubMed fetch
            fulltext_links: List of full-text link URLs to try

        Returns:
            Best result dict with updated final_url, or None if PubMed is best
        """
        if not fulltext_links:
            return None

        import asyncio

        # Get baseline content for comparison
        from .content_processor import ContentProcessor

        processor = ContentProcessor()
        baseline_md = processor.refine_article(pubmed_result["markdown_content"])

        # Fetch full-text links concurrently with semaphore
        semaphore = asyncio.Semaphore(PUBMED_MAX_CONCURRENT_LINKS)

        async def try_single_link(link: str):
            """Try fetching a single full-text link."""
            async with semaphore:
                try:
                    # Fetch using the appropriate method (PDF or HTML)
                    if self._is_pdf_url(link):
                        result = await self.fetch_pdf(link, retry=False)
                    else:
                        result = await self.fetch_html(link, retry=False)

                    # Process and compare
                    candidate_md = processor.refine_article(result["markdown_content"])

                    if self._is_content_superset(candidate_md, baseline_md):
                        return (link, result, len(candidate_md))
                    else:
                        return None
                except Exception:
                    # Silently ignore failures for individual links
                    return None

        # Try all links concurrently
        tasks = [try_single_link(link) for link in fulltext_links]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Filter successful results and find best one (longest content)
        successful_results = [r for r in results if r is not None]

        if not successful_results:
            return None

        # Return the result with the most content
        best_link, best_result, best_length = max(
            successful_results, key=lambda x: x[2]
        )

        # Update final_url to point to the best full-text link
        best_result["final_url"] = best_link

        if self.verbose:
            self._debug_console.print(
                f"\r[green]Found better full-text at {best_link} "
                f"({best_length} vs {len(baseline_md)} chars)[/green]"
            )

        return best_result

    def _should_retry_with_stealth(self, fetch_result: Dict[str, str]) -> bool:
        """Decide whether to retry fetching with stealth/full-text instrumentation."""
        raw_markdown = fetch_result.get("markdown_content", "")
        raw_html = fetch_result.get("raw_content", "")
        text = ((raw_markdown or "") + "\n" + (raw_html or "")).lower()

        # Initialize flags
        no_markdown = not raw_markdown
        short_markdown = bool(
            raw_markdown and len(raw_markdown) < STEALTH_RETRY_THRESHOLD
        )
        no_recognized_sections = not self._has_recognized_sections(raw_markdown or "")
        bot_challenge = "verifying you are human" in text or "are you a robot" in text
        free_full_text_available = "free full text" in text

        # Retry if we have a bot challenge OR content is short AND free full text available OR content is short/lacking sections
        should_retry = (
            bot_challenge
            or (short_markdown and free_full_text_available)
            or ((short_markdown or no_markdown) and no_recognized_sections)
        )

        return should_retry

    def _has_recognized_sections(self, markdown: str) -> bool:
        """Detect if markdown contains recognized academic sections based on headings."""
        try:
            from .content_processor import extract_headings, classify_heading_relevance

            heads = extract_headings(markdown)
            for h in heads:
                cls = classify_heading_relevance(h["text"])
                if cls.get("is_relevant"):
                    return True
        except Exception:
            pass
        return False
