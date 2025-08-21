"""
Enhanced PageFetcher with intelligent URL handling and granular status updates.

Features:
- Single URL and batch URL fetching with automatic concurrency
- Granular status updates showing detailed crawl4ai stages
- Automatic PDF vs HTML detection based on URL extension
- DOI extraction from HTML pages using XPath selectors
- References section removal from markdown content
- Efficient lazy imports for faster startup
- Progress bars for batch operations, detailed status for single URLs
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union, List, Any
import aiofiles
import aiofiles.os
from rich.console import Console

if TYPE_CHECKING:
    from .settings import IfetcherConfig

import re

# Content type configurations for serialization/deserialization
CONTENT_TYPE_CONFIG = {
    "html": {"extension": "html", "serialize": str, "deserialize": str},
    "pdf": {"extension": "pdf", "serialize": str, "deserialize": str},
    "markdown": {"extension": "md", "serialize": str, "deserialize": str},
    "raw_markdown": {"extension": "raw.md", "serialize": str, "deserialize": str},
    "chunks": {
        "extension": "json",
        "serialize": lambda x: json.dumps(x, indent=2, ensure_ascii=False),
        "deserialize": lambda x: json.loads(x)
    },
    "doi": {"extension": "doi", "serialize": str, "deserialize": lambda x: x.strip()}
}

# Heading classification for academic content processing
RELEVANT_HEADINGS = {
    'abstract', 'summary', 'introduction', 'methods', 'methodology', 'results',
    'discussion', 'conclusion', 'conclusions', 'background', 'objectives',
    'findings', 'analysis', 'materials', 'procedure', 'approach',
    'appendix', 'appendices', 'limitations', 'future work', 'implications',
    'main text'
}

IRRELEVANT_HEADINGS = {
    'references', 'keywords', 'bibliography', 'citations', 'authors', 'author',
    'acknowledgements', 'acknowledgments', 'share', 'sharing',
    'funding', 'conflicts', 'conflict of interest', 'competing interests',
    'data availability', 'supplementary', 'supporting information',
    'copyright', 'license', 'permissions', 'ethics', 'rights',
    'affiliations', 'corresponding', 'cite this', 'reprints',
    'related articles',
}

# Pattern lists for content classification
IRRELEVANT_PATTERNS = [
    r'.*login.*', r'.*sign\s*in.*', r'.*sign\s*up.*', r'.*register.*',
    r'.*cookies?.*', r'.*privacy.*', r'.*terms.*', r'.*subscribe.*',
    r'.*newsletter.*', r'.*follow.*', r'.*social.*', r'.*menu.*',
    r'.*navigation.*', r'.*search.*', r'.*contact.*', r'.*about.*',
    r'.*create.*account.*', r'.*free.*account.*', r'.*read.*content.*',
    r'similar content.*', r'.*viewed by others', r'recommended.*',
    r'supplementary\s+.*', r'supplemental\s+.*', r'.*metrics.*',
    r'declaration\s+.*', r'.*availability.*', r'.*privacy.*'
]

PAYWALL_PATTERNS = [
    r'.*login.*', r'.*sign\s*in.*', r'.*sign\s*up.*', r'.*register.*',
    r'.*create.*account.*', r'.*free.*account.*', r'.*subscription.*',
    r'.*paywall.*', r'.*access.*denied.*', r'.*premium.*content.*',
    r'.*unlock.*content.*', r'.*full.*access.*'
]

# DOI extraction schema for XPath-based extraction from academic publishers
DOI_EXTRACTION_SCHEMA = {
    "name": "DOI extractor (XPath)",
    "baseSelector": "/html",
    "fields": [
        {
            "name": "doi_meta_cite",
            "selector": "//meta[@name='citation_doi']",
            "type": "attribute",
            "attribute": "content"
        },
        {
            "name": "doi_meta_pub",
            "selector": "//meta[@name='publication_doi']",
            "type": "attribute",
            "attribute": "content"
        },
        {
            "name": "doi_dc",
            "selector": "//meta[translate(@name,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='dc.identifier' and contains(@content, 'doi.org/')]",
            "type": "attribute",
            "attribute": "content"
        },
        {
            "name": "doi_dc_doi",
            "selector": "//meta[translate(@name,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='dc.identifier.doi']",
            "type": "attribute",
            "attribute": "content"
        },
        {
            "name": "doi_canonical",
            "selector": "//link[@rel='canonical' and contains(@href, 'doi.org/')]",
            "type": "attribute",
            "attribute": "href"
        }
    ]
}

# =============================================================================
# Content Processing Helper Functions
# =============================================================================

def refine_article_content(markdown: str) -> str:
    """
    Refine article content by removing irrelevant sections using intelligent heading analysis.

    Identifies relevant academic sections (Abstract, Methods, Results, Discussion, Conclusion)
    and irrelevant sections (References, Authors, Acknowledgements, navigation elements).
    Removes irrelevant content before first relevant section and after last relevant section.

    Args:
        markdown: Raw markdown content from academic article webpage

    Returns:
        Cleaned markdown content with only relevant academic sections

    Example:
        >>> content = "# Title\\n## Abstract\\nContent...\\n## Authors\\nMore content..."
        >>> refined = refine_article_content(content)
        >>> # Returns: "# Title\\n## Abstract\\nContent..."
    """

    # Use module-level constants
    relevant_headings = RELEVANT_HEADINGS
    irrelevant_headings = IRRELEVANT_HEADINGS
    irrelevant_patterns = IRRELEVANT_PATTERNS
    paywall_patterns = PAYWALL_PATTERNS

    # Find all headings with their positions and levels
    heading_pattern = r'^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?\s*$'
    headings = []

    for match in re.finditer(heading_pattern, markdown, re.MULTILINE):
        level = len(match.group(1))  # Number of # characters
        text = match.group(2).strip()
        start_pos = match.start()
        end_pos = match.end()

        # Classify heading as relevant or irrelevant
        text_lower = text.lower()

        # Check for irrelevant patterns first (higher priority)
        is_irrelevant = (
            any(irrel in text_lower for irrel in irrelevant_headings) or
            any(re.match(pattern, text_lower, re.IGNORECASE) for pattern in irrelevant_patterns)
        )

        # Check for paywall patterns
        is_paywall = any(re.match(pattern, text_lower, re.IGNORECASE) for pattern in paywall_patterns)

        # Only check for relevant patterns if not already marked as irrelevant
        is_relevant = not is_irrelevant and any(rel in text_lower for rel in relevant_headings)

        headings.append({
            'text': text,
            'level': level,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'is_relevant': is_relevant,
            'is_irrelevant': is_irrelevant,
            'is_paywall': is_paywall,
            'full_match': match.group(0)
        })

    if not headings:
        return markdown

    # Find first and last relevant headings
    relevant_headings_list = [h for h in headings if h['is_relevant']]
    irrelevant_headings_list = [h for h in headings if h['is_irrelevant']]
    paywall_headings_list = [h for h in headings if h['is_paywall']]

    # Build list of sections to remove
    sections_to_remove = []

    if not relevant_headings_list:
        # No relevant headings found - check if this is paywall content
        if paywall_headings_list:
            # Paywall content detected with no scientific content - remove everything
            return ""
        else:
            # No relevant headings found - be more aggressive with irrelevant section removal
            for heading in irrelevant_headings_list:
                # Find end of this irrelevant section (next heading of same/higher level or end of doc)
                section_end = len(markdown)
                for next_heading in headings:
                    if (next_heading['start_pos'] > heading['start_pos'] and
                        next_heading['level'] <= heading['level']):
                        section_end = next_heading['start_pos']
                        break
                sections_to_remove.append((heading['start_pos'], section_end))
    else:
        first_relevant = relevant_headings_list[0]
        last_relevant = relevant_headings_list[-1]

        # Look for Abstract, Summary, or Introduction section first
        abstract_heading = None
        for heading in relevant_headings_list:
            text_lower = heading['text'].lower()
            if ('abstract' in text_lower) or ('summary' in text_lower) or ('introduction' in text_lower):
                abstract_heading = heading
                break

        # If Abstract exists, try to preserve title before it
        title_to_preserve = None
        if abstract_heading is not None:
            # Look for a larger heading immediately before Abstract
            for heading in reversed(headings):  # Start from closest to Abstract
                if (heading['start_pos'] < abstract_heading['start_pos'] and
                    heading['level'] < abstract_heading['level']):  # Any heading larger than Abstract
                    # Check if there are any RELEVANT headings between this title and the Abstract
                    # Ignore irrelevant headings like "Permissions", "Copyright", etc.
                    has_headings_between = False
                    for between_heading in headings:
                        if (heading['end_pos'] < between_heading['start_pos'] < abstract_heading['start_pos']):
                            # Only consider this heading as blocking if it's not clearly irrelevant
                            # and is of equal or higher importance than the Abstract (level <= Abstract level)
                            between_text_lower = between_heading['text'].lower()
                            is_between_irrelevant = (
                                any(irrel in between_text_lower for irrel in irrelevant_headings) or
                                any(re.match(pattern, between_text_lower, re.IGNORECASE) for pattern in irrelevant_patterns) or
                                between_heading.get('is_paywall', False)
                            )
                            is_blocking_level = between_heading['level'] <= abstract_heading['level']
                            if not is_between_irrelevant and is_blocking_level:
                                has_headings_between = True
                                break

                    if not has_headings_between:
                        # Found title with no headings in between - preserve it
                        title_to_preserve = heading
                        break

            # Apply title preservation logic for Abstract papers
            if title_to_preserve is not None:
                # Remove content before title
                sections_to_remove.append((0, title_to_preserve['start_pos']))
                # Remove content between title and Abstract, but preserve the newline after title
                title_line_end = title_to_preserve['end_pos']
                while title_line_end < len(markdown) and markdown[title_line_end] != '\n':
                    title_line_end += 1
                if title_line_end < len(markdown):
                    title_line_end += 1  # Include the newline
                sections_to_remove.append((title_line_end, abstract_heading['start_pos']))
            else:
                # No title found, remove everything before Abstract
                sections_to_remove.append((0, abstract_heading['start_pos']))

        else:
            # No Abstract/Summary/Introduction section - start at the first
            # non-irrelevant, non-paywall heading regardless of level.
            first_academic_heading = None
            for heading in headings:
                if (not heading['is_irrelevant'] and not heading['is_paywall']):
                    first_academic_heading = heading
                    break

            if first_academic_heading is not None:
                sections_to_remove.append((0, first_academic_heading['start_pos']))
            else:
                # Fallback: remove before first relevant heading if no academic heading found
                sections_to_remove.append((0, first_relevant['start_pos']))

        # Remove irrelevant sections after last relevant heading
        for heading in headings:
            if heading['start_pos'] <= last_relevant['start_pos']:
                continue
            if heading['is_irrelevant']:
                # Remove from this heading to end of document
                sections_to_remove.append((heading['start_pos'], len(markdown)))
                break  # Once we find the first irrelevant heading after last relevant, remove everything

    # Apply removals in reverse order to maintain position accuracy
    sections_to_remove.sort(key=lambda x: x[0], reverse=True)
    result = markdown

    for start_pos, end_pos in sections_to_remove:
        result = result[:start_pos] + result[end_pos:]

    # Safety check: only apply when we don't have clear academic structure
    # For content with clear relevant sections and many irrelevant sections, aggressive removal is desired
    has_clear_academic_structure = (
        len(relevant_headings_list) > 0 and
        len(irrelevant_headings_list) > 5 and
        any((('abstract' in h['text'].lower()) or ('summary' in h['text'].lower()) or ('introduction' in h['text'].lower())) for h in relevant_headings_list)
    )

    # Alternative academic structure: papers without Abstract but with standard sections
    has_alternative_academic_structure = (
        len(relevant_headings_list) >= 3 and  # At least 3 relevant sections
        len(irrelevant_headings_list) > 5 and  # Many irrelevant sections to remove
        any(section in h['text'].lower() for h in relevant_headings_list
            for section in ['methods', 'results', 'discussion', 'conclusion'])
    )

    # Skip safety check for clear academic content (e.g., abstract pages with lots of website chrome)
    if not has_clear_academic_structure and not has_alternative_academic_structure:
        reduction_ratio = len(result.strip()) / len(markdown.strip()) if len(markdown.strip()) > 0 else 1
        if markdown.strip() and result.strip() and reduction_ratio < 0.3:
            return markdown.strip()

    return result.strip()

def extract_headings(markdown: str) -> List[dict]:
    """
    Extract all headings from markdown content with metadata.

    Args:
        markdown: Markdown content to analyze

    Returns:
        List of heading dictionaries with text, level, and position information
    """
    heading_pattern = r'^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?\s*$'
    headings = []

    for match in re.finditer(heading_pattern, markdown, re.MULTILINE):
        level = len(match.group(1))
        text = match.group(2).strip()
        start_pos = match.start()
        end_pos = match.end()

        headings.append({
            'text': text,
            'level': level,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'full_match': match.group(0)
        })

    return headings

def classify_heading_relevance(heading_text: str) -> dict:
    """
    Classify a heading as relevant, irrelevant, or paywall content.

    Args:
        heading_text: The text content of the heading

    Returns:
        Dictionary with 'is_relevant', 'is_irrelevant', 'is_paywall' boolean flags
    """
    # Use module-level constants
    relevant_headings = RELEVANT_HEADINGS
    irrelevant_headings = IRRELEVANT_HEADINGS
    irrelevant_patterns = IRRELEVANT_PATTERNS
    paywall_patterns = PAYWALL_PATTERNS

    text_lower = heading_text.lower()

    is_irrelevant = (
        any(irrel in text_lower for irrel in irrelevant_headings) or
        any(re.match(pattern, text_lower, re.IGNORECASE) for pattern in irrelevant_patterns)
    )

    is_paywall = any(re.match(pattern, text_lower, re.IGNORECASE) for pattern in paywall_patterns)
    is_relevant = not is_irrelevant and any(rel in text_lower for rel in relevant_headings)

    return {
        'is_relevant': is_relevant,
        'is_irrelevant': is_irrelevant,
        'is_paywall': is_paywall
    }

# =============================================================================
# Lazy Import Cache Variables
# =============================================================================

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
            from crawl4ai.async_logger import AsyncLoggerBase
            _crawl4ai_html_imports = (AsyncWebCrawler, CrawlerRunConfig, JsonXPathExtractionStrategy, AsyncLoggerBase)
        except ImportError:
            raise RuntimeError("crawl4ai package is required for fetching. Install with: pip install crawl4ai")
    return _crawl4ai_html_imports

def _get_crawl4ai_pdf_imports():
    """Lazy import crawl4ai PDF modules with simple caching."""
    global _crawl4ai_pdf_imports
    if _crawl4ai_pdf_imports is None:
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
            from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
            from crawl4ai.async_logger import AsyncLoggerBase
            _crawl4ai_pdf_imports = (AsyncWebCrawler, CrawlerRunConfig, PDFCrawlerStrategy, PDFContentScrapingStrategy, AsyncLoggerBase)
        except ImportError:
            raise RuntimeError("crawl4ai package is required for fetching. Install with: pip install crawl4ai")
    return _crawl4ai_pdf_imports

def _get_granular_logger(status_display=None):
    """
    Create a logger that provides granular stage information to Rich status displays.

    Args:
        status_display: Rich status object to update with progress information

    Returns:
        GranularLogger instance that updates the status display with crawl4ai stages
    """
    _, _, _, AsyncLoggerBase = _get_crawl4ai_imports()

    class GranularLogger(AsyncLoggerBase):
        """Logger that translates crawl4ai stages into Rich status updates."""

        def __init__(self, status_display=None):
            self.status_display = status_display
        def debug(self, message: str, tag: str = "DEBUG", **kwargs): pass
        def info(self, message: str, tag: str = "INFO", **kwargs): pass
        def success(self, message: str, tag: str = "SUCCESS", **kwargs): pass
        def warning(self, message: str, tag: str = "WARNING", **kwargs): pass
        def error(self, message: str, tag: str = "ERROR", **kwargs): pass
        def error_status(self, url: str, error: str, tag: str = "ERROR", url_length: int = 100): pass

        def url_status(self, url: str, success: bool, timing: float,
                       tag: str = "FETCH", url_length: int = 100) -> None:
            if self.status_display is None:
                return

            domain = url.split('//')[1].split('/')[0] if '//' in url else url[:20]

            stage_messages = {
                "FETCH": f"[cyan]Fetching from[/cyan] [bold]{domain}[/bold]",
                "SCRAPE": f"[blue]Processing content from[/blue] [bold]{domain}[/bold]",
                "EXTRACT": f"[yellow]Extracting data from[/yellow] [bold]{domain}[/bold]",
                "COMPLETE": f"[green]✓ Completed[/green] [bold]{domain}[/bold] [dim]({timing:.1f}s)[/dim]"
            }

            if tag in stage_messages and hasattr(self.status_display, 'update'):
                self.status_display.update(stage_messages[tag])

    return GranularLogger(status_display)

def _get_chunker():
    """Lazy import and initialize chonkie chunker with simple caching."""
    global _chunker
    if _chunker is None:
        from chonkie import SDPMChunker
        _chunker = SDPMChunker(
            embedding_model="minishlab/potion-base-8M", # Default model
            threshold=0.5,                              # Similarity threshold (0-1)
            chunk_size=4096,                            # Maximum tokens per chunk
            min_sentences=2,                            # Initial sentences per chunk
            skip_window=1                               # Number of chunks to skip when looking for similarities
        )
    return _chunker

def _get_quiet_logger():
    """Get a silent logger for crawl4ai when no status updates are needed."""
    global _quiet_logger
    if _quiet_logger is None:
        _quiet_logger = _get_granular_logger(status_display=None)
    return _quiet_logger

class StatusDisplay:
    """Manages Rich status indicators for fetch operations."""

    def __init__(self, show_status: bool = True):
        self.show_status = show_status
        self.console = Console() if show_status else None

    def create_status(self, message: str, progress_info=None):
        """
        Create appropriate status indicator based on context.

        Args:
            message: Initial status message to display
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Rich status object or DummyStatus for no-op
        """
        if progress_info is not None:
            progress_instance, task_id = progress_info
            progress_instance.update(task_id, description=message)
            return DummyStatus()
        elif self.show_status and self.console:
            return self.console.status(message, spinner="dots")
        else:
            return DummyStatus()

class DummyStatus:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, text: str):
        pass
def url_to_hash(url: str) -> str:
    """Convert URL to a filesystem-safe hash (backward compatibility)."""
    normalized_url = normalize_url(url)
    return hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()[:16]

def normalize_url(url: str) -> str:
    """Normalize URL by removing fragment and other client-side only components."""
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    # Remove fragment (everything after #) as it's client-side only
    return urlunparse(parsed._replace(fragment=''))

def url_to_hash_base36(url: str) -> str:
    """Convert URL to a base36 hash for compact representation."""
    normalized_url = normalize_url(url)
    # Use deterministic SHA256 hash instead of Python's non-deterministic hash()
    hash_bytes = hashlib.sha256(normalized_url.encode('utf-8')).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')  # Use first 8 bytes

    # Manual base36 conversion
    if hash_int == 0:
        return "0"

    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while hash_int:
        result = digits[hash_int % 36] + result
        hash_int //= 36
    return result

class URLCache:
    """File-based cache for URLs with support for HTML, PDF, and Markdown content."""

    def __init__(self, config: IfetcherConfig):
        self.config = config
        self.base_path = config.abspath(config.output.cache)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def _get_paths(self, url: str, *extensions: str) -> tuple[Path, ...]:
        """Get file paths for specific extensions with collision resolution."""
        normalized_url = normalize_url(url)
        base_hash = url_to_hash_base36(url)
        probe = 0

        while probe < 1000:  # Safety limit
            if probe == 0:
                hash_str = base_hash
            else:
                hash_str = f"{base_hash}_{probe}"

            url_path = self.base_path / f"{hash_str}.url"

            # If no URL file exists, this slot is available
            if not url_path.exists():
                return tuple(self.base_path / f"{hash_str}.{ext}" for ext in extensions)

            # URL file exists, check if it's for the same URL
            try:
                async with aiofiles.open(url_path, 'r', encoding='utf-8') as f:
                    stored_url = (await f.read()).strip()
                if stored_url == normalized_url:
                    # Found existing entry for this URL
                    return tuple(self.base_path / f"{hash_str}.{ext}" for ext in extensions)
            except (OSError, UnicodeDecodeError):
                # Corrupted file, skip this slot
                pass

            probe += 1

        raise RuntimeError(f"Too many hash collisions for URL: {url}")

    async def _store_url_mapping(self, url: str, final_url: Optional[str] = None) -> None:
        """Store the URL mapping and redirect info in sidecar files with atomic operations."""
        normalized_url = normalize_url(url)
        url_path, redir_path = await self._get_paths(url, "url", "redir")

        # Atomic file creation: write to temp file then rename
        temp_url_path = url_path.with_suffix('.url.tmp')
        try:
            async with aiofiles.open(temp_url_path, 'w', encoding='utf-8') as f:
                await f.write(normalized_url)
            # Atomic rename - prevents race conditions
            await aiofiles.os.rename(temp_url_path, url_path)
        except Exception:
            # Clean up temp file on failure
            if temp_url_path.exists():
                temp_url_path.unlink()
            raise

        # Store redirect mapping if final URL differs from original
        if final_url and normalize_url(final_url) != normalized_url:
            temp_redir_path = redir_path.with_suffix('.redir.tmp')
            try:
                async with aiofiles.open(temp_redir_path, 'w', encoding='utf-8') as f:
                    await f.write(normalize_url(final_url))
                await aiofiles.os.rename(temp_redir_path, redir_path)
            except Exception:
                if temp_redir_path.exists():
                    temp_redir_path.unlink()
                raise
        elif redir_path.exists():
            # Remove stale redirect file if URLs now match
            redir_path.unlink()

    async def _verify_url_mapping(self, url: str) -> bool:
        """Verify the URL mapping matches what's stored."""
        normalized_url = normalize_url(url)
        url_path, = await self._get_paths(url, "url")
        if url_path.exists():
            try:
                async with aiofiles.open(url_path, 'r', encoding='utf-8') as f:
                    stored_url = (await f.read()).strip()
                return stored_url == normalized_url
            except (OSError, UnicodeDecodeError):
                return False
        return False

    async def has_path(self, url: str, content_type: str) -> bool:
        """Check if content with given content type is cached for URL (including redirected URLs)."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(f"Unknown content type '{content_type}'. Available: {available}")

        extension = CONTENT_TYPE_CONFIG[content_type]["extension"]
        content_path, redir_path = await self._get_paths(url, extension, "redir")
        if content_path.exists():
            if await self._verify_url_mapping(url):
                return True
        # Check if this URL redirected to another URL that has content
        if redir_path.exists():
            try:
                async with aiofiles.open(redir_path, 'r', encoding='utf-8') as f:
                    final_url = (await f.read()).strip()
                final_content_path, = await self._get_paths(final_url, extension)
                return final_content_path.exists()
            except (OSError, UnicodeDecodeError):
                pass
        return False

    async def get_path(self, url: str, content_type: str) -> str:
        """Get cached content for URL with given content type (following redirects if needed)."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(f"Unknown content type '{content_type}'. Available: {available}")

        extension = CONTENT_TYPE_CONFIG[content_type]["extension"]
        content_path, redir_path = await self._get_paths(url, extension, "redir")

        # Check if content exists
        if content_path.exists():
            if await self._verify_url_mapping(url):
                async with aiofiles.open(content_path, 'r', encoding='utf-8') as f:
                    return await f.read()

        # Check for redirected content
        if redir_path.exists():
            try:
                async with aiofiles.open(redir_path, 'r', encoding='utf-8') as f:
                    final_url = (await f.read()).strip()
                final_content_path, = await self._get_paths(final_url, extension)
                if final_content_path.exists():
                    async with aiofiles.open(final_content_path, 'r', encoding='utf-8') as f:
                        return await f.read()
            except (OSError, UnicodeDecodeError):
                pass

        raise KeyError(f"Content with extension '{extension}' not cached for URL: {url}")

    async def set_path(self, url: str, content_type: str, content: str, final_url: Optional[str] = None) -> None:
        """Store content for URL and content type with atomic operations."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(f"Unknown content type '{content_type}'. Available: {available}")

        extension = CONTENT_TYPE_CONFIG[content_type]["extension"]
        # If there's a redirect, store content at the final URL location
        storage_url = final_url if final_url else url
        path, = await self._get_paths(storage_url, extension)

        # Atomic content write
        temp_path = path.with_suffix(f'.{extension}.tmp')
        try:
            async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            await aiofiles.os.rename(temp_path, path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        # Store URL mapping for both original and final URLs
        await self._store_url_mapping(url, final_url)
        if final_url and final_url != url:
            # Also store mapping for final URL to itself (for direct access)
            await self._store_url_mapping(final_url, None)

    async def get_content(self, url: str, content_type: str):
        """Get cached content for URL with given type (following redirects if needed)."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(f"Unknown content type '{content_type}'. Available: {available}")

        if content_type == "doi":
            try:
                raw_content = await self.get_path(url, content_type)
                return CONTENT_TYPE_CONFIG[content_type]["deserialize"](raw_content)
            except KeyError:
                return None
        else:
            raw_content = await self.get_path(url, content_type)
            return CONTENT_TYPE_CONFIG[content_type]["deserialize"](raw_content)

    async def set_content(self, url: str, content_type: str, content, final_url: Optional[str] = None) -> None:
        """Store content for URL with given type."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(f"Unknown content type '{content_type}'. Available: {available}")

        serialized_content = CONTENT_TYPE_CONFIG[content_type]["serialize"](content)
        await self.set_path(url, content_type, serialized_content, final_url)

    def _get_extension(self, content_type: str) -> str:
        """Get file extension for a content type."""
        return CONTENT_TYPE_CONFIG[content_type]["extension"]

    async def has_url(self, url: str) -> bool:
        """Check if URL is cached in any format."""
        for content_type in CONTENT_TYPE_CONFIG.keys():
            if await self.has_path(url, content_type):
                return True
        return False

    async def get_source_type(self, url: str) -> Optional[str]:
        """Get the source content type for a URL ('html' or 'pdf')."""
        if await self.has_path(url, "html"):
            return "html"
        elif await self.has_path(url, "pdf"):
            return "pdf"
        return None

    async def get_source_content(self, url: str) -> Optional[str]:
        """Get the raw source content (HTML or PDF) for a URL."""
        if await self.has_path(url, "html"):
            return await self.get_content(url, "html")
        elif await self.has_path(url, "pdf"):
            return await self.get_content(url, "pdf")
        return None

    async def clear_url(self, url: str) -> None:
        """Remove all cached content for a URL."""
        extensions = [config["extension"] for config in CONTENT_TYPE_CONFIG.values()]
        extensions.extend(["url", "redir"])  # Add metadata file extensions
        paths = await self._get_paths(url, *extensions)
        for path in paths:
            if path.exists():
                path.unlink()

    async def get_url_hash(self, url: str) -> str:
        """Get the actual hash string used for a URL (including probe suffix if any)."""
        url_path, = await self._get_paths(url, "url")
        # Extract hash from the path name
        return url_path.stem  # Remove .url extension to get the hash

    async def get_original_url(self, hash_str: str) -> Optional[str]:
        """Get the original URL from a hash string."""
        url_path = self.base_path / f"{hash_str}.url"
        if url_path.exists():
            try:
                async with aiofiles.open(url_path, 'r', encoding='utf-8') as f:
                    return (await f.read()).strip()
            except (OSError, UnicodeDecodeError):
                return None
        return None

    async def get_redirect_info(self, url: str) -> Optional[str]:
        """Get the final URL if this URL redirected, None otherwise."""
        redir_path, = await self._get_paths(url, "redir")
        if redir_path.exists():
            try:
                async with aiofiles.open(redir_path, 'r', encoding='utf-8') as f:
                    content = (await f.read()).strip()
                # Validate that the content looks like a URL
                if content and '://' in content:
                    return content
            except (OSError, UnicodeDecodeError):
                pass
        return None

    async def list_cached_urls(self) -> list[str]:
        """Get a list of all cached URLs."""
        urls = []
        for url_file in self.base_path.glob("*.url"):
            try:
                async with aiofiles.open(url_file, 'r', encoding='utf-8') as f:
                    url = (await f.read()).strip()
                # Verify at least one content file exists
                if await self.has_url(url):
                    urls.append(url)
            except (OSError, UnicodeDecodeError):
                continue
        return urls

class PageFetcher:
    """High-level interface for fetching and caching web pages."""

    def __init__(self, config: 'IfetcherConfig', show_status: bool = True):
        self.cache = URLCache(config)
        self.config = config
        self.status_display = StatusDisplay(show_status=show_status)

    async def _fetch_html_url(self, url: str, progress_info=None) -> dict[str, str]:
        """
        Fetch HTML URL using Crawl4AI with DOI extraction.

        Args:
            url: URL to fetch
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Dict containing raw_content, markdown_content, final_url, and doi
        """
        AsyncWebCrawler, CrawlerRunConfig, JsonXPathExtractionStrategy, _ = _get_crawl4ai_imports()
        domain = url.split('//')[1].split('/')[0] if '//' in url else url

        # Create crawler configuration with DOI extraction and proper timeouts
        cfg = CrawlerRunConfig(
            extraction_strategy=JsonXPathExtractionStrategy(DOI_EXTRACTION_SCHEMA, verbose=False),
            page_timeout=self.config.tools.crawl4ai.timeout * 1000,
            word_count_threshold = 10,
            excluded_tags = ["nav", "footer", "aside", "form", "dialog"],
            excluded_selector = "[role=dialog]",
            verbose=False
        )

        if progress_info is not None:
            progress_instance, task_id = progress_info
            progress_instance.update(task_id, description=f"[green]Fetching HTML from[/green] [bold]{domain}[/bold]")
            granular_logger = _get_quiet_logger()
            async with AsyncWebCrawler(logger=granular_logger) as crawler:
                result = await crawler.arun(url=url, config=cfg)
                if not result.success:
                    raise RuntimeError(f"{result.status_code} error fetching {url}: {result.error_message}")
        else:
            with self.status_display.create_status(f"[dim]Initializing...[/dim]") as status:
                granular_logger = _get_granular_logger(status_display=status)
                async with AsyncWebCrawler(logger=granular_logger) as crawler:
                    result = await crawler.arun(url=url, config=cfg)
                    if not result.success:
                        raise RuntimeError(f"{result.status_code} error fetching {url}: {result.error_message}")

        final_url = self._extract_final_url(result, url)
        raw_content = result.html or ''
        raw_markdown_content = result.markdown.raw_markdown
        markdown_content = self._refine_article_content(raw_markdown_content)
        doi = self._extract_doi(result)

        return {
            'raw_content': raw_content,
            'raw_markdown_content': raw_markdown_content,
            'markdown_content': markdown_content,
            'final_url': final_url,
            'doi': doi
        }

    async def _fetch_pdf_url(self, url: str, progress_info=None) -> dict[str, str]:
        """
        Fetch PDF URL using Crawl4AI PDF processing strategy.

        Args:
            url: PDF URL to fetch
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Dict containing raw_content, markdown_content, final_url (no DOI extraction for PDFs yet)
        """
        AsyncWebCrawler, CrawlerRunConfig, PDFCrawlerStrategy, PDFContentScrapingStrategy, _ = _get_crawl4ai_pdf_imports()
        domain = url.split('//')[1].split('/')[0] if '//' in url else url

        pdf_crawler_cfg = PDFCrawlerStrategy()
        pdf_scraping_cfg = PDFContentScrapingStrategy()

        cfg = CrawlerRunConfig(
            scraping_strategy=pdf_scraping_cfg,
            page_timeout=self.config.tools.crawl4ai.timeout * 1000,
            verbose=False
        )

        if progress_info is not None:
            progress_instance, task_id = progress_info
            progress_instance.update(task_id, description=f"[red]Fetching PDF from[/red] [bold]{domain}[/bold]")
            granular_logger = _get_quiet_logger()
            async with AsyncWebCrawler(crawler_strategy=pdf_crawler_cfg, logger=granular_logger) as crawler:
                result = await crawler.arun(url=url, config=cfg)
                if not result.success:
                    raise RuntimeError(f"{result.status_code} error fetching PDF {url}: {result.error_message}")
        else:
            with self.status_display.create_status(f"[dim]Initializing...[/dim]") as status:
                granular_logger = _get_granular_logger(status_display=status)
                async with AsyncWebCrawler(crawler_strategy=pdf_crawler_cfg, logger=granular_logger) as crawler:
                    result = await crawler.arun(url=url, config=cfg)
                    if not result.success:
                        raise RuntimeError(f"{result.status_code} error fetching PDF {url}: {result.error_message}")

        final_url = result.url
        raw_markdown_content = result.markdown.raw_content
        markdown_content = self._refine_article_content(raw_markdown_content)
        raw_content = result.pdf or ''

        return {
            'raw_content': raw_content,
            'raw_markdown_content': raw_markdown_content,
            'markdown_content': markdown_content,
            'content_type': 'application/pdf',
            'final_url': final_url,
            'doi': ''
        }

    def _extract_final_url(self, result, original_url: str) -> str:
        """Extract the final URL after any redirects."""
        final_url = original_url
        if hasattr(result, '_results') and result._results:
            first_result = result._results[0]
            if hasattr(first_result, 'redirected_url') and first_result.redirected_url:
                final_url = first_result.redirected_url
            elif hasattr(first_result, 'url'):
                final_url = first_result.url

        if final_url == original_url and hasattr(result, 'url'):
            final_url = result.url

        return final_url

    def _extract_doi(self, result) -> str:
        """Extract DOI from crawl4ai extracted content."""
        doi = ''
        try:
            if hasattr(result, 'extracted_content') and result.extracted_content:
                extracted_data = json.loads(result.extracted_content)
                for doi_data in extracted_data:

                    for field in ["doi_meta_pub", "doi_meta_cite", "doi_dc_doi", "doi_dc"]:
                        if field in doi_data and doi_data[field]:
                            doi = doi_data[field]
                            break

                    if not doi and "doi_canonical" in doi_data and doi_data["doi_canonical"]:
                        canonical_url = doi_data["doi_canonical"]
                        if 'doi.org/' in canonical_url:
                            doi = canonical_url.split('doi.org/')[1]

                    if doi:
                        break
        except (json.JSONDecodeError, AttributeError, KeyError, IndexError):
            pass
        return doi

    def _refine_article_content(self, markdown: str) -> str:
        """
        Refine article content by removing irrelevant sections using intelligent heading analysis.

        Delegates to the global refine_article_content function.
        """
        return refine_article_content(markdown)

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file based on extension."""
        from urllib.parse import urlparse
        return urlparse(url).path.lower().endswith('.pdf')

    def _create_chunks(self, markdown_content: str) -> List[str]:
        """
        Chunk markdown content and return as list of strings.

        Args:
            markdown_content: The markdown content to chunk

        Returns:
            List of chunk strings
        """
        chunker = _get_chunker()

        # Chunk the content
        chunks = chunker(markdown_content)

        # Return just the text content of each chunk
        return [chunk.text for chunk in chunks]

    async def _fetch_and_cache(self, url: str, progress=True, progress_info=None) -> dict[str, str]:
        """
        Fetch content from URL and cache it based on content type.

        Args:
            url: URL to fetch
            progress: True for status display, False for silent, tuple for progress bar integration
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Dict containing content information
        """
        is_pdf = self._is_pdf_url(url)

        if progress is True:
            if is_pdf:
                content_info = await self._fetch_pdf_url(url)
                content_info['source_type'] = 'pdf'
            else:
                content_info = await self._fetch_html_url(url)
                content_info['source_type'] = 'html'
        elif progress is False:
            old_show_status = self.status_display.show_status
            self.status_display.show_status = False
            try:
                if is_pdf:
                    content_info = await self._fetch_pdf_url(url)
                    content_info['source_type'] = 'pdf'
                else:
                    content_info = await self._fetch_html_url(url)
                    content_info['source_type'] = 'html'
            finally:
                self.status_display.show_status = old_show_status
        else:
            if is_pdf:
                content_info = await self._fetch_pdf_url(url, progress_info=progress)
                content_info['source_type'] = 'pdf'
            else:
                content_info = await self._fetch_html_url(url, progress_info=progress)
                content_info['source_type'] = 'html'

        final_url = content_info.get('final_url', url)

        with self.status_display.create_status(f"[yellow]Caching content[/yellow]", progress_info):
            if is_pdf:
                await self.cache.set_content(url, "pdf", content_info['raw_content'], final_url)
            else:
                await self.cache.set_content(url, "html", content_info['raw_content'], final_url)

            await self.cache.set_content(url, "markdown", content_info['markdown_content'], final_url)

            # Store raw markdown if available (for both HTML and PDF sources)
            if content_info.get('raw_markdown_content'):
                await self.cache.set_content(url, "raw_markdown", content_info['raw_markdown_content'], final_url)

            if content_info.get('doi'):
                await self.cache.set_content(url, "doi", content_info['doi'], final_url)

        return content_info

    async def get_html(self, url: Union[str, List[str]], progress=True) -> Union[str, List[str]]:
        """
        Get HTML content for single URL or multiple URLs with concurrent fetching.

        Args:
            url: Single URL string or list of URLs to fetch
            progress: True for status display, False for silent, tuple for progress bar integration

        Returns:
            Single content string for single URL, list of content strings for multiple URLs
        """
        if isinstance(url, str):
            if not await self.cache.has_path(url, "html"):
                await self._fetch_and_cache(url, progress=progress)
            return await self.cache.get_content(url, "html")

        elif isinstance(url, list):
            if len(url) == 0:
                return []

            results = await self._fetch_multiple(url, "html", progress=progress)
            if isinstance(progress, tuple):
                # Custom progress returns content directly
                return results

            return self._process_multiple_results(results, progress)

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    async def get_pdf(self, url: Union[str, List[str]], progress=True) -> Union[str, List[str]]:
        """
        Get PDF content for single URL or multiple URLs with concurrent fetching.

        Args:
            url: Single URL string or list of URLs to fetch
            progress: True for status display, False for silent, tuple for progress bar integration

        Returns:
            Single content string for single URL, list of content strings for multiple URLs
        """
        if isinstance(url, str):
            if not await self.cache.has_path(url, "pdf"):
                await self._fetch_and_cache(url, progress=progress)
            return await self.cache.get_content(url, "pdf")

        elif isinstance(url, list):
            if len(url) == 0:
                return []

            results = await self._fetch_multiple(url, "pdf", progress=progress)
            if isinstance(progress, tuple):
                # Custom progress returns content directly
                return results

            return self._process_multiple_results(results, progress)

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    async def get_markdown(self, url: Union[str, List[str]], progress=True) -> Union[str, List[str]]:
        """
        Get Markdown content for single URL or multiple URLs with concurrent fetching.
        Automatically detects and handles both HTML and PDF URLs.

        Args:
            url: Single URL string or list of URLs to fetch
            progress: True for status display, False for silent, tuple for progress bar integration

        Returns:
            Single content string for single URL, list of content strings for multiple URLs
        """
        if isinstance(url, str):
            if await self.cache.has_path(url, "markdown"):
                return await self.cache.get_path(url, "markdown")

            # Check if HTML already exists - if so, convert it instead of re-fetching
            if await self.cache.has_path(url, "html"):
                html_content = await self.cache.get_content(url, "html")

                # Create a temporary file with the HTML content
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
                    temp_file.write(html_content)
                    temp_file_path = temp_file.name

                try:
                    # Convert the file:// URL to markdown using existing infrastructure
                    file_url = f"file://{temp_file_path}"
                    content_info = await self._fetch_html_url(file_url, progress_info=progress if isinstance(progress, tuple) else None)

                    # Get redirect info if this URL was originally redirected
                    final_url = await self.cache.get_redirect_info(url)

                    # Store the markdown
                    await self.cache.set_path(url, "markdown", content_info['markdown_content'], final_url)

                    # Also store the raw (pre-processed) markdown sidecar if available
                    if content_info.get('raw_markdown_content'):
                        await self.cache.set_content(url, "raw_markdown", content_info['raw_markdown_content'], final_url)

                    return content_info['markdown_content']
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
            else:
                # No HTML cached, fetch from scratch
                await self._fetch_and_cache(url, progress=progress)
                return await self.cache.get_path(url, "markdown")

        elif isinstance(url, list):
            if len(url) == 0:
                return []

            # Check which URLs need fetching vs are already cached
            urls_to_fetch = []
            results = {}

            for u in url:
                if await self.cache.has_path(u, "markdown"):
                    # Already cached, get from cache
                    try:
                        results[u] = await self.cache.get_path(u, "markdown")
                    except KeyError:
                        urls_to_fetch.append(u)
                else:
                    urls_to_fetch.append(u)

            # Fetch any URLs that aren't cached
            if urls_to_fetch:
                fetch_results = await self._fetch_multiple(urls_to_fetch, "markdown", progress=progress)

                # Handle different return formats
                if isinstance(progress, tuple):
                    # Custom progress returns content directly
                    for i, u in enumerate(urls_to_fetch):
                        if i < len(fetch_results):
                            results[u] = fetch_results[i]
                else:
                    # Default/silent progress returns result dictionaries
                    for i, u in enumerate(urls_to_fetch):
                        if i < len(fetch_results) and not isinstance(fetch_results[i], Exception):
                            if isinstance(fetch_results[i], dict) and "content" in fetch_results[i]:
                                results[u] = fetch_results[i]["content"]
                            else:
                                results[u] = fetch_results[i]

            # Return results in original order
            return [results[u] for u in url if u in results]

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    async def get_raw(self, url: str) -> str:
        """Get raw content (HTML or PDF) for URL, fetching if necessary."""
        # Check cache first
        source_content = await self.cache.get_source_content(url)
        if source_content is not None:
            return source_content

        # Not cached, fetch it
        content_info = await self._fetch_and_cache(url)
        return content_info['raw_content']

    def _process_multiple_results(self, results: List, progress) -> List:
        """
        Process mixed results from _fetch_multiple, filtering out Exception objects.

        Args:
            results: List containing mix of Exception objects and success data
            progress: Progress mode (tuple means custom progress with direct content)

        Returns:
            List of successful content only
        """
        if isinstance(progress, tuple):
            # Custom progress returns content directly, just filter exceptions
            return [result for result in results if not isinstance(result, Exception)]

        # Default/silent progress returns dicts or exceptions
        successful_content = []
        for result in results:
            if not isinstance(result, Exception):
                if isinstance(result, dict) and "content" in result:
                    successful_content.append(result["content"])
                else:
                    # Handle other success formats
                    successful_content.append(result)
        return successful_content



    async def get_raw_markdown(self, url: str) -> str:
        """Get raw (pre-cleaned) markdown content for URL, fetching if necessary."""
        # Check cache first
        if await self.cache.has_path(url, "raw_markdown"):
            return await self.cache.get_path(url, "raw_markdown")

        # Not cached, fetch it
        content_info = await self._fetch_and_cache(url)
        return content_info.get('raw_markdown_content', '')

    async def is_cached(self, url: str) -> bool:
        """Check if URL is cached (any content type)."""
        return await self.cache.has_url(url)

    async def get_source_type(self, url: str) -> Optional[str]:
        """Get the source content type for a cached URL."""
        return await self.cache.get_source_type(url)

    async def clear_cache(self, url: str) -> None:
        """Clear cached content for URL."""
        await self.cache.clear_url(url)

    async def get_chunks(self, url: Union[str, List[str]], progress=True) -> Union[List[str], List[List[str]]]:
        """
        Get chunked content for single URL or multiple URLs as lists of strings.
        Automatically fetches markdown first if not cached, then chunks it.

        Args:
            url: Single URL string or list of URLs to fetch
            progress: True for status display, False for silent, tuple for progress bar integration

        Returns:
            Single list of strings for single URL, list of lists of strings for multiple URLs
        """
        if isinstance(url, str):
            # Check if chunks are already cached
            if await self.cache.has_path(url, "chunks"):
                return await self.cache.get_content(url, "chunks")

            # Get markdown content first
            markdown_content = await self.get_markdown(url, progress=progress)

            # Get final URL from redirect info if it exists
            final_url = await self.cache.get_redirect_info(url)

            # Chunk it and cache
            with self.status_display.create_status(f"[blue]Chunking content[/blue]") as status:
                chunks = self._create_chunks(markdown_content)
                await self.cache.set_content(url, "chunks", chunks, final_url)

            return chunks

        elif isinstance(url, list):
            if len(url) == 0:
                return []

            results = await self._fetch_multiple(url, "chunks", progress=progress)
            return self._process_multiple_results(results, progress)

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    async def prefetch(self, urls: list[str], progress=True) -> None:
        """
        Prefetch multiple URLs for faster subsequent access.

        Args:
            urls: List of URLs to prefetch
            progress: True for status display, False for silent mode
        """
        for url in urls:
            try:
                await self.get_markdown(url, progress=progress)
            except Exception as e:
                print(f"Failed to prefetch {url}: {e}")

    async def get_doi(self, url: str) -> Optional[str]:
        """
        Get DOI for URL if available.

        Args:
            url: URL to get DOI for

        Returns:
            DOI string if found, None otherwise
        """
        if not await self.cache.has_path(url, "doi") and not await self.cache.has_url(url):
            await self._fetch_and_cache(url)
        return await self.cache.get_content(url, "doi")

    async def _get_content_by_type(self, url: str, content_type: str, progress_info=None) -> Any:
        """Get content of specified type, fetching if necessary."""
        if content_type == "html":
            if not await self.cache.has_path(url, "html"):
                await self._fetch_and_cache(url, progress=progress_info)
            return await self.cache.get_path(url, "html")
        elif content_type == "pdf":
            if not await self.cache.has_path(url, "pdf"):
                await self._fetch_and_cache(url, progress=progress_info)
            return await self.cache.get_path(url, "pdf")
        elif content_type == "markdown":
            if await self.cache.has_path(url, "markdown"):
                return await self.cache.get_path(url, "markdown")
            else:
                # Fetch appropriate content type to generate markdown
                if self._is_pdf_url(url):
                    if not await self.cache.has_path(url, "pdf"):
                        await self._fetch_and_cache(url, progress=progress_info)
                else:
                    if not await self.cache.has_path(url, "html"):
                        await self._fetch_and_cache(url, progress=progress_info)
                return await self.cache.get_path(url, "markdown")
        elif content_type == "chunks":
            if await self.cache.has_path(url, "chunks"):
                return await self.cache.get_content(url, "chunks")
            else:
                # Get markdown content first
                if await self.cache.has_path(url, "markdown"):
                    markdown_content = await self.cache.get_path(url, "markdown")
                else:
                    # Fetch appropriate content type to generate markdown
                    if self._is_pdf_url(url):
                        if not await self.cache.has_path(url, "pdf"):
                            await self._fetch_and_cache(url, progress=progress_info)
                    else:
                        if not await self.cache.has_path(url, "html"):
                            await self._fetch_and_cache(url, progress=progress_info)
                    markdown_content = await self.cache.get_path(url, "markdown")

                # Chunk the markdown content
                chunks = self._create_chunks(markdown_content)
                await self.cache.set_content(url, "chunks", chunks)
                return chunks
        else:
            return await self.get_raw(url)

    def _calculate_content_size(self, content: Any, content_type: str) -> int:
        """Calculate appropriate size for different content types."""
        if content_type == "chunks":
            return sum(len(chunk) for chunk in content)
        return len(content)

    async def _fetch_multiple(self, urls: List[str], content_type: str, progress=None, max_concurrent: int = 5) -> Union[List[dict], List[str]]:
        """
        Unified method to fetch multiple URLs with different progress modes.

        Args:
            urls: List of URLs to fetch
            content_type: Type of content ("html", "pdf", "markdown", "chunks")
            progress: True for default progress, False for silent, tuple for custom progress
            max_concurrent: Maximum concurrent fetches

        Returns:
            List of result dictionaries for default progress, list of content for custom progress
        """
        if progress is True:
            # Default progress with concurrent processing
            return await fetch_urls_concurrent_with_progress(
                urls,
                self.config,
                content_type=content_type,
                max_concurrent=max_concurrent
            )
        elif progress is False:
            # Silent concurrent processing
            import asyncio
            semaphore = asyncio.Semaphore(max_concurrent)

            async def fetch_one(url: str):
                async with semaphore:
                    try:
                        content = await self._get_content_by_type(url, content_type, progress_info=False)
                        size = self._calculate_content_size(content, content_type)
                        return {
                            "url": url,
                            "content": content,
                            "size": size,
                            "status": "success"
                        }
                    except Exception as e:
                        return e

            tasks = [fetch_one(url) for url in urls]
            return await asyncio.gather(*tasks)
        else:
            # Custom progress (sequential processing)
            progress_instance, task_id = progress
            results = []

            for i, url in enumerate(urls):
                try:
                    domain = url.split('//')[1].split('/')[0] if '//' in url else url
                    progress_instance.update(task_id, description=f"[blue]Fetching {content_type} from[/blue] [bold]{domain}[/bold] ({i+1}/{len(urls)})")

                    # Handle chunking progress separately
                    if content_type == "chunks" and not await self.cache.has_path(url, "chunks"):
                        content = await self._get_content_by_type(url, content_type, progress)
                        progress_instance.update(task_id, description=f"[blue]Chunking content from[/blue] [bold]{domain}[/bold] ({i+1}/{len(urls)})")
                    else:
                        content = await self._get_content_by_type(url, content_type, progress)

                    results.append(content)
                except Exception as e:
                    # Return exception directly instead of hiding it
                    results.append(e)

            return results
async def fetch_urls_with_progress(
    urls: list[str],
    config: 'IfetcherConfig',
    content_type: str = "html",
    progress_description: str = "Fetching URLs..."
) -> list[dict]:
    """
    Fetch multiple URLs with clean progress display.

    Args:
        urls: List of URLs to fetch
        config: Configuration for PageFetcher
        content_type: Type of content to fetch ("html", "pdf", "markdown")
        progress_description: Description for progress bar

    Returns:
        List of result dicts with url, content, and status
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    import math
    from urllib.parse import urlparse

    # Use direct method calls instead of string lookup
    fetcher = PageFetcher(config)

    # Calculate optimal domain width using 75th percentile for better preservation
    domains = [urlparse(url).netloc for url in urls]
    domain_lengths = [len(domain) for domain in domains if domain]
    if domain_lengths:
        import statistics
        percentile_75 = statistics.quantiles(domain_lengths, n=4)[2] if len(domain_lengths) > 1 else domain_lengths[0]
        optimal_width = min(20, int(percentile_75))
    else:
        optimal_width = 15

    def format_domain(url: str) -> str:
        domain = urlparse(url).netloc
        if len(domain) <= optimal_width:
            return domain.ljust(optimal_width)
        else:
            return domain[:optimal_width-1] + "…"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="blue", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:

        task = progress.add_task(progress_description, total=len(urls))
        fetcher = PageFetcher(config, show_status=False)

        for url in urls:
            try:
                if content_type == "html":
                    content = await fetcher.get_html(url, progress=(progress, task))
                elif content_type == "pdf":
                    content = await fetcher.get_pdf(url, progress=(progress, task))
                elif content_type == "markdown":
                    content = await fetcher.get_markdown(url, progress=(progress, task))
                elif content_type == "chunks":
                    content = await fetcher.get_chunks(url, progress=(progress, task))
                else:
                    content = await fetcher.get_raw(url)

                # Calculate size appropriately based on content type
                if content_type == "chunks":
                    size = sum(len(chunk) for chunk in content)
                else:
                    size = len(content)

                results.append({
                    "url": url,
                    "content": content,
                    "size": size,
                    "status": "success"
                })

                formatted_domain = format_domain(url)
                progress.update(task, description=f"[green]✓ {formatted_domain} ({len(content):,} chars)")

            except Exception as e:
                results.append({
                    "url": url,
                    "error": str(e),
                    "status": "error"
                })
                domain = url.split('//')[1].split('/')[0] if '//' in url else url
                progress.update(task, description=f"[red]✗ {domain} (failed)")

            progress.advance(task)

    return results

async def fetch_urls_concurrent_with_progress(
    urls: list[str],
    config: 'IfetcherConfig',
    content_type: str = "html",
    max_concurrent: int = 5
) -> list[dict]:
    """
    Fetch multiple URLs concurrently with progress display.

    Args:
        urls: List of URLs to fetch
        config: Configuration for PageFetcher
        content_type: Type of content to fetch
        max_concurrent: Maximum concurrent fetches

    Returns:
        List of result dicts
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    import asyncio
    import math
    from urllib.parse import urlparse

    # Use direct method calls instead of string lookup

    # Calculate optimal domain width using 75th percentile for better preservation
    domains = [urlparse(url).netloc for url in urls]
    domain_lengths = [len(domain) for domain in domains if domain]
    if domain_lengths:
        import statistics
        percentile_75 = statistics.quantiles(domain_lengths, n=4)[2] if len(domain_lengths) > 1 else domain_lengths[0]
        optimal_width = min(20, int(percentile_75))
    else:
        optimal_width = 15

    def format_domain(url: str) -> str:
        domain = urlparse(url).netloc
        if len(domain) <= optimal_width:
            return domain.ljust(optimal_width)
        else:
            return domain[:optimal_width-1] + "…"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="blue", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:

        task = progress.add_task("Fetching...", total=len(urls))
        completed = {"count": 0}

        async def fetch_one(url: str):
            fetcher = PageFetcher(config, show_status=False)
            formatted_domain = format_domain(url)

            try:
                if content_type == "html":
                    content = await fetcher.get_html(url, progress=False)
                elif content_type == "pdf":
                    content = await fetcher.get_pdf(url, progress=False)
                elif content_type == "markdown":
                    content = await fetcher.get_markdown(url, progress=False)
                elif content_type == "chunks":
                    content = await fetcher.get_chunks(url, progress=False)
                else:
                    content = await fetcher.get_raw(url)

                completed["count"] += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"[blue]Fetching... ({completed['count']}/{len(urls)}) {formatted_domain}"
                )

                # Calculate size appropriately based on content type
                if content_type == "chunks":
                    size = sum(len(chunk) for chunk in content)
                else:
                    size = len(content)

                return {
                    "url": url,
                    "content": content,
                    "size": size,
                    "status": "success"
                }

            except Exception as e:
                completed["count"] += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"[blue]Fetching... ({completed['count']}/{len(urls)}) - Failed: {formatted_domain}"
                )

                return e

        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_limit(url):
            async with semaphore:
                return await fetch_one(url)

        tasks = [fetch_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks)

    return results

    async def list_cached_urls(self) -> list[str]:
        """Get a list of all cached URLs."""
        return await self.cache.list_cached_urls()

    async def get_doi(self, url: str) -> Optional[str]:
        """Get DOI for URL if available."""
        if not await self.cache.has_path(url, "doi") and not await self.cache.has_url(url):
            # Try to fetch and extract DOI
            await self._fetch_and_cache(url)

        return await self.cache.get_content(url, "doi")
