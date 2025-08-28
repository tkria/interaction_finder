"""URL caching functionality for the fetcher package."""

import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
import aiofiles
import aiofiles.os

from .utils import normalize_url, url_to_hash_base36

if TYPE_CHECKING:
    from ..settings import IfetcherConfig

# Caching constants
COLLISION_PROBE_LIMIT = 10

# Content type configurations for serialization/deserialization
CONTENT_TYPE_CONFIG = {
    "html": {"extension": "html", "serialize": str, "deserialize": str},
    "pdf": {"extension": "pdf", "serialize": str, "deserialize": str},
    "markdown": {"extension": "md", "serialize": str, "deserialize": str},
    "raw_markdown": {"extension": "raw.md", "serialize": str, "deserialize": str},
    "chunks": {
        "extension": "json",
        "serialize": lambda x: json.dumps(x, indent=2, ensure_ascii=False),
        "deserialize": lambda x: json.loads(x),
    },
    "doi": {"extension": "doi", "serialize": str, "deserialize": lambda x: x.strip()},
}


class URLCache:
    """File-based cache for URLs with support for HTML, PDF, and Markdown content."""

    def __init__(self, config: "IfetcherConfig"):
        self.config = config
        self.base_path = Path(config.abspath(config.output.cache))
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def _get_paths(self, url: str, *extensions: str) -> tuple[Path, ...]:
        """Get file paths for specific extensions with collision resolution."""
        normalized_url = normalize_url(url)
        base_hash = url_to_hash_base36(url)
        probe = 0

        while probe < COLLISION_PROBE_LIMIT:
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
                async with aiofiles.open(url_path, "r", encoding="utf-8") as f:
                    stored_url = (await f.read()).strip()
                if stored_url == normalized_url:
                    # Found existing entry for this URL
                    return tuple(
                        self.base_path / f"{hash_str}.{ext}" for ext in extensions
                    )
            except (OSError, UnicodeDecodeError):
                # Corrupted file, skip this slot
                pass

            probe += 1

        raise RuntimeError(f"Too many hash collisions for URL: {url}")

    async def _store_url_mapping(
        self, url: str, final_url: Optional[str] = None
    ) -> None:
        """Store the URL mapping and redirect info in sidecar files with atomic operations."""
        normalized_url = normalize_url(url)
        url_path, redir_path = await self._get_paths(url, "url", "redir")

        # Atomic file creation: write to temp file then rename
        temp_url_path = url_path.with_suffix(".url.tmp")
        try:
            async with aiofiles.open(temp_url_path, "w", encoding="utf-8") as f:
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
            temp_redir_path = redir_path.with_suffix(".redir.tmp")
            try:
                async with aiofiles.open(temp_redir_path, "w", encoding="utf-8") as f:
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
        (url_path,) = await self._get_paths(url, "url")
        if url_path.exists():
            try:
                async with aiofiles.open(url_path, "r", encoding="utf-8") as f:
                    stored_url = (await f.read()).strip()
                return stored_url == normalized_url
            except (OSError, UnicodeDecodeError):
                return False
        return False

    async def has_path(self, url: str, content_type: str) -> bool:
        """Check if content with given content type is cached for URL (including redirected URLs)."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(
                f"Unknown content type '{content_type}'. Available: {available}"
            )

        extension = CONTENT_TYPE_CONFIG[content_type]["extension"]
        content_path, redir_path = await self._get_paths(url, extension, "redir")
        if content_path.exists():
            if await self._verify_url_mapping(url):
                return True
        # Check if this URL redirected to another URL that has content
        if redir_path.exists():
            try:
                async with aiofiles.open(redir_path, "r", encoding="utf-8") as f:
                    final_url = (await f.read()).strip()
                (final_content_path,) = await self._get_paths(final_url, extension)
                return final_content_path.exists()
            except (OSError, UnicodeDecodeError):
                pass
        return False

    async def get_path(self, url: str, content_type: str) -> str:
        """Get cached content for URL with given content type (following redirects if needed)."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(
                f"Unknown content type '{content_type}'. Available: {available}"
            )

        extension = CONTENT_TYPE_CONFIG[content_type]["extension"]
        content_path, redir_path = await self._get_paths(url, extension, "redir")

        # Check if content exists
        if content_path.exists():
            if await self._verify_url_mapping(url):
                async with aiofiles.open(content_path, "r", encoding="utf-8") as f:
                    return await f.read()

        # Check for redirected content
        if redir_path.exists():
            try:
                async with aiofiles.open(redir_path, "r", encoding="utf-8") as f:
                    final_url = (await f.read()).strip()
                (final_content_path,) = await self._get_paths(final_url, extension)
                if final_content_path.exists():
                    async with aiofiles.open(
                        final_content_path, "r", encoding="utf-8"
                    ) as f:
                        return await f.read()
            except (OSError, UnicodeDecodeError):
                pass

        raise KeyError(
            f"Content with extension '{extension}' not cached for URL: {url}"
        )

    async def set_path(
        self, url: str, content_type: str, content: str, final_url: Optional[str] = None
    ) -> None:
        """Store content for URL and content type with atomic operations."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(
                f"Unknown content type '{content_type}'. Available: {available}"
            )

        extension = CONTENT_TYPE_CONFIG[content_type]["extension"]
        # If there's a redirect, store content at the final URL location
        storage_url = final_url if final_url else url
        (path,) = await self._get_paths(storage_url, extension)

        # Atomic content write
        temp_path = path.with_suffix(f".{extension}.tmp")
        try:
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
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
            raise ValueError(
                f"Unknown content type '{content_type}'. Available: {available}"
            )

        if content_type == "doi":
            try:
                raw_content = await self.get_path(url, content_type)
                return CONTENT_TYPE_CONFIG[content_type]["deserialize"](raw_content)
            except KeyError:
                return None
        else:
            raw_content = await self.get_path(url, content_type)
            return CONTENT_TYPE_CONFIG[content_type]["deserialize"](raw_content)

    async def set_content(
        self, url: str, content_type: str, content, final_url: Optional[str] = None
    ) -> None:
        """Store content for URL with given type."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(
                f"Unknown content type '{content_type}'. Available: {available}"
            )

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
        """Remove all cached content for a URL and its entire redirect chain."""
        # Collect all URLs in the redirect chain
        urls_to_clear = {url}
        current = url
        visited = set()

        # Follow redirect chain forward with safety limit
        while current and current not in visited and len(visited) < 10:
            visited.add(current)
            try:
                redir = await self.get_redirect_info(current)
                if redir and redir != current:  # Avoid self-redirects
                    urls_to_clear.add(redir)
                    current = redir
                else:
                    break
            except (FileNotFoundError, KeyError, OSError):
                # No redirect info available, stop traversal
                break

        # Clear all URLs in the chain
        extensions = [config["extension"] for config in CONTENT_TYPE_CONFIG.values()]
        extensions.extend(["url", "redir", "failed"])  # Add metadata file extensions

        for url_to_clear in urls_to_clear:
            try:
                paths = await self._get_paths(url_to_clear, *extensions)
                for path in paths:
                    if path.exists():
                        path.unlink()
            except (FileNotFoundError, KeyError, OSError):
                # URL may not exist in cache, continue with others
                continue

    async def get_url_hash(self, url: str) -> str:
        """Get the actual hash string used for a URL (including probe suffix if any)."""
        (url_path,) = await self._get_paths(url, "url")
        # Extract hash from the path name
        return url_path.stem  # Remove .url extension to get the hash

    async def get_original_url(self, hash_str: str) -> Optional[str]:
        """Get the original URL from a hash string."""
        url_path = self.base_path / f"{hash_str}.url"
        if url_path.exists():
            try:
                async with aiofiles.open(url_path, "r", encoding="utf-8") as f:
                    return (await f.read()).strip()
            except (OSError, UnicodeDecodeError):
                return None
        return None

    async def get_redirect_info(self, url: str) -> Optional[str]:
        """Get the final URL if this URL redirected, None otherwise."""
        (redir_path,) = await self._get_paths(url, "redir")
        if redir_path.exists():
            try:
                async with aiofiles.open(redir_path, "r", encoding="utf-8") as f:
                    content = (await f.read()).strip()
                # Validate that the content looks like a URL
                if content and "://" in content:
                    return content
            except (OSError, UnicodeDecodeError):
                pass
        return None

    async def get_failed_reason(self, url: str) -> Optional[str]:
        """Read and return the stored failure reason for this exact URL if present."""
        (failed_path,) = await self._get_paths(url, "failed")
        if failed_path.exists():
            try:
                async with aiofiles.open(failed_path, "r", encoding="utf-8") as f:
                    return (await f.read()).strip()
            except (OSError, UnicodeDecodeError):
                return None
        return None

    async def is_failed(self, url: str) -> bool:
        """Check if a previous failure sentinel exists for this exact URL.

        Separation of concerns: redirect resolution should be handled by the
        caller (e.g., PageFetcher), which can then call `is_failed` on the
        resolved URL if desired.
        """
        (failed_path,) = await self._get_paths(url, "failed")
        return failed_path.exists()

    async def mark_failed(
        self, url: str, final_url: Optional[str] = None, reason: Optional[str] = None
    ) -> None:
        """Create/overwrite a .failed sentinel for this URL.

        Writes the plain-text error message to `<hash>.failed` for the provided URL.
        We do not duplicate writes for redirect targets; inheritance is handled at
        read-time by `is_failed()` following the `.redir` chain.
        """
        content = (reason or "").strip()
        (path,) = await self._get_paths(url, "failed")
        tmp = path.with_suffix(".failed.tmp")
        try:
            async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
                await f.write(content)
            await aiofiles.os.rename(tmp, path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    async def clear_failed(self, url: str) -> None:
        """Remove any existing .failed sentinel for this exact URL only."""
        (failed_path,) = await self._get_paths(url, "failed")
        if failed_path.exists():
            failed_path.unlink()

    async def list_cached_urls(self) -> list[str]:
        """Get a list of all cached URLs."""
        urls = []
        for url_file in self.base_path.glob("*.url"):
            try:
                async with aiofiles.open(url_file, "r", encoding="utf-8") as f:
                    url = (await f.read()).strip()
                # Verify at least one content file exists
                if await self.has_url(url):
                    urls.append(url)
            except (OSError, UnicodeDecodeError):
                continue
        return urls

    async def get_content_batch(
        self, urls: List[str], content_type: str
    ) -> List[Tuple[str, Optional[Any]]]:
        """Batch content retrieval - returns (url, content_or_None) pairs."""
        results = []
        for url in urls:
            try:
                content = (
                    await self.get_content(url, content_type)
                    if await self.has_path(url, content_type)
                    else None
                )
                results.append((url, content))
            except Exception:
                results.append((url, None))
        return results
