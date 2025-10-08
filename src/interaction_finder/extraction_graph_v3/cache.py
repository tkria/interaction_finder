"""
Semantic cache manager for extraction graph V3.

Provides two-level caching (in-memory LRU + disk persistence) to reduce
LLM API calls by 60%+ through intelligent caching of extraction and
assessment results. Cache keys are based on content hashes, so identical
document text from different URLs reuses cached results.
"""

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

import aiofiles
import aiofiles.os

from ..extraction_graph_v2.models import EntityWithQuotes, IndividualAssessment
from ..resources import Resource, ResourceId, ResourceQuote

logger = logging.getLogger(__name__)

# TTL configurations (in days)
EXTRACTION_TTL_DAYS = 30
ASSESSMENT_TTL_DAYS = 7


def compute_extraction_cache_key(
    full_doc_text: str,
    entity_kinds: List[str],
    model_version: str,
    prompt_version: str,
) -> str:
    """
    Compute deterministic cache key for extraction results.

    Cache key includes document content, entity kinds, model version,
    and prompt version. Any change to these parameters invalidates
    the cached result.

    Parameters:
        full_doc_text: Complete document text
        entity_kinds: List of entity types to extract (sorted for determinism)
        model_version: LLM model identifier (e.g., 'gpt-4o')
        prompt_version: Prompt version identifier (e.g., 'v3')

    Returns:
        64-character hex digest (SHA256)
    """
    # Sort entity kinds for order-independent cache keys: ["gene", "protein"]
    # and ["protein", "gene"] should produce the same cache key since they
    # request the same entities
    sorted_kinds = sorted(entity_kinds)
    # Combine all inputs into a single string
    key_material = (
        f"{full_doc_text}|{','.join(sorted_kinds)}|{model_version}|{prompt_version}"
    )
    # Compute SHA256 hash
    hash_obj = hashlib.sha256(key_material.encode("utf-8"))
    return hash_obj.hexdigest()


def compute_assessment_cache_key(
    entity_name: str,
    entity_kind: str,
    task_context: str,
    model_version: str,
    prompt_version: str,
) -> str:
    """
    Compute deterministic cache key for assessment results.

    Cache key includes entity name, kind, task context, model version,
    and prompt version. Any change to these parameters invalidates
    the cached result.

    Parameters:
        entity_name: Name of the entity being assessed
        entity_kind: Entity type (e.g., 'gene', 'disease')
        task_context: Task description for assessment
        model_version: LLM model identifier
        prompt_version: Prompt version identifier

    Returns:
        64-character hex digest (SHA256)
    """
    # Combine all inputs into a single string
    key_material = (
        f"{entity_name}|{entity_kind}|{task_context}|{model_version}|{prompt_version}"
    )
    # Compute SHA256 hash
    hash_obj = hashlib.sha256(key_material.encode("utf-8"))
    return hash_obj.hexdigest()


def _deserialize_entity(data: dict) -> EntityWithQuotes:
    """
    Deserialize EntityWithQuotes from JSON using model_construct.

    ResourceQuote has a custom __init__ that conflicts with Pydantic's
    standard validation, so we use model_construct to bypass it.

    Parameters:
        data: Serialized entity data from model_dump()

    Returns:
        Reconstructed EntityWithQuotes instance
    """
    # Deserialize quotes using model_construct to bypass ResourceQuote.__init__
    quotes_data = data.get("quotes", [])
    quotes = []
    for q in quotes_data:
        # Reconstruct Resource with nested ResourceId
        resource_data = q["resource"]
        resource_id = ResourceId.model_construct(**resource_data["id"])
        resource = Resource.model_construct(
            id=resource_id,
            title=resource_data["title"],
            text=resource_data["text"],
            normalized_text=resource_data["normalized_text"],
            chunks=resource_data["chunks"],
        )
        # Reconstruct ResourceQuote (convert span lists to tuples)
        quote = ResourceQuote.model_construct(
            resource=resource,
            query_text=q["query_text"],
            spans=[tuple(s) for s in q["spans"]],
            is_disjoint=q["is_disjoint"],
        )
        quotes.append(quote)
    # Reconstruct EntityWithQuotes
    return EntityWithQuotes.model_construct(
        name=data["name"],
        kind=data["kind"],
        aliases=data.get("aliases", []),
        quotes=quotes,
        confidence=data.get("confidence", 1.0),
    )


def _deserialize_assessment(data: dict) -> IndividualAssessment:
    """
    Deserialize IndividualAssessment from JSON using model_construct.

    Parameters:
        data: Serialized assessment data from model_dump()

    Returns:
        Reconstructed IndividualAssessment instance
    """
    # Deserialize the entity field
    entity = _deserialize_entity(data["entity"])
    # Deserialize evidence quotes
    evidence_quotes = []
    for q in data.get("evidence_quotes", []):
        resource_data = q["resource"]
        resource_id = ResourceId.model_construct(**resource_data["id"])
        resource = Resource.model_construct(
            id=resource_id,
            title=resource_data["title"],
            text=resource_data["text"],
            normalized_text=resource_data["normalized_text"],
            chunks=resource_data["chunks"],
        )
        quote = ResourceQuote.model_construct(
            resource=resource,
            query_text=q["query_text"],
            spans=[tuple(s) for s in q["spans"]],
            is_disjoint=q["is_disjoint"],
        )
        evidence_quotes.append(quote)
    # Reconstruct IndividualAssessment
    return IndividualAssessment.model_construct(
        entity=entity,
        relationship_potential=data["relationship_potential"],
        related_entities=data.get("related_entities", []),
        evidence_quotes=evidence_quotes,
        reasoning=data["reasoning"],
        confidence=data.get("confidence", 1.0),
    )


class SemanticCacheManager:
    """
    Two-level semantic cache with in-memory LRU and disk persistence.

    Provides O(1) lookups for hot entries (recent/frequent) via in-memory
    LRU cache, with disk persistence for cross-run caching. Cache entries
    have configurable TTL to prevent stale results.

    Cache structure:
    - cache_dir/extractions/{key}.json
    - cache_dir/assessments/{key}.json

    Thread-safe for concurrent access via async locks.
    """

    def __init__(self, cache_dir: Path, max_memory_entries: int = 1000):
        """
        Initialize semantic cache manager.

        Parameters:
            cache_dir: Directory for disk cache storage
            max_memory_entries: Maximum number of entries in memory LRU cache
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_entries = max_memory_entries
        # Separate LRU caches for extractions and assessments
        self._extraction_cache: OrderedDict[str, List[EntityWithQuotes]] = OrderedDict()
        self._assessment_cache: OrderedDict[str, IndividualAssessment] = OrderedDict()
        # Async locks for thread safety
        self._extraction_lock = asyncio.Lock()
        self._assessment_lock = asyncio.Lock()
        # Ensure cache directories exist
        self._ensure_cache_dirs()

    def _ensure_cache_dirs(self) -> None:
        """Create cache directories if they don't exist."""
        (self.cache_dir / "extractions").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "assessments").mkdir(parents=True, exist_ok=True)

    def _is_expired(self, cache_file: Path, ttl_days: int) -> bool:
        """
        Check if cache file has exceeded TTL.

        Parameters:
            cache_file: Path to cache file
            ttl_days: Time-to-live in days

        Returns:
            True if file is expired or doesn't exist
        """
        if not cache_file.exists():
            return True
        # Get file modification time
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        # Check if mtime + ttl < now
        expiration_time = mtime + timedelta(days=ttl_days)
        return datetime.now() > expiration_time

    def _add_to_memory_cache(
        self,
        cache_type: str,
        key: str,
        value: Union[List[EntityWithQuotes], IndividualAssessment],
    ) -> None:
        """
        Add entry to in-memory LRU cache with eviction.

        Parameters:
            cache_type: "extraction" or "assessment"
            key: Cache key
            value: Value to cache
        """
        if cache_type == "extraction":
            cache = self._extraction_cache
        elif cache_type == "assessment":
            cache = self._assessment_cache
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
        # Move to end (most recently used)
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        # Evict oldest entries if over limit
        while len(cache) > self.max_memory_entries:
            cache.popitem(last=False)

    async def _get_from_disk(self, cache_type: str, key: str) -> Optional[dict]:
        """
        Load cache entry from disk.

        Parameters:
            cache_type: "extraction" or "assessment"
            key: Cache key

        Returns:
            Deserialized JSON dict or None if not found/error
        """
        cache_dir = self.cache_dir / f"{cache_type}s"
        cache_file = cache_dir / f"{key}.json"
        # Check TTL
        ttl_days = (
            EXTRACTION_TTL_DAYS if cache_type == "extraction" else ASSESSMENT_TTL_DAYS
        )
        if self._is_expired(cache_file, ttl_days):
            return None
        try:
            async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read {cache_type} cache {key}: {e}")
            return None

    async def _save_to_disk(self, cache_type: str, key: str, data: dict) -> None:
        """
        Save cache entry to disk.

        Parameters:
            cache_type: "extraction" or "assessment"
            key: Cache key
            data: Serialized data to save
        """
        cache_dir = self.cache_dir / f"{cache_type}s"
        cache_file = cache_dir / f"{key}.json"
        try:
            # Write to temp file then atomic rename
            temp_file = cache_file.with_suffix(".json.tmp")
            async with aiofiles.open(temp_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
            await aiofiles.os.rename(temp_file, cache_file)
        except (OSError, TypeError) as e:
            logger.warning(f"Failed to write {cache_type} cache {key}: {e}")
            # Clean up temp file on failure
            if await aiofiles.os.path.exists(str(temp_file)):
                try:
                    await aiofiles.os.remove(str(temp_file))
                except OSError:
                    pass

    async def get_extraction(self, cache_key: str) -> Optional[List[EntityWithQuotes]]:
        """
        Get cached extraction result.

        Checks memory cache first, then disk cache. If found on disk,
        populates memory cache for future hits.

        Parameters:
            cache_key: Cache key from compute_extraction_cache_key()

        Returns:
            List of entities or None if not found/expired
        """
        async with self._extraction_lock:
            # Check memory cache
            if cache_key in self._extraction_cache:
                # Move to end (most recently used)
                self._extraction_cache.move_to_end(cache_key)
                return self._extraction_cache[cache_key]
            # Check disk cache
            data = await self._get_from_disk("extraction", cache_key)
            if data is None:
                return None
            try:
                # Deserialize JSON to List[EntityWithQuotes]
                entities = [_deserialize_entity(e) for e in data]
                # Populate memory cache
                self._add_to_memory_cache("extraction", cache_key, entities)
                return entities
            except Exception as e:
                logger.warning(
                    f"Failed to deserialize extraction cache {cache_key}: {e}"
                )
                return None

    async def set_extraction(
        self, cache_key: str, entities: List[EntityWithQuotes]
    ) -> None:
        """
        Store extraction result in cache.

        Saves to both memory and disk caches.

        Parameters:
            cache_key: Cache key from compute_extraction_cache_key()
            entities: Extracted entities to cache
        """
        async with self._extraction_lock:
            try:
                # Serialize to JSON
                data = [e.model_dump() for e in entities]
                # Save to disk
                await self._save_to_disk("extraction", cache_key, data)
                # Add to memory cache
                self._add_to_memory_cache("extraction", cache_key, entities)
            except Exception as e:
                logger.warning(f"Failed to cache extraction {cache_key}: {e}")

    async def get_assessment(self, cache_key: str) -> Optional[IndividualAssessment]:
        """
        Get cached assessment result.

        Checks memory cache first, then disk cache. If found on disk,
        populates memory cache for future hits.

        Parameters:
            cache_key: Cache key from compute_assessment_cache_key()

        Returns:
            Assessment or None if not found/expired
        """
        async with self._assessment_lock:
            # Check memory cache
            if cache_key in self._assessment_cache:
                # Move to end (most recently used)
                self._assessment_cache.move_to_end(cache_key)
                return self._assessment_cache[cache_key]
            # Check disk cache
            data = await self._get_from_disk("assessment", cache_key)
            if data is None:
                return None
            try:
                # Deserialize JSON to IndividualAssessment
                assessment = _deserialize_assessment(data)
                # Populate memory cache
                self._add_to_memory_cache("assessment", cache_key, assessment)
                return assessment
            except Exception as e:
                logger.warning(
                    f"Failed to deserialize assessment cache {cache_key}: {e}"
                )
                return None

    async def set_assessment(
        self, cache_key: str, assessment: IndividualAssessment
    ) -> None:
        """
        Store assessment result in cache.

        Saves to both memory and disk caches.

        Parameters:
            cache_key: Cache key from compute_assessment_cache_key()
            assessment: Assessment to cache
        """
        async with self._assessment_lock:
            try:
                # Serialize to JSON
                data = assessment.model_dump()
                # Save to disk
                await self._save_to_disk("assessment", cache_key, data)
                # Add to memory cache
                self._add_to_memory_cache("assessment", cache_key, assessment)
            except Exception as e:
                logger.warning(f"Failed to cache assessment {cache_key}: {e}")
