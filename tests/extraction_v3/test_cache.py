"""
Tests for V3 semantic caching functionality.

Tests cache manager, cache key computation, and cache effectiveness.
"""

import pytest

from interaction_finder.extraction_graph_v3.cache import (
    SemanticCacheManager,
    compute_extraction_cache_key,
    compute_assessment_cache_key,
)
from interaction_finder.extraction_graph_v2.models import EntityWithQuotes
from interaction_finder.resources import ResourcePool


@pytest.fixture
def cache_dir(tmp_path):
    """Create temporary cache directory."""
    return tmp_path / "cache"


@pytest.fixture
def cache_manager(cache_dir):
    """Create cache manager instance."""
    return SemanticCacheManager(cache_dir)


@pytest.fixture
def sample_entities(tmp_path):
    """Create sample entities with quotes for caching."""
    resource_pool = ResourcePool()
    resource = resource_pool.add(
        "http://example.com/doc1",
        "Test Document",
        "BRCA1 is a tumor suppressor gene associated with breast cancer.",
    )

    entity1 = EntityWithQuotes(
        name="BRCA1",
        kind="gene",
        quotes=[resource.quote("BRCA1 is a tumor suppressor gene")],
    )

    entity2 = EntityWithQuotes(
        name="breast cancer",
        kind="disease",
        quotes=[resource.quote("breast cancer")],
    )

    return [entity1, entity2]


def test_cache_key_computation_stable():
    """Test cache keys are stable for same inputs."""
    doc_text = "Sample document text here"
    entity_kinds = ["gene", "disease"]
    model_version = "openai:gpt-4o-mini"
    prompt_version = "v3"

    key1 = compute_extraction_cache_key(
        doc_text, entity_kinds, model_version, prompt_version
    )
    key2 = compute_extraction_cache_key(
        doc_text, entity_kinds, model_version, prompt_version
    )

    assert key1 == key2, "Cache keys should be stable for identical inputs"


def test_cache_key_computation_differs_on_text():
    """Test cache keys differ when document text changes."""
    entity_kinds = ["gene", "disease"]
    model_version = "openai:gpt-4o-mini"
    prompt_version = "v3"

    key1 = compute_extraction_cache_key(
        "Text A", entity_kinds, model_version, prompt_version
    )
    key2 = compute_extraction_cache_key(
        "Text B", entity_kinds, model_version, prompt_version
    )

    assert key1 != key2, "Cache keys should differ for different texts"


def test_cache_key_computation_differs_on_model():
    """Test cache keys differ when model changes."""
    doc_text = "Sample document"
    entity_kinds = ["gene", "disease"]
    prompt_version = "v3"

    key1 = compute_extraction_cache_key(
        doc_text, entity_kinds, "openai:gpt-4o-mini", prompt_version
    )
    key2 = compute_extraction_cache_key(
        doc_text, entity_kinds, "openai:gpt-4o", prompt_version
    )

    assert key1 != key2, "Cache keys should differ for different models"


def test_assessment_cache_key_computation():
    """Test assessment cache keys include entity and context."""
    entity_name = "BRCA1"
    entity_kind = "gene"
    context_text = "BRCA1 is a tumor suppressor gene"
    model_version = "openai:gpt-4o-mini"
    prompt_version = "v3"

    key1 = compute_assessment_cache_key(
        entity_name, entity_kind, context_text, model_version, prompt_version
    )
    key2 = compute_assessment_cache_key(
        entity_name, entity_kind, context_text, model_version, prompt_version
    )

    assert key1 == key2, "Assessment cache keys should be stable"


def test_assessment_cache_key_differs_on_context():
    """Test assessment cache keys differ when context changes."""
    entity_name = "BRCA1"
    entity_kind = "gene"
    model_version = "openai:gpt-4o-mini"
    prompt_version = "v3"

    key1 = compute_assessment_cache_key(
        entity_name, entity_kind, "Context A", model_version, prompt_version
    )
    key2 = compute_assessment_cache_key(
        entity_name, entity_kind, "Context B", model_version, prompt_version
    )

    assert key1 != key2, "Assessment keys should differ for different contexts"


@pytest.mark.asyncio
async def test_cache_manager_save_and_load(cache_manager, sample_entities):
    """Test saving and loading entities from cache."""
    cache_key = "test_key_123"

    # Save entities
    await cache_manager.set_extraction(cache_key, sample_entities)

    # Load entities
    loaded_entities = await cache_manager.get_extraction(cache_key)

    assert loaded_entities is not None, "Cached entities should be loaded"
    assert len(loaded_entities) == len(sample_entities), "Should load all entities"
    assert loaded_entities[0].name == sample_entities[0].name, (
        "Entity name should match"
    )
    assert loaded_entities[0].kind == sample_entities[0].kind, (
        "Entity kind should match"
    )


@pytest.mark.asyncio
async def test_cache_manager_missing_key(cache_manager):
    """Test loading non-existent key returns None."""
    result = await cache_manager.get_extraction("nonexistent_key")
    assert result is None, "Missing cache key should return None"


@pytest.mark.asyncio
async def test_cache_manager_assessment_save_load(cache_manager):
    """Test saving and loading assessment results."""
    from interaction_finder.extraction_graph_v2.models import (
        IndividualAssessment,
        EntityWithQuotes,
    )

    # Create sample assessment
    resource_pool = ResourcePool()
    resource = resource_pool.add(
        "http://example.com/doc1", "Test", "BRCA1 gene content"
    )
    entity = EntityWithQuotes(
        name="BRCA1",
        kind="gene",
        quotes=[resource.quote("BRCA1")],
    )

    assessment = IndividualAssessment(
        entity=entity,
        relationship_potential="high",
        related_entities=["breast cancer"],
        evidence_quotes=[],
        reasoning="BRCA1 is strongly associated with breast cancer",
    )

    cache_key = "assessment_test_key"

    # Save and load
    await cache_manager.set_assessment(cache_key, assessment)
    loaded_assessment = await cache_manager.get_assessment(cache_key)

    assert loaded_assessment is not None, "Cached assessment should be loaded"
    assert loaded_assessment.entity.name == assessment.entity.name, (
        "Entity name should match"
    )
    assert (
        loaded_assessment.relationship_potential == assessment.relationship_potential
    ), "Potential should match"
    assert loaded_assessment.related_entities == assessment.related_entities, (
        "Related entities should match"
    )


@pytest.mark.asyncio
async def test_cache_manager_handles_invalid_data(cache_manager, cache_dir):
    """Test cache manager handles corrupted cache files."""
    # Write invalid JSON to cache
    cache_file = cache_dir / "extractions" / "invalid_key.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("invalid json {{{")

    # Should return None for invalid cache
    result = await cache_manager.get_extraction("invalid_key")
    assert result is None, "Invalid cache should return None"


@pytest.mark.asyncio
async def test_cache_manager_clear(cache_manager, sample_entities, cache_dir):
    """Test clearing cache by removing files."""
    cache_key = "test_key_clear"

    # Save entities
    await cache_manager.set_extraction(cache_key, sample_entities)
    assert await cache_manager.get_extraction(cache_key) is not None

    # Clear cache by removing directory (cache manager has in-memory cache too)
    import shutil

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "extractions").mkdir(exist_ok=True)
    (cache_dir / "assessments").mkdir(exist_ok=True)

    # Also clear in-memory cache
    cache_manager._extraction_cache.clear()

    # Should not be loadable after clear
    assert await cache_manager.get_extraction(cache_key) is None


@pytest.mark.asyncio
async def test_cache_effectiveness_in_practice(cache_manager, tmp_path):
    """
    Test cache reduces redundant work in realistic scenario.

    Simulates extracting from same document twice.
    """
    doc_text = "BRCA1 is a tumor suppressor gene associated with breast cancer."
    entity_kinds = ["gene", "disease"]
    model_version = "openai:gpt-4o-mini"
    prompt_version = "v3"

    # Compute cache key
    cache_key = compute_extraction_cache_key(
        doc_text, entity_kinds, model_version, prompt_version
    )

    # First extraction (cache miss)
    cached_result_1 = await cache_manager.get_extraction(cache_key)
    assert cached_result_1 is None, "First access should be cache miss"

    # Create and save result
    resource_pool = ResourcePool()
    resource = resource_pool.add("http://example.com/doc1", "Test", doc_text)
    entities = [
        EntityWithQuotes(
            name="BRCA1",
            kind="gene",
            quotes=[resource.quote("BRCA1")],
        )
    ]
    await cache_manager.set_extraction(cache_key, entities)

    # Second extraction (cache hit)
    cached_result_2 = await cache_manager.get_extraction(cache_key)
    assert cached_result_2 is not None, "Second access should be cache hit"
    assert len(cached_result_2) == 1, "Should retrieve saved entities"
    assert cached_result_2[0].name == "BRCA1", "Entity name should match"
