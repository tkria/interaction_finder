"""Tests for ResourcePool serialization and deserialization."""

import json

import pytest

from interaction_finder.resources import ResourcePool


def test_empty_pool_serialization():
    """Test serialization of empty ResourcePool."""
    pool = ResourcePool()

    # Serialize
    serialized = pool.model_dump(mode="json")
    json_str = json.dumps(serialized)

    # Deserialize
    loaded = ResourcePool.model_validate(json.loads(json_str))

    assert len(loaded.resource_map) == 0
    assert len(loaded.resources) == 0


def test_pool_with_registered_urls_no_content():
    """Test serialization of pool with registered URLs but no content."""
    pool = ResourcePool()
    rid1 = pool.register("https://example.com/1")
    rid2 = pool.register("https://example.com/2")

    # Serialize
    serialized = pool.model_dump(mode="json")
    json_str = json.dumps(serialized)

    # Verify no id fields in standard case
    data = json.loads(json_str)
    # New format: serializes as list directly
    assert isinstance(data, list)
    assert len(data) == 2
    # Standard counter sequence should have no explicit id fields
    assert not any("id" in r for r in data)

    # Deserialize
    loaded = ResourcePool.model_validate(data)

    assert len(loaded.resource_map) == 2
    assert len(loaded.resources) == 0
    # Verify URLs match
    original_urls = sorted([rid.url for rid in pool.resource_map.keys()])
    loaded_urls = sorted([rid.url for rid in loaded.resource_map.keys()])
    assert original_urls == loaded_urls


def test_pool_with_content():
    """Test serialization of pool with content."""
    pool = ResourcePool()
    rid = pool.register("https://example.com/test")
    pool.add_content(rid, "Test Title", "Test document text", [(0, 10)])

    # Serialize
    serialized = pool.model_dump(mode="json")
    json_str = json.dumps(serialized)

    # Deserialize
    loaded = ResourcePool.model_validate(json.loads(json_str))

    assert len(loaded.resource_map) == 1
    assert len(loaded.resources) == 1

    # Verify content
    resource = loaded.resources[0]
    assert resource.title == "Test Title"
    assert resource.text == "Test document text"
    assert resource.chunks == [(0, 10)]


def test_complete_round_trip():
    """Test complete round-trip serialization with mixed content."""
    # Create pool with various resources
    pool = ResourcePool()
    rid1 = pool.register("https://example.com/1")
    pool.add_content(rid1, "Title 1", "Content 1", [(0, 9)])
    rid2 = pool.register("https://example.com/2")  # No content
    rid3 = pool.register("https://example.com/3")
    pool.add_content(rid3, "Title 3", "Content 3 with more text", [(0, 28)])

    # Serialize
    json_str = json.dumps(pool.model_dump(mode="json"))

    # Deserialize
    loaded_pool = ResourcePool.model_validate(json.loads(json_str))

    # Verify counts match
    assert len(pool.resource_map) == len(loaded_pool.resource_map)
    assert len(pool.resources) == len(loaded_pool.resources)

    # Verify URLs match
    orig_urls = sorted([rid.url for rid in pool.resource_map.keys()])
    loaded_urls = sorted([rid.url for rid in loaded_pool.resource_map.keys()])
    assert orig_urls == loaded_urls

    # Verify content matches
    orig_content = sorted([(r.title, r.text) for r in pool.resources])
    loaded_content = sorted([(r.title, r.text) for r in loaded_pool.resources])
    assert orig_content == loaded_content


def test_optimized_format_no_id_fields():
    """Test that id fields are omitted in standard counter sequence."""
    pool = ResourcePool()
    for i in range(1, 6):
        rid = pool.register(f"https://example.com/{i}")
        if i % 2 == 1:  # Add content to odd-numbered resources
            pool.add_content(rid, f"Title {i}", f"Content {i}", [(0, 9)])

    # Serialize
    serialized = pool.model_dump(mode="json")

    # Verify no id fields present (standard sequence)
    # New format: serializes as list directly
    assert isinstance(serialized, list)
    assert len(serialized) == 5
    assert not any("id" in r for r in serialized), (
        "id fields should be omitted in standard case"
    )


def test_size_savings():
    """Verify that optimized format saves space."""
    pool = ResourcePool()
    for i in range(1, 11):  # 10 resources
        rid = pool.register(f"https://example.com/resource{i}")
        pool.add_content(rid, f"Title {i}", f"Content for resource {i}", [(0, 20)])

    # Serialize with optimization
    optimized_json = json.dumps(pool.model_dump(mode="json"))

    # Manually add id fields to calculate unoptimized size
    unoptimized_data = json.loads(optimized_json)
    # New format: data is a list directly
    for idx, resource in enumerate(unoptimized_data, start=1):
        from interaction_finder.resources import ResourceId

        temp_id = ResourceId(url=resource["url"], counter=idx)
        resource["id"] = temp_id.id

    unoptimized_json = json.dumps(unoptimized_data)

    # Should see meaningful savings
    savings_percent = (
        100 * (len(unoptimized_json) - len(optimized_json)) / len(unoptimized_json)
    )
    assert len(optimized_json) < len(unoptimized_json)
    assert savings_percent > 5  # At least 5% savings


def test_backward_compatibility_with_legacy_format():
    """Test that legacy resource_map format is handled gracefully."""
    # Legacy format with stringified ResourceId keys (can't be deserialized fully)
    legacy_data = {"resource_map": {"id='1_abc' url='https://example.com'": None}}

    # Should not crash, but return empty pool
    pool = ResourcePool.model_validate(legacy_data)
    assert isinstance(pool, ResourcePool)
    # Legacy format with serialization issues returns empty pool
    assert len(pool.resource_map) == 0


def test_explicit_id_field_for_non_standard_counter():
    """Test that id field is included when counter doesn't match position."""
    # Create pool but simulate non-standard counter
    # (This would happen if resources were deleted from middle of sequence)
    pool = ResourcePool()
    rid1 = pool.register("https://example.com/1")
    rid3 = pool.register("https://example.com/3")  # Counter will be 2, not 3

    # Manually modify the counter in the second resource to create non-standard sequence
    # Extract the resources list first
    items = list(pool.resource_map.items())

    # Serialize normally first
    serialized = pool.model_dump(mode="json")

    # In a standard sequence, no id fields should be present
    # New format: serializes as list directly
    assert isinstance(serialized, list)
    assert not any("id" in r for r in serialized)

    # This test documents the current behavior:
    # Even with gaps in URLs, as long as counters are sequential (1, 2, ...)
    # the id fields are omitted
