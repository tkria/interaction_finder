"""Tests for parallel document renderer."""

from interaction_finder.report.parallel_renderer import render_documents_parallel
from interaction_finder.resources import (
    Resource,
    ResourceId,
    ResourcePool,
    ResourceQuote,
)


def test_parallel_rendering_basic():
    """Test basic parallel rendering with simple documents."""
    # Create test resources
    pool = ResourcePool()

    resources = []
    doc_to_quotes = {}
    doc_to_entities = {}

    for i in range(3):
        resource = pool.add(
            url=f"https://example.com/doc{i}",
            title=f"Test Doc {i}",
            document_text=f"Test content {i} with some **bold** text.",
        )
        resources.append(resource)

        # Add quote for this document
        quote = ResourceQuote(
            resource=resource,
            query_text=f"Test content {i}",
            spans=[(0, 16)],
        )
        doc_to_quotes[resource.id.id] = [quote]

        # Add entity data
        doc_to_entities[resource.id.id] = {
            0: {
                "entity1": {"name": "Entity1", "kind": "type", "aliases": []},
                "entity2": {"name": "Entity2", "kind": "type", "aliases": []},
            }
        }

    # Render in parallel
    documents, document_html = render_documents_parallel(
        resources, doc_to_quotes, doc_to_entities
    )

    # Verify all documents were rendered
    assert len(documents) == 3
    assert len(document_html) == 3

    # Verify structure of results
    for resource in resources:
        doc_id = resource.id.id
        assert doc_id in documents
        assert doc_id in document_html

        # Check metadata structure
        doc_meta = documents[doc_id]
        assert doc_meta["id"] == doc_id
        assert doc_meta["url"] == resource.id.url
        assert doc_meta["title"] == resource.title
        assert "quote_map" in doc_meta
        assert "entity_map" in doc_meta

        # Check HTML was generated
        html = document_html[doc_id]
        assert isinstance(html, str)
        assert len(html) > 0
        assert "<strong>bold</strong>" in html  # Markdown was converted


def test_parallel_rendering_progress_callback():
    """Test that progress callback is called correctly."""
    pool = ResourcePool()

    resources = []
    doc_to_quotes = {}
    doc_to_entities = {}

    for i in range(5):
        resource = pool.add(
            url=f"https://example.com/doc{i}",
            title=f"Test Doc {i}",
            document_text="Simple text",
        )
        resources.append(resource)
        doc_to_quotes[resource.id.id] = []
        doc_to_entities[resource.id.id] = {}

    # Track progress callback invocations
    progress_count = 0

    def progress_callback():
        nonlocal progress_count
        progress_count += 1

    # Render with progress tracking
    documents, document_html = render_documents_parallel(
        resources, doc_to_quotes, doc_to_entities, progress_callback=progress_callback
    )

    # Verify callback was called for each document
    assert progress_count == 5
    assert len(documents) == 5


def test_parallel_rendering_empty_list():
    """Test parallel rendering with empty document list."""
    documents, document_html = render_documents_parallel([], {}, {})

    assert documents == {}
    assert document_html == {}


def test_parallel_rendering_with_complex_markdown():
    """Test parallel rendering preserves complex markdown formatting."""
    pool = ResourcePool()

    markdown_text = """# Header 1

This is a paragraph with **bold** and *italic* text.

## Header 2

- List item 1
- List item 2

Some `code` here."""

    resource = pool.add(
        url="https://example.com/complex",
        title="Complex Doc",
        document_text=markdown_text,
    )

    # Render
    documents, document_html = render_documents_parallel(
        [resource], {resource.id.id: []}, {resource.id.id: {}}
    )

    html = document_html[resource.id.id]

    # Check formatting was preserved
    assert "<h1>Header 1</h1>" in html
    assert "<h2>Header 2</h2>" in html
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html
    assert "<code>code</code>" in html
    assert "<li>List item 1</li>" in html


def test_parallel_rendering_quote_spans():
    """Test that quote spans are correctly annotated in parallel rendering."""
    pool = ResourcePool()

    resource = pool.add(
        url="https://example.com/quotes",
        title="Quote Doc",
        document_text="This is a test document with quoted text.",
    )

    # Create quote spanning "quoted text"
    quote = ResourceQuote(resource=resource, query_text="quoted text", spans=[(30, 41)])

    doc_to_quotes = {resource.id.id: [quote]}
    doc_to_entities = {resource.id.id: {}}

    documents, document_html = render_documents_parallel(
        [resource], doc_to_quotes, doc_to_entities
    )

    # Check quote metadata was created
    doc_meta = documents[resource.id.id]
    assert len(doc_meta["quote_map"]) == 1

    # Check HTML contains quote span (includes document-specific ID)
    html = document_html[resource.id.id]
    assert 'class="quote-span' in html
