"""
Tests for document grouping functionality.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, patch
from interaction_finder.fetcher.document_grouper import (
    DocumentGrouper,
    compute_group_cohesion,
)


class TestDocumentGrouper:
    def test_parse_constraint_count(self):
        """Test parsing count constraint."""
        constraint_type, min_val, max_val = DocumentGrouper.parse_constraint(
            "count:3-8"
        )
        assert constraint_type == "count"
        assert min_val == 3
        assert max_val == 8

    def test_parse_constraint_words(self):
        """Test parsing word count constraint."""
        constraint_type, min_val, max_val = DocumentGrouper.parse_constraint(
            "words:5000-15000"
        )
        assert constraint_type == "words"
        assert min_val == 5000
        assert max_val == 15000

    def test_parse_constraint_invalid(self):
        """Test invalid constraint format."""
        with pytest.raises(ValueError):
            DocumentGrouper.parse_constraint("invalid")

    def test_compute_document_embeddings(self):
        """Test computing document embeddings from chunk data."""
        grouper = DocumentGrouper()

        chunk_data = {
            "doc1": [
                {"text": "chunk1", "wordcount": 10, "embedding": [1.0, 2.0, 3.0]},
                {"text": "chunk2", "wordcount": 15, "embedding": [2.0, 3.0, 4.0]},
            ],
            "doc2": [
                {"text": "chunk3", "wordcount": 12, "embedding": [3.0, 4.0, 5.0]},
            ],
        }

        doc_embeddings = grouper._compute_document_embeddings(
            ["doc1", "doc2"], chunk_data
        )

        assert "doc1" in doc_embeddings
        assert "doc2" in doc_embeddings

        # doc1 should have average of [1,2,3] and [2,3,4] = [1.5, 2.5, 3.5]
        np.testing.assert_array_equal(doc_embeddings["doc1"], [1.5, 2.5, 3.5])

        # doc2 should have [3,4,5]
        np.testing.assert_array_equal(doc_embeddings["doc2"], [3.0, 4.0, 5.0])

    def test_compute_word_counts(self):
        """Test computing word counts from chunk data."""
        grouper = DocumentGrouper()

        chunk_data = {
            "doc1": [
                {"text": "chunk1", "wordcount": 10, "embedding": [1.0, 2.0, 3.0]},
                {"text": "chunk2", "wordcount": 15, "embedding": [2.0, 3.0, 4.0]},
            ],
            "doc2": [
                {"text": "chunk3", "wordcount": 12, "embedding": [3.0, 4.0, 5.0]},
            ],
        }

        word_counts = grouper._compute_word_counts(["doc1", "doc2"], chunk_data)

        assert word_counts["doc1"] == 25  # 10 + 15
        assert word_counts["doc2"] == 12

    def test_group_documents_simple(self):
        """Test basic document grouping."""
        grouper = DocumentGrouper()

        # Create test data with similar embeddings for doc1 and doc2, different for doc3
        chunk_data = {
            "doc1": [{"text": "text1", "wordcount": 10, "embedding": [1.0, 0.0, 0.0]}],
            "doc2": [
                {"text": "text2", "wordcount": 12, "embedding": [0.9, 0.1, 0.0]}
            ],  # Similar to doc1
            "doc3": [
                {"text": "text3", "wordcount": 15, "embedding": [0.0, 0.0, 1.0]}
            ],  # Different
        }

        groups = grouper.group_documents(
            documents=["doc1", "doc2", "doc3"],
            chunk_data=chunk_data,
            constraint_type="count",
            min_size=1,
            max_size=3,  # Allow larger groups
        )

        # Should create groups based on similarity
        assert len(groups) >= 1

        # Check that all documents are grouped (or at least doc1 and doc2 are together)
        all_grouped_docs = set()
        for group in groups:
            all_grouped_docs.update(group)

        # Should have at least doc1 and doc2 grouped together
        assert "doc1" in all_grouped_docs
        assert "doc2" in all_grouped_docs

        # Check that similar documents (doc1, doc2) are in the same group
        doc1_group = None
        doc2_group = None
        for i, group in enumerate(groups):
            if "doc1" in group:
                doc1_group = i
            if "doc2" in group:
                doc2_group = i

        # doc1 and doc2 should be in the same group due to similarity
        assert doc1_group == doc2_group

    def test_group_documents_word_constraint(self):
        """Test grouping with word count constraints."""
        grouper = DocumentGrouper()

        chunk_data = {
            "doc1": [{"text": "text1", "wordcount": 100, "embedding": [1.0, 0.0, 0.0]}],
            "doc2": [{"text": "text2", "wordcount": 150, "embedding": [0.9, 0.1, 0.0]}],
            "doc3": [{"text": "text3", "wordcount": 200, "embedding": [0.8, 0.2, 0.0]}],
        }

        groups = grouper.group_documents(
            documents=["doc1", "doc2", "doc3"],
            chunk_data=chunk_data,
            constraint_type="words",
            min_size=100,
            max_size=300,  # doc1+doc2 = 250, doc1+doc2+doc3 = 450 (too much)
        )

        # Should respect word count constraints
        for group in groups:
            total_words = sum(
                sum(chunk["wordcount"] for chunk in chunk_data[doc]) for doc in group
            )
            assert 100 <= total_words <= 300


def test_compute_group_cohesion():
    """Test group cohesion calculation."""
    doc_embeddings = {
        "doc1": np.array([1.0, 0.0, 0.0]),
        "doc2": np.array([0.9, 0.1, 0.0]),  # Similar to doc1
        "doc3": np.array([0.0, 0.0, 1.0]),  # Different
    }

    # High cohesion group (similar documents)
    cohesion_similar = compute_group_cohesion(["doc1", "doc2"], doc_embeddings)
    assert cohesion_similar > 0.8

    # Low cohesion group (dissimilar documents)
    cohesion_dissimilar = compute_group_cohesion(["doc1", "doc3"], doc_embeddings)
    assert cohesion_dissimilar < 0.5

    # Single document should have cohesion 1.0
    cohesion_single = compute_group_cohesion(["doc1"], doc_embeddings)
    assert cohesion_single == 1.0


@pytest.mark.asyncio
class TestPageFetcherGrouping:
    """Test PageFetcher grouping integration."""

    async def test_get_chunks_with_embeddings_cached(self):
        """Test getting chunks with embeddings from cache."""
        from interaction_finder.fetcher.page_fetcher import PageFetcher
        from interaction_finder import IfetcherConfig

        config = IfetcherConfig()
        fetcher = PageFetcher(config)

        # Mock the cache to return chunk objects format
        fetcher.cache.has_path = AsyncMock(return_value=True)
        fetcher.cache.get_content = AsyncMock(
            return_value={
                "version": 2,
                "chunks": [
                    {"text": "chunk1", "wordcount": 10, "embedding": [1.0, 2.0, 3.0]},
                    {"text": "chunk2", "wordcount": 15, "embedding": [2.0, 3.0, 4.0]},
                ],
                "metadata": {"model": "test"},
            }
        )

        chunks = await fetcher.get_chunks_with_embeddings("test_url")

        assert len(chunks) == 2
        assert chunks[0]["text"] == "chunk1"
        assert chunks[0]["wordcount"] == 10
        assert chunks[0]["embedding"] == [1.0, 2.0, 3.0]

    async def test_get_chunks_returns_strings(self):
        """Test that get_chunks returns strings from chunk objects."""
        from interaction_finder.fetcher.page_fetcher import PageFetcher
        from interaction_finder import IfetcherConfig

        config = IfetcherConfig()
        fetcher = PageFetcher(config)

        # Mock get_chunks_with_embeddings
        fetcher.get_chunks_with_embeddings = AsyncMock(
            return_value=[
                {"text": "chunk1", "wordcount": 10, "embedding": [1.0, 2.0, 3.0]},
                {"text": "chunk2", "wordcount": 15, "embedding": [2.0, 3.0, 4.0]},
            ]
        )

        chunks = await fetcher.get_chunks("test_url")

        # Should return just the text strings
        assert chunks == ["chunk1", "chunk2"]
