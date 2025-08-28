"""
Comprehensive tests for chunking functionality.

Tests cover:
- WebClient.create_chunks() method with weighted averaging
- Position and length-based weighting logic
- Chonkie integration and SemanticChunker behavior
- Edge cases and error conditions
- Dynamic embedding dimensions
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from interaction_finder.fetcher.web_client import WebClient, _get_chunker


class MockSentence:
    """Mock sentence object with embedding."""

    def __init__(self, text: str, embedding: list = None):
        self.text = text
        self.embedding = np.array(embedding) if embedding else None


class MockChunk:
    """Mock chunk object from chonkie."""

    def __init__(self, text: str, sentences: list = None):
        self.text = text
        self.sentences = sentences or []


class TestChunkCreation:
    """Test basic chunk creation functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30
        return config

    @pytest.fixture
    def web_client(self, mock_config):
        """Create WebClient instance with mock config."""
        return WebClient(mock_config, verbose=False)

    def test_create_chunks_basic_functionality(self, web_client):
        """Test that create_chunks returns properly structured data."""
        # Mock the chunker
        mock_sentences = [
            MockSentence("First sentence about genes.", [1.0, 0.0, 0.0]),
            MockSentence("Second sentence about proteins.", [0.0, 1.0, 0.0]),
        ]
        mock_chunks = [
            MockChunk(
                "First sentence about genes. Second sentence about proteins.",
                mock_sentences,
            )
        ]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test markdown content.")

            assert len(result) == 1
            chunk = result[0]

            # Check basic structure
            assert "text" in chunk
            assert "wordcount" in chunk
            assert "embedding" in chunk

            # Check values
            assert (
                chunk["text"]
                == "First sentence about genes. Second sentence about proteins."
            )
            assert chunk["wordcount"] == 8  # Word count of the text
            assert chunk["embedding"] is not None
            assert len(chunk["embedding"]) == 3  # Embedding dimension

    def test_create_chunks_no_embeddings(self, web_client):
        """Test chunking when sentences have no embeddings."""
        mock_sentences = [
            MockSentence("First sentence.", None),
            MockSentence("Second sentence.", None),
        ]
        mock_chunks = [MockChunk("First sentence. Second sentence.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test content.")

            assert len(result) == 1
            assert result[0]["embedding"] is None

    def test_create_chunks_empty_content(self, web_client):
        """Test chunking with empty content."""
        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = []
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("")

            assert result == []


class TestWeightedAveraging:
    """Test weighted averaging logic for chunk embeddings."""

    @pytest.fixture
    def web_client(self):
        """Create WebClient instance."""
        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30
        return WebClient(config, verbose=False)

    def test_position_weighting_first_last_sentences(self, web_client):
        """Test that first and last sentences get higher weights."""
        # Create 5 sentences with distinct embeddings
        mock_sentences = [
            MockSentence("First important sentence.", [1.0, 0.0, 0.0]),  # First 20%
            MockSentence("Middle sentence one.", [0.0, 1.0, 0.0]),  # Middle
            MockSentence("Middle sentence two.", [0.0, 0.0, 1.0]),  # Middle
            MockSentence("Middle sentence three.", [1.0, 1.0, 0.0]),  # Middle
            MockSentence("Last important sentence.", [0.0, 0.0, 2.0]),  # Last 20%
        ]
        mock_chunks = [MockChunk("Combined text.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test content.")

            # The result should be influenced more by first and last sentences
            # due to their higher position weights (1.5x)
            embedding = np.array(result[0]["embedding"])
            assert embedding is not None
            assert len(embedding) == 3

            # First sentence [1,0,0] and last [0,0,2] should have more influence
            # than middle sentences due to position weighting
            assert embedding[0] > 0  # Influenced by first sentence
            assert embedding[2] > embedding[1]  # Last sentence influence > middle

    def test_length_weighting_longer_sentences(self, web_client):
        """Test that longer sentences get higher weights."""
        # Create sentences with different lengths
        mock_sentences = [
            MockSentence("Short.", [1.0, 0.0, 0.0]),  # Length: 6
            MockSentence(
                "This is a much longer sentence with more words and content.",
                [0.0, 1.0, 0.0],
            ),  # Length: 66
        ]
        mock_chunks = [MockChunk("Combined.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test content.")

            embedding = np.array(result[0]["embedding"])

            # Longer sentence should have more influence due to length weighting
            # Second sentence is much longer, so embedding[1] should be > embedding[0]
            assert embedding[1] > embedding[0]

    def test_combined_position_and_length_weighting(self, web_client):
        """Test combined position and length weighting."""
        # Create scenario where position and length weights interact
        mock_sentences = [
            MockSentence(
                "First sentence is quite long with multiple words and detailed content.",
                [1.0, 0.0, 0.0],
            ),  # First + Long
            MockSentence("Mid short.", [0.0, 1.0, 0.0]),  # Middle + Short
            MockSentence(
                "Last sentence is also very long and contains significant content.",
                [0.0, 0.0, 1.0],
            ),  # Last + Long
        ]
        mock_chunks = [MockChunk("Combined.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test content.")

            embedding = np.array(result[0]["embedding"])

            # First and last sentences should dominate due to:
            # - Position weight: 1.5x (vs 1.0x for middle)
            # - Length weight: much higher (long sentences vs short)
            # Middle sentence should have minimal influence
            assert embedding[1] < embedding[0]  # Middle < First
            assert embedding[1] < embedding[2]  # Middle < Last

    def test_weight_normalization(self, web_client):
        """Test that weights are properly normalized."""
        mock_sentences = [
            MockSentence("Test sentence one.", [1.0, 0.0, 0.0]),
            MockSentence("Test sentence two.", [0.0, 1.0, 0.0]),
        ]
        mock_chunks = [MockChunk("Combined.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test content.")

            # Test that we get a valid result (weights were normalized)
            embedding = result[0]["embedding"]
            assert embedding is not None
            assert len(embedding) == 3
            assert all(isinstance(x, (int, float)) for x in embedding)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def web_client(self):
        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30
        return WebClient(config, verbose=False)

    def test_chunk_with_no_sentences(self, web_client):
        """Test chunk that has no sentences attribute."""
        mock_chunks = [MockChunk("Text without sentences.")]  # No sentences list

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test content.")

            assert len(result) == 1
            assert result[0]["text"] == "Text without sentences."
            assert result[0]["embedding"] is None

    def test_sentences_without_embeddings_mixed(self, web_client):
        """Test mix of sentences with and without embeddings."""
        mock_sentences = [
            MockSentence("Has embedding.", [1.0, 0.0, 0.0]),
            MockSentence("No embedding.", None),
            MockSentence("Also has embedding.", [0.0, 1.0, 0.0]),
        ]
        mock_chunks = [MockChunk("Mixed content.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test content.")

            # Should still create embedding from available sentences
            assert result[0]["embedding"] is not None
            assert len(result[0]["embedding"]) == 3

    def test_all_sentences_same_length(self, web_client):
        """Test when all sentences have the same length."""
        mock_sentences = [
            MockSentence("Same length sentence here.", [1.0, 0.0, 0.0]),
            MockSentence("Same length sentence also.", [0.0, 1.0, 0.0]),
        ]
        mock_chunks = [MockChunk("Same lengths.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test content.")

            # Should handle equal length weighting without errors
            assert result[0]["embedding"] is not None

    def test_single_sentence_chunk(self, web_client):
        """Test chunk with only one sentence."""
        mock_sentences = [
            MockSentence("Only one sentence here.", [1.0, 2.0, 3.0]),
        ]
        mock_chunks = [MockChunk("Single sentence.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test content.")

            # Single sentence should get full weight
            embedding = result[0]["embedding"]
            assert embedding == [1.0, 2.0, 3.0]


class TestChonkieIntegration:
    """Test integration with chonkie SemanticChunker."""

    def test_chunker_initialization(self):
        """Test that chunker is properly initialized."""
        with patch("chonkie.SemanticChunker") as mock_semantic:
            mock_chunker_instance = Mock()
            mock_semantic.return_value = mock_chunker_instance

            # Clear the cached chunker first
            import interaction_finder.fetcher.web_client

            interaction_finder.fetcher.web_client._chunker = None

            chunker = _get_chunker()

            # Check that SDPMChunker was called with correct parameters
            mock_semantic.assert_called_once_with(
                embedding_model="minishlab/potion-base-8M",
                threshold=0.5,  # CHUNK_SIMILARITY_THRESHOLD
                chunk_size=4096,  # CHUNK_SIZE_TOKENS
                min_sentences=2,  # CHUNK_MIN_SENTENCES
                similarity_window=1,  # CHUNK_SKIP_WINDOW (renamed parameter)
            )
            assert chunker == mock_chunker_instance

    def test_chunker_caching(self):
        """Test that chunker instance is cached."""
        with patch("chonkie.SemanticChunker") as mock_semantic:
            mock_semantic.return_value = Mock()

            # Clear the cached chunker first
            import interaction_finder.fetcher.web_client

            interaction_finder.fetcher.web_client._chunker = None

            # Call multiple times
            chunker1 = _get_chunker()
            chunker2 = _get_chunker()

            # Should only initialize once due to caching
            mock_semantic.assert_called_once()
            assert chunker1 is chunker2

    def test_chunker_error_handling(self):
        """Test error handling when chonkie import fails."""
        with patch(
            "chonkie.SemanticChunker",
            side_effect=ImportError("chonkie not found"),
        ):
            # Clear the cached chunker first
            import interaction_finder.fetcher.web_client

            interaction_finder.fetcher.web_client._chunker = None

            with pytest.raises(ImportError, match="chonkie not found"):
                _get_chunker()


class TestDynamicEmbeddingDimensions:
    """Test handling of different embedding dimensions."""

    @pytest.fixture
    def web_client(self):
        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30
        return WebClient(config, verbose=False)

    @pytest.mark.parametrize("dimension", [256, 384, 768, 1024])
    def test_different_embedding_dimensions(self, web_client, dimension):
        """Test chunking with different embedding dimensions."""
        # Create embeddings of specified dimension
        embedding1 = np.random.rand(dimension).tolist()
        embedding2 = np.random.rand(dimension).tolist()

        mock_sentences = [
            MockSentence("First sentence.", embedding1),
            MockSentence("Second sentence.", embedding2),
        ]
        mock_chunks = [MockChunk("Combined text.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test content.")

            # Should handle any embedding dimension
            assert len(result[0]["embedding"]) == dimension
            assert all(isinstance(x, (int, float)) for x in result[0]["embedding"])

    def test_inconsistent_embedding_dimensions(self, web_client):
        """Test handling of inconsistent embedding dimensions within a chunk."""
        mock_sentences = [
            MockSentence("First.", [1.0, 0.0, 0.0]),  # 3D
            MockSentence("Second.", [0.0, 1.0]),  # 2D - Different dimension
        ]
        mock_chunks = [MockChunk("Text.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            # This should either handle gracefully or raise a clear error
            # Implementation detail: numpy.average should handle this
            try:
                result = web_client.create_chunks("Test content.")
                # If it succeeds, check the result makes sense
                if result[0]["embedding"] is not None:
                    assert len(result[0]["embedding"]) > 0
            except (ValueError, TypeError) as e:
                # Acceptable to fail on inconsistent dimensions
                assert "shape" in str(e).lower() or "dimension" in str(e).lower()


class TestPerformance:
    """Test performance characteristics of chunking."""

    @pytest.fixture
    def web_client(self):
        config = Mock()
        config.tools = Mock()
        config.tools.crawl4ai = Mock()
        config.tools.crawl4ai.timeout = 30
        return WebClient(config, verbose=False)

    def test_large_number_of_sentences(self, web_client):
        """Test chunking with many sentences."""
        # Create 100 sentences with embeddings
        mock_sentences = []
        for i in range(100):
            embedding = [float(i % 3 == 0), float(i % 3 == 1), float(i % 3 == 2)]
            mock_sentences.append(MockSentence(f"Sentence number {i}.", embedding))

        mock_chunks = [MockChunk("Large chunk.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Large content.")

            # Should handle large number of sentences efficiently
            assert len(result) == 1
            assert result[0]["embedding"] is not None
            assert len(result[0]["embedding"]) == 3

    def test_large_embedding_dimensions(self, web_client):
        """Test with high-dimensional embeddings."""
        dimension = 1536  # OpenAI embedding dimension

        mock_sentences = [
            MockSentence("First.", np.random.rand(dimension).tolist()),
            MockSentence("Second.", np.random.rand(dimension).tolist()),
        ]
        mock_chunks = [MockChunk("Text.", mock_sentences)]

        with patch(
            "interaction_finder.fetcher.web_client._get_chunker"
        ) as mock_get_chunker:
            mock_chunker = Mock()
            mock_chunker.return_value = mock_chunks
            mock_get_chunker.return_value = mock_chunker

            result = web_client.create_chunks("Test content.")

            # Should handle high dimensions efficiently
            assert len(result[0]["embedding"]) == dimension
