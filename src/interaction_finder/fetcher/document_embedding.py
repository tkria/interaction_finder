"""
Document embedding computation with multiple strategies.

This module provides different approaches to computing document-level embeddings
from chunk embeddings, including simple averaging and corpus-aware IDF-like weighting.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ChunkData:
    """Type-safe representation of a document chunk."""

    text: str
    embedding: np.ndarray  # L2-normalized
    wordcount: int
    metadata: Optional[Dict] = None


class DocumentEmbedder(ABC):
    """Base class for all document embedding strategies."""

    @abstractmethod
    def compute_embeddings(
        self,
        documents: List[str],
        chunk_data: Dict[str, List[ChunkData]],
        status: Optional["StatusProtocol"] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute document embeddings from chunks.

        Args:
            documents: List of document identifiers
            chunk_data: Dict mapping document ID to list of chunks
            status: Optional status object for progress reporting

        Returns:
            Dict mapping document ID to unit-norm embedding vector
        """
        pass


class SimpleAverageEmbedder(DocumentEmbedder):
    """Simple averaging of chunk embeddings for document representation."""

    def compute_embeddings(
        self,
        documents: List[str],
        chunk_data: Dict[str, List[ChunkData]],
        status: Optional["StatusProtocol"] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute document embeddings as simple averages of chunk embeddings."""
        if status:
            status.update("Computing document embeddings (uniform weighting)...")
        doc_embeddings = {}

        for doc in documents:
            chunks = chunk_data.get(doc, [])
            if not chunks:
                continue

            embeddings = []
            for chunk in chunks:
                if chunk.embedding is not None:
                    embeddings.append(chunk.embedding)

            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(avg_embedding)
                if norm > 1e-8:
                    doc_embeddings[doc] = avg_embedding / norm

        return doc_embeddings


class IDFEmbedder(DocumentEmbedder):
    """
    Corpus-aware IDF-like document embeddings with PC1 removal.

    Implements the exact 5-step pipeline:
    1. Compute chunk specificity using cross-document similarities
    2. Convert to weights with monotone sharpening
    3. Weighted average with small unweighted blend
    4. Remove PC1 component
    5. Ensure unit norm
    """

    def __init__(
        self,
        similarity_threshold: float = 0.3,
        sharpening_power: float = 2.0,
        blend_ratio: float = 0.05,
    ):
        """
        Initialize IDF embedder.

        Args:
            similarity_threshold: Soft threshold for match strength computation
            sharpening_power: Power for monotone sharpening (>1 increases contrast)
            blend_ratio: Fraction of unweighted mean to blend for stability
        """
        self.similarity_threshold = similarity_threshold
        self.sharpening_power = sharpening_power
        self.blend_ratio = blend_ratio

    def compute_embeddings(
        self,
        documents: List[str],
        chunk_data: Dict[str, List[ChunkData]],
        status: Optional["StatusProtocol"] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute IDF-like document embeddings following 5-step pipeline."""
        if status:
            status.update("Computing document embeddings (idf weighting)...")
        if not documents:
            return {}

        # Step 1: Compute chunk specificity
        specificities = self._compute_chunk_specificity(documents, chunk_data)

        # Step 2: Convert to weights with sharpening
        weights = self._compute_chunk_weights(specificities)

        # Step 3: Build weighted document vectors
        doc_vectors = self._build_document_vectors(documents, chunk_data, weights)

        # Step 4: Remove PC1 component
        doc_embeddings = self._remove_pc1(doc_vectors)

        # Step 5: Ensure unit norm
        return {
            doc: vec / (np.linalg.norm(vec) + 1e-8)
            for doc, vec in doc_embeddings.items()
        }

    def _compute_chunk_specificity(
        self, documents: List[str], chunk_data: Dict[str, List[ChunkData]]
    ) -> Dict[str, np.ndarray]:
        """
        Step 1: Compute corpus-aware specificity per chunk.

        For each chunk, compute its spread across OTHER documents using
        inverse participation ratio of match strengths.

        Returns:
            Dict mapping doc -> array of specificity scores per chunk
        """
        # Early return for single document
        if len(documents) <= 1:
            return {doc: np.ones(len(chunk_data.get(doc, []))) for doc in documents}

        # Build chunk similarity matrix and mappings
        all_chunks = []
        chunk_to_doc = {}
        doc_to_chunks = {doc: [] for doc in documents}

        for doc in documents:
            chunks = chunk_data.get(doc, [])
            for chunk in chunks:
                if chunk.embedding is not None:
                    chunk_idx = len(all_chunks)
                    all_chunks.append(chunk.embedding)
                    chunk_to_doc[chunk_idx] = doc
                    doc_to_chunks[doc].append(chunk_idx)

        if not all_chunks:
            return {}

        # Vectorized similarity computation
        chunk_matrix = np.stack(all_chunks)
        similarities = chunk_matrix @ chunk_matrix.T

        specificities = {}

        # Process each document's chunks
        for doc in documents:
            chunk_indices = doc_to_chunks[doc]
            if not chunk_indices:
                continue

            doc_specificities = []

            for chunk_idx in chunk_indices:
                # Get max similarities to OTHER documents' chunks using vectorized operations
                other_doc_maxes = []

                for other_doc in documents:
                    if other_doc == doc:
                        continue

                    other_chunk_indices = doc_to_chunks[other_doc]
                    if other_chunk_indices:
                        # Vectorized max operation
                        max_sim = np.max(similarities[chunk_idx, other_chunk_indices])
                        other_doc_maxes.append(max_sim)

                if not other_doc_maxes:
                    specificity = 1.0  # Single document case
                else:
                    # Convert to match strengths using soft threshold (hinge)
                    other_doc_maxes = np.array(other_doc_maxes)
                    strengths = np.maximum(
                        0, other_doc_maxes - self.similarity_threshold
                    )

                    # Normalize to distribution over documents
                    total_strength = np.sum(strengths) + 1e-8
                    distribution = strengths / total_strength

                    # Compute spread using inverse participation ratio
                    participation_ratio = np.sum(distribution**2)
                    if participation_ratio > 0:
                        spread = 1.0 / participation_ratio
                        # Normalize by max possible spread (uniform distribution)
                        max_spread = len(distribution) if len(distribution) > 0 else 1.0
                        normalized_spread = spread / max_spread
                    else:
                        normalized_spread = 0.0

                    # Specificity = inverse of spread, tempered by total strength
                    base_specificity = 1.0 - normalized_spread
                    strength_factor = 1.0 - np.exp(-total_strength / 0.5)
                    specificity = (
                        base_specificity * strength_factor + (1 - strength_factor) * 0.1
                    )

                doc_specificities.append(specificity)

            if doc_specificities:
                specificities[doc] = np.array(doc_specificities)

        return specificities

    def _compute_chunk_weights(
        self, specificities: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Step 2: Apply monotone sharpening and normalize weights per document.

        Args:
            specificities: Dict mapping doc -> array of specificity scores

        Returns:
            Dict mapping doc -> array of normalized weights summing to 1
        """
        weights = {}

        for doc, specs in specificities.items():
            # Apply monotone sharpening (power > 1)
            sharpened = np.power(specs, self.sharpening_power)

            # Keep strictly positive with small minimum
            sharpened = sharpened + 1e-6

            # Normalize within document to sum to 1
            weights[doc] = sharpened / np.sum(sharpened)

        return weights

    def _build_document_vectors(
        self,
        documents: List[str],
        chunk_data: Dict[str, List[ChunkData]],
        weights: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Step 3: Build weighted document vectors with robust pooling.

        Computes weighted average of chunk embeddings with small blend
        of unweighted mean for stability.
        """
        doc_vectors = {}

        for doc in documents:
            chunks = chunk_data.get(doc, [])
            if not chunks:
                continue

            embeddings = []
            for chunk in chunks:
                if chunk.embedding is not None:
                    embeddings.append(chunk.embedding)

            if not embeddings:
                continue

            embeddings_array = np.stack(embeddings)
            doc_weights = weights.get(doc, np.ones(len(chunks)) / len(chunks))

            # Ensure weights match embeddings count
            if len(doc_weights) != len(embeddings):
                doc_weights = np.ones(len(embeddings)) / len(embeddings)

            # Weighted average
            weighted_embedding = np.average(
                embeddings_array, weights=doc_weights, axis=0
            )

            # Small blend with unweighted mean for stability
            unweighted_embedding = np.mean(embeddings_array, axis=0)
            blended_embedding = (
                1 - self.blend_ratio
            ) * weighted_embedding + self.blend_ratio * unweighted_embedding

            # L2 normalize
            norm = np.linalg.norm(blended_embedding)
            if norm > 1e-8:
                doc_vectors[doc] = blended_embedding / norm

        return doc_vectors

    def _remove_pc1(
        self, doc_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Step 4: Remove PC1 component and re-normalize.

        Subtracts projection onto top principal component to remove
        corpus-common background signal.
        """
        if len(doc_embeddings) < 2:
            return doc_embeddings

        docs = list(doc_embeddings.keys())
        embedding_matrix = np.stack([doc_embeddings[doc] for doc in docs])

        try:
            # Mean center
            mean_embedding = np.mean(embedding_matrix, axis=0)
            centered_matrix = embedding_matrix - mean_embedding

            # Compute PC1 via SVD
            U, S, Vt = np.linalg.svd(centered_matrix, full_matrices=False)
            pc1 = Vt[0]  # First principal component

            # Remove PC1 projection
            projections = centered_matrix @ pc1[:, np.newaxis]
            pc1_reconstructed = projections @ pc1[np.newaxis, :]
            denoised_matrix = centered_matrix - pc1_reconstructed

            # Re-normalize to unit length
            denoised_embeddings = {}
            for i, doc in enumerate(docs):
                denoised_vec = denoised_matrix[i]
                norm = np.linalg.norm(denoised_vec)
                if norm > 1e-8:
                    denoised_embeddings[doc] = denoised_vec / norm
                else:
                    # Fallback to original if PC1 removal eliminates all signal
                    denoised_embeddings[doc] = doc_embeddings[doc]

            return denoised_embeddings

        except Exception:
            # Fallback to original embeddings if PC1 removal fails
            return doc_embeddings


def convert_legacy_chunk_data(
    legacy_chunks: Dict[str, List[Dict]],
) -> Dict[str, List[ChunkData]]:
    """
    Convert legacy chunk format to typed ChunkData format.

    Args:
        legacy_chunks: Dict mapping doc -> list of chunk dicts

    Returns:
        Dict mapping doc -> list of ChunkData objects
    """
    typed_chunks = {}

    for doc, chunks in legacy_chunks.items():
        typed_list = []
        for chunk in chunks:
            chunk_data = ChunkData(
                text=chunk.get("text", ""),
                embedding=np.array(chunk["embedding"])
                if chunk.get("embedding") is not None
                else None,
                wordcount=chunk.get("wordcount", 0),
                metadata=chunk.get("metadata"),
            )
            typed_list.append(chunk_data)
        typed_chunks[doc] = typed_list

    return typed_chunks
