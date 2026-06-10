from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Abstract text embedder returning L2-normalized vectors.

    Implementations must guarantee that ``embed(texts)`` returns a
    ``(len(texts), dim)`` float32 array of unit vectors, so that downstream
    cosine similarity reduces to a plain inner product.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings of shape ``(len(texts), dim)``.

        For an empty input list, returns an empty ``(0, dim)`` array rather
        than raising — callers can compose without guard-clauses everywhere.
        """

    @staticmethod
    def average_unit_vector(vectors: np.ndarray) -> np.ndarray:
        """Mean of unit vectors, renormalized to unit length.

        Used to compose a single aggregate query vector from multiple
        keypoint vectors. Falls back to the zero mean if the average
        degenerates to a zero vector.
        """
        if vectors.size == 0:
            raise ValueError("average_unit_vector: empty input")
        avg = vectors.mean(axis=0)
        norm = float(np.linalg.norm(avg))
        if norm == 0.0:
            return avg.astype(np.float32)
        return (avg / norm).astype(np.float32)
