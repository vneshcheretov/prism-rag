import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient

from prism import Embedder, PrismGraph, QdrantBackend


class FakeEmbedder(Embedder):
    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self._dim), dtype=np.float32)


def _backend(**kwargs) -> QdrantBackend:
    return QdrantBackend(
        AsyncQdrantClient(location=":memory:"),
        collection_name="test",
        **kwargs,
    )


def test_graph_inherits_dim_from_embedder():
    qdrant = _backend()
    assert qdrant.vector_size is None
    PrismGraph(qdrant, FakeEmbedder(dim=512))
    assert qdrant.vector_size == 512


def test_graph_accepts_matching_explicit_dim():
    qdrant = _backend(vector_size=512)
    PrismGraph(qdrant, FakeEmbedder(dim=512))
    assert qdrant.vector_size == 512


def test_graph_rejects_dim_mismatch():
    qdrant = _backend(vector_size=1024)
    with pytest.raises(ValueError, match="dimension mismatch"):
        PrismGraph(qdrant, FakeEmbedder(dim=512))


async def test_ensure_collection_requires_dim():
    with pytest.raises(ValueError, match="vector_size is not set"):
        await _backend().ensure_collection()


async def test_ensure_collection_rejects_existing_dim_mismatch():
    client = AsyncQdrantClient(location=":memory:")
    first = QdrantBackend(client, collection_name="test", vector_size=512)
    await first.ensure_collection()

    second = QdrantBackend(client, collection_name="test", vector_size=1024)
    with pytest.raises(ValueError, match="already exists with"):
        await second.ensure_collection()
