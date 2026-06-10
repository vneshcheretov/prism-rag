"""Prism — keypoint-decomposition RAG engine over a Qdrant-backed knowledge graph."""

from .core import (
    Chunk,
    MarkdownChunker,
    NodeBlueprint,
    Prism,
    PrismGraph,
    PrismNode,
    SearchResult,
)
from .embeddings import Embedder, SonarEmbedder
from .llm import LLMClient
from .storage import QdrantBackend

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "Embedder",
    "LLMClient",
    "MarkdownChunker",
    "NodeBlueprint",
    "Prism",
    "PrismGraph",
    "PrismNode",
    "QdrantBackend",
    "SearchResult",
    "SonarEmbedder",
]
