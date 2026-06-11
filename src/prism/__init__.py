"""Prism — keypoint-decomposition RAG engine over a Qdrant-backed knowledge graph."""

from .core import (
    AnswerResult,
    Chunk,
    MarkdownChunker,
    NodeBlueprint,
    Prism,
    PrismGraph,
    PrismNode,
    SearchResult,
)
from .embeddings import Embedder, SonarEmbedder
from .llm import LLMClient, LLMProvider
from .storage import QdrantBackend

__version__ = "0.1.0"

__all__ = [
    "AnswerResult",
    "Chunk",
    "Embedder",
    "LLMClient",
    "LLMProvider",
    "MarkdownChunker",
    "NodeBlueprint",
    "Prism",
    "PrismGraph",
    "PrismNode",
    "QdrantBackend",
    "SearchResult",
    "SonarEmbedder",
]
