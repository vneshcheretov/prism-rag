"""Graph rehydration: a restarted process must recover the in-memory graph
from Qdrant alone, without re-ingesting."""

import numpy as np
from qdrant_client import AsyncQdrantClient

from prism import Embedder, MarkdownChunker, Prism, PrismGraph, QdrantBackend
from prism.schemas.llm_outputs import (
    CorpusSummary,
    NodeExtraction,
    QueryKeypoints,
    RelevanceFilter,
    Summarization,
)

MARKDOWN = """# Hotel Handbook

## Pets

Pets up to 5 kg are allowed in all rooms.

## Checkout

Checkout is at noon.
"""


class ConstantEmbedder(Embedder):
    """Same unit vector for every text — retrieval matches everything, so the
    test depends only on what was indexed, not on a real embedding model."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        vec[0] = 1.0
        return np.tile(vec, (len(texts), 1))


class FakeLLM:
    async def complete_structured(self, system, user, schema, *, tier="fast"):
        if schema is NodeExtraction:
            return NodeExtraction(
                header="Section", summary="A section.", key_phrases=["a", "b"]
            )
        if schema is QueryKeypoints:
            return QueryKeypoints(
                is_searchable=True,
                short_summary="pets",
                key_phrases=["pets"],
                synonyms=["pets"],
            )
        if schema is RelevanceFilter:
            return RelevanceFilter(answer="Pets up to 5 kg are allowed.", is_correct=True)
        if schema is CorpusSummary:
            return CorpusSummary(summary="Handbook about a hotel.")
        if schema is Summarization:
            return Summarization(summary="Yes.", final_summary="Данные об отеле")
        raise AssertionError(f"unexpected schema {schema}")

    async def complete_text(self, system, user, *, tier="fast"):
        return "en"


def _backend(client: AsyncQdrantClient) -> QdrantBackend:
    return QdrantBackend(client, collection_name="persist_test")


async def _fresh_prism(client: AsyncQdrantClient) -> Prism:
    graph = await PrismGraph.create(_backend(client), ConstantEmbedder(), recreate=True)
    return Prism(graph, FakeLLM(), MarkdownChunker(max_tokens=256, min_section_tokens=5),
                 language="en")


async def test_load_rehydrates_nodes_summary_and_language():
    # One client = one persistent store, shared across two graph "lifetimes".
    client = AsyncQdrantClient(location=":memory:")

    original = await _fresh_prism(client)
    nodes = await original.ingest(MARKDOWN)
    assert nodes
    before = {n.index: (n.name, n.text, n.keypoints) for n in nodes}
    search_before = await original.search("can I bring a pet?")

    # New process: fresh graph object over the SAME collection, rehydrated.
    graph2 = await PrismGraph.create(_backend(client), ConstantEmbedder(), recreate=False)
    reloaded = await Prism.load(graph2, FakeLLM(), MarkdownChunker())

    # Nodes restored identically.
    assert {i: (n.name, n.text, n.keypoints) for i, n in graph2.nodes.items()} == before
    # Engine-level state restored.
    assert reloaded.language == "en"
    assert reloaded.corpus_summary == original.corpus_summary
    # Search over the reloaded graph returns the same paragraphs.
    search_after = await reloaded.search("can I bring a pet?")
    assert search_after.paragraphs == search_before.paragraphs


async def test_load_continues_id_numbering_for_incremental_ingest():
    client = AsyncQdrantClient(location=":memory:")
    original = await _fresh_prism(client)
    first = await original.ingest(MARKDOWN)
    max_id_before = max(n.index for n in first)

    graph2 = await PrismGraph.create(_backend(client), ConstantEmbedder(), recreate=False)
    reloaded = await Prism.load(graph2, FakeLLM(), MarkdownChunker(max_tokens=256, min_section_tokens=5))
    more = await reloaded.ingest(
        "# Extra\n\n## Parking\n\n"
        "Free parking is available on site for all hotel guests around the clock.\n"
    )

    # New nodes get fresh ids, no collision with rehydrated ones.
    assert min(n.index for n in more) > max_id_before


async def test_load_on_empty_collection_is_noop():
    client = AsyncQdrantClient(location=":memory:")
    graph = await PrismGraph.create(_backend(client), ConstantEmbedder(), recreate=True)
    prism = await Prism.load(graph, FakeLLM(), MarkdownChunker())
    assert graph.nodes == {}
    assert prism.language is None
    assert prism.corpus_summary == ""
