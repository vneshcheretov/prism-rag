from __future__ import annotations

import numpy as np
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient

from prism import Embedder, MarkdownChunker, Prism, PrismGraph, QdrantBackend
from prism.api import create_app
from prism.schemas.llm_outputs import (
    CorpusSummary,
    NodeExtraction,
    QueryKeypoints,
    RelevanceFilter,
    Summarization,
)

MARKDOWN = """# Hotel Handbook

## Pets

Pets up to 5 kg are allowed in all rooms. Please notify reception in advance
so a room on the ground floor can be prepared for your dog or cat.
"""


class ConstantEmbedder(Embedder):
    """Returns the same unit vector for every text.

    Makes cosine similarity 1.0 for every query/index pair, so retrieval
    deterministically returns whatever was indexed without depending on a
    real embedding model.
    """

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
    """Canned structured responses, one per schema used by the pipeline."""

    async def complete_structured(self, system, user, schema, *, tier="fast"):
        if schema is NodeExtraction:
            return NodeExtraction(
                header="Pets",
                summary="Pet policy for the hotel.",
                key_phrases=["pets", "dogs"],
            )
        if schema is QueryKeypoints:
            return QueryKeypoints(
                is_searchable=True,
                short_summary="pet policy",
                key_phrases=["pets"],
                synonyms=["pets", "dogs"],
            )
        if schema is RelevanceFilter:
            return RelevanceFilter(
                answer="Pets up to 5 kg are allowed in all rooms.",
                is_correct=True,
            )
        if schema is CorpusSummary:
            return CorpusSummary(summary="A hotel handbook covering pet policy.")
        if schema is Summarization:
            return Summarization(
                summary="Yes, pets up to 5 kg are allowed.",
                final_summary="Данные об условиях проживания с животными",
            )
        raise AssertionError(f"unexpected schema {schema}")

    async def complete_text(self, system, user, *, tier="fast"):
        return "en"


async def _build_client() -> AsyncClient:
    qdrant = QdrantBackend(AsyncQdrantClient(location=":memory:"), collection_name="test")
    graph = await PrismGraph.create(qdrant, ConstantEmbedder(), recreate=True)
    prism = Prism(
        graph,
        FakeLLM(),
        MarkdownChunker(max_tokens=256, min_section_tokens=10),
        language="en",
    )
    app = create_app(prism=prism)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health():
    async with await _build_client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_ingest_search_answer_flow():
    async with await _build_client() as client:
        ingest_resp = await client.post("/ingest", json={"markdown": MARKDOWN})
        assert ingest_resp.status_code == 200
        ingest_body = ingest_resp.json()
        assert ingest_body["language"] == "en"
        assert len(ingest_body["nodes"]) == 1
        assert ingest_body["nodes"][0]["name"] == "Pets"
        assert ingest_body["corpus_summary"] == "A hotel handbook covering pet policy."

        search_resp = await client.post("/search", json={"query": "can I bring my dog?"})
        assert search_resp.status_code == 200
        search_body = search_resp.json()
        assert search_body["keypoints"] == ["pets", "dogs"]
        assert search_body["paragraphs"] == ["Pets up to 5 kg are allowed in all rooms."]
        assert search_body["note"] is None

        answer_resp = await client.post("/answer", json={"query": "can I bring my dog?"})
        assert answer_resp.status_code == 200
        answer_body = answer_resp.json()
        assert answer_body["answer"] == "Yes, pets up to 5 kg are allowed."
        assert answer_body["search"]["paragraphs"] == [
            "Pets up to 5 kg are allowed in all rooms."
        ]


async def test_search_empty_query_returns_note():
    async with await _build_client() as client:
        resp = await client.post("/search", json={"query": "   "})
    assert resp.status_code == 200
    body = resp.json()
    assert body["paragraphs"] == []
    assert body["note"] == "empty query"
