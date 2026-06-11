"""End-to-end Prism demo on the Aiso Hotel sample corpus.

Walks through the pipeline:

1. wire up SONAR + Qdrant + OpenAI;
2. ingest a tiny Russian markdown about a hotel (language auto-detected);
3. run a few searches and one end-to-end ``answer()`` call;
4. show what a cross-language query (English over a Russian corpus) returns.

Prereqs:
- ``OPENAI_API_KEY`` in ``.env``;
- Qdrant up (``docker compose up qdrant``).

    python demo/aiso_hotel_demo.py
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient

from prism import (
    LLMClient,
    MarkdownChunker,
    Prism,
    PrismGraph,
    QdrantBackend,
    SonarEmbedder,
)

# The corpus file lives next to this script. We promote a few logical
# section labels to markdown headers so the chunker has something to
# split on — otherwise the whole file becomes one chunk and the demo is
# uninteresting.
DATA_FILE = Path(__file__).parent / "aiso_hotel.txt"

_SECTION_HEADERS = (
    "Описание отеля",
    "Услуги и удобства",
    "Правила проживания",
)


def load_corpus() -> str:
    raw = DATA_FILE.read_text(encoding="utf-8")
    lines: list[str] = []
    for line in raw.splitlines():
        if line.strip() in _SECTION_HEADERS:
            lines.append(f"## {line.strip()}")
        else:
            lines.append(line)
    # Title becomes the top-level header.
    body = "\n".join(lines)
    return f"# Aiso Hotel\n\n{body}\n"


QUERIES = [
    "можно ли с собакой?",
    "во сколько заезд и выезд?",
    "какие удобства есть в отеле?",
    "есть ли тренажёрный зал?",
]


async def main() -> None:
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set in .env"

    logging.basicConfig(
        level=os.getenv("PRISM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("fairseq2").setLevel(logging.WARNING)

    embedder = SonarEmbedder(device=os.getenv("PRISM_SONAR_DEVICE", "cpu"))
    qdrant = QdrantBackend(
        AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333")),
        collection_name="prism_aiso_hotel",
    )

    graph = await PrismGraph.create(qdrant, embedder, recreate=True)
    llm = LLMClient(temperature=None)
    prism = Prism(
        graph,
        llm,
        chunker=MarkdownChunker(max_tokens=256, min_section_tokens=10),
    )

    try:
        corpus = load_corpus()
        print(f"\n=== INGEST ({len(corpus)} chars) ===")
        nodes = await prism.ingest(corpus)
        print(f"language detected: {prism.language}")
        print(f"ingested nodes:    {len(nodes)}")
        for n in nodes:
            print(f"  - {n.name}: keypoints={n.keypoints[:5]}")
        print(f"\ncorpus summary:\n  {prism.corpus_summary}")

        for q in QUERIES:
            print(f"\n=== SEARCH: {q!r} ===")
            res = await prism.search(q)
            print(f"keypoints: {res.keypoints}")
            print(f"note:      {res.note}")
            for i, p in enumerate(res.paragraphs, 1):
                print(f"  [{i}] {p}")

        # End-to-end answer
        target = "можно ли с собакой?"
        print(f"\n=== ANSWER: {target!r} ===")
        ans = await prism.answer(target)
        print(f"answer:        {ans.answer}")
        print(f"final_summary: {ans.final_summary}")
        if ans.note:
            print(f"note:          {ans.note}")

        # Cross-language: ask in English over a Russian corpus.
        print("\n=== CROSS-LANGUAGE (en query, ru corpus) ===")
        en = await prism.answer("can I bring my dog?", query_language="en")
        print(f"answer: {en.answer}")
        print(f"note:   {en.note}")
    finally:
        await qdrant.client.close()


if __name__ == "__main__":
    asyncio.run(main())
