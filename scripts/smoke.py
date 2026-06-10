"""End-to-end smoke test for Prism.

Runs: SONAR → Qdrant (dense + lexical payload) → OpenAI ingest + search.
Requires Docker Qdrant up and OPENAI_API_KEY in .env.

    python scripts/smoke.py
"""
from __future__ import annotations

import asyncio
import logging
import os

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

SAMPLE_MD = """# Архитектура Prism

Prism — это RAG-движок, который раскладывает запрос пользователя в набор атомарных ключевых фраз и ищет каждую из них по гибридному индексу.

## Хранилища

Векторы и лексический payload хранятся в одном Qdrant с HNSW-настройкой m=256, ef_construct=512. Поверх dense-каналу работает payload-фильтр по стемам Snowball, дающий строгое AND с проверкой proximity внутри предложения.

## Эмбеддинги

Текст кодируется моделью Meta SONAR. Это 1024-мерные мультиязычные эмбеддинги, поддерживающие 200 языков через коды FLORES-200. Все векторы L2-нормализованы, поэтому косинус сводится к скалярному произведению.

## Поиск

Запрос сначала разбирается LLM на ключевые фразы и синонимы. Каждая фраза эмбеддится отдельно, плюс считается агрегатный вектор как среднее. По каждому вектору идёт независимый Qdrant-поиск с динамическим порогом по перцентилю распределения скоров — это устойчивее фиксированного порога косинуса.

# Гибридное ранжирование

После векторного поиска подключается лексический канал — AND по стемам ключевой фразы с post-фильтром на уровне предложения. Результаты объединяются, дедуплицируются, затем LLM-судья отфильтровывает нерелевантные параграфы перед тем как вернуть их клиенту.
"""

QUERY = "какие модели эмбеддингов используются и почему 1024 размерность?"


async def main() -> None:
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set in .env"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    embedder = SonarEmbedder(
        source_lang="rus_Cyrl",
        device=os.getenv("PRISM_SONAR_DEVICE", "cpu"),
    )
    qdrant = QdrantBackend(
        AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333")),
        collection_name="prism_smoke",
    )

    graph = await PrismGraph.create(qdrant, embedder, recreate=True)
    llm = LLMClient(temperature=None)
    prism = Prism(
        graph,
        llm,
        chunker=MarkdownChunker(max_tokens=256, min_section_tokens=10),
    )

    try:
        nodes = await prism.ingest(SAMPLE_MD)
        print(f"\nINGESTED {len(nodes)} nodes:")
        for n in nodes:
            print(f"  - {n.name}  (keypoints: {n.keypoints[:3]}...)")

        result = await prism.search(QUERY)
        print(f"\nQUERY: {result.query}")
        print(f"KEYPOINTS: {result.keypoints}")
        print(f"NOTE: {result.note}")
        print(f"\n--- {len(result.paragraphs)} paragraph(s) ---")
        for i, p in enumerate(result.paragraphs, 1):
            print(f"\n[{i}] {p}")
    finally:
        await qdrant.client.close()


if __name__ == "__main__":
    asyncio.run(main())
