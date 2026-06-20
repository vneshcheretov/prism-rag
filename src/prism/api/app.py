from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from ..core.chunker import MarkdownChunker
from ..core.engine import Prism
from ..core.graph import PrismGraph
from ..embeddings.sonar import SonarEmbedder
from ..llm.client import LLMClient
from ..storage.qdrant import QdrantBackend
from .config import Settings
from .routes import router

log = logging.getLogger(__name__)


@asynccontextmanager
async def _default_lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_dotenv()
    settings = Settings.from_env()

    embedder = SonarEmbedder(device=settings.sonar_device)
    qdrant = QdrantBackend(
        AsyncQdrantClient(url=settings.qdrant_url),
        collection_name=settings.collection_name,
    )
    graph = await PrismGraph.create(qdrant, embedder, recreate=settings.recreate_collection)
    llm = LLMClient()
    app.state.prism = Prism(graph, llm, MarkdownChunker(), language=settings.language)
    log.info(
        "prism API ready (collection=%s, qdrant=%s)",
        settings.collection_name,
        settings.qdrant_url,
    )

    try:
        yield
    finally:
        await llm.aclose()
        await qdrant.client.close()


def create_app(prism: Prism | None = None) -> FastAPI:
    """Build the Prism FastAPI app.

    With ``prism=None`` (the default, used when running the service) the
    embedder, Qdrant connection, and LLM client are built from environment
    variables on startup and torn down on shutdown.

    Passing a pre-built ``prism`` instance skips that startup wiring
    entirely and is set immediately — used by tests to inject fakes.
    """
    app = FastAPI(title="Prism", lifespan=_default_lifespan if prism is None else None)
    if prism is not None:
        app.state.prism = prism
    app.include_router(router)
    return app
