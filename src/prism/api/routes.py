from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ..core.engine import Prism
from .dependencies import get_prism
from .schemas import (
    AnswerRequest,
    AnswerResponse,
    IngestedNode,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is up and the event loop is responsive.

    Shallow on purpose — wire this to a liveness probe. It must not depend
    on Qdrant, or a transient storage blip would trigger a pod restart.
    """
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness: the engine is loaded and Qdrant is reachable.

    Wire this to a readiness probe — a failure pulls the instance out of
    rotation (no restart) until its dependencies recover.
    """
    prism = getattr(request.app.state, "prism", None)
    if prism is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "engine not initialized"},
        )
    try:
        await prism.graph.qdrant.ping()
    except Exception as e:
        log.warning("readiness: qdrant unreachable: %s: %s", type(e).__name__, e)
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "qdrant unreachable"},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "nodes": len(prism.graph.nodes)},
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, prism: Prism = Depends(get_prism)) -> IngestResponse:
    nodes = await prism.ingest(req.markdown, summarize=req.summarize)
    return IngestResponse(
        language=prism.language,
        nodes=[
            IngestedNode(
                index=n.index,
                name=n.name,
                keypoints=n.keypoints,
                paragraph_id=n.paragraph_id,
            )
            for n in nodes
        ],
        corpus_summary=prism.corpus_summary,
    )


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, prism: Prism = Depends(get_prism)) -> SearchResponse:
    result = await prism.search(
        req.query,
        filter_relevance=req.filter_relevance,
        query_language=req.query_language,
        history=[m.model_dump() for m in req.history],
    )
    return SearchResponse(
        query=result.query,
        keypoints=result.keypoints,
        paragraphs=result.paragraphs,
        note=result.note,
    )


@router.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest, prism: Prism = Depends(get_prism)) -> AnswerResponse:
    result = await prism.answer(
        req.query,
        filter_relevance=req.filter_relevance,
        query_language=req.query_language,
        history=[m.model_dump() for m in req.history],
    )
    search_out = None
    if result.search is not None:
        search_out = SearchResponse(
            query=result.search.query,
            keypoints=result.search.keypoints,
            paragraphs=result.search.paragraphs,
            note=result.search.note,
        )
    return AnswerResponse(
        query=result.query,
        answer=result.answer,
        final_summary=result.final_summary,
        note=result.note,
        search=search_out,
    )
