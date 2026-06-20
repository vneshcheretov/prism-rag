from __future__ import annotations

from fastapi import APIRouter, Depends

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

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
