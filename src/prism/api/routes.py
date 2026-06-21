from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Body, Depends, Request, UploadFile
from fastapi.responses import JSONResponse

from .. import __version__
from ..core.engine import Prism
from ..core.structuring import MarkdownStructurer
from ..utils.text import normalize_newlines
from .dependencies import get_prism
from .schemas import (
    AnswerRequest,
    AnswerResponse,
    ConvertResponse,
    IngestedNode,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root() -> dict[str, str]:
    """Service banner — so hitting the base URL isn't a bare 404."""
    return {
        "service": "prism",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


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


@router.post("/ingest/markdown", response_model=IngestResponse)
async def ingest_markdown(
    markdown: str = Body(
        ...,
        media_type="text/markdown",
        description="Markdown document to ingest — sent as the raw request body.",
    ),
    prism: Prism = Depends(get_prism),
) -> IngestResponse:
    nodes = await prism.ingest(markdown)
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


async def _convert_to_markdown(file: UploadFile) -> tuple[str, str | None] | JSONResponse:
    """Convert an uploaded file to markdown via markitdown.

    Returns ``(markdown, title)`` or a ``JSONResponse`` error (503 when the
    ``convert`` extra is missing, 422 when conversion fails).
    """
    try:
        from markitdown import MarkItDown
    except ImportError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "conversion_unavailable",
                "detail": "install the conversion extra: pip install 'prism-rag[convert]'",
            },
        )

    data = await file.read()
    suffix = Path(file.filename).suffix if file.filename else None
    try:
        result = MarkItDown().convert_stream(BytesIO(data), file_extension=suffix)
    except Exception as e:
        log.warning("convert: failed for %r: %s: %s", file.filename, type(e).__name__, e)
        return JSONResponse(
            status_code=422,
            content={"error": "conversion_failed", "detail": "could not convert the file"},
        )
    return normalize_newlines(result.markdown), result.title


@router.post("/convert/file", response_model=ConvertResponse)
async def convert_file(file: UploadFile) -> ConvertResponse | JSONResponse:
    """Convert an uploaded document to markdown (conversion only).

    Supported inputs (via markitdown): PDF, Word (.docx), PowerPoint
    (.pptx), Excel (.xlsx/.xls), HTML, CSV, JSON, XML, EPUB, ZIP.

    **Heading structure caveat:** markdown headings (`#`) are produced only
    when the source file carries real heading formatting — e.g. Word
    "Heading 1/2/3" paragraph styles. PDFs (which have no heading
    semantics) and documents that fake headings with bold text convert to
    flat markdown without `#`. Since Prism chunks by headers, a flat result
    yields a single coarse section. If you are unsure about the source's
    structure, use ``/convert/structured`` (LLM-assisted heading inference).

    The returned markdown is meant to be reviewed before being sent to
    ``/ingest/markdown``. Independent of the engine, so it works even when
    Qdrant is down. Requires the ``convert`` extra (markitdown).
    """
    out = await _convert_to_markdown(file)
    if isinstance(out, JSONResponse):
        return out
    markdown, title = out
    return ConvertResponse(markdown=markdown, title=title)


@router.post("/convert/structured", response_model=ConvertResponse)
async def convert_structured(
    file: UploadFile, prism: Prism = Depends(get_prism)
) -> ConvertResponse | JSONResponse:
    """Convert a document to markdown **and** infer heading structure via an LLM.

    Same conversion as ``/convert/file``, plus an extra LLM pass that adds
    markdown headings — for sources with no usable structure (PDFs, docs
    that fake headings with bold text). Heavier (one LLM call per ~70
    sentences); needs both the ``convert`` extra and a configured LLM.
    """
    out = await _convert_to_markdown(file)
    if isinstance(out, JSONResponse):
        return out
    markdown, title = out
    structurer = MarkdownStructurer(prism.llm)
    structured = await structurer.structure(markdown)
    # markitdown rarely detects a title for unstructured sources (e.g. PDFs);
    # infer one from the opening of the document in that case.
    if not title:
        title = await structurer.generate_title(markdown)
    return ConvertResponse(markdown=structured, title=title)


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, prism: Prism = Depends(get_prism)) -> SearchResponse:
    result = await prism.search(
        req.query,
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
