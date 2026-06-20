"""HTTP error mapping for the Prism API.

The engine already degrades most failures gracefully via ``note`` (a 200
response explaining why nothing came back). What remains uncaught are
dependency failures — Qdrant during retrieval/upsert, SONAR/OpenAI when
they error past their retries — plus a hard ingest failure. These handlers
turn those into honest status codes with a consistent JSON body instead of
an opaque 500.
"""

from __future__ import annotations

import logging

import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException

from ..core.engine import IngestError

log = logging.getLogger(__name__)


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "detail": detail})


def register_error_handlers(app: FastAPI) -> None:
    """Map engine/dependency exceptions to HTTP responses on ``app``."""

    @app.exception_handler(IngestError)
    async def _on_ingest_error(_: Request, exc: IngestError) -> JSONResponse:
        return _error(422, "ingest_failed", str(exc))

    @app.exception_handler(openai.APIError)
    async def _on_llm_error(_: Request, exc: openai.APIError) -> JSONResponse:
        log.warning("LLM upstream error: %s: %s", type(exc).__name__, exc)
        return _error(502, "llm_unavailable", "the language model is unavailable")

    @app.exception_handler(ResponseHandlingException)
    @app.exception_handler(ApiException)
    async def _on_storage_error(_: Request, exc: Exception) -> JSONResponse:
        log.warning("storage error: %s: %s", type(exc).__name__, exc)
        return _error(503, "storage_unavailable", "the vector store is unavailable")

    @app.exception_handler(Exception)
    async def _on_unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error: %s", exc)
        return _error(500, "internal_error", "an unexpected error occurred")
