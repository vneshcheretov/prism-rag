from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..embeddings.sonar import resolve_flores_code
from ..llm.client import LLMClient
from ..llm.prompts import RELEVANCE_FILTER_PROMPT, build_prompts
from ..schemas.llm_outputs import (
    CorpusSummary,
    NodeExtraction,
    QueryKeypoints,
    RelevanceFilter,
    Summarization,
)
from ..utils.language import detect_language, english_name, format_mismatch_message
from .chunker import Chunk, MarkdownChunker
from .graph import PrismGraph
from .node import NodeBlueprint, PrismNode

log = logging.getLogger(__name__)

_EMPTY_QUERY = "empty query"
_NON_SEARCHABLE = "query is not an information request"
_NO_KEYPOINTS = "no searchable keypoints could be extracted"
_LANGUAGE_MISMATCH = "language mismatch"


@dataclass
class SearchResult:
    """End-to-end retrieval result.

    ``paragraphs`` is the final answer material the caller should hand to
    a downstream LLM (or display to the user); ``nodes`` and ``keypoints``
    are exposed for debugging, tracing, and evaluation harnesses.

    ``note`` is set when the pipeline short-circuited before retrieval
    (e.g. the query was empty or judged non-searchable) — in that case
    ``paragraphs`` is empty and ``note`` explains why.
    """

    query: str
    keypoints: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    nodes: list[PrismNode] = field(default_factory=list)
    note: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


@dataclass
class AnswerResult:
    """End-to-end answer result returned by :py:meth:`Prism.answer`.

    Wraps the underlying :class:`SearchResult` (kept on ``search`` for
    inspection, citations, tracing) and adds the synthesized answer.

    ``answer`` is empty when retrieval returned nothing or the
    summarization call failed — in that case ``note`` explains why.
    """

    query: str
    answer: str = ""
    final_summary: str = ""
    search: SearchResult | None = None
    note: str | None = None


class Prism:
    """End-to-end RAG orchestrator.

    Wires together:
    - :class:`MarkdownChunker` — markdown → token-bounded chunks
    - :class:`LLMClient` — keypoint extraction (per chunk and per query),
      relevance filtering, corpus summarization
    - :class:`PrismGraph` — Qdrant storage (dense + lexical payload), hybrid retrieval

    Two public flows:

    - :py:meth:`ingest` (markdown → graph): chunk → LLM keypoint extraction
      per chunk → graph batch-embeds, indexes, and builds neighbor edges →
      optional corpus-level summary used as context for future queries.

    - :py:meth:`search` (natural-language query → relevant paragraphs):
      LLM decomposes the query into keypoints + synonyms → hybrid retrieval
      (vector search with 1-hop expansion + BM25) → paragraph reconstruction
      from chunks → optional per-paragraph LLM relevance filter.
    """

    def __init__(
        self,
        graph: PrismGraph,
        llm: LLMClient,
        chunker: MarkdownChunker | None = None,
        *,
        language: str | None = None,
        chunk_concurrency: int = 10,
        chunk_max_retries: int = 3,
    ) -> None:
        self.graph = graph
        self.llm = llm
        self.chunker = chunker or MarkdownChunker()
        self.corpus_summary: str = ""
        self.language: str | None = None
        # Prompt set is rebuilt every time language is locked in (explicit or
        # auto-detected). Default to English so the instance is usable for
        # API surface inspection before any ingest.
        self._prompts: dict[str, str] = build_prompts("English")

        self._chunk_sem = asyncio.Semaphore(chunk_concurrency)
        self._chunk_max_retries = chunk_max_retries

        if language:
            self._set_language(language)

    def _set_language(self, lang_iso: str) -> None:
        """Lock in the corpus language and propagate to the SONAR embedder.

        Stored as an ISO 639-1 code on ``self.language``; the embedder gets
        the corresponding FLORES-200 code so SONAR encodes correctly.
        Duck-typed on the embedder so non-SONAR implementations are silently
        skipped — they just won't be language-aware.
        """
        self.language = lang_iso
        lang_en = english_name(lang_iso)
        self._prompts = build_prompts(lang_en)
        # Hand the LLM to the graph too — it's used as the stemmer fallback
        # for languages outside Snowball (kk, uk, ja, zh, …). For supported
        # languages the graph ignores the LLM and uses Snowball locally.
        self.graph.set_language(lang_iso, llm=self.llm, language_name=lang_en)
        embedder = getattr(self.graph, "embedder", None)
        if embedder is not None and hasattr(embedder, "source_lang"):
            embedder.source_lang = resolve_flores_code(lang_iso)
            log.info(
                "Prism: language=%s (embedder source_lang=%s)",
                lang_iso,
                embedder.source_lang,
            )
        else:
            log.info("Prism: language=%s (embedder is not language-aware)", lang_iso)

    async def ingest(self, markdown: str, *, summarize: bool = True) -> list[PrismNode]:
        """Ingest a markdown document into the graph.

        If ``language`` was not set at construction time, the language of
        the first ingested document is auto-detected (heuristic + optional
        LLM fallback) and locked in for the lifetime of the Prism instance.

        Returns the list of newly created nodes. Chunks that fail LLM
        extraction after retries are skipped with a warning rather than
        failing the whole ingest — partial success is more useful than
        no result for large documents.
        """
        if self.language is None:
            detected = await detect_language(markdown, llm=self.llm)
            self._set_language(detected)

        chunks = self.chunker.chunk(markdown)
        if not chunks:
            log.warning("ingest: no chunks produced from input")
            return []

        log.info("ingest: extracting %d chunks", len(chunks))
        results = await asyncio.gather(*(self._chunk_to_blueprint(c) for c in chunks))
        blueprints = [r for r in results if r is not None]

        failed = len(chunks) - len(blueprints)
        if failed:
            log.warning("ingest: %d/%d chunks failed extraction", failed, len(chunks))
        if not blueprints:
            raise RuntimeError("ingest: all chunks failed extraction")

        log.info("ingest: indexing %d nodes", len(blueprints))
        nodes = await self.graph.add_nodes(blueprints)

        if summarize:
            await self._refresh_corpus_summary(blueprints)

        return nodes

    async def _chunk_to_blueprint(self, chunk: Chunk) -> NodeBlueprint | None:
        """LLM extracts a node from a chunk, with manual retry on validation failures."""
        delay = 1.0
        for attempt in range(1, self._chunk_max_retries + 1):
            try:
                async with self._chunk_sem:
                    extraction = await self.llm.complete_structured(
                        system=self._prompts["node_extraction"],
                        user=f"Markdown text:\n{chunk.text}",
                        schema=NodeExtraction,
                        tier="fast",
                    )
                keypoints = list(
                    dict.fromkeys([extraction.header, *extraction.key_phrases])
                )
                return NodeBlueprint(
                    name=extraction.header,
                    text=chunk.text,
                    keypoints=keypoints,
                    paragraph_id=chunk.paragraph_id,
                )
            except Exception as e:
                if attempt >= self._chunk_max_retries:
                    log.error(
                        "chunk extraction failed after %d attempts: %s: %s",
                        self._chunk_max_retries,
                        type(e).__name__,
                        e,
                    )
                    return None
                log.debug(
                    "chunk extraction attempt %d/%d failed: %s",
                    attempt,
                    self._chunk_max_retries,
                    e,
                )
                await asyncio.sleep(delay)
                delay *= 2
        return None

    async def _refresh_corpus_summary(self, blueprints: list[NodeBlueprint]) -> None:
        thumbnails = [
            f"{bp.name}: " + ", ".join(bp.keypoints[:5]) for bp in blueprints
        ]
        joined = "\n".join(thumbnails)
        try:
            result = await self.llm.complete_structured(
                system=self._prompts["corpus_summary"],
                user=f"Section thumbnails:\n{joined}",
                schema=CorpusSummary,
                tier="strong",
            )
            self.corpus_summary = result.summary
            log.info("ingest: corpus summary refreshed (%d chars)", len(self.corpus_summary))
        except Exception as e:
            log.warning(
                "ingest: corpus summarization failed: %s: %s", type(e).__name__, e
            )

    def _language_mismatch_message(self, query_language: str | None) -> str | None:
        """Return a localized hint when the query language differs from the corpus.

        Returns ``None`` (no mismatch) when either side is unknown — we don't
        block retrieval just because the caller didn't tag the query.
        """
        if not query_language or not self.language:
            return None
        if query_language == self.language:
            return None
        return format_mismatch_message(query_language, self.language)

    async def search(
        self,
        query: str,
        *,
        filter_relevance: bool = True,
        query_language: str | None = None,
    ) -> SearchResult:
        q = query.strip() if query else ""
        if not q:
            return SearchResult(query=query, note=_EMPTY_QUERY)

        mismatch = self._language_mismatch_message(query_language)
        if mismatch:
            return SearchResult(query=query, note=mismatch)

        try:
            kp = await self._extract_query_keypoints(q)
        except Exception as e:
            log.error("search: keypoint extraction failed: %s: %s", type(e).__name__, e)
            return SearchResult(query=query, note=f"keypoint extraction error: {e}")

        if not kp.is_searchable:
            return SearchResult(query=query, note=_NON_SEARCHABLE)

        keypoints = list(
            dict.fromkeys(
                s.strip() for s in [*kp.key_phrases, *kp.synonyms] if s and s.strip()
            )
        )
        if not keypoints:
            return SearchResult(query=query, note=_NO_KEYPOINTS)

        log.debug("search: keypoints=%s", keypoints)

        nodes = await self._hybrid_retrieve(keypoints)
        paragraphs = self.graph.collect_paragraphs(nodes)

        if filter_relevance and paragraphs:
            paragraphs = await self._filter_paragraphs(query, paragraphs)

        return SearchResult(
            query=query,
            keypoints=keypoints,
            paragraphs=paragraphs,
            nodes=nodes,
        )

    async def answer(
        self,
        query: str,
        *,
        filter_relevance: bool = True,
        query_language: str | None = None,
    ) -> AnswerResult:
        """End-to-end: retrieve relevant fragments and synthesize an answer.

        Thin opt-in wrapper over :py:meth:`search` plus a single LLM call
        that grounds the answer in the retrieved fragments. Returns the
        underlying ``SearchResult`` on ``AnswerResult.search`` so callers
        can still show citations or fall back to raw fragments.

        When ``query_language`` is given and differs from the corpus
        language, returns immediately with a localized message in the
        query language explaining the mismatch — no retrieval, no LLM
        calls. Use this to politely refuse cross-language queries.

        Short-circuits without calling the summarization LLM when
        retrieval yielded nothing — the ``note`` on the search result is
        propagated.
        """
        mismatch = self._language_mismatch_message(query_language)
        if mismatch:
            return AnswerResult(
                query=query,
                answer=mismatch,
                note=_LANGUAGE_MISMATCH,
            )

        search = await self.search(
            query,
            filter_relevance=filter_relevance,
            query_language=query_language,
        )

        if not search.paragraphs:
            return AnswerResult(
                query=query,
                search=search,
                note=search.note or "no relevant fragments retrieved",
            )

        joined = "\n\n".join(f"- {p}" for p in search.paragraphs)
        user_msg = f"REQUEST:\n{query}\n\nDATA FRAGMENTS:\n{joined}"

        try:
            result = await self.llm.complete_structured(
                system=self._prompts["summarization"],
                user=user_msg,
                schema=Summarization,
                tier="strong",
            )
        except Exception as e:
            log.error("answer: summarization failed: %s: %s", type(e).__name__, e)
            return AnswerResult(
                query=query,
                search=search,
                note=f"summarization error: {e}",
            )

        return AnswerResult(
            query=query,
            answer=result.summary,
            final_summary=result.final_summary,
            search=search,
        )

    async def _extract_query_keypoints(self, query: str) -> QueryKeypoints:
        corpus = self.corpus_summary or "(no corpus summary available yet)"
        user_msg = f"DATA CONTEXT:\n{corpus}\n\nInput:\n{query}"
        return await self.llm.complete_structured(
            system=self._prompts["query_keypoints"],
            user=user_msg,
            schema=QueryKeypoints,
            tier="fast",
        )

    async def _hybrid_retrieve(self, keypoints: list[str]) -> list[PrismNode]:
        vec_nodes, query_vectors = await self.graph.vector_search(keypoints)
        vec_nodes = await self.graph.expand_neighbors(vec_nodes, query_vectors)
        lex_nodes = await self.graph.lexical_search(keypoints)

        seen: set[int] = set()
        combined: list[PrismNode] = []
        for n in (*vec_nodes, *lex_nodes):
            if n.index in seen:
                continue
            seen.add(n.index)
            combined.append(n)
        return combined

    async def _filter_paragraphs(self, query: str, paragraphs: list[str]) -> list[str]:
        async def _judge(paragraph: str) -> str | None:
            user_msg = (
                f"REQUEST:\n{query}\n\n"
                f"INFORMATION:\n```\n{paragraph}\n```"
            )
            try:
                result = await self.llm.complete_structured(
                    system=RELEVANCE_FILTER_PROMPT,
                    user=user_msg,
                    schema=RelevanceFilter,
                    tier="fast",
                )
            except Exception as e:
                log.error(
                    "relevance filter failed: %s: %s", type(e).__name__, e
                )
                return None

            if not result.is_correct:
                return None
            excerpt = result.answer.strip()
            return excerpt or None

        verdicts = await asyncio.gather(*(_judge(p) for p in paragraphs))
        return [p for p in verdicts if p is not None]
