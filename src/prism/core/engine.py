from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..llm.client import LLMClient
from ..llm.prompts import (
    CORPUS_SUMMARY_PROMPT,
    NODE_EXTRACTION_PROMPT,
    QUERY_KEYPOINTS_PROMPT,
    RELEVANCE_FILTER_PROMPT,
)
from ..schemas.llm_outputs import (
    CorpusSummary,
    NodeExtraction,
    QueryKeypoints,
    RelevanceFilter,
)
from .chunker import Chunk, MarkdownChunker
from .graph import PrismGraph
from .node import NodeBlueprint, PrismNode

log = logging.getLogger(__name__)

_EMPTY_QUERY = "empty query"
_NON_SEARCHABLE = "query is not an information request"
_NO_KEYPOINTS = "no searchable keypoints could be extracted"


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
        chunk_concurrency: int = 10,
        chunk_max_retries: int = 3,
    ) -> None:
        self.graph = graph
        self.llm = llm
        self.chunker = chunker or MarkdownChunker()
        self.corpus_summary: str = ""

        self._chunk_sem = asyncio.Semaphore(chunk_concurrency)
        self._chunk_max_retries = chunk_max_retries

    async def ingest(self, markdown: str, *, summarize: bool = True) -> list[PrismNode]:
        """Ingest a markdown document into the graph.

        Returns the list of newly created nodes. Chunks that fail LLM
        extraction after retries are skipped with a warning rather than
        failing the whole ingest — partial success is more useful than
        no result for large documents.
        """
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
                        system=NODE_EXTRACTION_PROMPT,
                        user=f"Chunk:\n{chunk.text}",
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
                system=CORPUS_SUMMARY_PROMPT,
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

    async def search(
        self,
        query: str,
        *,
        filter_relevance: bool = True,
    ) -> SearchResult:
        q = query.strip() if query else ""
        if not q:
            return SearchResult(query=query, note=_EMPTY_QUERY)

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

    async def _extract_query_keypoints(self, query: str) -> QueryKeypoints:
        corpus = self.corpus_summary or "(no corpus summary available yet)"
        user_msg = f"Corpus context:\n{corpus}\n\nQuery:\n{query}"
        return await self.llm.complete_structured(
            system=QUERY_KEYPOINTS_PROMPT,
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
            try:
                result = await self.llm.complete_structured(
                    system=RELEVANCE_FILTER_PROMPT,
                    user=f"PARAGRAPH:\n{paragraph}\n\nREQUEST:\n{query}",
                    schema=RelevanceFilter,
                    tier="fast",
                )
                return paragraph if result.is_relevant else None
            except Exception as e:
                log.error(
                    "relevance filter failed: %s: %s", type(e).__name__, e
                )
                return None

        verdicts = await asyncio.gather(*(_judge(p) for p in paragraphs))
        return [p for p in verdicts if p is not None]
