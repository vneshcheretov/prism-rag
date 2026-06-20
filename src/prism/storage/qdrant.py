from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    SearchParams,
    VectorParams,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

log = logging.getLogger(__name__)

T = TypeVar("T")

# Connection-level failures worth retrying on reads. ``ResponseHandlingException``
# wraps transport errors (timeouts, dropped/closed connections); HTTP 4xx
# surface as ``UnexpectedResponse`` and are deliberately not retried.
_READ_RETRYABLE: tuple[type[BaseException], ...] = (ResponseHandlingException,)


@dataclass(slots=True)
class QdrantHit:
    """A search hit translated into the domain types used by the graph."""

    point_id: int
    node_id: int
    score: float


@dataclass(slots=True)
class LexicalCandidate:
    """A chunk surviving the stem AND-filter, ready for sentence post-filter.

    ``sentence_stems`` is fetched from the Qdrant payload of the node's
    lexical anchor point, so the client side can check the "all query
    stems share one sentence" invariant without a second round-trip.
    """

    point_id: int
    node_id: int
    sentence_stems: list[list[str]]


@dataclass(slots=True)
class StoredPoint:
    """A raw point read back from Qdrant during graph rehydration."""

    point_id: int
    payload: dict[str, Any]
    vector: list[float] | None = None


class QdrantBackend:
    """Thin async wrapper around ``AsyncQdrantClient`` for Prism.

    Collapses the Qdrant API surface to the four operations the rest of
    the codebase actually needs (``ensure_collection``, ``upsert``,
    ``query``, ``delete``) and adds:

    - HNSW defaults matching the production tuning of the source library
      (``m=256``, ``ef_construct=512``, search-time ``hnsw_ef=1024``).
    - Concurrency-bounded, retrying bulk upsert. Batches are pushed in
      parallel with a small semaphore so a large ingestion does not flood
      Qdrant, and each batch retries on transient errors.
    - Returns flat ``QdrantHit`` records with ``node_id`` already lifted
      out of the payload, so the graph layer never touches Qdrant types.
    """

    DEFAULT_HNSW_CONFIG = HnswConfigDiff(m=256, ef_construct=512)
    DEFAULT_SEARCH_EF = 1024
    DEFAULT_BATCH_SIZE = 300
    DEFAULT_BATCH_CONCURRENCY = 3
    DEFAULT_READ_RETRIES = 3

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        *,
        vector_size: int | None = None,
        distance: Distance = Distance.COSINE,
        hnsw_config: HnswConfigDiff | None = None,
        hnsw_ef: int = DEFAULT_SEARCH_EF,
    ) -> None:
        # ``vector_size`` may be left as None: PrismGraph fills it in from
        # ``embedder.dim``, which keeps the embedder the single source of
        # truth for dimensionality. Set it explicitly only when using the
        # backend standalone; PrismGraph will then verify it matches the
        # embedder.
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.hnsw_config = hnsw_config or self.DEFAULT_HNSW_CONFIG
        self.hnsw_ef = hnsw_ef

    async def ensure_collection(self, recreate: bool = False) -> None:
        if self.vector_size is None:
            raise ValueError(
                "QdrantBackend.vector_size is not set. Pass vector_size=... "
                "explicitly or create the graph via PrismGraph(qdrant, embedder), "
                "which propagates embedder.dim."
            )
        exists = await self.client.collection_exists(self.collection_name)
        if exists and recreate:
            await self.client.delete_collection(self.collection_name)
            exists = False
        if exists:
            await self._check_existing_dim()
        else:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=self.distance
                ),
                hnsw_config=self.hnsw_config,
            )
            log.info(
                "created Qdrant collection %s (dim=%d, distance=%s)",
                self.collection_name,
                self.vector_size,
                self.distance,
            )
        await self._ensure_payload_indexes()

    async def _check_existing_dim(self) -> None:
        """Fail fast when reusing a collection built for another embedder."""
        info = await self.client.get_collection(self.collection_name)
        params = info.config.params.vectors
        existing = params.size if isinstance(params, VectorParams) else None
        if existing is not None and existing != self.vector_size:
            raise ValueError(
                f"collection {self.collection_name!r} already exists with "
                f"dim={existing}, but the configured embedding dim is "
                f"{self.vector_size}. Changing the embedder requires "
                "reindexing — recreate the collection (recreate=True) or "
                "use a different collection_name."
            )

    async def _ensure_payload_indexes(self) -> None:
        """Create keyword indexes on lexical payload fields.

        ``stems`` powers the AND-filter pre-pass for lexical search;
        ``node_id`` powers vector-search filtering and bulk deletes. Both
        need to be inverted-indexed on the server side for the operations
        to stay sub-millisecond as the collection grows.
        """
        for field, schema in (
            ("stems", PayloadSchemaType.KEYWORD),
            ("node_id", PayloadSchemaType.INTEGER),
            ("kind", PayloadSchemaType.KEYWORD),
        ):
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as e:
                log.debug("payload index %s already present or skipped: %s", field, e)

    async def upsert(
        self,
        points: list[PointStruct],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        concurrency: int = DEFAULT_BATCH_CONCURRENCY,
        max_retries: int = 4,
    ) -> None:
        if not points:
            return

        batches = [
            points[i : i + batch_size] for i in range(0, len(points), batch_size)
        ]
        sem = asyncio.Semaphore(concurrency)

        async def _upsert_one(batch: list[PointStruct]) -> None:
            async with sem:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(max_retries),
                    wait=wait_random_exponential(multiplier=1, min=2, max=20),
                    retry=retry_if_exception_type(Exception),
                    reraise=True,
                ):
                    with attempt:
                        await self.client.upsert(
                            collection_name=self.collection_name,
                            points=batch,
                            wait=True,
                        )

        await asyncio.gather(*(_upsert_one(b) for b in batches))
        log.info(
            "upserted %d points to %s in %d batches",
            len(points),
            self.collection_name,
            len(batches),
        )

    async def _read(self, call: Callable[[], Awaitable[T]]) -> T:
        """Run a single read against Qdrant, retrying transient transport errors.

        Reads are idempotent, so a dropped/timed-out connection is safe to
        retry — a transient blip during search no longer fails the request.
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.DEFAULT_READ_RETRIES),
            wait=wait_random_exponential(multiplier=0.5, max=5),
            retry=retry_if_exception_type(_READ_RETRYABLE),
            reraise=True,
        ):
            with attempt:
                return await call()
        raise AssertionError("unreachable")  # pragma: no cover

    async def query(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        node_id_filter: list[int] | None = None,
        point_id_filter: list[int] | None = None,
    ) -> list[QdrantHit]:
        qfilter: Filter | None = None
        must: list[Any] = []
        if node_id_filter:
            must.append(
                FieldCondition(key="node_id", match=MatchAny(any=node_id_filter))
            )
        if point_id_filter:
            must.append(HasIdCondition(has_id=point_id_filter))
        if must:
            qfilter = Filter(must=must)

        response = await self._read(
            lambda: self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
                query_filter=qfilter,
                search_params=SearchParams(hnsw_ef=self.hnsw_ef),
            )
        )
        hits: list[QdrantHit] = []
        for p in response.points:
            payload = p.payload or {}
            node_id = payload.get("node_id")
            if node_id is None:
                continue
            hits.append(
                QdrantHit(point_id=int(p.id), node_id=int(node_id), score=float(p.score))
            )
        return hits

    async def lexical_scroll(
        self,
        query_stems: list[str],
        *,
        node_id_filter: list[int] | None = None,
        page_size: int = 256,
        max_candidates: int = 4096,
    ) -> list[LexicalCandidate]:
        """AND-filter the collection on ``stems`` and return lexical candidates.

        Returns at most one record per node (only the lexical anchor
        points carry ``stems`` payload — keypoint vector points do not).
        Caller is expected to apply the sentence-level post-filter using
        :func:`prism.utils.lexical.sentence_contains_all` on the returned
        ``sentence_stems``.

        ``max_candidates`` is a safety cap: with ~200k nodes a popular
        single-stem query could match tens of thousands of points, and
        post-filtering every one of them is pointless. The dense channel
        will recover anything we drop here.
        """
        if not query_stems:
            return []

        must: list[Any] = [
            FieldCondition(key="stems", match=MatchValue(value=s)) for s in query_stems
        ]
        if node_id_filter:
            must.append(
                FieldCondition(key="node_id", match=MatchAny(any=node_id_filter))
            )
        qfilter = Filter(must=must)

        candidates: list[LexicalCandidate] = []
        offset: Any = None
        while True:
            points, next_offset = await self._read(
                partial(
                    self.client.scroll,
                    collection_name=self.collection_name,
                    scroll_filter=qfilter,
                    limit=page_size,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
            )
            for p in points:
                payload = p.payload or {}
                node_id = payload.get("node_id")
                if node_id is None:
                    continue
                sentence_stems = payload.get("sentence_stems") or []
                candidates.append(
                    LexicalCandidate(
                        point_id=int(p.id),
                        node_id=int(node_id),
                        sentence_stems=sentence_stems,
                    )
                )
                if len(candidates) >= max_candidates:
                    return candidates
            if next_offset is None:
                break
            offset = next_offset

        return candidates

    async def scroll_kind(
        self,
        kind: str,
        *,
        with_vectors: bool = False,
        page_size: int = 512,
    ) -> list[StoredPoint]:
        """Return every point whose payload ``kind`` matches, paging through all.

        Used at startup to rehydrate the in-memory graph: ``kind="anchor"``
        carries node metadata, ``kind="aggregate"`` carries the per-node
        aggregate vector. Keypoint points are unmarked and never scrolled
        here, so the bulk of the vectors stays on the server.
        """
        qfilter = Filter(must=[FieldCondition(key="kind", match=MatchValue(value=kind))])
        results: list[StoredPoint] = []
        offset: Any = None
        while True:
            points, next_offset = await self._read(
                partial(
                    self.client.scroll,
                    collection_name=self.collection_name,
                    scroll_filter=qfilter,
                    limit=page_size,
                    with_payload=True,
                    with_vectors=with_vectors,
                    offset=offset,
                )
            )
            for p in points:
                vector = None
                raw_vector = p.vector
                if with_vectors and isinstance(raw_vector, list):
                    vector = [float(x) for x in raw_vector if isinstance(x, int | float)]
                results.append(
                    StoredPoint(point_id=int(p.id), payload=p.payload or {}, vector=vector)
                )
            if next_offset is None:
                break
            offset = next_offset
        return results

    async def retrieve_payload(self, point_id: int) -> dict[str, Any] | None:
        """Read a single point's payload by id, or ``None`` if it is absent."""
        points = await self._read(
            lambda: self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        )
        if not points:
            return None
        return points[0].payload or {}

    async def delete_by_node_ids(self, node_ids: list[int]) -> None:
        if not node_ids:
            return
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="node_id", match=MatchAny(any=node_ids))]
            ),
        )

    async def delete_by_point_ids(self, point_ids: list[int]) -> None:
        if not point_ids:
            return
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=list(point_ids)),
        )

    async def close(self) -> None:
        await self.client.close()
