from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
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
    DEFAULT_VECTOR_SIZE = 1024
    DEFAULT_BATCH_SIZE = 300
    DEFAULT_BATCH_CONCURRENCY = 3

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        *,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: Distance = Distance.COSINE,
        hnsw_config: HnswConfigDiff | None = None,
        hnsw_ef: int = DEFAULT_SEARCH_EF,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.hnsw_config = hnsw_config or self.DEFAULT_HNSW_CONFIG
        self.hnsw_ef = hnsw_ef

    async def ensure_collection(self, recreate: bool = False) -> None:
        exists = await self.client.collection_exists(self.collection_name)
        if exists and recreate:
            await self.client.delete_collection(self.collection_name)
            exists = False
        if not exists:
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

        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=qfilter,
            search_params=SearchParams(hnsw_ef=self.hnsw_ef),
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
            points, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qfilter,
                limit=page_size,
                with_payload=True,
                with_vectors=False,
                offset=offset,
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
