from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

import numpy as np
from qdrant_client.http.models import PointStruct

from ..embeddings import Embedder
from ..llm.base import LLMProvider
from ..storage.qdrant import QdrantBackend
from ..utils.lexical import (
    ISO_TO_SNOWBALL,
    chunk_lexical_payload,
    chunk_lexical_payload_llm,
    keypoint_stems,
    keypoint_stems_llm,
    sentence_contains_all,
)
from .node import NodeBlueprint, PrismNode

log = logging.getLogger(__name__)


class PrismGraph:
    """In-memory knowledge graph backed entirely by Qdrant.

    Each ingested node contributes ``N + 2`` Qdrant points:

    - One *lexical anchor* point with a placeholder vector (zeros) and the
      lexical payload (``stems`` + ``sentence_stems``). It is the only
      point that carries lexical fields, so a single scroll over the
      AND-filter returns one record per matching node.
    - ``N`` keypoint vectors, one per LLM-extracted phrase, used for
      keypoint-decomposition dense retrieval.
    - One aggregate (mean-of-keypoints) vector. All keypoint and aggregate
      points carry the same ``node_id`` payload so dense hits map back to
      a node regardless of which vector matched.

    Two retrieval modes are exposed:

    - :py:meth:`vector_search` — dense retrieval. Embeds every query
      keypoint plus their aggregate, runs one Qdrant query per vector
      and applies a per-query percentile cutoff so the threshold adapts
      to the score distribution rather than being a fixed cosine value.
    - :py:meth:`lexical_search` — strict AND over Snowball stems of the
      chunk text, plus a sentence-level proximity post-filter (flavor S):
      every query stem must show up *and* coexist in at least one
      sentence of the chunk.

    Plus a 1-hop expansion step (:py:meth:`expand_neighbors`) that pulls
    in pre-computed neighbors of the retrieved nodes when they themselves
    are close to any of the query vectors.
    """

    LEXICAL_ANCHOR_VECTOR_VALUE = 0.0

    def __init__(
        self,
        qdrant: QdrantBackend,
        embedder: Embedder,
        *,
        retrieval_percentile: float = 70.0,
        neighbor_top_k: int = 5,
        neighbor_threshold: float = 0.7,
        expand_threshold: float = 0.8,
    ) -> None:
        self.qdrant = qdrant
        self.embedder = embedder

        # The embedder is the single source of truth for dimensionality.
        # An unset backend inherits embedder.dim; an explicitly set one
        # must match it — anything else fails on the first upsert with an
        # opaque Qdrant error, so we fail fast here instead.
        if qdrant.vector_size is None:
            qdrant.vector_size = embedder.dim
        elif qdrant.vector_size != embedder.dim:
            raise ValueError(
                f"embedding dimension mismatch: QdrantBackend(vector_size="
                f"{qdrant.vector_size}) vs embedder.dim={embedder.dim}. "
                "Omit vector_size to derive it from the embedder."
            )

        self.retrieval_percentile = retrieval_percentile
        self.neighbor_top_k = neighbor_top_k
        self.neighbor_threshold = neighbor_threshold
        self.expand_threshold = expand_threshold

        # ISO 639-1 code used by the lexical pipeline (Snowball stemmer
        # selection). ``None`` means identity-stem — see ``lexical.py``.
        self.lang_iso: str | None = None
        # LLM-backed stemmer for languages outside Snowball. ``None`` means
        # the sync Snowball/identity path is used. Set via ``set_language``.
        self._stem_llm: LLMProvider | None = None
        self._stem_language_name: str | None = None

        self.nodes: dict[int, PrismNode] = {}
        self._paragraph_to_nodes: dict[str, list[int]] = defaultdict(list)
        self._next_node_id = 1
        self._next_point_id = 1
        self._ingest_lock = asyncio.Lock()

    def set_language(
        self,
        iso: str | None,
        *,
        llm: LLMProvider | None = None,
        language_name: str | None = None,
    ) -> None:
        """Set the ISO 639-1 code used by the lexical pipeline.

        Affects future ``add_nodes`` and ``lexical_search`` calls. Already
        indexed payloads keep whatever stems they were built with — callers
        that mix languages in one collection must reindex.

        If ``llm`` is provided together with ``language_name`` (English),
        the LLM is used as the stemmer for any language outside Snowball.
        For Snowball-supported languages the LLM is ignored — the local
        algorithm is always cheaper and more deterministic.
        """
        self.lang_iso = iso
        if llm is not None and language_name and iso and iso not in ISO_TO_SNOWBALL:
            self._stem_llm = llm
            self._stem_language_name = language_name
            log.info("PrismGraph: lexical stemmer = LLM (%s)", language_name)
        else:
            self._stem_llm = None
            self._stem_language_name = None

    @classmethod
    async def create(
        cls,
        qdrant: QdrantBackend,
        embedder: Embedder,
        *,
        recreate: bool = False,
        **kwargs: object,
    ) -> PrismGraph:
        graph = cls(qdrant, embedder, **kwargs)  # type: ignore[arg-type]
        await graph.qdrant.ensure_collection(recreate=recreate)
        return graph

    async def add_nodes(self, blueprints: list[NodeBlueprint]) -> list[PrismNode]:
        if not blueprints:
            return []

        async with self._ingest_lock:
            phrases, counts = self._collect_phrases(blueprints)

            vectors = await self.embedder.embed(phrases)
            log.debug("embedded %d phrases for %d nodes", len(phrases), len(blueprints))

            new_nodes = self._materialize_nodes(blueprints, vectors, counts)
            await self._upsert_to_qdrant(new_nodes, vectors, counts)

        await self.build_neighbors(node_ids=[n.index for n in new_nodes])
        return new_nodes

    @staticmethod
    def _collect_phrases(
        blueprints: list[NodeBlueprint],
    ) -> tuple[list[str], list[int]]:
        """Flatten ``[name, *keypoints]`` per blueprint into a single embed call."""
        phrases: list[str] = []
        counts: list[int] = []
        for bp in blueprints:
            node_phrases = [bp.name, *bp.keypoints]
            phrases.extend(node_phrases)
            counts.append(len(node_phrases))
        return phrases, counts

    def _materialize_nodes(
        self,
        blueprints: list[NodeBlueprint],
        vectors: np.ndarray,
        counts: list[int],
    ) -> list[PrismNode]:
        nodes: list[PrismNode] = []
        offset = 0
        for bp, count in zip(blueprints, counts, strict=True):
            node_vecs = vectors[offset : offset + count]
            offset += count

            avg = Embedder.average_unit_vector(node_vecs)
            node = PrismNode(
                index=self._next_node_id,
                name=bp.name,
                text=bp.text,
                keypoints=bp.keypoints,
                vector=avg,
                paragraph_id=bp.paragraph_id,
            )
            self._next_node_id += 1
            self.nodes[node.index] = node
            self._paragraph_to_nodes[bp.paragraph_id].append(node.index)
            nodes.append(node)
        return nodes

    async def _build_lexical_payload(self, text: str) -> dict[str, list[str] | list[list[str]]]:
        """Dispatch to the LLM stemmer when wired in, else the sync path."""
        if self._stem_llm is not None and self._stem_language_name is not None:
            return await chunk_lexical_payload_llm(
                text, self._stem_llm, self._stem_language_name
            )
        return chunk_lexical_payload(text, self.lang_iso)

    async def _stem_query_keypoint(self, q: str) -> list[str]:
        """Dispatch to the LLM stemmer when wired in, else the sync path."""
        if self._stem_llm is not None and self._stem_language_name is not None:
            return await keypoint_stems_llm(q, self._stem_llm, self._stem_language_name)
        return keypoint_stems(q, self.lang_iso)

    def _lexical_anchor_vector(self) -> list[float]:
        """Zero vector used as a placeholder for the lexical anchor point.

        Qdrant requires every point to carry the named vector; the anchor
        is never returned by dense search (cosine to zero is undefined and
        Qdrant filters it out) so the value does not matter — but the
        shape must match the collection's vector size, which __init__
        guarantees to equal ``embedder.dim``.
        """
        return [self.LEXICAL_ANCHOR_VECTOR_VALUE] * self.embedder.dim

    async def _upsert_to_qdrant(
        self,
        nodes: list[PrismNode],
        keypoint_vectors: np.ndarray,
        counts: list[int],
    ) -> None:
        # Build lexical payloads up front and (when the LLM stemmer is
        # active) in parallel — one network call per node otherwise serialises
        # the whole ingest.
        lex_payloads = await asyncio.gather(
            *(self._build_lexical_payload(n.text) for n in nodes)
        )

        points: list[PointStruct] = []
        offset = 0
        for node, count, lex in zip(nodes, counts, lex_payloads, strict=True):
            node_vecs = keypoint_vectors[offset : offset + count]
            offset += count

            # Lexical anchor: only point carrying stems + sentence_stems.
            points.append(
                PointStruct(
                    id=self._next_point_id,
                    vector=self._lexical_anchor_vector(),
                    payload={
                        "node_id": node.index,
                        "stems": lex["stems"],
                        "sentence_stems": lex["sentence_stems"],
                    },
                )
            )
            self._next_point_id += 1

            # Keypoint vectors plus the aggregate. Dedup near-identical
            # embeddings so we don't burn HNSW slots on duplicates.
            seen: set[tuple] = set()
            for vec in np.vstack([node_vecs, node.vector[np.newaxis, :]]):
                key = tuple(np.round(vec, 6))
                if key in seen:
                    continue
                seen.add(key)
                points.append(
                    PointStruct(
                        id=self._next_point_id,
                        vector=vec.flatten().tolist(),
                        payload={"node_id": node.index},
                    )
                )
                self._next_point_id += 1

        await self.qdrant.upsert(points)

    async def build_neighbors(
        self,
        node_ids: list[int] | None = None,
        *,
        max_concurrent: int = 8,
    ) -> None:
        """Wire up 1-hop edges via cosine similarity.

        Edges are mutual: if ``A`` finds ``B`` similar above the threshold,
        both ``A.neighbors`` and ``B.neighbors`` are updated.
        """
        targets = (
            [self.nodes[nid] for nid in node_ids if nid in self.nodes]
            if node_ids is not None
            else list(self.nodes.values())
        )
        if not targets:
            return

        sem = asyncio.Semaphore(max_concurrent)

        async def _find(node: PrismNode) -> None:
            if node.vector.size == 0:
                return
            async with sem:
                hits = await self.qdrant.query(
                    vector=node.vector.flatten().tolist(),
                    limit=self.neighbor_top_k + 1,
                )

            neighbors: list[int] = []
            seen: set[int] = set()
            for hit in hits:
                if hit.node_id == node.index or hit.node_id in seen:
                    continue
                if hit.score < self.neighbor_threshold:
                    continue
                if hit.node_id not in self.nodes:
                    continue
                seen.add(hit.node_id)
                neighbors.append(hit.node_id)

            await node.add_neighbors(*neighbors)
            for nid in neighbors:
                await self.nodes[nid].add_neighbors(node.index)

        await asyncio.gather(*(_find(n) for n in targets))

    async def vector_search(
        self,
        keypoints: list[str],
        *,
        top_k: int = 10,
        relevant_ids: list[int] | None = None,
    ) -> tuple[list[PrismNode], list[np.ndarray]]:
        """Dense retrieval over Qdrant with keypoint decomposition.

        Embeds each keypoint of the query plus the aggregate of all
        keypoints, runs one Qdrant query per vector, and applies a
        per-query percentile cutoff so the threshold adapts to the
        score distribution rather than being a fixed cosine value.

        Returns the matched nodes (deduplicated) and the list of query
        vectors actually used — the caller can reuse them for neighbor
        expansion against the same query family.
        """
        clean = [kp.strip() for kp in keypoints if kp and kp.strip()]
        if not clean:
            return [], []

        keypoint_vecs = await self.embedder.embed(clean)
        aggregate = Embedder.average_unit_vector(keypoint_vecs)
        query_vectors = [aggregate, *list(keypoint_vecs)]

        node_ids: set[int] = set()
        for qv in query_vectors:
            hits = await self.qdrant.query(
                vector=qv.flatten().tolist(),
                limit=top_k,
                node_id_filter=relevant_ids,
            )
            for hit in self._percentile_filter(hits):
                if hit.node_id in self.nodes:
                    node_ids.add(hit.node_id)

        nodes = [self.nodes[nid] for nid in node_ids]
        log.info(
            "vector search: %d nodes: %s",
            len(nodes),
            [(n.index, n.name) for n in nodes],
        )
        return nodes, query_vectors

    def _percentile_filter(
        self, hits: list, percentile: float | None = None
    ) -> list:
        if not hits:
            return []
        p = percentile if percentile is not None else self.retrieval_percentile
        threshold = float(np.percentile([h.score for h in hits], p))
        return [h for h in hits if h.score >= threshold]

    async def lexical_search(
        self,
        queries: list[str],
        *,
        node_ids: list[int] | None = None,
    ) -> list[PrismNode]:
        """Strict AND-on-stems + sentence-level proximity retrieval.

        For each query string we:

        1. Stem it via Snowball (matching the index-time tokenization).
        2. Hit Qdrant with an AND-filter on the ``stems`` payload — this
           returns lexical anchor points of every chunk that contains
           all of those stems somewhere.
        3. Post-filter each candidate by checking that at least one
           sentence of the chunk contains every query stem together
           (the proximity invariant).

        Results across queries are deduplicated by node id; order
        roughly follows the order queries were passed in.
        """
        if not queries:
            return []

        async def _one(q: str) -> list[int]:
            stems = await self._stem_query_keypoint(q)
            if not stems:
                return []
            cands = await self.qdrant.lexical_scroll(
                stems, node_id_filter=node_ids
            )
            return [
                c.node_id
                for c in cands
                if sentence_contains_all(c.sentence_stems, stems)
            ]

        results = await asyncio.gather(*(_one(q) for q in queries))
        seen: set[int] = set()
        nodes: list[PrismNode] = []
        for batch in results:
            for nid in batch:
                if nid in seen or nid not in self.nodes:
                    continue
                seen.add(nid)
                nodes.append(self.nodes[nid])
        log.info(
            "lexical search: %d nodes: %s",
            len(nodes),
            [(n.index, n.name) for n in nodes],
        )
        return nodes

    async def expand_neighbors(
        self,
        found: list[PrismNode],
        query_vectors: list[np.ndarray],
        *,
        threshold: float | None = None,
    ) -> list[PrismNode]:
        """Pull in 1-hop neighbors that score above ``threshold`` against
        any of the query vectors.

        Cheap and effective for recovering the surrounding context of a
        good hit without doing a second Qdrant query.
        """
        if not found:
            return []
        t = threshold if threshold is not None else self.expand_threshold

        base = {n.index: n for n in found}
        extra: dict[int, PrismNode] = {}

        for node in found:
            for neighbor_id in node.neighbors:
                if neighbor_id in base or neighbor_id in extra:
                    continue
                neighbor = self.nodes.get(neighbor_id)
                if neighbor is None or neighbor.vector.size == 0:
                    continue
                for qv in query_vectors:
                    sim = float(np.dot(qv.flatten(), neighbor.vector.flatten()))
                    if sim >= t:
                        extra[neighbor_id] = neighbor
                        break

        if extra:
            log.info(
                "neighbor expansion: +%d nodes: %s",
                len(extra),
                [(n.index, n.name) for n in extra.values()],
            )
        return list(base.values()) + list(extra.values())

    def collect_paragraphs(self, nodes: list[PrismNode]) -> list[str]:
        """Reconstruct full paragraphs from the retrieved chunks.

        Multiple chunks share a ``paragraph_id`` when they came from the
        same markdown section. For every paragraph touched by the result
        set, we re-join *all* of its chunks (not only the retrieved ones)
        ordered by node index, so the downstream LLM sees the full
        section, not an isolated fragment.
        """
        seen: set[str] = set()
        paragraphs: list[str] = []
        for node in nodes:
            pid = node.paragraph_id
            if pid in seen:
                continue
            seen.add(pid)
            sibling_ids = sorted(self._paragraph_to_nodes.get(pid, []))
            joined = " ".join(self.nodes[i].text for i in sibling_ids if i in self.nodes)
            if joined:
                paragraphs.append(joined)
        return paragraphs

    async def delete_nodes(self, node_ids: list[int]) -> None:
        if not node_ids:
            return
        await self.qdrant.delete_by_node_ids(node_ids)
        for nid in node_ids:
            node = self.nodes.pop(nid, None)
            if node is None:
                continue
            siblings = self._paragraph_to_nodes.get(node.paragraph_id, [])
            if nid in siblings:
                siblings.remove(nid)
            if not siblings:
                self._paragraph_to_nodes.pop(node.paragraph_id, None)
            for other in self.nodes.values():
                if nid in other.neighbors:
                    other.neighbors.remove(nid)
