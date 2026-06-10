from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class NodeBlueprint:
    """Input to ``PrismGraph.add_nodes``.

    Carries everything needed to materialize a ``PrismNode`` *except* the
    embedding vector — the graph batches embedding calls across all
    blueprints before constructing the actual nodes.
    """

    name: str
    text: str
    keypoints: list[str]
    paragraph_id: str


@dataclass(slots=True)
class PrismNode:
    """One node in the knowledge graph.

    ``vector`` is the aggregate keypoint vector (mean of unit vectors,
    renormalized) — used for neighbor scoring and as one of the query
    vectors during retrieval. Individual keypoint vectors live in Qdrant
    only; we don't keep them on the node to save memory.

    ``neighbors`` is the list of 1-hop similar nodes (by cosine on the
    aggregate vector) above a similarity threshold. Edges are mutual —
    when ``A.add_neighbors(B)`` is called, ``B`` should also be told of
    ``A`` by the caller.
    """

    index: int
    name: str
    text: str
    keypoints: list[str]
    vector: np.ndarray
    paragraph_id: str
    neighbors: list[int] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    async def add_neighbors(self, *ids: int) -> None:
        async with self._lock:
            for nid in ids:
                if nid != self.index and nid not in self.neighbors:
                    self.neighbors.append(nid)

    def __hash__(self) -> int:
        return hash(self.index)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PrismNode) and self.index == other.index
