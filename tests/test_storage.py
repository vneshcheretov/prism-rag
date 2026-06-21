"""QdrantBackend read-path retries: a transient transport blip during a read
should be retried, not surfaced as a failure."""

from types import SimpleNamespace

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from prism import QdrantBackend


class FlakyClient:
    """Fake Qdrant client: fail the first ``fail_times`` calls, then succeed."""

    def __init__(self, fail_times: int, error: Exception | None = None) -> None:
        self.fail_times = fail_times
        self.error = error or ResponseHandlingException(RuntimeError("connection blip"))
        self.calls = 0

    async def query_points(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return SimpleNamespace(
            points=[SimpleNamespace(id=1, payload={"node_id": 5}, score=0.9)]
        )


async def test_query_retries_transient_then_succeeds():
    backend = QdrantBackend(FlakyClient(fail_times=2), collection_name="t")
    hits = await backend.query([0.0, 0.0, 0.0, 0.0], limit=1)
    assert hits[0].node_id == 5
    assert backend.client.calls == 3  # two failures + one success


async def test_query_gives_up_after_retries():
    backend = QdrantBackend(FlakyClient(fail_times=99), collection_name="t")
    with pytest.raises(ResponseHandlingException):
        await backend.query([0.0, 0.0, 0.0, 0.0], limit=1)
    assert backend.client.calls == QdrantBackend.DEFAULT_READ_RETRIES


async def test_query_does_not_retry_non_transient():
    # A 4xx-style error (UnexpectedResponse) is deterministic — surface it now.
    err = UnexpectedResponse(400, "Bad Request", b"bad", headers=None)
    backend = QdrantBackend(FlakyClient(fail_times=99, error=err), collection_name="t")
    with pytest.raises(UnexpectedResponse):
        await backend.query([0.0, 0.0, 0.0, 0.0], limit=1)
    assert backend.client.calls == 1  # no retry
