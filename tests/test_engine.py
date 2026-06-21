from typing import TypeVar

import numpy as np
import pytest
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from prism import ChatSession, Embedder, IngestError, Prism, PrismGraph, QdrantBackend
from prism.core.engine import _NON_SEARCHABLE

T = TypeVar("T", bound=BaseModel)


class FakeEmbedder(Embedder):
    @property
    def dim(self) -> int:
        return 8

    async def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 8), dtype=np.float32)


class StubLLM:
    """LLMProvider stub that records calls and rejects every query as
    non-searchable, so search() short-circuits before touching Qdrant."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete_structured(
        self, system: str, user: str, schema: type[T], *, tier: str = "fast"
    ) -> T:
        self.calls.append((system, user))
        return schema(
            language="en",
            is_searchable=False,
            short_summary="x",
            key_phrases=[],
            synonyms=[],
        )

    async def complete_text(
        self, system: str, user: str, *, tier: str = "fast"
    ) -> str:
        return "en"


def _prism(llm: StubLLM, *, language: str | None = None) -> Prism:
    qdrant = QdrantBackend(
        AsyncQdrantClient(location=":memory:"), collection_name="test"
    )
    return Prism(PrismGraph(qdrant, FakeEmbedder()), llm, language=language)


async def test_search_flags_translated_when_query_language_differs():
    # corpus ru, decomposition reports query language en -> translated
    res = await _prism(StubLLM(), language="ru").search("can I bring my dog?")
    assert res.translated is True


async def test_search_not_translated_when_no_corpus_language():
    # no corpus language locked yet -> never flagged as translated
    res = await _prism(StubLLM()).search("can I bring my dog?")
    assert res.translated is False


async def test_ingest_blank_input_raises_before_locking_language():
    prism = _prism(StubLLM())
    with pytest.raises(IngestError):
        await prism.ingest("   \n  ")
    # blank input must not lock in a language
    assert prism.language is None


async def test_search_passes_history_to_query_decomposition():
    llm = StubLLM()
    res = await _prism(llm).search(
        "а с кошкой?",
        history=[
            {"role": "user", "content": "можно ли с собакой?"},
            {"role": "assistant", "content": "Да, до 5 кг."},
        ],
    )
    assert res.note == _NON_SEARCHABLE
    _, user_msg = llm.calls[0]
    assert "DIALOGUE HISTORY" in user_msg
    assert "User: можно ли с собакой?" in user_msg
    assert "Assistant: Да, до 5 кг." in user_msg
    assert user_msg.rstrip().endswith("а с кошкой?")


async def test_search_without_history_has_no_history_block():
    llm = StubLLM()
    await _prism(llm).search("можно ли с собакой?")
    _, user_msg = llm.calls[0]
    assert "DIALOGUE HISTORY" not in user_msg


async def test_chat_session_accumulates_history_and_feeds_it_back():
    llm = StubLLM()
    session = ChatSession(_prism(llm))

    first = await session.ask("можно ли с собакой?")
    assert first.note == _NON_SEARCHABLE
    # user turn recorded; no empty assistant reply for a short-circuited turn
    assert session.messages == [
        {"role": "user", "content": "можно ли с собакой?"}
    ]

    await session.ask("а с кошкой?")
    _, second_user_msg = llm.calls[1]
    assert "User: можно ли с собакой?" in second_user_msg

    session.clear()
    assert session.messages == []


def test_format_history_truncates_and_skips_blanks():
    messages = []
    for i in range(10):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    block = Prism._format_history(messages)
    assert "q3" not in block  # only the last 12 messages survive
    assert "q4" in block and "q9" in block

    assert Prism._format_history(None) == ""
    assert Prism._format_history([]) == ""
    assert Prism._format_history([{"role": "user", "content": "  "}]) == ""
    # non-dialogue roles are skipped, not rendered
    assert Prism._format_history([{"role": "system", "content": "x"}]) == ""
    # a user message without an assistant reply yet is still rendered
    assert "User: q" in Prism._format_history([{"role": "user", "content": "q"}])
