from __future__ import annotations

import logging

from .engine import AnswerResult, Prism

log = logging.getLogger(__name__)


class ChatSession:
    """Stateful convenience wrapper for one conversation over a stateless engine.

    :class:`Prism` deliberately takes dialogue ``history`` as an argument —
    one engine instance serves many concurrent conversations and never
    stores per-user state. ``ChatSession`` is the thin layer that owns that
    state for a single dialogue: it keeps an OpenAI-style message log and
    feeds it back into :py:meth:`Prism.answer` on every turn.

    Create one session per conversation; sessions share the engine freely.

    >>> session = ChatSession(prism)
    >>> await session.ask("можно ли с собакой?")
    >>> await session.ask("а с кошкой?")   # resolved against the history
    """

    def __init__(self, prism: Prism) -> None:
        self.prism = prism
        self.messages: list[dict[str, str]] = []

    async def ask(
        self,
        query: str,
        *,
        filter_relevance: bool = True,
        query_language: str | None = None,
    ) -> AnswerResult:
        """Answer ``query`` in the context of this session's history.

        The user message is always recorded; the assistant message only
        when an answer was actually produced — short-circuited turns
        (chit-chat, language mismatch) leave no empty replies behind.
        """
        result = await self.prism.answer(
            query,
            filter_relevance=filter_relevance,
            query_language=query_language,
            history=self.messages,
        )
        self.messages.append({"role": "user", "content": query})
        if result.answer:
            self.messages.append({"role": "assistant", "content": result.answer})
        return result

    def clear(self) -> None:
        """Forget the conversation, keep the engine."""
        self.messages.clear()
