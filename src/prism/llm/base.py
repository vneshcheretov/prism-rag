"""Structural typing for the LLM layer.

``Prism`` does not require the bundled OpenAI-backed :class:`~prism.llm.client.LLMClient`
— any object satisfying the :class:`LLMProvider` protocol can drive the
pipeline: keypoint extraction, relevance filtering, summarization, language
detection and the LLM-stemmer fallback all go through these two methods.

Implementations are responsible for their own retries and rate limiting;
the pipeline treats every call as a single attempt that either returns or
raises.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

Tier = Literal["fast", "strong"]
"""Intent-level model selector.

``fast`` is used for high-volume calls (per-chunk extraction, query
decomposition, relevance filtering); ``strong`` for one-off heavy calls
(corpus summary, answer synthesis). Implementations may map both tiers to
the same model.
"""

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    """Minimal LLM interface the Prism pipeline depends on."""

    async def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        tier: Tier = "fast",
    ) -> T:
        """Call the model and return a validated instance of ``schema``.

        Must either return a schema instance or raise — the pipeline never
        inspects raw text from this method.
        """
        ...

    async def complete_text(
        self,
        system: str,
        user: str,
        *,
        tier: Tier = "fast",
    ) -> str:
        """Call the model and return the raw text response."""
        ...
