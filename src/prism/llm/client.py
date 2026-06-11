from __future__ import annotations

import logging
import os
from typing import TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .base import Tier

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class LLMClient:
    """Async OpenAI client tuned for the Prism RAG pipeline.

    Exposes two model tiers so that the rest of the codebase declares
    *intent* (``fast`` vs ``strong``) rather than hard-coding model names:

    - ``fast`` — high-volume calls: keypoint extraction, result filtration,
      per-chunk node generation. Defaults to ``gpt-4o-mini``.
    - ``strong`` — one-off heavy reasoning: corpus-wide summarization,
      judge-style evaluation. Defaults to ``gpt-4o``.

    All calls retry on transient errors (rate limits, timeouts, 5xx,
    connection drops) with random-exponential backoff. 4xx errors are
    treated as deterministic and surfaced immediately.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        fast_model: str | None = None,
        strong_model: str | None = None,
        temperature: float | None = 0.0,
        max_retries: int = 5,
        timeout: float = 60.0,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.fast_model = fast_model or os.getenv("PRISM_LLM_FAST_MODEL", "gpt-4o-mini")
        self.strong_model = strong_model or os.getenv("PRISM_LLM_STRONG_MODEL", "gpt-4o")
        self.temperature = temperature
        self.max_retries = max_retries

        self.client = client or AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=timeout,
        )

    def _model_for(self, tier: Tier) -> str:
        return self.strong_model if tier == "strong" else self.fast_model

    def _retryer(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(_RETRYABLE_ERRORS),
            reraise=True,
        )

    def _build_messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _resolve_temperature(self, override: float | None) -> float | None:
        if override is not None:
            return override
        return self.temperature

    async def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        tier: Tier = "fast",
        temperature: float | None = None,
        model: str | None = None,
    ) -> T:
        """Call the model and parse the response into ``schema``.

        Uses OpenAI's structured-output mode: the SDK validates the model's
        JSON output against the Pydantic schema before returning, so the
        caller gets a typed instance or an exception — no manual parsing.
        """
        model_name = model or self._model_for(tier)
        temp = self._resolve_temperature(temperature)
        messages = self._build_messages(system, user)

        log.debug("llm.parse model=%s schema=%s", model_name, schema.__name__)

        async for attempt in self._retryer():
            with attempt:
                kwargs: dict[str, object] = {
                    "model": model_name,
                    "messages": messages,
                    "response_format": schema,
                }
                if temp is not None:
                    kwargs["temperature"] = temp

                response = await self.client.beta.chat.completions.parse(**kwargs)
                message = response.choices[0].message

                if message.refusal:
                    raise RuntimeError(f"Model refused to answer: {message.refusal}")

                if message.parsed is None:
                    raise RuntimeError(
                        f"Model returned no parsed content for schema {schema.__name__}"
                    )

                return message.parsed

        raise RuntimeError("retry loop exited without returning")  # pragma: no cover

    async def complete_text(
        self,
        system: str,
        user: str,
        *,
        tier: Tier = "fast",
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        """Call the model and return the raw text response."""
        model_name = model or self._model_for(tier)
        temp = self._resolve_temperature(temperature)
        messages = self._build_messages(system, user)

        log.debug("llm.text model=%s", model_name)

        async for attempt in self._retryer():
            with attempt:
                kwargs: dict[str, object] = {
                    "model": model_name,
                    "messages": messages,
                }
                if temp is not None:
                    kwargs["temperature"] = temp

                response = await self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""

        raise RuntimeError("retry loop exited without returning")  # pragma: no cover

    async def aclose(self) -> None:
        await self.client.close()
