"""Language detection and query-vs-corpus language mismatch messaging.

Two-stage detection cascade by design:

1. Fast heuristic via ``langdetect`` — millisecond, free, deterministic.
   Accurate enough for any text longer than a sentence in a major language.

2. Optional LLM fallback when the heuristic is below ``threshold`` confidence
   and an ``LLMClient`` is passed in. Used for very short or mixed inputs.

A non-empty ``fallback`` ISO code (default ``"en"``) is returned when both
stages fail to produce anything — callers always get a usable code.

For cross-language queries (user asks in language X about a corpus indexed
in language Y), ``format_mismatch_message`` produces a localized message in
the *query* language explaining that the corpus is in language Y, so the
user can actually read the explanation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langdetect import DetectorFactory, LangDetectException, detect_langs

if TYPE_CHECKING:  # pragma: no cover
    from ..llm.base import LLMProvider

# Make langdetect deterministic across runs — by default it seeds its
# internal RNG from the system, which produces different results on the
# same input across invocations.
DetectorFactory.seed = 0

log = logging.getLogger(__name__)

_LLM_PROMPT = (
    "Identify the language of the given text. "
    "Reply with ONLY the lowercase ISO 639-1 code (two letters, e.g. 'ru', 'en', 'es', 'ja'). "
    "No punctuation, no explanation."
)


async def detect_language(
    text: str,
    *,
    llm: LLMProvider | None = None,
    threshold: float = 0.90,
    fallback: str = "en",
    sample_chars: int = 2000,
) -> str:
    """Detect the language of ``text``, returning an ISO 639-1 code.

    ``threshold`` is the langdetect confidence below which the LLM fallback
    is invoked (when ``llm`` is provided). On a short, ambiguous input
    both stages can still fail — in that case the heuristic's best guess
    is returned, or ``fallback`` if even that is unavailable.
    """
    sample = (text or "").strip()[:sample_chars]
    if not sample:
        return fallback

    low_conf_guess: str | None = None
    try:
        results = detect_langs(sample)
        if results:
            top = results[0]
            if top.prob >= threshold:
                return top.lang
            low_conf_guess = top.lang
            log.debug(
                "language: low-confidence heuristic guess %s (%.2f)",
                top.lang,
                top.prob,
            )
    except LangDetectException as e:
        log.debug("langdetect failed: %s", e)

    if llm is not None:
        try:
            raw = await llm.complete_text(
                system=_LLM_PROMPT,
                user=sample,
                tier="fast",
            )
            iso = raw.strip().lower()[:2]
            if len(iso) == 2 and iso.isalpha():
                log.debug("language: LLM fallback returned %s", iso)
                return iso
            log.warning("language: LLM returned unusable value %r", raw)
        except Exception as e:
            log.warning("language: LLM fallback failed: %s: %s", type(e).__name__, e)

    return low_conf_guess or fallback


# English names of languages, used to interpolate into prompts as the
# instruction-target language (e.g. "your output language MUST be Russian").
# Modern LLMs follow English language names reliably for major languages.
LANGUAGE_ENGLISH_NAMES: dict[str, str] = {
    "ru": "Russian",
    "en": "English",
    "es": "Spanish",
    "ja": "Japanese",
    "kk": "Kazakh",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ar": "Arabic",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "pl": "Polish",
    "nl": "Dutch",
}


def english_name(iso: str) -> str:
    """Return the English name of an ISO 639-1 code, or the code itself as fallback."""
    return LANGUAGE_ENGLISH_NAMES.get(iso, iso)
