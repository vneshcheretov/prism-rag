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
    from ..llm.client import LLMClient

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
    llm: LLMClient | None = None,
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


# Native names of the corpus language, used as a standalone proper-noun slot
# in the mismatch templates so we don't have to inflect grammar per template.
LANGUAGE_NATIVE_NAMES: dict[str, str] = {
    "ru": "Русский",
    "en": "English",
    "es": "Español",
    "ja": "日本語",
    "kk": "Қазақша",
    "de": "Deutsch",
    "fr": "Français",
    "it": "Italiano",
    "pt": "Português",
    "zh": "中文",
    "ar": "العربية",
    "tr": "Türkçe",
    "uk": "Українська",
    "pl": "Polski",
    "nl": "Nederlands",
}

# Message templates keyed by *query* language. ``{language}`` is the native
# name of the corpus language (so the script may differ from the rest of the
# sentence — that's intentional and clearer than inflecting names per locale).
# Unknown query languages fall back to the English template.
MISMATCH_TEMPLATES: dict[str, str] = {
    "en": "The corpus is available in {language} only. Please ask your question in this language.",
    "ru": "Корпус доступен только на языке {language}. Пожалуйста, задайте вопрос на этом языке.",
    "es": "El corpus está disponible solo en {language}. Por favor, haga su pregunta en este idioma.",
    "ja": "コーパスは{language}のみで利用可能です。その言語で質問してください。",
    "kk": "Корпус тек {language} тілінде қолжетімді. Сұрағыңызды осы тілде қойыңыз.",
    "de": "Der Korpus ist nur auf {language} verfügbar. Bitte stellen Sie Ihre Frage in dieser Sprache.",
    "fr": "Le corpus est disponible uniquement en {language}. Veuillez poser votre question dans cette langue.",
    "it": "Il corpus è disponibile solo in {language}. Per favore, ponga la domanda in questa lingua.",
    "pt": "O corpus está disponível apenas em {language}. Por favor, faça sua pergunta neste idioma.",
    "zh": "语料库仅支持{language}。请使用该语言提问。",
    "ar": "المجموعة متاحة فقط بـ {language}. يرجى طرح سؤالك بهذه اللغة.",
    "tr": "Külliyat yalnızca {language} dilinde mevcuttur. Lütfen sorunuzu bu dilde sorun.",
    "uk": "Корпус доступний лише {language} мовою. Будь ласка, ставте питання цією мовою.",
    "pl": "Korpus jest dostępny tylko w języku {language}. Proszę zadać pytanie w tym języku.",
    "nl": "Het corpus is alleen beschikbaar in {language}. Stel uw vraag in deze taal.",
}


def format_mismatch_message(query_language: str, corpus_language: str) -> str:
    """Localized message telling the user to query in the corpus language.

    The template is picked by ``query_language`` (so the user can read it),
    falling back to English for unknown query languages. The corpus name
    slot uses ``LANGUAGE_NATIVE_NAMES``; an unknown corpus code is rendered
    as its raw ISO code rather than substituted with something fake.
    """
    template = MISMATCH_TEMPLATES.get(query_language, MISMATCH_TEMPLATES["en"])
    name = LANGUAGE_NATIVE_NAMES.get(corpus_language, corpus_language)
    return template.format(language=name)
