"""Lexical features: stemming + per-sentence stem decomposition.

These two payloads are what Prism stores in Qdrant to implement an
AND-style "the query words must actually appear (in their stemmed form),
and they must co-occur inside one sentence" filter — flavor S in design notes.

Index time, for each chunk:

- ``stems``: flat set of unique stems in the chunk. Drives the cheap
  inverted-index pre-filter ("does the chunk contain *all* query stems
  somewhere?").
- ``sentence_stems``: list of per-sentence stem sets. Drives the
  client-side post-filter ("is there a single sentence that contains
  *all* query stems together?").

Stemming tiers (picked per ``lang`` ISO 639-1 code):

1. Snowball — fast, deterministic, no external calls. Used for every
   language in :data:`ISO_TO_SNOWBALL` (25 languages at the time of writing).
2. LLM fallback — async, one call per chunk/query. Used for languages
   outside Snowball when an ``LLMClient`` is wired in (see the ``_llm``
   variants below). Necessary for agglutinative languages like Kazakh
   where Snowball has no algorithm.
3. Identity — no stemming, just lowercase. Last-resort fallback when
   neither Snowball nor an LLM is available (still produces a usable
   AND-filter, just with word-form rather than lemma granularity).

Stopword filtering is intentionally omitted: ``len(s) > 1`` already drops
single-letter noise, and language-specific stopword lists would be a
maintenance burden that pays back only marginal selectivity (common words
appear in most chunks anyway).
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import snowballstemmer

from ..schemas.llm_outputs import LLMStems
from .text import sentence_tokenize

if TYPE_CHECKING:  # pragma: no cover
    from ..llm.client import LLMClient

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)

# ISO 639-1 → Snowball algorithm name. Covers every language Snowball
# ships with at the time of writing; codes outside this set fall back to
# the identity stemmer in :func:`_stemmer_for`.
ISO_TO_SNOWBALL: dict[str, str] = {
    "ar": "arabic",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "fa": "persian",
    "fi": "finnish",
    "fr": "french",
    "hi": "hindi",
    "hu": "hungarian",
    "id": "indonesian",
    "it": "italian",
    "lt": "lithuanian",
    "ne": "nepali",
    "nl": "dutch",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sr": "serbian",
    "sv": "swedish",
    "ta": "tamil",
    "tr": "turkish",
}


def _identity_stem(words: list[str]) -> list[str]:
    return list(words)


@lru_cache(maxsize=8)
def _stemmer_for(iso: str | None) -> Callable[[list[str]], list[str]]:
    """Return a ``words -> stems`` callable for ``iso`` (ISO 639-1).

    Unknown or ``None`` codes get the identity stemmer — lexical AND still
    works on raw lowercased word forms, just without morphological folding.
    """
    name = ISO_TO_SNOWBALL.get(iso or "")
    if name is None:
        return _identity_stem
    return snowballstemmer.stemmer(name).stemWords


def stem_tokens(text: str, lang: str | None = None) -> list[str]:
    """Tokenize ``text`` into a list of stems.

    Order is preserved (caller can use it to compute window pairs etc.),
    duplicates are *not* removed here — that is a concern of whoever
    builds the payload. Single-character stems are dropped as noise.
    """
    if not text:
        return []
    words = _WORD_RE.findall(text.lower())
    stems = _stemmer_for(lang)(words)
    return [s for s in stems if s and len(s) > 1]


def chunk_lexical_payload(
    text: str,
    lang: str | None = None,
) -> dict[str, list[str] | list[list[str]]]:
    """Compute the lexical payload of a chunk.

    Returns a dict shaped for Qdrant payload:

    - ``stems``: list of unique stems in the chunk (order preserved by
      first occurrence so debug dumps stay readable).
    - ``sentence_stems``: list of sentence-level unique stem sets, in
      document order. Each inner list is what the client-side post-filter
      checks against ("does any sentence contain all query stems?").

    Empty/whitespace-only sentences are dropped.
    """
    sentences = sentence_tokenize(text)
    sentence_stems: list[list[str]] = []
    chunk_seen: dict[str, None] = {}

    for sent in sentences:
        stems = stem_tokens(sent, lang)
        if not stems:
            continue
        sent_unique: list[str] = []
        sent_seen: set[str] = set()
        for s in stems:
            if s not in sent_seen:
                sent_seen.add(s)
                sent_unique.append(s)
            if s not in chunk_seen:
                chunk_seen[s] = None
        sentence_stems.append(sent_unique)

    return {
        "stems": list(chunk_seen),
        "sentence_stems": sentence_stems,
    }


def keypoint_stems(text: str, lang: str | None = None) -> list[str]:
    """Stem a query keypoint, returning unique stems in order of first occurrence.

    Same conventions as :func:`chunk_lexical_payload` so the index and
    query sides are guaranteed to agree on tokenization.
    """
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for s in stem_tokens(text, lang):
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def sentence_contains_all(sentence_stems: list[list[str]], required: list[str]) -> bool:
    """Client-side post-filter: is there *any* sentence in the chunk that
    contains all of ``required`` stems simultaneously?

    This is the proximity invariant of flavor S — the AND filter on
    ``stems`` already guarantees that every required stem is *somewhere*
    in the chunk; this function adds "and in the same sentence".
    """
    if not required:
        return True
    req = set(required)
    return any(req.issubset(set(s)) for s in sentence_stems)


# --- LLM fallback stemmer -------------------------------------------------
#
# For languages without a Snowball algorithm (kk, uk, ja, zh, ...) we can
# ask the LLM to do morphological folding. This is asynchronous and costs
# one call per chunk/query, so it should only be used when Snowball is
# unavailable. Determinism is best-effort: index time and query time go
# through the same prompt at temperature 0, but minor drift is possible.

_LLM_STEMMER_PROMPT = """\
You are a morphological analyzer for {language}.

You will receive a JSON array of lowercased words from a single text.
Return the base form (lemma / stem) of each word, preserving order.

Rules:
- Output array length MUST equal the input array length.
- Lowercase everything.
- For proper nouns, foreign words, abbreviations and numbers — return them unchanged (lowercased).
- Strip morphological affixes typical of the language (e.g. for Kazakh strip case/possessive/plural suffixes; for Ukrainian strip case/gender endings).
- Do not translate, paraphrase, or replace synonyms.

Output strictly as JSON of the form: {"stems": ["...", "..."]}
"""


async def _llm_stem_words(
    words: list[str],
    llm: LLMClient,
    language_name: str,
) -> list[str]:
    """Ask the LLM to stem a list of words. Falls back to identity on failure.

    A length mismatch in the response is treated as failure — we can't
    safely map mismatched output back to per-sentence positions.
    """
    if not words:
        return []
    system = _LLM_STEMMER_PROMPT.replace("{language}", language_name)
    try:
        result = await llm.complete_structured(
            system=system,
            user=json.dumps(words, ensure_ascii=False),
            schema=LLMStems,
            tier="fast",
        )
    except Exception as e:
        log.warning(
            "llm stemmer failed (%s: %s) — falling back to identity for %d words",
            type(e).__name__, e, len(words),
        )
        return list(words)
    if len(result.stems) != len(words):
        log.warning(
            "llm stemmer length mismatch (got %d, expected %d) — falling back to identity",
            len(result.stems), len(words),
        )
        return list(words)
    return [s.lower().strip() for s in result.stems]


async def chunk_lexical_payload_llm(
    text: str,
    llm: LLMClient,
    language_name: str,
) -> dict[str, list[str] | list[list[str]]]:
    """LLM-backed version of :func:`chunk_lexical_payload`.

    One LLM call per chunk: words from all sentences are flattened, sent
    together, then re-mapped back to per-sentence position. This keeps the
    payload shape identical to the Snowball path so downstream filters
    don't need to know which stemmer ran.
    """
    sentences = sentence_tokenize(text)
    sentence_words = [_WORD_RE.findall(s.lower()) for s in sentences]
    flat_words = [w for sw in sentence_words for w in sw]
    if not flat_words:
        return {"stems": [], "sentence_stems": []}

    stems_flat = await _llm_stem_words(flat_words, llm, language_name)

    sentence_stems: list[list[str]] = []
    chunk_seen: dict[str, None] = {}
    idx = 0
    for words in sentence_words:
        n = len(words)
        sent_raw = stems_flat[idx : idx + n]
        idx += n
        sent_unique: list[str] = []
        sent_seen: set[str] = set()
        for s in sent_raw:
            if not s or len(s) <= 1:
                continue
            if s not in sent_seen:
                sent_seen.add(s)
                sent_unique.append(s)
            if s not in chunk_seen:
                chunk_seen[s] = None
        if sent_unique:
            sentence_stems.append(sent_unique)

    return {"stems": list(chunk_seen), "sentence_stems": sentence_stems}


async def keypoint_stems_llm(
    text: str,
    llm: LLMClient,
    language_name: str,
) -> list[str]:
    """LLM-backed version of :func:`keypoint_stems` for a single query."""
    if not text:
        return []
    words = _WORD_RE.findall(text.lower())
    if not words:
        return []
    raw = await _llm_stem_words(words, llm, language_name)
    out: list[str] = []
    seen: set[str] = set()
    for s in raw:
        if not s or len(s) <= 1:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
