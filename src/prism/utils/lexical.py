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

Snowball is used because it covers Russian declension cases that a
naive tokenizer would miss (``новая/новые/новой → нов``), and supports
the same set of languages we already use downstream — without pulling
in NLTK or any heavy NLP dependency.
"""
from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

from .text import sentence_tokenize

_WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)

# A small Russian + English function-word list. We strip these before
# stemming because they'd be present in essentially every chunk —
# carrying them through to the AND filter only inflates payload size
# without changing selectivity.
_STOPWORDS: frozenset[str] = frozenset(
    {
        # Russian
        "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
        "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
        "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
        "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
        "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
        "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
        "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
        "может", "они", "тут", "где", "есть", "надо", "ней", "для",
        "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто",
        "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда",
        "кто", "этот", "того", "потому", "этого", "какой", "совсем",
        "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы",
        "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда",
        "можно", "при", "наконец", "два", "об", "другой", "хоть",
        "после", "над", "больше", "тот", "через", "эти", "нас", "про",
        "всего", "них", "какая", "много", "разве", "три", "эту", "моя",
        "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше",
        "чуть", "том", "нельзя", "такой", "им", "более", "всегда",
        "конечно", "всю", "между", "это", "эта",
        # English
        "a", "an", "the", "and", "or", "but", "if", "while", "of", "at",
        "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "having", "do",
        "does", "did", "doing", "will", "would", "should", "could",
        "may", "might", "must", "this", "that", "these", "those", "i",
        "you", "he", "she", "it", "we", "they", "them", "their", "what",
        "which", "who", "whom", "as", "not", "no", "so",
    }
)


@lru_cache(maxsize=4)
def _stemmer(lang: str) -> snowballstemmer.RussianStemmer:
    return snowballstemmer.stemmer(lang)


def stem_tokens(text: str, lang: str = "russian") -> list[str]:
    """Tokenize ``text`` into a list of stems with stopwords dropped.

    Order is preserved (caller can use it to compute window pairs etc.),
    duplicates are *not* removed here — that is a concern of whoever
    builds the payload.
    """
    if not text:
        return []
    words = _WORD_RE.findall(text.lower())
    stems = _stemmer(lang).stemWords(words)
    return [s for s in stems if s and s not in _STOPWORDS and len(s) > 1]


def chunk_lexical_payload(
    text: str,
    lang: str = "russian",
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


def keypoint_stems(text: str, lang: str = "russian") -> list[str]:
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
