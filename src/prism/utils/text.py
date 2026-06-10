from __future__ import annotations

import re
from functools import lru_cache

import tiktoken

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?]) +|(?=\s*(?<!\d)(?<!\d[.?!])\d+\.\s)|(?<=\n)")
_ENUM_HEAD_RE = re.compile(r"^[.!?]?\s*\d+\.")


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Token count using the given tiktoken encoding.

    Defaults to ``cl100k_base`` (gpt-4 / gpt-3.5 family) since the exact
    tokenizer rarely matters for chunk sizing — any reasonable BPE gives
    counts within a few percent of each other.
    """
    return len(_get_encoding(encoding_name).encode(text))


def sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences.

    Friendly to Russian and English, and keeps numbered list items
    (``"1. ..."``) glued to their content rather than splitting the
    number off as its own "sentence".
    """
    raw = _SENT_SPLIT_RE.split(text)
    sentences: list[str] = []
    i = 0
    while i < len(raw):
        s = raw[i].strip() if raw[i] else ""
        if not s:
            i += 1
            continue
        if _ENUM_HEAD_RE.match(s) and (i + 1) < len(raw):
            nxt = raw[i + 1].strip() if raw[i + 1] else ""
            sentences.append(f"{s} {nxt}".strip())
            i += 2
        else:
            sentences.append(s)
            i += 1
    return sentences
