"""LLM-assisted markdown heading inference.

Adds markdown headings to flat text that has none (e.g. extracted from a
PDF). The text is split into numbered-sentence chunks; an LLM proposes
headings keyed by line number, and the headings are merged back into the
original text. Ported from the heading-structuring logic of the source
project (datakeeper-lib).
"""

from __future__ import annotations

import logging
import re

from ..llm.base import LLMProvider, Tier
from ..llm.prompts import (
    MD_HEADERS_CONTINUE_PROMPT,
    MD_HEADERS_START_PROMPT,
    MD_TITLE_PROMPT,
)
from ..schemas.llm_outputs import DocumentTitle
from ..utils.text import normalize_simple, sentence_tokenize

log = logging.getLogger(__name__)

# Sentences per chunk, and how many trailing sentences overlap into the next
# chunk so the model keeps context across the boundary.
_BLOCK_SIZE = 70
_OVERLAP = 5

_NUMBERED_RE = re.compile(r"(\d+\))(.+?)(?=\d+\)|$)", re.DOTALL)
_MARKDOWN_FENCE_RE = re.compile(r"```markdown\s*(.*?)\s*```", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```\s*(.*?)\s*```", re.DOTALL)
_DEEP_HEADER_RE = re.compile(r"^#{5,}", re.MULTILINE)


class MarkdownStructurer:
    """Infer markdown headings for unstructured text via an LLM.

    One instance per document. Calls the LLM once per chunk (``tier`` —
    defaults to ``strong`` since heading inference is a reasoning task);
    this is an opt-in heavy operation, used when the source has no usable
    heading structure.
    """

    def __init__(self, llm: LLMProvider, *, tier: Tier = "strong") -> None:
        self.llm = llm
        self.tier = tier

    def _split_data(self, data: str) -> list[str]:
        """Number every sentence and group into overlapping blocks."""
        sentences = sentence_tokenize(data)
        blocks: list[list[str]] = []
        current: list[str] = []
        for i in range(len(sentences)):
            current.append(f"{i + 1}) {sentences[i]}\n")
            if (i + 1) % _BLOCK_SIZE == 0:
                blocks.append(current)
                current = current[-_OVERLAP:]
        if current:
            blocks.append(current)
        return ["".join(block) for block in blocks]

    @staticmethod
    def _extract_markdown(text: str) -> str:
        """Pull the headings out of the model reply, tolerating fence drift.

        Prefer a ```markdown fence, fall back to any ``` fence, and finally
        accept the raw reply — downstream parsing only keeps ``N)`` lines, so
        stray prose is ignored either way. (Strictly requiring the labeled
        fence silently dropped every heading when a model omitted it.)
        """
        match = _MARKDOWN_FENCE_RE.search(text) or _ANY_FENCE_RE.search(text)
        return match.group(1) if match else text.strip()

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _add_linebreaks(self, text: str) -> str:
        """Collapse numbered entries to one-per-line, de-duplicated."""
        matches = _NUMBERED_RE.findall(text)
        return "\n".join(
            self._dedupe(["".join(m).replace("\n", "") for m in matches])
        )

    async def _add_headers(self, blocks: list[str]) -> str:
        """Feed each block to the LLM, carrying the running header structure."""
        result = ""
        for i, block in enumerate(blocks):
            log.info("structuring: chunk %d/%d", i + 1, len(blocks))
            if i == 0:
                system = MD_HEADERS_START_PROMPT
                user = f"CHUNK: \n{block}"
            else:
                system = MD_HEADERS_CONTINUE_PROMPT
                user = (
                    f"Input data: \nPrevious headlines structure: "
                    f"{result} \nCHUNK: \n{block}"
                )
            raw = await self.llm.complete_text(system=system, user=user, tier=self.tier)
            result += self._extract_markdown(raw)
        return result

    @staticmethod
    def _parse_numbered(text: str) -> dict[int, str]:
        out: dict[int, str] = {}
        for num, body in _NUMBERED_RE.findall(text):
            out[int(num[:-1])] = body[1:]
        return out

    @staticmethod
    def _remove_deep_headers(text: str) -> str:
        """Drop headings deeper than level 4 (5+ hashes)."""
        return _DEEP_HEADER_RE.sub("", text)

    def _merge(self, headers: dict[int, str], sentences: dict[int, str]) -> str:
        """Splice headings into the numbered text by line number.

        If a heading is just a restatement of its line, only the hashes are
        added; otherwise the heading is prepended before the line.
        """
        for num, header in headers.items():
            sentence = sentences.get(num)
            if not sentence:
                continue
            # Header on its own line — otherwise it glues to the following
            # sentence and the markdown chunker won't parse it as a heading.
            header_line = self._remove_deep_headers(header).strip()
            if normalize_simple(sentence) == normalize_simple(header):
                sentences[num] = f"\n{header_line}\n"
            else:
                sentences[num] = f"\n{header_line}\n{sentence}"
        return " ".join(sentences.values())

    async def structure(self, data: str) -> str:
        """Return ``data`` with LLM-inferred markdown headings spliced in."""
        blocks = self._split_data(data)
        if not blocks:
            return data
        headers_str = await self._add_headers(blocks)

        all_headers = self._add_linebreaks(headers_str)
        all_sentences = self._add_linebreaks("\n".join(blocks))
        headers_dict = self._parse_numbered(all_headers)
        sentences_dict = self._parse_numbered(all_sentences)
        return self._merge(headers_dict, sentences_dict)

    async def generate_title(self, data: str) -> str | None:
        """Infer a short title from the opening of the document.

        Uses only the first chunk of sentences (titles live at the top) and
        a single ``fast`` structured call. Returns ``None`` for empty input.
        """
        sentences = sentence_tokenize(data)
        if not sentences:
            return None
        first_chunk = " ".join(sentences[:_BLOCK_SIZE])
        result = await self.llm.complete_structured(
            system=MD_TITLE_PROMPT,
            user=first_chunk,
            schema=DocumentTitle,
            tier="fast",
        )
        return result.title.strip() or None
