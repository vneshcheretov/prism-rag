from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from ..utils.text import count_tokens, sentence_tokenize


@dataclass(slots=True)
class Chunk:
    """A retrieval unit produced by ``MarkdownChunker``.

    ``text`` always starts with the full markdown header path of its
    section, so an embedding of the chunk carries the section context.
    ``paragraph_id`` groups chunks coming from the same markdown section —
    during retrieval we can re-join them to recover the original paragraph
    instead of returning a single isolated chunk.
    """

    text: str
    paragraph_id: str
    headers: list[str] = field(default_factory=list)

    @property
    def header_path(self) -> str:
        return " > ".join(self.headers)


class MarkdownChunker:
    """Header-aware markdown chunker.

    Splits a markdown document into chunks of at most ``max_tokens`` tokens.
    Three invariants the chunker guarantees:

    1. Every chunk begins with its full header chain (``# A\\n## B\\n...``),
       so chunks are self-describing — useful both for embedding context
       and for displaying retrieval results to a user.
    2. Chunks never end mid-sentence; sentence boundaries come from
       ``sentence_tokenize`` which handles Russian/English and numbered
       list items.
    3. Chunks from the same section share a ``paragraph_id`` (UUID) so
       the retrieval pipeline can stitch them back into the original
       paragraph rather than returning a single fragment.
    """

    HEADER_RE = re.compile(r"^(#{1,5})\s+(.*)")

    def __init__(
        self,
        max_tokens: int = 512,
        min_section_tokens: int = 20,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.max_tokens = max_tokens
        self.min_section_tokens = min_section_tokens
        self.encoding_name = encoding_name

    def split_by_headers(self, text: str) -> list[tuple[list[str], str]]:
        """Split markdown into ``(headers, body)`` sections.

        ``body`` contains only the section content — the header chain is
        carried separately in ``headers`` and prepended back onto each
        emitted chunk later. A header of level ``n`` pops the stack down
        to depth ``n - 1`` before pushing itself, matching standard
        markdown nesting semantics.
        """
        sections: list[tuple[list[str], str]] = []
        body_lines: list[str] = []
        stack: list[str] = []

        def _flush(current_headers: list[str]) -> None:
            if not current_headers and not body_lines:
                return
            body = "\n".join(body_lines).strip()
            if body:
                sections.append((current_headers.copy(), body))

        for line in text.splitlines():
            m = self.HEADER_RE.match(line)
            if m:
                _flush(stack)
                body_lines.clear()
                level = len(m.group(1))
                heading = m.group(2).strip()
                stack = stack[: level - 1]
                stack.append(heading)
            else:
                body_lines.append(line)
        _flush(stack)
        return sections

    @staticmethod
    def _header_prefix(headers: list[str]) -> str:
        return "\n".join(f"{'#' * (i + 1)} {h}" for i, h in enumerate(headers))

    @staticmethod
    def _emit_chunk(
        buf: list[str],
        header_prefix: str,
        paragraph_id: str,
        headers: list[str],
    ) -> Chunk:
        joined = " ".join(buf)
        text = f"{header_prefix}\n{joined}" if header_prefix else joined
        return Chunk(text=text, paragraph_id=paragraph_id, headers=list(headers))

    def chunk(self, text: str) -> list[Chunk]:
        sections = self.split_by_headers(text)
        chunks: list[Chunk] = []

        for headers, body in sections:
            if count_tokens(body, self.encoding_name) <= self.min_section_tokens:
                continue

            paragraph_id = str(uuid.uuid4())
            header_prefix = self._header_prefix(headers)
            sentences = sentence_tokenize(body)

            buf: list[str] = []
            buf_tokens = 0

            for sent in sentences:
                sent_tokens = count_tokens(sent, self.encoding_name)
                if buf and (buf_tokens + sent_tokens) > self.max_tokens:
                    chunks.append(
                        self._emit_chunk(buf, header_prefix, paragraph_id, headers)
                    )
                    buf = []
                    buf_tokens = 0
                buf.append(sent)
                buf_tokens += sent_tokens

            if buf:
                chunks.append(
                    self._emit_chunk(buf, header_prefix, paragraph_id, headers)
                )

        return chunks
