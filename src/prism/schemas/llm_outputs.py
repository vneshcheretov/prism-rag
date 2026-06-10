from __future__ import annotations

from pydantic import BaseModel, Field


class NodeExtraction(BaseModel):
    """Per-chunk extraction produced during ingestion.

    Output is fed into the knowledge graph: ``header`` becomes the node
    name, ``key_phrases`` become the indexed terms whose embeddings power
    keypoint-decomposition retrieval.
    """

    header: str = Field(
        description=(
            "Lowest-level markdown header of the chunk, or the main topic name. "
            "Russian."
        )
    )
    summary: str = Field(
        description=(
            "1-2 sentence overview of the chunk's main idea(s) in Russian, up to ~60 words. "
            "No step-by-step instructions, UI actions, or numbers."
        )
    )
    key_phrases: list[str] = Field(
        description=(
            "General concepts and overarching ideas from the chunk, in Russian. "
            "Each phrase must be no more than two words. "
            "All top-level (#) markdown headers must be included."
        )
    )


class QueryKeypoints(BaseModel):
    """Decomposition of a natural-language query into searchable terms."""

    is_searchable: bool = Field(
        description=(
            "False only for inputs with no information-seeking intent at all: "
            "greetings, chit-chat, meta-commands, gibberish. Default to true when in doubt."
        )
    )
    short_summary: str = Field(
        description=(
            "Compact declarative restatement of the input in Russian, content nouns only. "
            "Non-empty even when is_searchable is false (e.g. 'приветствие')."
        )
    )
    key_phrases: list[str] = Field(
        description=(
            "Monolithic noun-based phrases lifted from the query (Russian). "
            "Each phrase at most two words. "
            "Empty list allowed when is_searchable is false."
        )
    )
    synonyms: list[str] = Field(
        description=(
            "Flat list: each key phrase followed by up to two Russian noun-based "
            "synonyms or alternative phrasings. Domain-aware when DATA CONTEXT is provided."
        )
    )


class RelevanceFilter(BaseModel):
    """Per-paragraph relevance verdict applied after retrieval.

    Mirrors the datakeeper-lib results_filtration schema: the model both
    judges relevance and lifts a verbatim excerpt useful for answering
    the request. Paraphrasing and summarization are forbidden — callers
    can rely on ``answer`` being copied verbatim from the input fragment.
    """

    answer: str = Field(
        description=(
            "Verbatim excerpt from the INFORMATION that directly helps answer the REQUEST. "
            "Empty string when nothing relevant. Never paraphrased."
        )
    )
    is_correct: bool = Field(
        description=(
            "True if the INFORMATION contains a clear, directly relevant excerpt for the REQUEST. "
            "Broad requests (services, facilities, room types, etc.) are satisfied by a relevant "
            "section or list."
        )
    )


class Summarization(BaseModel):
    """Final answer synthesized over retrieved paragraphs.

    Two-field schema mirrors datakeeper-lib summarization: ``summary`` is
    the user-facing answer to the request; ``final_summary`` is a catalog-
    style meta-description that always opens with ``"Данные об "`` and is
    useful as a section title or index entry.
    """

    summary: str = Field(
        description=(
            "1-2 sentence Russian answer to the REQUEST, grounded only in the DATA FRAGMENTS. "
            "No invented facts."
        )
    )
    final_summary: str = Field(
        description=(
            "Catalog entry in Russian, ALWAYS starts with 'Данные об ', "
            "followed by 5-10 words describing the topic."
        )
    )


class CorpusSummary(BaseModel):
    """High-level overview of an ingested corpus, used as context for query understanding."""

    summary: str = Field(
        description=(
            "Concise Russian summary (under 6 sentences) of what the corpus contains, "
            "its main topics, and any obvious structure."
        )
    )


class LLMStems(BaseModel):
    """Per-word stems produced by the LLM fallback stemmer.

    The list must be the same length as the input word list — the caller
    matches stems back to words by position to preserve per-sentence
    structure. A mismatch is treated as a transient failure and the caller
    falls back to identity stemming for that batch.
    """

    stems: list[str] = Field(
        description=(
            "Lowercased base form (lemma/stem) for each input word, in input order. "
            "Length must equal the input array length. Proper nouns, foreign words "
            "and numbers are returned unchanged (still lowercased)."
        )
    )
