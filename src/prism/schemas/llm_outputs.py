from __future__ import annotations

from pydantic import BaseModel, Field


class NodeExtraction(BaseModel):
    """Per-chunk extraction produced during ingestion.

    Output is fed into the knowledge graph: ``header`` becomes the node
    name, ``key_phrases`` become the indexed terms whose embeddings power
    keypoint-decomposition retrieval.
    """

    header: str = Field(
        description="Short descriptive title for the chunk (3-7 words)."
    )
    summary: str = Field(
        description="One-sentence summary of what the chunk says."
    )
    key_phrases: list[str] = Field(
        description=(
            "Distinct noun phrases, named entities, or terms appearing in the chunk "
            "that someone might search for. Prefer specific over generic. 5-12 items."
        )
    )


class QueryKeypoints(BaseModel):
    """Decomposition of a natural-language query into searchable terms."""

    is_searchable: bool = Field(
        description=(
            "False for chit-chat, greetings, or messages that are not actual "
            "information requests. True otherwise."
        )
    )
    key_phrases: list[str] = Field(
        description="2-6 concrete terms or entities from the query to match against indexed content."
    )
    synonyms: list[str] = Field(
        description="0-6 alternative phrasings the indexed content might use."
    )


class RelevanceFilter(BaseModel):
    """Per-paragraph relevance verdict applied after retrieval."""

    is_relevant: bool = Field(
        description=(
            "True only if the paragraph contains material directly useful for "
            "answering the user's request. Tangential keyword matches are not relevant."
        )
    )


class CorpusSummary(BaseModel):
    """High-level overview of an ingested corpus, used as context for query understanding."""

    summary: str = Field(
        description="Concise multi-sentence summary (under 6 sentences) of what the corpus contains."
    )
