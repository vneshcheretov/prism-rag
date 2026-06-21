from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """One OpenAI-style dialogue turn for follow-up questions."""

    role: Literal["user", "assistant"]
    content: str


class ConvertResponse(BaseModel):
    markdown: str = Field(description="The converted document as markdown.")
    title: str | None = Field(default=None, description="Document title, if detected.")


class IngestedNode(BaseModel):
    index: int
    name: str
    keypoints: list[str]
    paragraph_id: str


class IngestResponse(BaseModel):
    language: str | None
    nodes: list[IngestedNode]
    corpus_summary: str


class SearchRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior dialogue turns (oldest first) for follow-up resolution.",
    )


class SearchResponse(BaseModel):
    query: str
    keypoints: list[str] = Field(default_factory=list)
    paragraphs: list[str] = Field(default_factory=list)
    note: str | None = None
    translated: bool = Field(
        default=False,
        description="True when the query was translated to the corpus language "
        "before retrieval (its script differed). Paragraphs stay in the corpus language.",
    )


class AnswerRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior dialogue turns (oldest first) for follow-up resolution.",
    )


class AnswerResponse(BaseModel):
    query: str
    answer: str = ""
    final_summary: str = ""
    note: str | None = None
    search: SearchResponse | None = None
    translated: bool = Field(
        default=False,
        description="True when the query language differed from the corpus; the "
        "answer is returned in the user's query language.",
    )
