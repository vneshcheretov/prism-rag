from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    markdown: str = Field(description="Markdown document to ingest.")
    summarize: bool = Field(
        default=True, description="Refresh the corpus summary after ingest."
    )


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
    filter_relevance: bool = True
    query_language: str | None = None


class SearchResponse(BaseModel):
    query: str
    keypoints: list[str] = Field(default_factory=list)
    paragraphs: list[str] = Field(default_factory=list)
    note: str | None = None


class AnswerRequest(BaseModel):
    query: str
    filter_relevance: bool = True
    query_language: str | None = None


class AnswerResponse(BaseModel):
    query: str
    answer: str = ""
    final_summary: str = ""
    note: str | None = None
    search: SearchResponse | None = None
