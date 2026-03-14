"""Pydantic schemas for API I/O and internal data structures."""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel


# ── API Models ──────────────────────────────────────────────────────────────


class Question(BaseModel):
    """A single question from the challenge."""

    id: str
    text: str


class ChallengeRequest(BaseModel):
    """Incoming request body for POST /challenge/run."""

    corpus_url: str
    questions: list[Question]
    llm_model: str | None = None  # Optional override for model benchmarking
    rerank_top_k: int | None = None  # Optional override for context tuning
    bypass_cache: bool = False  # Force cold start for fair benchmarking


class Source(BaseModel):
    """A document source reference with filename and page numbers."""

    filename: str
    pages: list[int]


class Answer(BaseModel):
    """A single answer with its source citations."""

    question_id: str
    answer: str
    sources: list[Source]


class ChallengeResponse(BaseModel):
    """Final response body returned from POST /challenge/run."""

    results: list[Answer]
    total_time: float | None = None
    total_tokens: int | None = None
    cache_hit: bool | None = None  # True if vector cache was warm
    phase_times: dict[str, float] | None = None  # Per-phase timing breakdown


# ── Internal Data Structures (zero-overhead TypedDicts) ─────────────────────


class Chunk(TypedDict):
    """A text chunk produced by Phase 1 extraction."""

    chunk_id: str
    filename: str
    page_nums: list[int]
    text: str


class DocMetadata(TypedDict):
    """Per-document metadata collected during extraction."""

    filename: str
    title: str
    page_count: int
