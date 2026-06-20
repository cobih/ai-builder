"""
Domain models for the AI Builder RAG system.

Every model represents a product decision:
- What data do we need to capture at each pipeline stage?
- What do we need to diagnose failures after the fact?
- What contract does each component expose to the next?

Using Pydantic v2 throughout for:
- Runtime validation (not just type hints)
- Clean serialisation to/from MongoDB
- Self-documenting field descriptions
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────


class CorpusSource(StrEnum):
    """Which documentation corpus a document came from."""
    MONGODB = "mongodb"
    DASH0 = "dash0"
    REAP = "reap"


class RouterAction(StrEnum):
    """
    The four actions the query router can take.

    RETRIEVE     — standard RAG: embed query, retrieve, generate
    REFORMULATE  — query is ambiguous; rewrite before retrieving
    DECOMPOSE    — multi-part question; split into sub-queries
    ESCALATE     — out of scope; tell the user we can't answer
    """
    RETRIEVE = "retrieve"
    REFORMULATE = "reformulate"
    DECOMPOSE = "decompose"
    ESCALATE = "escalate"


class EvalStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class ChunkStrategy(StrEnum):
    NAIVE = "naive"       # fixed-size chunks — fast, loses context at boundaries
    SEMANTIC = "semantic" # split on headers/paragraphs — slower, better for tech docs


# ── Document models ───────────────────────────────────────────────────────


class Document(BaseModel):
    """A chunk of source content stored in MongoDB Atlas Vector Search."""

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(min_length=1)
    source: CorpusSource
    url: str = Field(description="Origin URL")
    chunk_index: int = Field(ge=0)
    chunk_strategy: ChunkStrategy = Field(default=ChunkStrategy.SEMANTIC)
    title: str = Field(default="")
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict[str, Any]:
        data = self.model_dump()
        data["_id"] = str(data.pop("id"))
        data["source"] = data["source"]
        data["created_at"] = data["created_at"].isoformat()
        return data


# ── Router models ─────────────────────────────────────────────────────────


class RouterDecision(BaseModel):
    """
    Structured output from the QueryRouter LLM call.

    Using structured output (Pydantic model) rather than parsing
    free text means routing decisions are reliable and type-safe.
    This is the key pattern for production agentic systems.
    """

    action: RouterAction
    reasoning: str = Field(description="Why the router chose this action")
    source_filter: Literal["mongodb", "dash0", "reap", "all"] = Field(
        default="all",
        description="Which corpus to search — narrows retrieval for better precision",
    )
    reformulated_query: str | None = Field(
        default=None,
        description="Populated when action == REFORMULATE",
    )
    sub_queries: list[str] | None = Field(
        default=None,
        description="Populated when action == DECOMPOSE",
    )
    escalation_reason: str | None = Field(
        default=None,
        description="Populated when action == ESCALATE",
    )


# ── Query / Response models ───────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Incoming query — validated at the boundary."""

    query: str = Field(min_length=3)
    session_id: UUID = Field(default_factory=uuid4)
    top_k: int | None = Field(default=None)
    source_filter: CorpusSource | None = Field(
        default=None,
        description="Optional: restrict retrieval to one corpus",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 3:
            raise ValueError("Query too short")
        return stripped


class RetrievedContext(BaseModel):
    """A single chunk returned by vector search."""

    document_id: str
    content: str
    source: CorpusSource
    url: str
    title: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class RAGResponse(BaseModel):
    """
    Complete output of one RAG pipeline run.

    We capture every input to generation so we can:
    - Run RAGAS evaluation post-hoc
    - Debug failures by replaying the exact context
    - Detect when retrieval quality degrades before users notice
    """

    query: str
    answer: str
    contexts: list[RetrievedContext]
    session_id: UUID
    router_decision: RouterDecision
    model: str
    latency_ms: float = Field(ge=0.0)
    retrieval_latency_ms: float = Field(ge=0.0)
    generation_latency_ms: float = Field(ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def context_texts(self) -> list[str]:
        return [c.content for c in self.contexts]

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(c.url for c in self.contexts))

    @property
    def corpus_sources(self) -> list[CorpusSource]:
        return list(dict.fromkeys(c.source for c in self.contexts))


# ── Evaluation models ─────────────────────────────────────────────────────


class EvalResult(BaseModel):
    """
    RAGAS evaluation scores + pass/fail for one RAG response.

    Three metrics because they diagnose different root causes:
    ┌──────────────────────┬─────────────────────────────────────────┐
    │ faithfulness         │ Is the answer grounded in the context?  │
    │                      │ Low = hallucination                      │
    ├──────────────────────┼─────────────────────────────────────────┤
    │ answer_relevancy     │ Does the answer address the question?    │
    │                      │ Low = answer is off-topic               │
    ├──────────────────────┼─────────────────────────────────────────┤
    │ context_precision    │ Were the right chunks retrieved?        │
    │                      │ Low = wrong documents in context        │
    └──────────────────────┴─────────────────────────────────────────┘
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    query: str
    answer: str
    corpus_source: str
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    status: EvalStatus
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def overall_score(self) -> float:
        """
        Weighted average — faithfulness weighted highest.
        For trust-critical AI applications, hallucinations are the
        worst failure mode, so we penalise them most heavily.
        """
        return (
            self.faithfulness * 0.5
            + self.answer_relevancy * 0.3
            + self.context_precision * 0.2
        )

    def to_mongo(self) -> dict[str, Any]:
        data = self.model_dump()
        data["_id"] = str(data.pop("id"))
        data["session_id"] = str(data["session_id"])
        data["evaluated_at"] = data["evaluated_at"].isoformat()
        return data


# ── Drift monitoring models ───────────────────────────────────────────────


class DriftAlert(BaseModel):
    """
    Fired when a quality metric drops significantly from its baseline.

    Design decision: we alert on relative drops (15% from baseline)
    rather than absolute thresholds. A system that starts at 0.6
    faithfulness and drops to 0.5 is MORE concerning than one that
    starts at 0.9 and drops to 0.8, even though both drop by 0.1.
    """

    metric: str
    corpus_source: str
    baseline_value: float
    current_value: float
    drop_fraction: float
    window_hours: int
    fired_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def message(self) -> str:
        return (
            f"DRIFT ALERT: {self.metric} for '{self.corpus_source}' dropped "
            f"{self.drop_fraction:.1%} from baseline "
            f"({self.baseline_value:.3f} → {self.current_value:.3f}) "
            f"over the last {self.window_hours}h"
        )


class QualityDashboard(BaseModel):
    """Current quality state across all corpora."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_evaluations: int
    overall_pass_rate: float
    per_corpus: dict[str, dict[str, float]]
    active_alerts: list[DriftAlert]
    trend: str = Field(description="improving | stable | degrading")


# ── Chunking benchmark models ─────────────────────────────────────────────


class ChunkingBenchmarkResult(BaseModel):
    """
    Comparison of naive vs semantic chunking on the same query set.

    This is the 'documented failure and fix':
    naive chunking was our first approach, semantic chunking is why we changed.
    """

    strategy: ChunkStrategy
    query: str
    top_similarity_score: float
    avg_similarity_score: float
    chunk_count: int
    avg_chunk_length: int
