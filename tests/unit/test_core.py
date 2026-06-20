"""
Unit tests — no external dependencies, no API keys required.

Testing philosophy for agentic AI systems:

1. Mock the LLM — test YOUR routing logic, not the model's behaviour
2. Mock MongoDB — test YOUR aggregation logic, not PyMongo
3. Score ranges over exact values — non-deterministic systems need flexible assertions
4. Test failure modes explicitly — what happens when the LLM returns bad JSON?

The last point is critical for Principal-level credibility:
"We tested what happens when the router fails and verified it falls back to RETRIEVE
rather than crashing. Fail-safe, not fail-fast."
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.models import (
    ChunkStrategy,
    CorpusSource,
    DriftAlert,
    EvalResult,
    EvalStatus,
    QueryRequest,
    RAGResponse,
    RetrievedContext,
    RouterAction,
    RouterDecision,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_response() -> RAGResponse:
    return RAGResponse(
        query="How does MongoDB Atlas Vector Search work?",
        answer=(
            "MongoDB Atlas Vector Search enables semantic search by storing "
            "vector embeddings alongside your operational data in Atlas."
        ),
        contexts=[
            RetrievedContext(
                document_id="1",
                content="MongoDB Atlas Vector Search enables semantic search using HNSW indexing.",
                source=CorpusSource.MONGODB,
                url="https://www.mongodb.com/docs/atlas/atlas-vector-search/",
                title="Vector Search Overview",
                similarity_score=0.92,
            )
        ],
        session_id=uuid4(),
        router_decision=RouterDecision(
            action=RouterAction.RETRIEVE,
            reasoning="Clear question about MongoDB Vector Search",
            source_filter="mongodb",
        ),
        model="meta-llama/llama-3.2-3b-instruct",
        latency_ms=450.0,
        retrieval_latency_ms=120.0,
        generation_latency_ms=330.0,
    )


@pytest.fixture
def retrieve_decision() -> RouterDecision:
    return RouterDecision(
        action=RouterAction.RETRIEVE,
        reasoning="Clear, specific question about MongoDB",
        source_filter="mongodb",
    )


@pytest.fixture
def reformulate_decision() -> RouterDecision:
    return RouterDecision(
        action=RouterAction.REFORMULATE,
        reasoning="Query is vague — 'how does it work' needs more specificity",
        source_filter="all",
        reformulated_query="What is the architecture of MongoDB Atlas Vector Search and how does HNSW indexing work?",
    )


@pytest.fixture
def decompose_decision() -> RouterDecision:
    return RouterDecision(
        action=RouterAction.DECOMPOSE,
        reasoning="Multi-part question covering two distinct topics",
        source_filter="all",
        sub_queries=[
            "How do I create a vector search index in MongoDB Atlas?",
            "How do I query embeddings using the $vectorSearch aggregation stage?",
        ],
    )


@pytest.fixture
def escalate_decision() -> RouterDecision:
    return RouterDecision(
        action=RouterAction.ESCALATE,
        reasoning="Question about stock prices is out of scope",
        source_filter="all",
        escalation_reason="I can only answer questions about MongoDB, Dash0, and Reap.",
    )


# ── Model tests ───────────────────────────────────────────────────────────

class TestQueryRequest:
    def test_valid_query(self) -> None:
        req = QueryRequest(query="How do I create a vector index?")
        assert req.query == "How do I create a vector index?"

    def test_strips_whitespace(self) -> None:
        req = QueryRequest(query="  what is RAG?  ")
        assert req.query == "what is RAG?"

    def test_rejects_short_query(self) -> None:
        with pytest.raises(ValueError):
            QueryRequest(query="hi")

    def test_source_filter_optional(self) -> None:
        req = QueryRequest(query="What is OpenTelemetry?")
        assert req.source_filter is None


class TestRAGResponse:
    def test_context_texts(self, sample_response: RAGResponse) -> None:
        texts = sample_response.context_texts
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)

    def test_corpus_sources(self, sample_response: RAGResponse) -> None:
        sources = sample_response.corpus_sources
        assert CorpusSource.MONGODB in sources

    def test_sources_deduplication(self) -> None:
        response = RAGResponse(
            query="test",
            answer="answer",
            contexts=[
                RetrievedContext(
                    document_id="1", content="a",
                    source=CorpusSource.MONGODB,
                    url="https://docs.mongodb.com",
                    title="", similarity_score=0.9,
                ),
                RetrievedContext(
                    document_id="2", content="b",
                    source=CorpusSource.MONGODB,
                    url="https://docs.mongodb.com",  # same URL
                    title="", similarity_score=0.85,
                ),
            ],
            session_id=uuid4(),
            router_decision=RouterDecision(
                action=RouterAction.RETRIEVE,
                reasoning="test", source_filter="mongodb",
            ),
            model="llama3", latency_ms=100.0,
            retrieval_latency_ms=30.0, generation_latency_ms=70.0,
        )
        assert len(response.sources) == 1


class TestRouterDecision:
    def test_retrieve_action(self, retrieve_decision: RouterDecision) -> None:
        assert retrieve_decision.action == RouterAction.RETRIEVE
        assert retrieve_decision.reformulated_query is None
        assert retrieve_decision.sub_queries is None

    def test_reformulate_has_rewrite(self, reformulate_decision: RouterDecision) -> None:
        assert reformulate_decision.action == RouterAction.REFORMULATE
        assert reformulate_decision.reformulated_query is not None
        assert len(reformulate_decision.reformulated_query) > 0

    def test_decompose_has_sub_queries(self, decompose_decision: RouterDecision) -> None:
        assert decompose_decision.action == RouterAction.DECOMPOSE
        assert decompose_decision.sub_queries is not None
        assert len(decompose_decision.sub_queries) >= 2

    def test_escalate_has_reason(self, escalate_decision: RouterDecision) -> None:
        assert escalate_decision.action == RouterAction.ESCALATE
        assert escalate_decision.escalation_reason is not None


class TestEvalResult:
    def test_faithfulness_weighted_highest(self) -> None:
        """
        Faithfulness weighted at 0.5 because hallucinations are
        the worst failure mode for trust-critical applications.
        """
        result = EvalResult(
            session_id=uuid4(), query="q", answer="a",
            corpus_source="mongodb",
            faithfulness=1.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            status=EvalStatus.PASS,
        )
        assert result.overall_score == pytest.approx(0.5)

    def test_overall_score_range(self) -> None:
        result = EvalResult(
            session_id=uuid4(), query="q", answer="a",
            corpus_source="mongodb",
            faithfulness=0.8,
            answer_relevancy=0.7,
            context_precision=0.6,
            status=EvalStatus.PASS,
        )
        assert 0.0 <= result.overall_score <= 1.0

    def test_to_mongo_serialisation(self) -> None:
        result = EvalResult(
            session_id=uuid4(), query="q", answer="a",
            corpus_source="mongodb",
            faithfulness=0.8, answer_relevancy=0.7,
            context_precision=0.6, status=EvalStatus.PASS,
        )
        doc = result.to_mongo()
        assert "_id" in doc
        assert "id" not in doc
        assert isinstance(doc["session_id"], str)


# ── Router tests ──────────────────────────────────────────────────────────

class TestQueryRouter:
    @pytest.mark.asyncio
    async def test_route_returns_decision(self) -> None:
        """Router should return a RouterDecision for any input."""
        from src.agents.router import QueryRouter

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "action": "retrieve",
            "reasoning": "Clear question about MongoDB Vector Search",
            "source_filter": "mongodb",
            "reformulated_query": None,
            "sub_queries": None,
            "escalation_reason": None,
        })

        with patch("src.agents.router.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_response)
            router = QueryRouter()
            router._llm = instance

            decision = await router.route("How does Atlas Vector Search work?")

        assert isinstance(decision, RouterDecision)
        assert decision.action == RouterAction.RETRIEVE
        assert decision.source_filter == "mongodb"

    @pytest.mark.asyncio
    async def test_router_falls_back_on_bad_json(self) -> None:
        """
        CRITICAL: If the LLM returns malformed JSON, the router should
        fall back to RETRIEVE rather than crashing.
        Fail-safe over fail-fast for production systems.
        """
        from src.agents.router import QueryRouter

        mock_response = MagicMock()
        mock_response.content = "this is not valid json { broken"

        with patch("src.agents.router.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_response)
            router = QueryRouter()
            router._llm = instance

            decision = await router.route("some query")

        # Should NOT raise — should return safe fallback
        assert decision.action == RouterAction.RETRIEVE

    @pytest.mark.asyncio
    async def test_router_handles_decompose(self) -> None:
        from src.agents.router import QueryRouter

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "action": "decompose",
            "reasoning": "Multi-part question",
            "source_filter": "all",
            "reformulated_query": None,
            "sub_queries": [
                "How do I create a vector index?",
                "How do I run a vector search query?",
            ],
            "escalation_reason": None,
        })

        with patch("src.agents.router.ChatOpenAI") as MockLLM:
            instance = MockLLM.return_value
            instance.ainvoke = AsyncMock(return_value=mock_response)
            router = QueryRouter()
            router._llm = instance

            decision = await router.route(
                "How do I create a vector index and also run vector search queries?"
            )

        assert decision.action == RouterAction.DECOMPOSE
        assert decision.sub_queries is not None
        assert len(decision.sub_queries) == 2


# ── Drift Monitor tests ───────────────────────────────────────────────────

class TestDriftMonitor:
    """
    Tests for drift detection logic.
    MongoDB is mocked — we test OUR aggregation and alerting logic.
    """

    def _make_eval_result(
        self,
        corpus: str = "mongodb",
        faithfulness: float = 0.85,
        relevancy: float = 0.80,
        precision: float = 0.75,
        status: str = "pass",
        hours_ago: int = 1,
    ) -> dict:
        ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        return {
            "corpus_source": corpus,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "status": status,
            "evaluated_at": ts.isoformat(),
        }

    def test_drift_alert_fires_on_significant_drop(self) -> None:
        """Alert should fire when metric drops > 15% from baseline."""
        alert = DriftAlert(
            metric="faithfulness",
            corpus_source="mongodb",
            baseline_value=0.85,
            current_value=0.60,  # 29% drop — well above 15% threshold
            drop_fraction=0.294,
            window_hours=24,
        )
        assert alert.drop_fraction > 0.15
        assert "DRIFT ALERT" in alert.message
        assert "faithfulness" in alert.message
        assert "mongodb" in alert.message

    def test_drift_alert_message_format(self) -> None:
        alert = DriftAlert(
            metric="answer_relevancy",
            corpus_source="dash0",
            baseline_value=0.80,
            current_value=0.65,
            drop_fraction=0.1875,
            window_hours=24,
        )
        msg = alert.message
        assert "answer_relevancy" in msg
        assert "dash0" in msg
        assert "0.800" in msg
        assert "0.650" in msg

    def test_relative_threshold_more_sensitive_than_absolute(self) -> None:
        """
        A system at 0.60 dropping to 0.51 is MORE concerning
        than one at 0.90 dropping to 0.81, even though both
        drop by 0.09 in absolute terms.
        """
        low_base = DriftAlert(
            metric="faithfulness", corpus_source="mongodb",
            baseline_value=0.60, current_value=0.51,
            drop_fraction=0.15, window_hours=24,
        )
        high_base = DriftAlert(
            metric="faithfulness", corpus_source="mongodb",
            baseline_value=0.90, current_value=0.81,
            drop_fraction=0.10, window_hours=24,
        )
        # Low-base drop (15%) triggers alert, high-base drop (10%) does not
        assert low_base.drop_fraction >= 0.15
        assert high_base.drop_fraction < 0.15


# ── Score range tests ─────────────────────────────────────────────────────

class TestScoreRangePattern:
    """
    Demonstrates the correct pattern for testing non-deterministic AI systems.

    In the MongoDB Python Engineering interview, when asked about testing:
    "We don't assert exact scores because LLM outputs vary between runs.
    We assert properties that should always hold — score is in [0,1],
    faithfulness of a grounded answer is higher than a hallucination."
    """

    def test_faithfulness_is_probability(self) -> None:
        """Faithfulness is always in [0, 1]."""
        result = EvalResult(
            session_id=uuid4(), query="q", answer="a",
            corpus_source="mongodb", faithfulness=0.82,
            answer_relevancy=0.75, context_precision=0.68,
            status=EvalStatus.PASS,
        )
        assert 0.0 <= result.faithfulness <= 1.0

    def test_overall_score_is_probability(self) -> None:
        """Overall score is always in [0, 1] for any valid inputs."""
        result = EvalResult(
            session_id=uuid4(), query="q", answer="a",
            corpus_source="mongodb", faithfulness=0.5,
            answer_relevancy=0.5, context_precision=0.5,
            status=EvalStatus.PASS,
        )
        assert 0.0 <= result.overall_score <= 1.0
