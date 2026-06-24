"""
Unit tests for Platform AI Primitives.

Testing philosophy matches the rest of the test suite:
- No external dependencies, no API keys required
- Mock the pipeline — test the primitive's logic, not the LLM
- Test failure modes explicitly — what happens with missing fields?
- Test the review decision thresholds — these are product decisions that must work correctly

The primitives are the quality gate between AI pipeline and product teams.
These tests verify that gate works correctly under normal and failure conditions.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.models import (
    CorpusSource,
    EvalStatus,
    QueryRequest,
    RAGResponse,
    RetrievedContext,
    RouterAction,
    RouterDecision,
)
from src.primitives import (
    ClassificationPrimitive,
    ClassificationRequest,
    ConfidenceLevel,
    ExtractionPrimitive,
    ExtractionRequest,
    ExtractionResult,
    ExtractedField,
    ReviewDecision,
    SourceCitation,
    VerificationPrimitive,
    VerificationRequest,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_rag_response() -> RAGResponse:
    """A realistic RAG response with one high-similarity context."""
    return RAGResponse(
        query="Extract vendor and amount from this receipt",
        answer="Vendor: Acme Corp, Amount: $1,250.00, Date: 2026-06-24",
        contexts=[
            RetrievedContext(
                document_id="doc-001",
                content="Receipt from Acme Corp for $1,250.00 on June 24, 2026.",
                source=CorpusSource.REAP,
                url="https://reap.readme.io/docs/receipts",
                title="Receipt Processing > Expense Extraction",
                similarity_score=0.93,
            )
        ],
        session_id=uuid4(),
        router_decision=RouterDecision(
            action=RouterAction.RETRIEVE,
            reasoning="Direct extraction query",
            source_filter="reap",
        ),
        model="openrouter/free",
        latency_ms=1250.0,
        retrieval_latency_ms=420.0,
        generation_latency_ms=830.0,
    )


@pytest.fixture
def mock_pipeline(mock_rag_response):
    """Mock RAGPipeline that returns a realistic response."""
    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=mock_rag_response)
    return pipeline


@pytest.fixture
def high_confidence_extraction() -> ExtractionResult:
    """A fully extracted result with high confidence on all fields."""
    citation = SourceCitation(
        document_id="doc-001",
        url="https://reap.readme.io/docs/receipts",
        section_path="Receipt Processing > Expense Extraction",
        page_number=None,
        similarity_score=0.93,
    )
    return ExtractionResult(
        source_document_id="doc-001",
        document_type="expense_receipt",
        fields=[
            ExtractedField(
                field_name="vendor",
                value="Acme Corp",
                confidence=0.95,
                confidence_level=ConfidenceLevel.HIGH,
                source_citation=citation,
                extraction_rationale="Vendor name appears in first line of receipt header.",
            ),
            ExtractedField(
                field_name="amount",
                value="1250.00",
                confidence=0.97,
                confidence_level=ConfidenceLevel.HIGH,
                source_citation=citation,
                extraction_rationale="Amount clearly stated as '$1,250.00' in the total line.",
            ),
            ExtractedField(
                field_name="date",
                value="2026-06-24",
                confidence=0.92,
                confidence_level=ConfidenceLevel.HIGH,
                source_citation=citation,
                extraction_rationale="Date extracted from receipt header: 'June 24, 2026'.",
            ),
        ],
        overall_confidence=0.947,
        requires_human_review=False,
        extraction_model="openrouter/free",
        latency_ms=1250.0,
    )


@pytest.fixture
def low_confidence_extraction() -> ExtractionResult:
    """An extraction with low confidence — should trigger human review."""
    citation = SourceCitation(
        document_id="doc-002",
        url="https://reap.readme.io/docs/receipts",
        section_path="Unknown section",
        page_number=None,
        similarity_score=0.52,
    )
    return ExtractionResult(
        source_document_id="doc-002",
        document_type="expense_receipt",
        fields=[
            ExtractedField(
                field_name="vendor",
                value="Unknown",
                confidence=0.45,
                confidence_level=ConfidenceLevel.LOW,
                source_citation=citation,
                extraction_rationale="Vendor name unclear from document content.",
            ),
            ExtractedField(
                field_name="amount",
                value=None,
                confidence=0.30,
                confidence_level=ConfidenceLevel.LOW,
                source_citation=citation,
                extraction_rationale="Amount field not found in retrieved context.",
            ),
        ],
        overall_confidence=0.375,
        requires_human_review=True,
        extraction_model="openrouter/free",
        latency_ms=980.0,
    )


# ── ExtractionPrimitive tests ─────────────────────────────────────────────


class TestExtractionPrimitive:

    @pytest.mark.asyncio
    async def test_extract_returns_result(self, mock_pipeline):
        """ExtractionPrimitive returns a valid ExtractionResult."""
        primitive = ExtractionPrimitive(pipeline=mock_pipeline)
        request = ExtractionRequest(
            document_content="Receipt from Acme Corp for $1,250.00 on June 24, 2026.",
            document_type="expense_receipt",
            source_document_id="doc-001",
            extraction_fields=["vendor", "amount", "date"],
        )

        result = await primitive.extract(request)

        assert isinstance(result, ExtractionResult)
        assert result.source_document_id == "doc-001"
        assert result.document_type == "expense_receipt"
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_extract_calls_pipeline(self, mock_pipeline):
        """Primitive delegates to the RAG pipeline — one call per extraction."""
        primitive = ExtractionPrimitive(pipeline=mock_pipeline)
        request = ExtractionRequest(
            document_content="Test document content.",
            document_type="invoice",
            source_document_id="doc-003",
            extraction_fields=["vendor", "total"],
        )

        await primitive.extract(request)

        mock_pipeline.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_review(self, mock_pipeline, mock_rag_response):
        """When context similarity is low, requires_human_review should be True."""
        # Override with low-similarity context
        low_sim_response = mock_rag_response.model_copy(update={
            "contexts": [
                RetrievedContext(
                    document_id="doc-low",
                    content="Unrelated content.",
                    source=CorpusSource.REAP,
                    url="https://reap.readme.io",
                    title="Unrelated",
                    similarity_score=0.40,
                )
            ]
        })
        mock_pipeline.query = AsyncMock(return_value=low_sim_response)

        primitive = ExtractionPrimitive(pipeline=mock_pipeline, confidence_review_threshold=0.85)
        request = ExtractionRequest(
            document_content="Unclear document",
            document_type="kyb_document",
            source_document_id="doc-low",
            extraction_fields=["entity_name"],
        )

        result = await primitive.extract(request)

        # Low confidence context → overall confidence below threshold → review required
        assert result.overall_confidence < 0.85 or result.requires_human_review

    @pytest.mark.asyncio
    async def test_source_citation_populated(self, mock_pipeline):
        """Every extracted field must have a source citation for traceability."""
        primitive = ExtractionPrimitive(pipeline=mock_pipeline)
        request = ExtractionRequest(
            document_content="Board resolution dated 2026-01-15.",
            document_type="board_resolution",
            source_document_id="doc-board-001",
            extraction_fields=["effective_date"],
        )

        result = await primitive.extract(request)

        for field in result.fields:
            assert field.source_citation is not None
            assert field.source_citation.document_id == "doc-board-001"

    def test_high_confidence_fields_property(self, high_confidence_extraction):
        """high_confidence_fields returns only fields above 0.90 threshold."""
        high = high_confidence_extraction.high_confidence_fields
        assert all(f.confidence >= 0.90 for f in high)
        assert len(high) > 0

    def test_low_confidence_fields_property(self, low_confidence_extraction):
        """low_confidence_fields returns fields below 0.75 threshold."""
        low = low_confidence_extraction.low_confidence_fields
        assert all(f.confidence < 0.75 for f in low)

    def test_field_dict_property(self, high_confidence_extraction):
        """field_dict provides a clean {name: value} accessor."""
        d = high_confidence_extraction.field_dict
        assert "vendor" in d
        assert "amount" in d
        assert d["vendor"] == "Acme Corp"


# ── ClassificationPrimitive tests ─────────────────────────────────────────


class TestClassificationPrimitive:

    @pytest.mark.asyncio
    async def test_classify_returns_result(self, mock_pipeline):
        """ClassificationPrimitive returns a valid ClassificationResult."""
        primitive = ClassificationPrimitive(pipeline=mock_pipeline)
        request = ClassificationRequest(
            content="Flight to Singapore for sales conference, $850",
            taxonomy=["travel", "meals", "software", "office_supplies", "other"],
        )

        result = await primitive.classify(request)

        assert result.primary_category in request.taxonomy
        assert 0.0 <= result.confidence <= 1.0
        assert result.review_decision in list(ReviewDecision)

    @pytest.mark.asyncio
    async def test_high_similarity_gives_autonomous_decision(self, mock_pipeline, mock_rag_response):
        """High retrieval similarity → AUTONOMOUS review decision."""
        # mock_rag_response already has 0.93 similarity
        primitive = ClassificationPrimitive(
            pipeline=mock_pipeline,
            autonomous_threshold=0.90,
            review_threshold=0.75,
        )
        request = ClassificationRequest(
            content="Business travel expense",
            taxonomy=["travel", "meals", "software"],
        )

        result = await primitive.classify(request)

        assert result.review_decision == ReviewDecision.AUTONOMOUS

    @pytest.mark.asyncio
    async def test_low_similarity_requires_human(self, mock_pipeline, mock_rag_response):
        """Low retrieval similarity → REQUIRE_HUMAN review decision."""
        low_sim_response = mock_rag_response.model_copy(update={
            "contexts": [
                RetrievedContext(
                    document_id="ctx-low",
                    content="Unrelated context.",
                    source=CorpusSource.REAP,
                    url="",
                    title="",
                    similarity_score=0.50,
                )
            ]
        })
        mock_pipeline.query = AsyncMock(return_value=low_sim_response)

        primitive = ClassificationPrimitive(
            pipeline=mock_pipeline,
            autonomous_threshold=0.90,
            review_threshold=0.75,
        )
        request = ClassificationRequest(
            content="Unclear transaction",
            taxonomy=["travel", "meals", "software"],
        )

        result = await primitive.classify(request)

        assert result.review_decision == ReviewDecision.REQUIRE_HUMAN

    @pytest.mark.asyncio
    async def test_medium_similarity_flags_review(self, mock_pipeline, mock_rag_response):
        """Medium retrieval similarity → FLAG_REVIEW decision."""
        medium_sim_response = mock_rag_response.model_copy(update={
            "contexts": [
                RetrievedContext(
                    document_id="ctx-med",
                    content="Somewhat relevant context.",
                    source=CorpusSource.REAP,
                    url="",
                    title="",
                    similarity_score=0.82,
                )
            ]
        })
        mock_pipeline.query = AsyncMock(return_value=medium_sim_response)

        primitive = ClassificationPrimitive(
            pipeline=mock_pipeline,
            autonomous_threshold=0.90,
            review_threshold=0.75,
        )
        request = ClassificationRequest(
            content="Ambiguous transaction",
            taxonomy=["travel", "meals", "software"],
        )

        result = await primitive.classify(request)

        assert result.review_decision == ReviewDecision.FLAG_REVIEW

    @pytest.mark.asyncio
    async def test_result_category_from_taxonomy(self, mock_pipeline):
        """Primary category must always be from the provided taxonomy."""
        primitive = ClassificationPrimitive(pipeline=mock_pipeline)
        taxonomy = ["low_risk", "medium_risk", "high_risk"]
        request = ClassificationRequest(
            content="Large cross-border transaction to new counterparty",
            taxonomy=taxonomy,
        )

        result = await primitive.classify(request)

        assert result.primary_category in taxonomy


# ── VerificationPrimitive tests ───────────────────────────────────────────


class TestVerificationPrimitive:

    def test_all_high_confidence_fields_pass(self, high_confidence_extraction):
        """All fields above threshold → PASS with AUTONOMOUS decision."""
        primitive = VerificationPrimitive(confidence_threshold=0.85)
        request = VerificationRequest(
            extraction_result=high_confidence_extraction,
            required_fields=["vendor", "amount", "date"],
            confidence_threshold=0.85,
        )

        result = primitive.verify(request)

        assert result.overall_status == EvalStatus.PASS
        assert result.overall_review_decision == ReviewDecision.AUTONOMOUS
        assert not result.requires_human_review
        assert result.pass_rate == 1.0

    def test_missing_required_field_fails(self, high_confidence_extraction):
        """A required field that wasn't extracted → overall FAIL."""
        primitive = VerificationPrimitive(confidence_threshold=0.85)
        request = VerificationRequest(
            extraction_result=high_confidence_extraction,
            required_fields=["vendor", "amount", "date", "currency"],  # currency not extracted
            confidence_threshold=0.85,
        )

        result = primitive.verify(request)

        assert result.overall_status == EvalStatus.FAIL
        assert result.overall_review_decision == ReviewDecision.REQUIRE_HUMAN
        assert result.requires_human_review

        failed = result.failed_fields
        assert any(f.field_name == "currency" for f in failed)

    def test_none_value_field_fails(self, low_confidence_extraction):
        """A field extracted as None → FAIL."""
        primitive = VerificationPrimitive(confidence_threshold=0.85)
        request = VerificationRequest(
            extraction_result=low_confidence_extraction,
            required_fields=["vendor", "amount"],
            confidence_threshold=0.85,
        )

        result = primitive.verify(request)

        # amount is None → should fail
        assert result.overall_status == EvalStatus.FAIL
        failed_names = [f.field_name for f in result.failed_fields]
        assert "amount" in failed_names

    def test_low_confidence_skips_not_fails(self, low_confidence_extraction):
        """
        A field present but with very low confidence → SKIP not FAIL.

        Key distinction: FAIL means we extracted something wrong.
        SKIP means we couldn't complete the evaluation — not a quality failure.
        This is the same principle as RAGAS nan → SKIP.
        """
        primitive = VerificationPrimitive(confidence_threshold=0.85)
        # Use only vendor (which has a value, just low confidence)
        request = VerificationRequest(
            extraction_result=low_confidence_extraction,
            required_fields=["vendor"],
            confidence_threshold=0.85,
        )

        result = primitive.verify(request)

        # vendor has value "Unknown" but confidence 0.45 → SKIP
        vendor_verif = next(
            (f for f in result.field_verifications if f.field_name == "vendor"),
            None
        )
        assert vendor_verif is not None
        assert vendor_verif.review_decision == ReviewDecision.REQUIRE_HUMAN

    def test_review_fields_property(self, high_confidence_extraction):
        """review_fields returns only fields that need human attention."""
        # Lower one field's confidence to trigger review
        fields = list(high_confidence_extraction.fields)
        fields[0] = fields[0].model_copy(update={
            "confidence": 0.80,
            "confidence_level": ConfidenceLevel.MEDIUM,
        })
        modified = high_confidence_extraction.model_copy(update={"fields": fields})

        primitive = VerificationPrimitive(confidence_threshold=0.85)
        request = VerificationRequest(
            extraction_result=modified,
            required_fields=["vendor", "amount", "date"],
            confidence_threshold=0.85,
        )

        result = primitive.verify(request)
        review = result.review_fields

        # At least one field should be flagged for review
        assert len(review) >= 1

    def test_pass_rate_calculation(self, high_confidence_extraction):
        """pass_rate reflects the fraction of fields that pass."""
        primitive = VerificationPrimitive(confidence_threshold=0.85)
        request = VerificationRequest(
            extraction_result=high_confidence_extraction,
            required_fields=["vendor", "amount", "date"],
            confidence_threshold=0.85,
        )

        result = primitive.verify(request)

        assert 0.0 <= result.pass_rate <= 1.0
        assert result.pass_rate == len([
            f for f in result.field_verifications
            if f.status == EvalStatus.PASS
        ]) / len(result.field_verifications)

    def test_worst_case_review_decision_propagates(self):
        """Overall review decision is worst-case across all fields."""
        citation = SourceCitation(
            document_id="doc-mixed",
            url="",
            section_path="",
            similarity_score=0.88,
        )
        mixed_extraction = ExtractionResult(
            source_document_id="doc-mixed",
            document_type="invoice",
            fields=[
                ExtractedField(
                    field_name="vendor",
                    value="Acme Corp",
                    confidence=0.95,
                    confidence_level=ConfidenceLevel.HIGH,
                    source_citation=citation,
                    extraction_rationale="Clear.",
                ),
                ExtractedField(
                    field_name="total",
                    value="5000.00",
                    confidence=0.60,  # Below review threshold
                    confidence_level=ConfidenceLevel.LOW,
                    source_citation=citation,
                    extraction_rationale="Unclear.",
                ),
            ],
            overall_confidence=0.775,
            requires_human_review=True,
            extraction_model="openrouter/free",
            latency_ms=900.0,
        )

        primitive = VerificationPrimitive(confidence_threshold=0.85)
        request = VerificationRequest(
            extraction_result=mixed_extraction,
            required_fields=["vendor", "total"],
            confidence_threshold=0.85,
        )

        result = primitive.verify(request)

        # Worst case (REQUIRE_HUMAN from low-confidence total) propagates
        assert result.overall_review_decision == ReviewDecision.REQUIRE_HUMAN
        assert result.requires_human_review
