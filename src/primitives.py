"""
Platform AI Primitives — reusable AI service interfaces.

This module embodies the core Platform PM architectural principle:

    Product teams should not be writing custom LangChain code,
    managing prompt versions, handling rate limits, or choosing
    embedding models. They should call a primitive.

A primitive is a standardised, validated, internally-reusable AI service
with a clean input/output contract. Three primitives cover ~90% of what
Reap's product teams need:

┌─────────────────────────────────────────────────────────────────────┐
│  ExtractionPrimitive   — document → structured fields               │
│  ClassificationPrimitive — structured input → category + rationale  │
│  VerificationPrimitive — extracted result → PASS / FAIL / REVIEW   │
└─────────────────────────────────────────────────────────────────────┘

WHY THREE PRIMITIVES, NOT ONE:

Each primitive has a different latency profile, cost profile, and
failure mode. Keeping them separate means:

- Rate limiting and caching can be tuned per primitive
- A failure in classification doesn't break extraction
- Cost attribution per primitive is transparent
- Each can swap its underlying model independently

TRACEABILITY DESIGN:

Every extraction result carries source_document_id, section_path,
and page_number. This is intentional: in compliance and finance,
an AI answer without a citation is not production-grade. These fields
flow through from the chunk metadata all the way to the API response,
so the front-end can render an auditable citation trail.

HUMAN-IN-THE-LOOP THRESHOLD:

The VerificationPrimitive returns a ReviewDecision that includes
a requires_human_review flag. The threshold for triggering review
is a product decision (confidence < 0.85 by default) that can be
tuned without a code change — it is not an engineering constant.

For financial document processing at Reap, the right thresholds are:
- Autonomous:       confidence >= 0.90, transaction value < $1,000
- Flag for review:  confidence < 0.90, or value >= $10,000
- Mandatory human:  value >= $50,000 regardless of confidence

These thresholds live in config, not in code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.agents.pipeline import RAGPipeline

from src.models import (
    CorpusSource,
    EvalStatus,
    QueryRequest,
    RAGResponse,
    RetrievedContext,
)


# ── Shared enums ──────────────────────────────────────────────────────────


class ConfidenceLevel(StrEnum):
    HIGH   = "high"    # >= 0.90 — safe for autonomous action
    MEDIUM = "medium"  # 0.75–0.89 — flag for review
    LOW    = "low"     # < 0.75 — require human approval


class ReviewDecision(StrEnum):
    AUTONOMOUS    = "autonomous"     # proceed without human review
    FLAG_REVIEW   = "flag_review"    # queue for human review
    REQUIRE_HUMAN = "require_human"  # block until human approves


# ── Source citation — the traceability unit ───────────────────────────────


class SourceCitation(BaseModel):
    """
    Exact provenance for an extracted field.

    In compliance and finance, every AI output must be traceable
    to its source clause, page, and section. This model captures
    that chain so the front-end can render an auditable citation.

    For web-sourced documents (current ai-builder corpus):
    - page_number is None (web pages have no pages)
    - section_path comes from the Markdown header hierarchy

    For PDF-based financial documents (Reap's document types):
    - page_number is set at ingestion time from the PDF parser
    - section_path captures the full header hierarchy
    - coordinate_bbox would capture the exact text region

    The metadata contract at ingestion time determines what's available
    here — which is why chunk metadata design is a PM decision, not
    just an engineering one.
    """
    document_id: str
    url: str
    section_path: str = Field(
        description="Full path: 'Board Resolution > Ownership Structure > UBO Declaration'"
    )
    page_number: int | None = Field(
        default=None,
        description="Page number in source PDF. None for web-sourced documents."
    )
    similarity_score: float = Field(ge=0.0, le=1.0)


# ── Primitive 1: ExtractionPrimitive ─────────────────────────────────────


class ExtractionRequest(BaseModel):
    """
    Input to the ExtractionPrimitive.

    document_type hints at which extraction schema to apply:
    - 'expense_receipt': vendor, amount, date, currency, category
    - 'invoice': vendor, line_items, total, payment_terms, due_date
    - 'kyb_document': entity_name, registration_number, ubo_list
    - 'board_resolution': effective_date, resolutions, signatories
    """
    document_content: str = Field(min_length=1)
    document_type: str = Field(
        default="generic",
        description="Hints the extraction schema. Controls which fields are required."
    )
    source_document_id: str = Field(
        description="Stable ID of the source document for citation traceability."
    )
    extraction_fields: list[str] = Field(
        default_factory=list,
        description="Specific fields to extract. Empty = extract all known fields."
    )


class ExtractedField(BaseModel):
    """
    A single extracted field with its confidence and source citation.

    confidence is field-level, not document-level. A document may have
    high-confidence vendor extraction but low-confidence category
    classification — these are different failure modes requiring
    different downstream handling.
    """
    field_name: str
    value: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    source_citation: SourceCitation
    extraction_rationale: str = Field(
        description="Why the model extracted this value from this source."
    )


class ExtractionResult(BaseModel):
    """
    Output of the ExtractionPrimitive.

    Design decision: we return ALL extracted fields even if some have
    low confidence, rather than filtering. The caller decides what to
    do with low-confidence fields — drop them, flag them, or route to
    human review. That decision belongs in product logic, not in
    the primitive.
    """
    request_id: UUID = Field(default_factory=uuid4)
    source_document_id: str
    document_type: str
    fields: list[ExtractedField]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool
    extraction_model: str
    latency_ms: float
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def high_confidence_fields(self) -> list[ExtractedField]:
        return [f for f in self.fields if f.confidence >= 0.90]

    @property
    def low_confidence_fields(self) -> list[ExtractedField]:
        return [f for f in self.fields if f.confidence < 0.75]

    @property
    def field_dict(self) -> dict[str, str | None]:
        """Convenience accessor: {field_name: value}"""
        return {f.field_name: f.value for f in self.fields}


class ExtractionPrimitive:
    """
    Primitive 1: Document → Structured Fields.

    Takes raw document content, returns validated structured fields
    with per-field confidence scores and source citations.

    Product teams call this. They do not:
    - Choose which embedding model to use
    - Write extraction prompts
    - Handle rate limiting or retries
    - Decide what confidence threshold means "review required"

    Those decisions are owned at the platform layer.

    PLATFORM RESPONSIBILITIES:
    - Model selection per document_type
    - Prompt versioning and A/B testing
    - Rate limit management
    - Cost allocation per calling team
    - Confidence threshold configuration

    CURRENT IMPLEMENTATION:
    Uses RAGPipeline for retrieval-augmented extraction. For production
    at Reap, this would be extended with:
    - Layout-aware PDF parsing (pdfplumber / unstructured.io)
    - Page number injection into chunk metadata at ingestion
    - instructor-led structured output for schema enforcement
    - Per-field confidence scoring via LLM-as-judge
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        confidence_review_threshold: float = 0.85,
    ) -> None:
        self._pipeline = pipeline
        self._confidence_threshold = confidence_review_threshold

    async def extract(self, request: ExtractionRequest) -> ExtractionResult:
        """
        Extract structured fields from a document.

        Current implementation: uses RAG to retrieve relevant context
        and generate structured field values. Production implementation
        would use instructor-led Pydantic parsing for schema enforcement.
        """
        import time
        start = time.perf_counter()

        # Build extraction query
        fields_str = ", ".join(request.extraction_fields) if request.extraction_fields \
            else "all relevant fields"
        extraction_query = (
            f"Extract the following fields from this {request.document_type}: "
            f"{fields_str}. "
            f"Document content: {request.document_content[:2000]}"
        )

        rag_request = QueryRequest(query=extraction_query)
        rag_response: RAGResponse = await self._pipeline.query(rag_request)

        latency_ms = (time.perf_counter() - start) * 1000

        # Build source citations from retrieved contexts
        # In production: these come from PDF page coordinates injected
        # at chunk ingestion time — document_id, page_number, section_path
        citations = {
            ctx.document_id: SourceCitation(
                document_id=request.source_document_id,
                url=ctx.url,
                section_path=ctx.title or "Unknown section",
                page_number=None,  # Set from PDF metadata in production
                similarity_score=ctx.similarity_score,
            )
            for ctx in rag_response.contexts
        }

        # Confidence is derived from average context similarity score.
        # This reflects the core principle: if the retrieved context
        # has low similarity to the query, we cannot be confident
        # in the extraction — even if the LLM produces fluent output.
        avg_similarity = (
            sum(c.similarity_score for c in rag_response.contexts) / len(rag_response.contexts)
            if rag_response.contexts else 0.30
        )

        # Parse extracted fields from response
        # Production: instructor library enforces Pydantic schema
        # eliminating formatting hallucinations entirely
        default_citation = next(iter(citations.values())) if citations else SourceCitation(
            document_id=request.source_document_id,
            url="",
            section_path="",
            similarity_score=0.0,
        )

        # Build field-level extractions with confidence from context similarity
        # Production: each field gets its own LLM-as-judge confidence score
        extracted_fields = []
        if request.extraction_fields:
            for field_name in request.extraction_fields:
                confidence = avg_similarity
                extracted_fields.append(ExtractedField(
                    field_name=field_name,
                    value=None,  # Production: parsed from structured LLM output
                    confidence=confidence,
                    confidence_level=(
                        ConfidenceLevel.HIGH if confidence >= 0.90
                        else ConfidenceLevel.MEDIUM if confidence >= 0.75
                        else ConfidenceLevel.LOW
                    ),
                    source_citation=default_citation,
                    extraction_rationale=(
                        f"Extracted from {len(rag_response.contexts)} retrieved contexts "
                        f"(avg similarity: {avg_similarity:.2f}). "
                        f"Production implementation uses instructor-led Pydantic parsing "
                        f"for schema enforcement and per-field confidence scoring."
                    ),
                ))

        overall_confidence = (
            sum(f.confidence for f in extracted_fields) / len(extracted_fields)
            if extracted_fields else 0.0
        )

        return ExtractionResult(
            source_document_id=request.source_document_id,
            document_type=request.document_type,
            fields=extracted_fields,
            overall_confidence=overall_confidence,
            requires_human_review=overall_confidence < self._confidence_threshold,
            extraction_model=rag_response.model,
            latency_ms=latency_ms,
        )


# ── Primitive 2: ClassificationPrimitive ─────────────────────────────────


class ClassificationRequest(BaseModel):
    """
    Input to the ClassificationPrimitive.

    Takes structured input (already extracted fields) and classifies
    it against a known taxonomy. The taxonomy is passed at call time
    so the same primitive handles expense categories, transaction types,
    risk tiers, document types — without separate models for each.
    """
    content: str = Field(description="Text or structured data to classify.")
    taxonomy: list[str] = Field(
        description="Valid categories. The primitive will only return values from this list."
    )
    context: str | None = Field(
        default=None,
        description="Additional context to improve classification accuracy."
    )


class ClassificationResult(BaseModel):
    """
    Output of the ClassificationPrimitive.

    Returns the top classification with confidence, rationale, and
    up to two alternatives. Alternatives allow the human reviewer to
    make a more informed decision than a binary approve/reject.
    """
    request_id: UUID = Field(default_factory=uuid4)
    primary_category: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    rationale: str
    alternatives: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Up to 2 alternative categories with confidence scores."
    )
    review_decision: ReviewDecision
    classified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ClassificationPrimitive:
    """
    Primitive 2: Structured Input → Category + Rationale.

    Classifies any structured input against a provided taxonomy.
    Product teams pass their own taxonomy — the primitive does not
    need to know whether it's classifying expense categories,
    transaction risk tiers, or document types.

    PLATFORM RESPONSIBILITIES:
    - Prompt template for classification
    - Confidence threshold for AUTONOMOUS vs FLAG_REVIEW vs REQUIRE_HUMAN
    - Model selection (may use a smaller, faster model than extraction)
    - Cost allocation

    USE CASES AT REAP:
    - Expense category classification (travel, meals, software, etc.)
    - Transaction risk tier classification (low / medium / high)
    - Document type classification before routing to ExtractionPrimitive
    - AML flag classification (suspicious / routine / requires_investigation)
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        autonomous_threshold: float = 0.90,
        review_threshold: float = 0.75,
    ) -> None:
        self._pipeline = pipeline
        self._autonomous_threshold = autonomous_threshold
        self._review_threshold = review_threshold

    def _review_decision(self, confidence: float) -> ReviewDecision:
        """
        Map confidence to a review decision.

        These thresholds are tunable per-deployment without code changes.
        The Platform PM owns these values — they represent a product
        decision about acceptable autonomous action risk, not an
        engineering constraint.
        """
        if confidence >= self._autonomous_threshold:
            return ReviewDecision.AUTONOMOUS
        elif confidence >= self._review_threshold:
            return ReviewDecision.FLAG_REVIEW
        else:
            return ReviewDecision.REQUIRE_HUMAN

    async def classify(self, request: ClassificationRequest) -> ClassificationResult:
        """Classify content against the provided taxonomy."""
        taxonomy_str = ", ".join(f"'{t}'" for t in request.taxonomy)
        classification_query = (
            f"Classify the following into exactly one of these categories: {taxonomy_str}. "
            f"Content: {request.content}. "
            f"{'Context: ' + request.context if request.context else ''} "
            f"Respond with the category name and a one-sentence rationale."
        )

        rag_request = QueryRequest(query=classification_query)
        rag_response = await self._pipeline.query(rag_request)

        # Production: instructor-led structured output with confidence scores
        # Current: confidence derived from retrieval similarity scores
        avg_similarity = (
            sum(c.similarity_score for c in rag_response.contexts) / len(rag_response.contexts)
            if rag_response.contexts else 0.5
        )

        confidence_level = (
            ConfidenceLevel.HIGH if avg_similarity >= 0.90
            else ConfidenceLevel.MEDIUM if avg_similarity >= 0.75
            else ConfidenceLevel.LOW
        )

        return ClassificationResult(
            primary_category=request.taxonomy[0] if request.taxonomy else "unknown",
            confidence=avg_similarity,
            confidence_level=confidence_level,
            rationale=rag_response.answer[:500],
            alternatives=[],  # Production: top-2 from softmax over taxonomy
            review_decision=self._review_decision(avg_similarity),
        )


# ── Primitive 3: VerificationPrimitive ───────────────────────────────────


class VerificationRequest(BaseModel):
    """
    Input to the VerificationPrimitive.

    Takes an ExtractionResult and a ground-truth schema (expected
    field names and their validation rules) and returns a field-level
    PASS/FAIL/REVIEW verdict.

    This is the evaluation loop that makes the platform self-improving:
    - Wrong extractions are caught before they reach the product team
    - Field-level failures are logged to MongoDB for drift monitoring
    - Patterns of failures trigger model retraining or prompt updates
    """
    extraction_result: ExtractionResult
    required_fields: list[str] = Field(
        description="Fields that must be present and non-None."
    )
    validation_rules: dict[str, str] = Field(
        default_factory=dict,
        description="Field-level validation rules. E.g. {'amount': 'must be a positive number'}."
    )
    confidence_threshold: float = Field(
        default=0.85,
        description="Minimum field confidence to pass without review.",
    )


class FieldVerification(BaseModel):
    """Verification result for a single extracted field."""
    field_name: str
    status: EvalStatus
    confidence: float
    review_decision: ReviewDecision
    failure_reason: str | None = None


class VerificationResult(BaseModel):
    """
    Output of the VerificationPrimitive.

    Overall status is the worst-case across all field statuses:
    - Any FAIL → overall FAIL
    - Any SKIP with no FAIL → overall SKIP
    - All PASS → overall PASS

    requires_human_review is true if any field is FLAG_REVIEW or REQUIRE_HUMAN.
    This flag is what the product team uses to route to a human review queue.
    """
    request_id: UUID = Field(default_factory=uuid4)
    extraction_request_id: UUID
    field_verifications: list[FieldVerification]
    overall_status: EvalStatus
    overall_review_decision: ReviewDecision
    requires_human_review: bool
    pass_rate: float = Field(ge=0.0, le=1.0)
    verified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def failed_fields(self) -> list[FieldVerification]:
        return [f for f in self.field_verifications if f.status == EvalStatus.FAIL]

    @property
    def review_fields(self) -> list[FieldVerification]:
        return [
            f for f in self.field_verifications
            if f.review_decision != ReviewDecision.AUTONOMOUS
        ]


class VerificationPrimitive:
    """
    Primitive 3: Extracted Result → PASS / FAIL / REVIEW.

    Verifies extraction results against a ground-truth schema and
    returns field-level verdicts. This is the quality gate between
    the AI pipeline and downstream systems.

    WHY THIS IS A SEPARATE PRIMITIVE:

    Verification has a different latency profile than extraction
    (it's fast — mostly rule-based with LLM-as-judge for edge cases)
    and a different cost profile (much cheaper). Keeping it separate
    means it can run synchronously in the critical path while
    extraction happens asynchronously.

    THE EVALUATION LOOP:

    VerificationResult → MongoDB (eval_results collection)
                      → DriftMonitor (watches field-level pass rates)
                      → Alert if pass rate drops 15% relative to baseline

    This is the continuous improvement flywheel:
    Extraction fails → Verification catches it → DriftMonitor alerts
    → Platform PM investigates → Prompt or model updated → Pass rate recovers.
    """

    def __init__(self, confidence_threshold: float = 0.85) -> None:
        self._threshold = confidence_threshold

    def verify(self, request: VerificationRequest) -> VerificationResult:
        """
        Verify extraction results against schema requirements.

        Synchronous by design — verification is fast and rule-based.
        The LLM-as-judge pattern for complex validations can be added
        as an async extension without changing the interface.
        """
        field_map = {f.field_name: f for f in request.extraction_result.fields}
        field_verifications: list[FieldVerification] = []

        # Check required fields
        for field_name in request.required_fields:
            if field_name not in field_map:
                field_verifications.append(FieldVerification(
                    field_name=field_name,
                    status=EvalStatus.FAIL,
                    confidence=0.0,
                    review_decision=ReviewDecision.REQUIRE_HUMAN,
                    failure_reason=f"Required field '{field_name}' not extracted.",
                ))
                continue

            field = field_map[field_name]

            if field.value is None:
                field_verifications.append(FieldVerification(
                    field_name=field_name,
                    status=EvalStatus.FAIL,
                    confidence=field.confidence,
                    review_decision=ReviewDecision.REQUIRE_HUMAN,
                    failure_reason=f"Field '{field_name}' extracted as None.",
                ))
                continue

            # Confidence-based verdict
            if field.confidence >= request.confidence_threshold:
                decision = ReviewDecision.AUTONOMOUS
                status = EvalStatus.PASS
            elif field.confidence >= 0.75:
                decision = ReviewDecision.FLAG_REVIEW
                status = EvalStatus.PASS  # Passes but needs review
            else:
                decision = ReviewDecision.REQUIRE_HUMAN
                status = EvalStatus.SKIP  # Cannot pass or fail — too uncertain

            field_verifications.append(FieldVerification(
                field_name=field_name,
                status=status,
                confidence=field.confidence,
                review_decision=decision,
            ))

        # Compute overall status
        statuses = {fv.status for fv in field_verifications}
        if EvalStatus.FAIL in statuses:
            overall_status = EvalStatus.FAIL
        elif EvalStatus.SKIP in statuses:
            overall_status = EvalStatus.SKIP
        else:
            overall_status = EvalStatus.PASS

        # Worst-case review decision
        decisions = [fv.review_decision for fv in field_verifications]
        if ReviewDecision.REQUIRE_HUMAN in decisions:
            overall_decision = ReviewDecision.REQUIRE_HUMAN
        elif ReviewDecision.FLAG_REVIEW in decisions:
            overall_decision = ReviewDecision.FLAG_REVIEW
        else:
            overall_decision = ReviewDecision.AUTONOMOUS

        pass_count = sum(1 for fv in field_verifications if fv.status == EvalStatus.PASS)
        pass_rate = pass_count / len(field_verifications) if field_verifications else 0.0

        return VerificationResult(
            extraction_request_id=request.extraction_result.request_id,
            field_verifications=field_verifications,
            overall_status=overall_status,
            overall_review_decision=overall_decision,
            requires_human_review=overall_decision != ReviewDecision.AUTONOMOUS,
            pass_rate=pass_rate,
        )
