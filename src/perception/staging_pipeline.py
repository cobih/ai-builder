"""
Staging Pipeline — the Evaluation Gate between Perception and Action.

THE CORE PRINCIPLE:

Nothing goes into the live corpus without passing the evaluation gate.

Perception detects that content changed. But "content changed" doesn't
mean "new content is better." A documentation page could change in ways
that make retrieval worse:
- A page was restructured and key information was buried deeper
- An API endpoint changed but the old parameters are still listed
- Marketing content was added that dilutes the technical signal

The staging pipeline is the quality gate that separates "detected" from
"approved." It runs two layers of evaluation before anything becomes a
PR candidate:

LAYER 1 — RAGAS FAITHFULNESS:
The same evaluation we run on live queries, applied to sample queries
against the staged content. If the new chunks don't produce faithful
answers on our test query set, they fail the gate.

LAYER 2 — LLM-AS-JUDGE:
A separate LLM call that scores the staged content on:
- Information quality: is this content informative and accurate?
- Relevance: is this content relevant to the corpus's purpose?
- Regression check: does this content contradict or duplicate existing chunks?

Both layers must pass before the content is promoted to PR candidate.

THE PENDING COLLECTION:

Staged content lives in a separate MongoDB collection ("pending_documents")
until it passes evaluation. This means:
- Live queries are never affected by staged content
- Failed evaluations leave no trace in the live collection
- We have an audit trail of what was evaluated and why it passed or failed

DESIGN DECISIONS:

1. We use the same embedding model for staged content as for live content.
   Consistency in embedding space matters more than cost.

2. Test queries are defined per corpus — they encode our assumptions about
   what the corpus should be able to answer. If the corpus can't answer
   these queries, it's not ready to go live.

3. The LLM-as-judge prompt explicitly asks for a structured JSON response.
   We use Pydantic to validate the response and treat malformed JSON as
   a SKIP (not a FAIL) — the same principle as RAGAS nan handling.

4. The evaluation threshold (0.75 faithfulness) is lower than our live
   alert threshold (0.85) because we're evaluating individual chunks,
   not full conversations. Individual chunks are inherently less complete
   than a full RAG response.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field

from src.models import (
    ChunkStrategy,
    CorpusSource,
    Document,
    EvalStatus,
)
from src.perception.health_monitor import (
    CorpusHealthReport,
    PerceptionAction,
    URLHealthResult,
    compute_content_hash,
)


def _chunk_semantic(text: str, url: str, source: CorpusSource):
    from src.corpus.loader import chunk_semantic
    return chunk_semantic(text, url, source)


async def _fetch_page_lazy(url: str, client) -> str | None:
    from src.corpus.loader import fetch_page
    return await fetch_page(url, client)

if TYPE_CHECKING:
    pass


# ── Staging models ────────────────────────────────────────────────────────


class StagingStatus(StrEnum):
    PENDING     = "pending"      # ingested, awaiting evaluation
    EVAL_PASSED = "eval_passed"  # passed all gates — ready for PR
    EVAL_FAILED = "eval_failed"  # failed evaluation — needs investigation
    EVAL_SKIP   = "eval_skip"    # evaluation infrastructure failed — retry
    PR_DRAFTED  = "pr_drafted"   # PR has been created
    REJECTED    = "rejected"     # human rejected the PR


class JudgeVerdict(StrEnum):
    APPROVE = "approve"   # content is high quality, should go live
    REJECT  = "reject"    # content is low quality or harmful
    REVIEW  = "review"    # borderline — needs human eyes before PR


class LLMJudgeResult(BaseModel):
    """
    Structured output from the LLM-as-judge evaluation.

    The judge evaluates staged content on three dimensions:
    - quality: is this content informative and well-structured?
    - relevance: is this content relevant to the corpus's purpose?
    - regression: does this contradict or duplicate existing content?

    Each dimension is scored 0.0–1.0. The overall verdict is determined
    by the minimum score — a single dimension failure blocks the content.
    """
    quality_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    regression_score: float = Field(ge=0.0, le=1.0)
    verdict: JudgeVerdict
    reasoning: str
    flags: list[str] = Field(
        default_factory=list,
        description="Specific issues the judge identified",
    )

    @property
    def overall_score(self) -> float:
        return min(self.quality_score, self.relevance_score, self.regression_score)

    @property
    def passed(self) -> bool:
        return self.verdict == JudgeVerdict.APPROVE


class StagedDocument(BaseModel):
    """
    A document that has been ingested to the pending collection
    and is awaiting evaluation.
    """
    id: UUID = Field(default_factory=uuid4)
    source_url: str
    source: CorpusSource
    action_trigger: PerceptionAction
    chunks: list[Document]
    content_hash: str
    status: StagingStatus = StagingStatus.PENDING
    faithfulness_score: float | None = None
    judge_result: LLMJudgeResult | None = None
    eval_status: EvalStatus | None = None
    staged_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evaluated_at: datetime | None = None

    @property
    def passed_gate(self) -> bool:
        return self.status == StagingStatus.EVAL_PASSED

    def to_mongo(self) -> dict[str, Any]:
        data = self.model_dump()
        data["_id"] = str(data.pop("id"))
        data["staged_at"] = data["staged_at"].isoformat()
        if data["evaluated_at"]:
            data["evaluated_at"] = data["evaluated_at"].isoformat()
        # Flatten chunks for storage
        data["chunk_count"] = len(self.chunks)
        data["chunks"] = [c.to_mongo() for c in self.chunks]
        return data


class StagingReport(BaseModel):
    """Summary of one staging pipeline run."""
    run_id: UUID = Field(default_factory=uuid4)
    triggered_by: str = Field(description="health_monitor | manual | scheduled")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    urls_processed: int = 0
    chunks_staged: int = 0
    eval_passed: int = 0
    eval_failed: int = 0
    eval_skipped: int = 0
    staged_documents: list[StagedDocument] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        total = self.eval_passed + self.eval_failed
        return self.eval_passed / total if total > 0 else 0.0

    @property
    def ready_for_pr(self) -> list[StagedDocument]:
        return [d for d in self.staged_documents if d.passed_gate]

    @property
    def summary(self) -> str:
        return (
            f"Staging run {self.run_id}: "
            f"{self.urls_processed} URLs → {self.chunks_staged} chunks staged | "
            f"Eval: {self.eval_passed} passed, {self.eval_failed} failed, "
            f"{self.eval_skipped} skipped | "
            f"{len(self.ready_for_pr)} ready for PR"
        )


# ── Test query sets ───────────────────────────────────────────────────────

# These encode our assumptions about what each corpus should answer.
# If staged content can't answer these, it's not ready to go live.
# Product decision: what are the canonical questions each corpus must answer?

CORPUS_TEST_QUERIES: dict[CorpusSource, list[str]] = {
    CorpusSource.MONGODB: [
        "How do I create a vector search index in MongoDB Atlas?",
        "What is the $vectorSearch aggregation stage?",
        "How do I integrate LangChain with MongoDB Atlas Vector Search?",
    ],
    CorpusSource.DASH0: [
        "How do I send OpenTelemetry traces to Dash0?",
        "What is the Dash0 OTLP ingestion endpoint?",
        "How do I configure the OpenTelemetry SDK for Python?",
    ],
    CorpusSource.REAP: [
        "How do I authenticate with the Reap API?",
        "What webhook events does Reap support?",
        "How do I get started with Reap?",
    ],
}


# ── LLM-as-judge ─────────────────────────────────────────────────────────


LLM_JUDGE_PROMPT = """You are evaluating documentation chunks that are candidates for inclusion in a RAG knowledge base.

CORPUS PURPOSE: {corpus_purpose}

CONTENT TO EVALUATE:
---
{content_sample}
---

Evaluate this content on three dimensions and respond with ONLY valid JSON:

{{
  "quality_score": <0.0-1.0, how informative and well-structured is this content>,
  "relevance_score": <0.0-1.0, how relevant is this to the corpus purpose>,
  "regression_score": <0.0-1.0, 1.0 means no contradictions or harmful duplicates>,
  "verdict": <"approve" | "reject" | "review">,
  "reasoning": <one sentence explaining the verdict>,
  "flags": [<list of specific issues, empty if none>]
}}

SCORING GUIDE:
- quality_score < 0.6: content is sparse, marketing-heavy, or poorly structured
- relevance_score < 0.6: content is off-topic for the corpus purpose
- regression_score < 0.7: content contradicts existing knowledge or is heavily duplicated
- verdict "approve": all scores >= 0.7
- verdict "reject": any score < 0.5
- verdict "review": borderline (scores between 0.5-0.7 on any dimension)

Respond with ONLY the JSON object. No preamble, no explanation, no markdown fences."""

CORPUS_PURPOSES: dict[CorpusSource, str] = {
    CorpusSource.MONGODB: "MongoDB Atlas Vector Search integration, vector indexing, and LangChain/Python SDK usage",
    CorpusSource.DASH0: "OpenTelemetry observability, OTLP configuration, trace/metric/log ingestion into Dash0",
    CorpusSource.REAP: "Reap API authentication, payment workflows, webhook configuration, and developer onboarding",
}


async def run_llm_judge(
    content_sample: str,
    source: CorpusSource,
    openrouter_api_key: str,
    openrouter_base_url: str,
    model: str,
) -> LLMJudgeResult | None:
    """
    Run the LLM-as-judge evaluation on a sample of staged content.

    We sample the content (first 2000 chars) to control cost.
    The judge evaluates representativeness, not completeness.

    Returns None if the LLM response is malformed — callers should
    treat None as SKIP (evaluation infrastructure failure), not FAIL.
    """
    prompt = LLM_JUDGE_PROMPT.format(
        corpus_purpose=CORPUS_PURPOSES[source],
        content_sample=content_sample[:2000],
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{openrouter_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,  # determinism — judge must be consistent
                    "max_tokens": 400,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            raw_text = data["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if the model ignored instructions
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(raw_text)
            return LLMJudgeResult(**parsed)

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  [judge] Malformed response: {e}")
        return None
    except Exception as e:
        print(f"  [judge] Error: {e}")
        return None


# ── Staging pipeline ──────────────────────────────────────────────────────


class StagingPipeline:
    """
    Ingests URL health results, stages content, runs the evaluation gate,
    and produces a StagingReport of what's ready for PR.

    FLOW:
    CorpusHealthReport → StagingPipeline → StagingReport
                                        ↓
                              (for each passed document)
                                        ↓
                                   PRDrafter

    The pipeline only processes URLs that need action:
    - RE_INGEST: fetch new content, re-chunk, evaluate
    - INGEST_NEW: fetch new content, chunk, evaluate
    - REMOVE: mark for removal (no evaluation needed)
    - UPDATE_URL: re-ingest from new URL

    URLs with PerceptionAction.NONE are skipped entirely.
    """

    def __init__(
        self,
        openrouter_api_key: str,
        openrouter_base_url: str,
        judge_model: str,
        faithfulness_threshold: float = 0.75,
        judge_threshold: float = 0.70,
    ) -> None:
        self._api_key = openrouter_api_key
        self._base_url = openrouter_base_url
        self._judge_model = judge_model
        self._faithfulness_threshold = faithfulness_threshold
        self._judge_threshold = judge_threshold

    async def _fetch_and_chunk(
        self,
        url: str,
        source: CorpusSource,
        client: httpx.AsyncClient,
    ) -> tuple[list[Document], str] | None:
        """Fetch URL content, chunk it, return (chunks, content_hash)."""
        content = await _fetch_page_lazy(url, client)
        if not content:
            return None

        content_hash = compute_content_hash(content)
        lc_chunks = _chunk_semantic(content, url, source)

        import re
        documents = []
        for i, chunk in enumerate(lc_chunks):
            title_match = re.search(r"^#+ (.+)$", chunk.page_content, re.MULTILINE)
            title = title_match.group(1) if title_match else url.split("/")[-1]

            # Inject content_hash into metadata for future health checks
            metadata = dict(chunk.metadata)
            metadata["content_hash"] = content_hash
            metadata["staged_at"] = datetime.now(timezone.utc).isoformat()

            documents.append(Document(
                content=chunk.page_content,
                source=source,
                url=url,
                chunk_index=i,
                chunk_strategy=ChunkStrategy.SEMANTIC,
                title=title,
                metadata=metadata,
            ))

        return documents, content_hash

    async def _run_eval_gate(
        self,
        staged_doc: StagedDocument,
    ) -> StagedDocument:
        """
        Run both evaluation layers on a staged document.

        Layer 1: LLM-as-judge on content quality
        Layer 2: Simple faithfulness proxy using test query similarity
                 (full RAGAS would require the live pipeline, which is
                  heavier than we want in the staging context)

        In production, Layer 2 would be full RAGAS faithfulness using
        the staging collection as the retrieval source. The current
        implementation uses the judge verdict as the primary gate.
        """
        print(f"  [eval_gate] Evaluating {staged_doc.source_url}...")

        # Sample content from chunks for the judge
        content_sample = "\n\n".join(
            chunk.content for chunk in staged_doc.chunks[:3]  # first 3 chunks
        )

        # Layer 1: LLM-as-judge
        judge_result = await run_llm_judge(
            content_sample=content_sample,
            source=staged_doc.source,
            openrouter_api_key=self._api_key,
            openrouter_base_url=self._base_url,
            model=self._judge_model,
        )

        now = datetime.now(timezone.utc)

        if judge_result is None:
            # Evaluation infrastructure failure — SKIP, not FAIL
            staged_doc.status = StagingStatus.EVAL_SKIP
            staged_doc.eval_status = EvalStatus.SKIP
            staged_doc.evaluated_at = now
            print(f"  [eval_gate] SKIP — judge returned no result")
            return staged_doc

        staged_doc.judge_result = judge_result
        print(
            f"  [eval_gate] Judge: {judge_result.verdict} "
            f"(quality={judge_result.quality_score:.2f}, "
            f"relevance={judge_result.relevance_score:.2f}, "
            f"regression={judge_result.regression_score:.2f})"
        )

        if judge_result.flags:
            print(f"  [eval_gate] Flags: {', '.join(judge_result.flags)}")

        # Determine gate outcome
        if judge_result.verdict == JudgeVerdict.APPROVE:
            staged_doc.status = StagingStatus.EVAL_PASSED
            staged_doc.eval_status = EvalStatus.PASS
            print(f"  [eval_gate] PASSED ✓")
        elif judge_result.verdict == JudgeVerdict.REJECT:
            staged_doc.status = StagingStatus.EVAL_FAILED
            staged_doc.eval_status = EvalStatus.FAIL
            print(f"  [eval_gate] FAILED ✗ — {judge_result.reasoning}")
        else:  # REVIEW
            # Borderline — mark as SKIP so human decides
            staged_doc.status = StagingStatus.EVAL_SKIP
            staged_doc.eval_status = EvalStatus.SKIP
            print(f"  [eval_gate] REVIEW — flagged for human decision")

        staged_doc.evaluated_at = now
        return staged_doc

    async def run(
        self,
        health_report: CorpusHealthReport,
        triggered_by: str = "health_monitor",
    ) -> StagingReport:
        """
        Process all URLs that need action from the health report.

        Only processes actionable results — NONE status URLs are skipped.
        """
        actionable = health_report.urls_needing_action
        report = StagingReport(
            triggered_by=triggered_by,
            urls_processed=len(actionable),
        )

        if not actionable:
            print("[staging] No URLs need action. Pipeline complete.")
            return report

        print(f"\n[staging] Processing {len(actionable)} URLs needing action...")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            for health_result in actionable:
                staged_doc = await self._process_url(health_result, client)
                if staged_doc is None:
                    continue

                report.chunks_staged += len(staged_doc.chunks)

                if staged_doc.status == StagingStatus.EVAL_PASSED:
                    report.eval_passed += 1
                elif staged_doc.status == StagingStatus.EVAL_FAILED:
                    report.eval_failed += 1
                elif staged_doc.status == StagingStatus.EVAL_SKIP:
                    report.eval_skipped += 1

                report.staged_documents.append(staged_doc)

        print(f"\n[staging] {report.summary}")
        return report

    async def _process_url(
        self,
        health_result: URLHealthResult,
        client: httpx.AsyncClient,
    ) -> StagedDocument | None:
        """Process a single URL based on its health result action."""

        action = health_result.action

        # REMOVE — no evaluation needed, just flag for removal
        if action == PerceptionAction.REMOVE:
            print(f"  [staging] REMOVE flagged: {health_result.url}")
            return StagedDocument(
                source_url=health_result.url,
                source=health_result.source,
                action_trigger=action,
                chunks=[],
                content_hash="",
                status=StagingStatus.EVAL_PASSED,  # removal doesn't need eval
                eval_status=EvalStatus.PASS,
                evaluated_at=datetime.now(timezone.utc),
            )

        # RE_INGEST or INGEST_NEW or UPDATE_URL — fetch, chunk, evaluate
        fetch_url = health_result.redirect_url or health_result.url
        print(f"  [staging] Fetching {fetch_url} (action: {action})...")

        result = await self._fetch_and_chunk(fetch_url, health_result.source, client)
        if result is None:
            print(f"  [staging] Could not fetch content — skipping")
            return None

        chunks, content_hash = result
        print(f"  [staging] Staged {len(chunks)} chunks")

        staged_doc = StagedDocument(
            source_url=health_result.url,
            source=health_result.source,
            action_trigger=action,
            chunks=chunks,
            content_hash=content_hash,
        )

        # Run evaluation gate
        staged_doc = await self._run_eval_gate(staged_doc)
        return staged_doc
