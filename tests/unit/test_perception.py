"""
Unit tests for the Perception layer.

Testing philosophy: mock all network calls and external dependencies.
We're testing the decision logic, not the network.

Key things to test:
- URL health classification: does the right status map to the right action?
- Content hash detection: does a changed hash trigger RE_INGEST?
- Evaluation gate: does a judge rejection correctly set EVAL_FAILED?
- PR body generation: does the PR include all evidence fields?
- Edge cases: malformed judge responses, network errors, empty content
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx

from src.models import CorpusSource, EvalStatus
from src.perception.health_monitor import (
    CorpusHealthMonitor,
    CorpusHealthReport,
    URLHealthResult,
    URLStatus,
    PerceptionAction,
    compute_content_hash,
    check_url_health,
)
from src.perception.staging_pipeline import (
    JudgeVerdict,
    LLMJudgeResult,
    StagedDocument,
    StagingPipeline,
    StagingReport,
    StagingStatus,
)
from src.perception.pr_drafter import (
    GitHubPRDrafter,
    PRDraftStatus,
    build_pr_body,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def healthy_response():
    """Mock httpx response for a healthy URL."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.history = []
    resp.url = httpx.URL("https://example.com/docs/page")
    resp.text = "<html><body><main><h1>Test Page</h1><p>Content here.</p></main></body></html>"
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def not_found_response():
    """Mock httpx response for a 404."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 404
    resp.history = []
    resp.url = httpx.URL("https://example.com/docs/gone")
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("404", request=MagicMock(), response=resp)
    )
    return resp


@pytest.fixture
def redirect_response():
    """Mock httpx response for a redirect."""
    original = MagicMock(spec=httpx.Response)
    original.status_code = 301
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.history = [original]
    resp.url = httpx.URL("https://example.com/docs/new-location")
    resp.text = "<html><body><main><h1>Moved</h1></main></body></html>"
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def passing_judge():
    """LLM judge result that passes all thresholds."""
    return LLMJudgeResult(
        quality_score=0.92,
        relevance_score=0.88,
        regression_score=0.95,
        verdict=JudgeVerdict.APPROVE,
        reasoning="Content is high quality, relevant, and introduces no regressions.",
        flags=[],
    )


@pytest.fixture
def failing_judge():
    """LLM judge result that fails quality threshold."""
    return LLMJudgeResult(
        quality_score=0.40,
        relevance_score=0.85,
        regression_score=0.90,
        verdict=JudgeVerdict.REJECT,
        reasoning="Content is sparse marketing copy with no technical value.",
        flags=["Low information density", "No code examples or configuration details"],
    )


@pytest.fixture
def review_judge():
    """LLM judge result that needs human review."""
    return LLMJudgeResult(
        quality_score=0.72,
        relevance_score=0.65,
        regression_score=0.80,
        verdict=JudgeVerdict.REVIEW,
        reasoning="Content partially relevant but some sections are off-topic.",
        flags=["Off-topic section detected"],
    )


@pytest.fixture
def staged_doc_passed(passing_judge):
    """A staged document that passed the evaluation gate."""
    from src.models import Document, ChunkStrategy
    return StagedDocument(
        source_url="https://reap.readme.io/docs/getting-started",
        source=CorpusSource.REAP,
        action_trigger=PerceptionAction.RE_INGEST,
        chunks=[
            Document(
                content="# Getting Started\n\nWelcome to Reap API. Here's how to authenticate.",
                source=CorpusSource.REAP,
                url="https://reap.readme.io/docs/getting-started",
                chunk_index=0,
                chunk_strategy=ChunkStrategy.SEMANTIC,
                title="Getting Started",
                metadata={"content_hash": "abc123"},
            )
        ],
        content_hash="abc123def456",
        status=StagingStatus.EVAL_PASSED,
        eval_status=EvalStatus.PASS,
        judge_result=passing_judge,
        evaluated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def staged_doc_failed(failing_judge):
    """A staged document that failed the evaluation gate."""
    from src.models import Document, ChunkStrategy
    return StagedDocument(
        source_url="https://reap.readme.io/docs/marketing",
        source=CorpusSource.REAP,
        action_trigger=PerceptionAction.INGEST_NEW,
        chunks=[
            Document(
                content="Reap is the best payment platform in Asia!",
                source=CorpusSource.REAP,
                url="https://reap.readme.io/docs/marketing",
                chunk_index=0,
                chunk_strategy=ChunkStrategy.SEMANTIC,
                title="Marketing",
                metadata={},
            )
        ],
        content_hash="marketing456",
        status=StagingStatus.EVAL_FAILED,
        eval_status=EvalStatus.FAIL,
        judge_result=failing_judge,
        evaluated_at=datetime.now(timezone.utc),
    )


# ── Content hash tests ────────────────────────────────────────────────────


class TestComputeContentHash:

    def test_same_content_same_hash(self):
        """Identical content produces identical hash."""
        content = "# Test Page\n\nThis is the content."
        assert compute_content_hash(content) == compute_content_hash(content)

    def test_different_content_different_hash(self):
        """Changed content produces different hash."""
        original = "# Test Page\n\nOriginal content."
        changed = "# Test Page\n\nUpdated content with new information."
        assert compute_content_hash(original) != compute_content_hash(changed)

    def test_whitespace_normalised(self):
        """Trailing whitespace doesn't affect hash."""
        content_clean = "# Test\n\nContent here."
        content_trailing = "# Test   \n\nContent here.  "
        assert compute_content_hash(content_clean) == compute_content_hash(content_trailing)

    def test_empty_lines_normalised(self):
        """Multiple consecutive blank lines don't affect hash."""
        content_single = "# Test\n\nContent."
        content_multiple = "# Test\n\n\n\nContent."
        assert compute_content_hash(content_single) == compute_content_hash(content_multiple)

    def test_returns_hex_string(self):
        """Hash is a valid SHA-256 hex string."""
        result = compute_content_hash("test content")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


# ── URL health check tests ────────────────────────────────────────────────


class TestCheckURLHealth:

    @pytest.mark.asyncio
    async def test_404_returns_not_found(self):
        """404 response → NOT_FOUND status → REMOVE action."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.history = []
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await check_url_health(
            url="https://example.com/gone",
            source=CorpusSource.REAP,
            stored_hash="oldhash",
            client=mock_client,
        )

        assert result.status == URLStatus.NOT_FOUND
        assert result.action == PerceptionAction.REMOVE
        assert result.http_status_code == 404

    @pytest.mark.asyncio
    async def test_unchanged_content_returns_healthy(self):
        """Same content hash → HEALTHY status → NONE action."""
        content = "# Test\n\nContent that hasn't changed."
        stored_hash = compute_content_hash(content)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.history = []
        mock_response.url = httpx.URL("https://example.com/page")
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("src.perception.health_monitor._fetch_page", return_value=content):
            result = await check_url_health(
                url="https://example.com/page",
                source=CorpusSource.MONGODB,
                stored_hash=stored_hash,
                client=mock_client,
            )

        assert result.status == URLStatus.HEALTHY
        assert result.action == PerceptionAction.NONE
        assert result.content_changed is False

    @pytest.mark.asyncio
    async def test_changed_content_returns_changed(self):
        """Different content hash → CHANGED status → RE_INGEST action."""
        new_content = "# Test\n\nThis content has been updated with new information."
        old_hash = compute_content_hash("# Test\n\nOld content.")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.history = []
        mock_response.url = httpx.URL("https://example.com/page")
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("src.perception.health_monitor._fetch_page", return_value=new_content):
            result = await check_url_health(
                url="https://example.com/page",
                source=CorpusSource.MONGODB,
                stored_hash=old_hash,
                client=mock_client,
            )

        assert result.status == URLStatus.CHANGED
        assert result.action == PerceptionAction.RE_INGEST
        assert result.content_changed is True

    @pytest.mark.asyncio
    async def test_no_stored_hash_returns_new(self):
        """No stored hash (never ingested) → NEW status → INGEST_NEW action."""
        content = "# New Page\n\nThis is brand new documentation."

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.history = []
        mock_response.url = httpx.URL("https://example.com/new")
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("src.perception.health_monitor._fetch_page", return_value=content):
            result = await check_url_health(
                url="https://example.com/new",
                source=CorpusSource.DASH0,
                stored_hash=None,  # Never ingested
                client=mock_client,
            )

        assert result.status == URLStatus.NEW
        assert result.action == PerceptionAction.INGEST_NEW

    @pytest.mark.asyncio
    async def test_redirect_returns_redirected(self):
        """Redirect response → REDIRECTED status → UPDATE_URL action."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.history = [MagicMock()]  # Non-empty = redirect occurred
        mock_response.url = httpx.URL("https://example.com/new-location")
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await check_url_health(
            url="https://example.com/old-location",
            source=CorpusSource.REAP,
            stored_hash="oldhash",
            client=mock_client,
        )

        assert result.status == URLStatus.REDIRECTED
        assert result.action == PerceptionAction.UPDATE_URL
        assert result.redirect_url == "https://example.com/new-location"

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self):
        """Network timeout → ERROR status → INVESTIGATE action."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        result = await check_url_health(
            url="https://example.com/slow",
            source=CorpusSource.MONGODB,
            stored_hash="oldhash",
            client=mock_client,
        )

        assert result.status == URLStatus.ERROR
        assert result.action == PerceptionAction.INVESTIGATE
        assert "timed out" in result.error_message.lower()

    def test_needs_action_property(self):
        """needs_action is False for HEALTHY, True for everything else."""
        healthy = URLHealthResult(
            url="https://example.com",
            source=CorpusSource.REAP,
            status=URLStatus.HEALTHY,
            action=PerceptionAction.NONE,
        )
        assert not healthy.needs_action

        changed = URLHealthResult(
            url="https://example.com",
            source=CorpusSource.REAP,
            status=URLStatus.CHANGED,
            action=PerceptionAction.RE_INGEST,
        )
        assert changed.needs_action


# ── CorpusHealthReport tests ──────────────────────────────────────────────


class TestCorpusHealthReport:

    def test_health_score_all_healthy(self):
        """100% healthy URLs → health_score of 1.0."""
        results = [
            URLHealthResult(url=f"https://example.com/{i}", source=CorpusSource.REAP,
                          status=URLStatus.HEALTHY, action=PerceptionAction.NONE)
            for i in range(5)
        ]
        report = CorpusHealthReport(
            total_urls=5, healthy=5, changed=0,
            not_found=0, new_discovered=0, errors=0,
            results=results,
        )
        assert report.health_score == 1.0

    def test_health_score_partial(self):
        """3/5 healthy → health_score of 0.6."""
        report = CorpusHealthReport(
            total_urls=5, healthy=3, changed=1,
            not_found=1, new_discovered=0, errors=0,
            results=[],
        )
        assert report.health_score == pytest.approx(0.6)

    def test_urls_needing_action_filters_correctly(self):
        """urls_needing_action excludes HEALTHY/NONE results."""
        results = [
            URLHealthResult(url="https://a.com", source=CorpusSource.REAP,
                          status=URLStatus.HEALTHY, action=PerceptionAction.NONE),
            URLHealthResult(url="https://b.com", source=CorpusSource.REAP,
                          status=URLStatus.CHANGED, action=PerceptionAction.RE_INGEST),
            URLHealthResult(url="https://c.com", source=CorpusSource.REAP,
                          status=URLStatus.NOT_FOUND, action=PerceptionAction.REMOVE),
        ]
        report = CorpusHealthReport(
            total_urls=3, healthy=1, changed=1,
            not_found=1, new_discovered=0, errors=0,
            results=results,
        )
        actionable = report.urls_needing_action
        assert len(actionable) == 2
        assert all(r.needs_action for r in actionable)

    def test_empty_corpus_health_score(self):
        """Empty corpus → health_score of 1.0 (no problems)."""
        report = CorpusHealthReport(
            total_urls=0, healthy=0, changed=0,
            not_found=0, new_discovered=0, errors=0,
            results=[],
        )
        assert report.health_score == 1.0


# ── LLM judge tests ───────────────────────────────────────────────────────


class TestLLMJudgeResult:

    def test_overall_score_is_minimum(self, passing_judge):
        """overall_score is the minimum of the three dimension scores."""
        judge = LLMJudgeResult(
            quality_score=0.95,
            relevance_score=0.72,  # minimum
            regression_score=0.88,
            verdict=JudgeVerdict.APPROVE,
            reasoning="Good overall.",
        )
        assert judge.overall_score == pytest.approx(0.72)

    def test_approve_verdict_passes(self, passing_judge):
        assert passing_judge.passed is True

    def test_reject_verdict_fails(self, failing_judge):
        assert failing_judge.passed is False

    def test_review_verdict_does_not_pass(self, review_judge):
        assert review_judge.passed is False

    def test_flags_populated(self, failing_judge):
        assert len(failing_judge.flags) > 0
        assert any("density" in f.lower() for f in failing_judge.flags)


# ── Staging pipeline evaluation gate tests ───────────────────────────────


class TestStagingPipelineEvalGate:

    @pytest.mark.asyncio
    async def test_passing_judge_sets_eval_passed(self, passing_judge):
        """A judge APPROVE verdict → StagingStatus.EVAL_PASSED."""
        from src.models import Document, ChunkStrategy

        pipeline = StagingPipeline(
            openrouter_api_key="test",
            openrouter_base_url="https://openrouter.ai/api/v1",
            judge_model="test-model",
        )

        staged_doc = StagedDocument(
            source_url="https://example.com/docs",
            source=CorpusSource.REAP,
            action_trigger=PerceptionAction.RE_INGEST,
            chunks=[Document(
                content="# API Reference\n\nAuthentication details here.",
                source=CorpusSource.REAP,
                url="https://example.com/docs",
                chunk_index=0,
                chunk_strategy=ChunkStrategy.SEMANTIC,
                title="API Reference",
                metadata={},
            )],
            content_hash="testhash",
        )

        with patch(
            "src.perception.staging_pipeline.run_llm_judge",
            return_value=passing_judge
        ):
            result = await pipeline._run_eval_gate(staged_doc)

        assert result.status == StagingStatus.EVAL_PASSED
        assert result.eval_status == EvalStatus.PASS

    @pytest.mark.asyncio
    async def test_failing_judge_sets_eval_failed(self, failing_judge):
        """A judge REJECT verdict → StagingStatus.EVAL_FAILED."""
        from src.models import Document, ChunkStrategy

        pipeline = StagingPipeline(
            openrouter_api_key="test",
            openrouter_base_url="https://openrouter.ai/api/v1",
            judge_model="test-model",
        )

        staged_doc = StagedDocument(
            source_url="https://example.com/marketing",
            source=CorpusSource.REAP,
            action_trigger=PerceptionAction.INGEST_NEW,
            chunks=[Document(
                content="Reap is amazing! Best payments ever!",
                source=CorpusSource.REAP,
                url="https://example.com/marketing",
                chunk_index=0,
                chunk_strategy=ChunkStrategy.SEMANTIC,
                title="Marketing",
                metadata={},
            )],
            content_hash="marketinghash",
        )

        with patch(
            "src.perception.staging_pipeline.run_llm_judge",
            return_value=failing_judge
        ):
            result = await pipeline._run_eval_gate(staged_doc)

        assert result.status == StagingStatus.EVAL_FAILED
        assert result.eval_status == EvalStatus.FAIL

    @pytest.mark.asyncio
    async def test_none_judge_sets_eval_skip(self):
        """Judge returns None (infrastructure failure) → EVAL_SKIP not FAIL."""
        from src.models import Document, ChunkStrategy

        pipeline = StagingPipeline(
            openrouter_api_key="test",
            openrouter_base_url="https://openrouter.ai/api/v1",
            judge_model="test-model",
        )

        staged_doc = StagedDocument(
            source_url="https://example.com/docs",
            source=CorpusSource.REAP,
            action_trigger=PerceptionAction.RE_INGEST,
            chunks=[Document(
                content="Some content.",
                source=CorpusSource.REAP,
                url="https://example.com/docs",
                chunk_index=0,
                chunk_strategy=ChunkStrategy.SEMANTIC,
                title="Docs",
                metadata={},
            )],
            content_hash="testhash",
        )

        with patch(
            "src.perception.staging_pipeline.run_llm_judge",
            return_value=None  # Infrastructure failure
        ):
            result = await pipeline._run_eval_gate(staged_doc)

        # SKIP, not FAIL — infrastructure failure ≠ quality failure
        assert result.status == StagingStatus.EVAL_SKIP
        assert result.eval_status == EvalStatus.SKIP

    @pytest.mark.asyncio
    async def test_review_verdict_sets_skip(self, review_judge):
        """Judge REVIEW verdict → EVAL_SKIP (needs human decision)."""
        from src.models import Document, ChunkStrategy

        pipeline = StagingPipeline(
            openrouter_api_key="test",
            openrouter_base_url="https://openrouter.ai/api/v1",
            judge_model="test-model",
        )

        staged_doc = StagedDocument(
            source_url="https://example.com/borderline",
            source=CorpusSource.DASH0,
            action_trigger=PerceptionAction.RE_INGEST,
            chunks=[Document(
                content="Borderline content.",
                source=CorpusSource.DASH0,
                url="https://example.com/borderline",
                chunk_index=0,
                chunk_strategy=ChunkStrategy.SEMANTIC,
                title="Borderline",
                metadata={},
            )],
            content_hash="borderhash",
        )

        with patch(
            "src.perception.staging_pipeline.run_llm_judge",
            return_value=review_judge
        ):
            result = await pipeline._run_eval_gate(staged_doc)

        assert result.status == StagingStatus.EVAL_SKIP


# ── Staging report tests ──────────────────────────────────────────────────


class TestStagingReport:

    def test_ready_for_pr_filters_passed_only(self, staged_doc_passed, staged_doc_failed):
        """ready_for_pr returns only EVAL_PASSED documents."""
        report = StagingReport(
            triggered_by="test",
            urls_processed=2,
            eval_passed=1,
            eval_failed=1,
            staged_documents=[staged_doc_passed, staged_doc_failed],
        )
        ready = report.ready_for_pr
        assert len(ready) == 1
        assert ready[0].source_url == staged_doc_passed.source_url

    def test_pass_rate_calculation(self):
        report = StagingReport(
            triggered_by="test",
            urls_processed=4,
            eval_passed=3,
            eval_failed=1,
            staged_documents=[],
        )
        assert report.pass_rate == pytest.approx(0.75)

    def test_pass_rate_zero_when_no_evals(self):
        report = StagingReport(triggered_by="test")
        assert report.pass_rate == 0.0


# ── PR body generation tests ──────────────────────────────────────────────


class TestPRBodyGeneration:

    def test_pr_body_contains_url(self, staged_doc_passed):
        body = build_pr_body(staged_doc_passed)
        assert staged_doc_passed.source_url in body

    def test_pr_body_contains_eval_scores(self, staged_doc_passed):
        body = build_pr_body(staged_doc_passed)
        judge = staged_doc_passed.judge_result
        assert str(judge.quality_score) in body
        assert str(judge.relevance_score) in body

    def test_pr_body_contains_verdict(self, staged_doc_passed):
        body = build_pr_body(staged_doc_passed)
        assert "APPROVE" in body.upper()

    def test_pr_body_contains_machine_metadata(self, staged_doc_passed):
        """PR body must contain JSON metadata for automation parsing."""
        body = build_pr_body(staged_doc_passed)
        assert "staged_document_id" in body
        assert "content_hash" in body
        assert "judge_verdict" in body

    def test_pr_body_contains_reviewer_instructions(self, staged_doc_passed):
        body = build_pr_body(staged_doc_passed)
        assert "Reviewer Instructions" in body or "reviewer" in body.lower()

    def test_pr_body_no_judge_handles_gracefully(self):
        """PR body is still generated when judge result is None."""
        from src.models import Document, ChunkStrategy
        staged_no_judge = StagedDocument(
            source_url="https://example.com/docs",
            source=CorpusSource.REAP,
            action_trigger=PerceptionAction.INGEST_NEW,
            chunks=[Document(
                content="Content.",
                source=CorpusSource.REAP,
                url="https://example.com/docs",
                chunk_index=0,
                chunk_strategy=ChunkStrategy.SEMANTIC,
                title="Docs",
                metadata={},
            )],
            content_hash="testhash",
            status=StagingStatus.EVAL_PASSED,
            eval_status=EvalStatus.PASS,
            judge_result=None,  # Skipped
            evaluated_at=datetime.now(timezone.utc),
        )
        body = build_pr_body(staged_no_judge)
        assert "skipped" in body.lower() or "manual review" in body.lower()

    def test_pr_body_flags_appear_when_present(self, staged_doc_failed):
        body = build_pr_body(staged_doc_failed)
        for flag in staged_doc_failed.judge_result.flags:
            assert flag in body


# ── PR drafter skips non-passed documents ─────────────────────────────────


class TestGitHubPRDrafter:

    @pytest.mark.asyncio
    async def test_skips_failed_documents(self, staged_doc_failed):
        """PR drafter skips EVAL_FAILED documents without calling GitHub API."""
        drafter = GitHubPRDrafter(
            github_token="test-token",
            repo_owner="cobih",
            repo_name="ai-builder",
        )

        result = await drafter.draft_pr(staged_doc_failed)

        assert result.status == PRDraftStatus.SKIPPED
        assert result.pr_url is None

    @pytest.mark.asyncio
    async def test_draft_all_empty_staging_report(self):
        """draft_all returns empty report when no documents are ready."""
        drafter = GitHubPRDrafter(
            github_token="test-token",
            repo_owner="cobih",
            repo_name="ai-builder",
        )

        empty_report = StagingReport(
            triggered_by="test",
            staged_documents=[],
        )

        result = await drafter.draft_all(empty_report)

        assert result.prs_created == 0
        assert result.prs_failed == 0
        assert len(result.results) == 0
