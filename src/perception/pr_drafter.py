"""
PR Drafter — the Action layer.

THE PRINCIPLE:

Humans should only review content that the system has already validated.

Without an evaluation gate, a Perception system that detects corpus changes
would generate a constant stream of PRs — every documentation update,
every page restructure, every minor wording change. That's not useful.
That's noise.

The evaluation gate (StagingPipeline) ensures that by the time a PR
is created, the content has already passed:
1. LLM-as-judge quality, relevance, and regression scoring
2. Structured output validation (Pydantic)
3. Verdict classification (APPROVE / REJECT / REVIEW)

The PR is not a question — it's a recommendation with evidence.
The human reviewer's job is to verify, not to evaluate from scratch.

WHAT THE PR CONTAINS:

Each PR includes:
- Which URL changed and why (what the health monitor detected)
- What action was taken (RE_INGEST, INGEST_NEW, REMOVE, UPDATE_URL)
- How many chunks were staged
- The LLM-as-judge scores (quality, relevance, regression)
- The judge's reasoning and any flags raised
- A sample of the staged content for spot-checking
- Instructions for the reviewer

WHY GITHUB PRs, NOT JUST A DATABASE FLAG:

PRs give the review workflow:
- Assignment (who reviews this)
- Discussion (comments on the content)
- Approval (explicit human sign-off before anything goes live)
- Audit trail (every corpus change is tracked in git history)

For a compliance-sensitive financial platform like Reap, the audit trail
is not optional. Every change to the AI knowledge base needs a paper trail.

DESIGN DECISIONS:

1. We create PRs against a "corpus-updates" branch, not main directly.
   This allows batching multiple URL updates into one review cycle.

2. The PR description includes machine-readable JSON metadata so that
   a future automation can parse PR outcomes and update the staging
   collection status automatically (PR merged → promote to live).

3. We never merge automatically. Human approval is always required.
   The evaluation gate filters the noise; humans make the final call.
   This is the human-in-the-loop design for AI-managed corpus updates.

4. The GitHub token is required at runtime (environment variable).
   We never hardcode credentials — a design requirement, not just
   good practice.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field

from src.perception.staging_pipeline import (
    PerceptionAction,
    StagedDocument,
    StagingReport,
    StagingStatus,
)


# ── PR models ─────────────────────────────────────────────────────────────


class PRDraftStatus(StrEnum):
    CREATED = "created"
    FAILED  = "failed"
    SKIPPED = "skipped"  # no changes to draft


class PRDraftResult(BaseModel):
    """Result of one PR creation attempt."""
    staged_document_id: UUID
    source_url: str
    status: PRDraftStatus
    pr_url: str | None = None
    pr_number: int | None = None
    error_message: str | None = None
    drafted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PRDraftReport(BaseModel):
    """Summary of one PR drafting run."""
    run_id: UUID = Field(default_factory=uuid4)
    staging_run_id: UUID
    drafted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prs_created: int = 0
    prs_failed: int = 0
    prs_skipped: int = 0
    results: list[PRDraftResult] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"PR Draft run {self.run_id}: "
            f"{self.prs_created} created, "
            f"{self.prs_failed} failed, "
            f"{self.prs_skipped} skipped"
        )


# ── PR body generation ────────────────────────────────────────────────────


def _action_description(action: PerceptionAction) -> str:
    return {
        PerceptionAction.RE_INGEST:   "Content changed — re-ingested with updated chunks",
        PerceptionAction.INGEST_NEW:  "New URL discovered — ingested for the first time",
        PerceptionAction.REMOVE:      "URL returned 404 — chunks flagged for removal",
        PerceptionAction.UPDATE_URL:  "URL redirected — re-ingested from new location",
        PerceptionAction.INVESTIGATE: "Persistent errors detected — flagged for investigation",
        PerceptionAction.NONE:        "No action required",
    }.get(action, str(action))


def build_pr_body(staged_doc: StagedDocument) -> str:
    """
    Build the PR description with evaluation evidence.

    The PR body is the complete information package the human reviewer
    needs to make a decision. It includes:
    - What changed and why
    - Evaluation scores and verdict
    - Content sample for spot-checking
    - Machine-readable metadata for automation

    Design principle: the reviewer should be able to approve or reject
    without leaving the PR. All evidence is in the body.
    """
    judge = staged_doc.judge_result
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build the eval evidence section
    if judge:
        eval_section = f"""## 🔍 Evaluation Results

| Dimension | Score | Threshold |
|-----------|-------|-----------|
| Quality | {judge.quality_score:.2f} | 0.70 |
| Relevance | {judge.relevance_score:.2f} | 0.70 |
| Regression | {judge.regression_score:.2f} | 0.70 |
| **Overall** | **{judge.overall_score:.2f}** | **0.70** |

**Verdict:** `{judge.verdict.upper()}` — {judge.reasoning}

{"⚠️ **Flags raised:**" + chr(10) + chr(10).join(f"- {f}" for f in judge.flags) if judge.flags else "✅ No flags raised"}"""
    else:
        eval_section = """## 🔍 Evaluation Results

⚠️ LLM judge evaluation was skipped (infrastructure issue). Manual review required."""

    # Content sample (first chunk, truncated)
    content_sample = ""
    if staged_doc.chunks and staged_doc.action_trigger != PerceptionAction.REMOVE:
        sample_text = staged_doc.chunks[0].content[:600]
        if len(staged_doc.chunks[0].content) > 600:
            sample_text += "\n\n*[truncated — see full content in staging collection]*"
        content_sample = f"""## 📄 Content Sample (First Chunk)

```
{sample_text}
```"""

    # Machine-readable metadata for automation
    metadata = {
        "staged_document_id": str(staged_doc.id),
        "source_url": staged_doc.source_url,
        "source_corpus": staged_doc.source.value,
        "action": staged_doc.action_trigger.value,
        "chunk_count": len(staged_doc.chunks),
        "content_hash": staged_doc.content_hash,
        "eval_status": staged_doc.eval_status.value if staged_doc.eval_status else None,
        "judge_verdict": judge.verdict.value if judge else None,
        "judge_overall_score": round(judge.overall_score, 3) if judge else None,
        "staged_at": staged_doc.staged_at.isoformat(),
        "generated_at": now,
    }

    return f"""# 🤖 Corpus Update: {staged_doc.source.value.upper()}

**Generated:** {now}
**Source:** [{staged_doc.source_url}]({staged_doc.source_url})
**Action:** {_action_description(staged_doc.action_trigger)}
**Chunks:** {len(staged_doc.chunks)} chunk(s) staged

---

## 📋 What Changed

The corpus health monitor detected that this URL requires action:

- **Corpus:** `{staged_doc.source.value}`
- **URL:** `{staged_doc.source_url}`
- **Action trigger:** `{staged_doc.action_trigger.value}`
- **Content hash:** `{staged_doc.content_hash[:16]}...`

---

{eval_section}

---

{content_sample}

---

## ✅ Reviewer Instructions

1. **Check the content sample** — does it look accurate and relevant?
2. **Review the evaluation scores** — any scores below 0.70 need justification
3. **Check for flags** — each flag should be addressed or explicitly dismissed
4. **Approve** if the content is accurate, relevant, and passes your judgment
5. **Request changes** if the content needs adjustment before going live
6. **Close without merging** if the content should not be added to the corpus

> ⚠️ **Important:** Do not merge this PR manually. Merging triggers the
> automated pipeline that promotes staged chunks to the live collection.
> The CI check must pass before merge is allowed.

---

## 🔧 Machine-Readable Metadata

```json
{json.dumps(metadata, indent=2)}
```

*This PR was generated automatically by the ai-builder Corpus Health Monitor.*
*Staging pipeline: Perception → Evaluation Gate → PR Draft → Human Review → Live*"""


# ── GitHub API client ─────────────────────────────────────────────────────


class GitHubPRDrafter:
    """
    Creates GitHub PRs for staged corpus updates.

    Each PR represents one URL that needs to be updated in the corpus.
    The PR body contains all evaluation evidence the reviewer needs.

    AUTHENTICATION:
    Requires a GitHub personal access token with `repo` scope.
    Set via GITHUB_TOKEN environment variable — never hardcoded.

    BRANCH STRATEGY:
    Each PR is created from a unique branch: corpus-update/{source}/{timestamp}
    This allows multiple updates to be reviewed independently without
    blocking each other.

    PRODUCTION EXTENSION:
    In production, the PR merge would trigger a GitHub Action that:
    1. Reads the machine-readable metadata from the PR body
    2. Promotes the staged chunks from pending_documents to documents
    3. Removes any chunks flagged for deletion
    4. Updates content hashes for the affected URLs
    5. Posts a comment on the PR confirming promotion
    """

    GITHUB_API = "https://api.github.com"

    def __init__(
        self,
        github_token: str,
        repo_owner: str,
        repo_name: str,
        base_branch: str = "main",
    ) -> None:
        self._token = github_token
        self._owner = repo_owner
        self._repo = repo_name
        self._base_branch = base_branch

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _get_base_sha(self, client: httpx.AsyncClient) -> str | None:
        """Get the SHA of the base branch HEAD."""
        try:
            resp = await client.get(
                f"{self.GITHUB_API}/repos/{self._owner}/{self._repo}/git/ref/heads/{self._base_branch}",
                headers=self._headers(),
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()["object"]["sha"]
        except Exception as e:
            print(f"  [pr_drafter] Could not get base SHA: {e}")
            return None

    async def _create_branch(
        self,
        client: httpx.AsyncClient,
        branch_name: str,
        sha: str,
    ) -> bool:
        """Create a new branch for this PR."""
        try:
            resp = await client.post(
                f"{self.GITHUB_API}/repos/{self._owner}/{self._repo}/git/refs",
                headers=self._headers(),
                json={"ref": f"refs/heads/{branch_name}", "sha": sha},
                timeout=15.0,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"  [pr_drafter] Could not create branch '{branch_name}': {e}")
            return False

    async def _commit_metadata(
        self,
        client: httpx.AsyncClient,
        branch_name: str,
        staged_doc: StagedDocument,
    ) -> bool:
        """
        Commit a metadata file to the branch so the PR has a diff.

        GitHub PRs require at least one file change. We commit a
        machine-readable metadata file that records the staging result.
        In production, this would be the actual updated corpus chunks.
        """
        import base64

        file_path = f"corpus-updates/{staged_doc.source.value}/{staged_doc.id}.json"
        content = json.dumps(staged_doc.to_mongo() if hasattr(staged_doc, 'to_mongo') else {
            "id": str(staged_doc.id),
            "source_url": staged_doc.source_url,
            "source": staged_doc.source.value,
            "action": staged_doc.action_trigger.value,
            "chunk_count": len(staged_doc.chunks),
            "content_hash": staged_doc.content_hash,
            "eval_status": staged_doc.eval_status.value if staged_doc.eval_status else None,
            "judge_verdict": staged_doc.judge_result.verdict.value if staged_doc.judge_result else None,
            "staged_at": staged_doc.staged_at.isoformat(),
        }, indent=2)

        encoded = base64.b64encode(content.encode()).decode()

        try:
            resp = await client.put(
                f"{self.GITHUB_API}/repos/{self._owner}/{self._repo}/contents/{file_path}",
                headers=self._headers(),
                json={
                    "message": f"chore: stage corpus update for {staged_doc.source.value} [{staged_doc.action_trigger.value}]",
                    "content": encoded,
                    "branch": branch_name,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"  [pr_drafter] Could not commit metadata file: {e}")
            return False

    async def _create_pr(
        self,
        client: httpx.AsyncClient,
        branch_name: str,
        staged_doc: StagedDocument,
    ) -> dict[str, Any] | None:
        """Create the pull request."""
        source_short = staged_doc.source_url.split("/")[-1] or staged_doc.source_url
        action_label = staged_doc.action_trigger.value.replace("_", " ").title()

        title = f"[Corpus Update] {staged_doc.source.value.upper()}: {action_label} — {source_short}"
        body = build_pr_body(staged_doc)

        try:
            resp = await client.post(
                f"{self.GITHUB_API}/repos/{self._owner}/{self._repo}/pulls",
                headers=self._headers(),
                json={
                    "title": title,
                    "body": body,
                    "head": branch_name,
                    "base": self._base_branch,
                    "draft": True,  # Always draft first — human must explicitly ready it
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  [pr_drafter] Could not create PR: {e}")
            return None

    async def draft_pr(self, staged_doc: StagedDocument) -> PRDraftResult:
        """
        Create a GitHub PR for a single staged document.

        Only processes documents that passed the evaluation gate.
        Skips EVAL_FAILED and EVAL_SKIP documents.
        """
        if staged_doc.status not in (StagingStatus.EVAL_PASSED,):
            return PRDraftResult(
                staged_document_id=staged_doc.id,
                source_url=staged_doc.source_url,
                status=PRDraftStatus.SKIPPED,
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        branch_name = f"corpus-update/{staged_doc.source.value}/{timestamp}-{str(staged_doc.id)[:8]}"

        print(f"  [pr_drafter] Creating PR for {staged_doc.source_url}...")

        async with httpx.AsyncClient() as client:
            # Step 1: Get base branch SHA
            sha = await self._get_base_sha(client)
            if not sha:
                return PRDraftResult(
                    staged_document_id=staged_doc.id,
                    source_url=staged_doc.source_url,
                    status=PRDraftStatus.FAILED,
                    error_message="Could not get base branch SHA",
                )

            # Step 2: Create branch
            if not await self._create_branch(client, branch_name, sha):
                return PRDraftResult(
                    staged_document_id=staged_doc.id,
                    source_url=staged_doc.source_url,
                    status=PRDraftStatus.FAILED,
                    error_message=f"Could not create branch {branch_name}",
                )

            # Step 3: Commit metadata file (creates the diff)
            if not await self._commit_metadata(client, branch_name, staged_doc):
                return PRDraftResult(
                    staged_document_id=staged_doc.id,
                    source_url=staged_doc.source_url,
                    status=PRDraftStatus.FAILED,
                    error_message="Could not commit metadata file",
                )

            # Step 4: Create the PR
            pr_data = await self._create_pr(client, branch_name, staged_doc)
            if not pr_data:
                return PRDraftResult(
                    staged_document_id=staged_doc.id,
                    source_url=staged_doc.source_url,
                    status=PRDraftStatus.FAILED,
                    error_message="Could not create PR",
                )

        pr_url = pr_data.get("html_url", "")
        pr_number = pr_data.get("number")
        print(f"  [pr_drafter] PR created: {pr_url}")

        return PRDraftResult(
            staged_document_id=staged_doc.id,
            source_url=staged_doc.source_url,
            status=PRDraftStatus.CREATED,
            pr_url=pr_url,
            pr_number=pr_number,
        )

    async def draft_all(self, staging_report: StagingReport) -> PRDraftReport:
        """Create PRs for all documents that passed the evaluation gate."""
        ready = staging_report.ready_for_pr
        report = PRDraftReport(staging_run_id=staging_report.run_id)

        if not ready:
            print("[pr_drafter] No documents ready for PR.")
            return report

        print(f"\n[pr_drafter] Drafting PRs for {len(ready)} documents...")

        for staged_doc in ready:
            result = await self.draft_pr(staged_doc)
            report.results.append(result)

            if result.status == PRDraftStatus.CREATED:
                report.prs_created += 1
            elif result.status == PRDraftStatus.FAILED:
                report.prs_failed += 1
            else:
                report.prs_skipped += 1

        print(f"\n[pr_drafter] {report.summary}")
        return report
