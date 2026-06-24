"""
Corpus Health Monitor — the Perception layer.

This module solves the most underappreciated failure mode in production RAG:
CONTEXT GOES STALE.

Most RAG systems are built, demoed, and then left to quietly degrade:
- URLs return 404 because documentation moved
- Page content changes but the embedded chunks are still the old version
- New pages are added to the docs site but never ingested
- The system keeps answering confidently from outdated context

In a financial compliance context (Reap's world), this isn't just
quality degradation — it's a liability. An AI that cites a KYC policy
that changed six months ago is actively dangerous.

THE THREE PERCEPTION SIGNALS:

1. URL Health — is this URL still accessible? 404/redirect = action needed
2. Content Freshness — has the page changed since we ingested it?
   We detect this via content hash comparison, not timestamps.
   Timestamps lie (CDN caching). Content hashes don't.
3. New URL Discovery — are there URLs in the sitemap or navigation
   that we haven't ingested yet? Gap detection.

DESIGN DECISIONS:

- We check health on a schedule (daily), not on every query.
  Checking on every query would add latency and hammer the source servers.

- We store content hashes in MongoDB alongside the document chunks.
  This lets us detect changes without re-fetching the full page on every run.

- Health results flow into the StagingPipeline, not directly into the
  live collection. Nothing goes live without passing the evaluation gate.

- CorpusHealthStatus is the contract between Perception and the
  StagingPipeline. Perception detects; Staging validates; Action ships.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field

from src.models import CorpusSource

if TYPE_CHECKING:
    pass


def _get_corpus_urls() -> dict:
    """Lazy import to avoid pulling langchain at module load time."""
    from src.corpus.loader import CORPUS_URLS
    return CORPUS_URLS


async def _fetch_page(url: str, client: httpx.AsyncClient) -> str | None:
    """Lazy import wrapper for fetch_page."""
    from src.corpus.loader import fetch_page
    return await fetch_page(url, client)


# ── Health signal enums ───────────────────────────────────────────────────


class URLStatus(StrEnum):
    HEALTHY    = "healthy"     # 200, content unchanged
    CHANGED    = "changed"     # 200, but content hash differs from stored
    NOT_FOUND  = "not_found"   # 404 or 410 — document removed
    REDIRECTED = "redirected"  # URL moved — update the catalogue
    ERROR      = "error"       # network error, timeout, server error
    NEW        = "new"         # URL discovered that we haven't ingested


class PerceptionAction(StrEnum):
    NONE        = "none"        # healthy, no action needed
    RE_INGEST   = "re_ingest"   # content changed — re-fetch and stage
    REMOVE      = "remove"      # 404 — remove from live collection
    UPDATE_URL  = "update_url"  # redirect — update catalogue to new URL
    INGEST_NEW  = "ingest_new"  # new URL discovered — ingest and stage
    INVESTIGATE = "investigate" # persistent errors — flag for human


# ── Health models ─────────────────────────────────────────────────────────


class URLHealthResult(BaseModel):
    """
    Health check result for a single URL.

    content_hash is computed from the fetched Markdown (post-parse),
    not the raw HTML. This means cosmetic HTML changes (whitespace,
    attribute ordering) don't trigger false positives.
    """
    url: str
    source: CorpusSource
    status: URLStatus
    action: PerceptionAction
    http_status_code: int | None = None
    redirect_url: str | None = None
    content_hash: str | None = None
    stored_hash: str | None = None
    content_changed: bool = False
    error_message: str | None = None
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def needs_action(self) -> bool:
        return self.action != PerceptionAction.NONE

    @property
    def summary(self) -> str:
        parts = [f"[{self.status.upper()}] {self.url}"]
        if self.action != PerceptionAction.NONE:
            parts.append(f"→ {self.action}")
        if self.error_message:
            parts.append(f"({self.error_message})")
        if self.redirect_url:
            parts.append(f"→ {self.redirect_url}")
        return " ".join(parts)


class CorpusHealthReport(BaseModel):
    """
    Full health check across all corpora.

    This is what the scheduled job produces and what feeds into
    the StagingPipeline for action.
    """
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_urls: int
    healthy: int
    changed: int
    not_found: int
    new_discovered: int
    errors: int
    results: list[URLHealthResult]

    @property
    def urls_needing_action(self) -> list[URLHealthResult]:
        return [r for r in self.results if r.needs_action]

    @property
    def health_score(self) -> float:
        """Fraction of URLs that are healthy — the headline metric."""
        if self.total_urls == 0:
            return 1.0
        return self.healthy / self.total_urls

    @property
    def summary(self) -> str:
        return (
            f"Corpus Health: {self.health_score:.0%} healthy "
            f"({self.healthy}/{self.total_urls} URLs) | "
            f"Changed: {self.changed} | "
            f"Not found: {self.not_found} | "
            f"New: {self.new_discovered} | "
            f"Errors: {self.errors}"
        )


# ── Hash utilities ────────────────────────────────────────────────────────


def compute_content_hash(content: str) -> str:
    """
    SHA-256 of normalised content.

    We normalise before hashing to avoid false positives from:
    - Trailing whitespace differences
    - Line ending variations (CRLF vs LF)
    - Multiple consecutive blank lines
    """
    normalised = "\n".join(
        line.rstrip()
        for line in content.splitlines()
        if line.strip()
    )
    return hashlib.sha256(normalised.encode()).hexdigest()


# ── Stored hash retrieval ─────────────────────────────────────────────────


def get_stored_hashes(mongo_uri: str, database: str, collection: str) -> dict[str, str]:
    """
    Retrieve the content hashes we stored during the last ingestion.

    Returns {url: content_hash} for all documents in the live collection.
    If a URL appears multiple times (multiple chunks), we use any hash —
    we only need to know if the content changed, not which chunk changed.
    """
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri)
        db = client[database]
        col = db[collection]

        stored: dict[str, str] = {}
        cursor = col.find(
            {"metadata.content_hash": {"$exists": True}},
            {"url": 1, "metadata.content_hash": 1, "_id": 0}
        )
        for doc in cursor:
            url = doc.get("url", "")
            content_hash = doc.get("metadata", {}).get("content_hash", "")
            if url and content_hash:
                stored[url] = content_hash

        client.close()
        return stored

    except Exception as e:
        print(f"[health_monitor] Could not retrieve stored hashes: {e}")
        return {}


# ── Per-URL health check ──────────────────────────────────────────────────


async def check_url_health(
    url: str,
    source: CorpusSource,
    stored_hash: str | None,
    client: httpx.AsyncClient,
) -> URLHealthResult:
    """
    Check a single URL and determine what action (if any) is needed.

    We fetch the full page content (not just HEAD) because:
    1. HEAD requests don't always reflect content changes
    2. We need the content to compute the hash anyway
    3. We can reuse the fetched content in the staging pipeline
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; ai-builder-health-monitor/0.2.0; "
                "+https://github.com/cobih/ai-builder)"
            ),
            "Accept": "text/html,application/xhtml+xml",
        }

        response = await client.get(url, headers=headers, timeout=15.0)
        http_status = response.status_code

        # 404 / 410 — document is gone
        if http_status in (404, 410):
            return URLHealthResult(
                url=url,
                source=source,
                status=URLStatus.NOT_FOUND,
                action=PerceptionAction.REMOVE,
                http_status_code=http_status,
            )

        # Other client/server errors
        if http_status >= 400:
            return URLHealthResult(
                url=url,
                source=source,
                status=URLStatus.ERROR,
                action=PerceptionAction.INVESTIGATE,
                http_status_code=http_status,
                error_message=f"HTTP {http_status}",
            )

        # Redirect — URL moved
        if response.history and str(response.url) != url:
            return URLHealthResult(
                url=url,
                source=source,
                status=URLStatus.REDIRECTED,
                action=PerceptionAction.UPDATE_URL,
                http_status_code=http_status,
                redirect_url=str(response.url),
            )

        # 200 — fetch and parse content
        content = await _fetch_page(url, client)
        if not content:
            return URLHealthResult(
                url=url,
                source=source,
                status=URLStatus.ERROR,
                action=PerceptionAction.INVESTIGATE,
                http_status_code=http_status,
                error_message="Content parsing returned empty result",
            )

        current_hash = compute_content_hash(content)

        # No stored hash — we've never ingested this (shouldn't happen
        # for catalogue URLs, but possible for newly discovered URLs)
        if stored_hash is None:
            return URLHealthResult(
                url=url,
                source=source,
                status=URLStatus.NEW,
                action=PerceptionAction.INGEST_NEW,
                http_status_code=http_status,
                content_hash=current_hash,
            )

        # Hash matches — content unchanged
        if current_hash == stored_hash:
            return URLHealthResult(
                url=url,
                source=source,
                status=URLStatus.HEALTHY,
                action=PerceptionAction.NONE,
                http_status_code=http_status,
                content_hash=current_hash,
                stored_hash=stored_hash,
                content_changed=False,
            )

        # Hash differs — content has changed
        return URLHealthResult(
            url=url,
            source=source,
            status=URLStatus.CHANGED,
            action=PerceptionAction.RE_INGEST,
            http_status_code=http_status,
            content_hash=current_hash,
            stored_hash=stored_hash,
            content_changed=True,
        )

    except httpx.TimeoutException:
        return URLHealthResult(
            url=url,
            source=source,
            status=URLStatus.ERROR,
            action=PerceptionAction.INVESTIGATE,
            error_message="Request timed out",
        )
    except httpx.RequestError as e:
        return URLHealthResult(
            url=url,
            source=source,
            status=URLStatus.ERROR,
            action=PerceptionAction.INVESTIGATE,
            error_message=str(e),
        )


# ── Full corpus health check ──────────────────────────────────────────────


class CorpusHealthMonitor:
    """
    Runs scheduled health checks across all corpora and produces
    a CorpusHealthReport that feeds into the StagingPipeline.

    SCHEDULING:
    This is designed to be called by a cron job or Cloud Scheduler.
    The MVP schedule: once per day at 02:00 UTC, when source servers
    have low traffic and changes from the previous business day are
    already live.

    For production at Reap, the right schedule depends on how frequently
    their documentation changes. For a fast-moving startup: every 6 hours.
    For stable compliance docs: daily. This is a product decision
    configurable without code changes via the schedule parameter.

    DESIGN DECISION — CATALOGUE VS DISCOVERY:
    We currently check the predefined URL catalogue only.
    True discovery (sitemap crawling, nav link extraction) would catch
    entirely new documentation pages. That's the next iteration.
    The catalogue-based check catches the most common failure modes:
    404s, content changes, and redirects.
    """

    def __init__(
        self,
        mongo_uri: str,
        database: str,
        collection: str,
        concurrency: int = 5,
    ) -> None:
        self._mongo_uri = mongo_uri
        self._database = database
        self._collection = collection
        self._concurrency = concurrency

    async def run(
        self,
        sources: list[CorpusSource] | None = None,
    ) -> CorpusHealthReport:
        """
        Run a full health check and return the report.

        Checks are run concurrently (up to self._concurrency at once)
        to keep the total run time reasonable across large catalogues.
        """
        if sources is None:
            sources = list(CorpusSource)

        # Retrieve stored hashes once — one DB round trip for all URLs
        stored_hashes = get_stored_hashes(
            self._mongo_uri, self._database, self._collection
        )

        # Build the full list of (url, source) to check
        corpus_urls = _get_corpus_urls()
        checks: list[tuple[str, CorpusSource]] = []
        for source in sources:
            for url in corpus_urls.get(source, []):
                checks.append((url, source))

        print(f"\n[health_monitor] Checking {len(checks)} URLs across {len(sources)} corpora...")

        # Run checks with bounded concurrency
        semaphore = asyncio.Semaphore(self._concurrency)
        results: list[URLHealthResult] = []

        async def bounded_check(url: str, source: CorpusSource, client: httpx.AsyncClient) -> URLHealthResult:
            async with semaphore:
                result = await check_url_health(
                    url=url,
                    source=source,
                    stored_hash=stored_hashes.get(url),
                    client=client,
                )
                print(f"  {result.summary}")
                return result

        async with httpx.AsyncClient(follow_redirects=False) as client:
            tasks = [bounded_check(url, source, client) for url, source in checks]
            results = await asyncio.gather(*tasks)

        # Aggregate counts
        status_counts: dict[URLStatus, int] = {s: 0 for s in URLStatus}
        for r in results:
            status_counts[r.status] += 1

        report = CorpusHealthReport(
            total_urls=len(results),
            healthy=status_counts[URLStatus.HEALTHY],
            changed=status_counts[URLStatus.CHANGED],
            not_found=status_counts[URLStatus.NOT_FOUND],
            new_discovered=status_counts[URLStatus.NEW],
            errors=status_counts[URLStatus.ERROR] + status_counts[URLStatus.REDIRECTED],
            results=results,
        )

        print(f"\n[health_monitor] {report.summary}")
        return report
