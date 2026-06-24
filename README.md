# ai-builder

A production-grade agentic RAG system that answers questions about MongoDB, Dash0, and Reap using those products' own documentation as its knowledge base.

Built to demonstrate the full AI platform stack: **agentic routing → retrieval → evaluation → drift detection → corpus freshness → autonomous action**.

```bash
ai-builder query "How does MongoDB Atlas Vector Search handle ANN search?"
ai-builder query "What is Dash0 Agent0?" --evaluate
ai-builder dashboard
```

**75/78 tests passing** (3 pre-existing router tests require live API keys)

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                        PERCEPTION LAYER                              ║
║                                                                      ║
║  CorpusHealthMonitor → content hash comparison → HEALTHY/CHANGED/   ║
║                        NOT_FOUND/NEW/REDIRECTED                      ║
║        │                                                             ║
║        ▼                                                             ║
║  StagingPipeline → fetch → chunk → LLM-as-judge gate                ║
║        │           APPROVE → EVAL_PASSED                            ║
║        │           REJECT  → EVAL_FAILED (logged, not surfaced)     ║
║        │           REVIEW  → EVAL_SKIP (human decides)              ║
║        │                                                             ║
║        ▼                                                             ║
║  PRDrafter → GitHub PR with eval scores + content sample            ║
║              Human approves → chunks promoted to live corpus        ║
╚══════════════════════════════════════════════════════════════════════╝
                                │
                    (live corpus, quality-gated)
                                │
╔══════════════════════════════════════════════════════════════════════╗
║                        REASONING LAYER                               ║
║                                                                      ║
║  User Query                                                          ║
║      │                                                               ║
║      ▼                                                               ║
║  QueryRouter (LLM + structured Pydantic output)                      ║
║      RETRIEVE    → standard RAG                                      ║
║      REFORMULATE → rewrite vague query + RAG                        ║
║      DECOMPOSE   → parallel RAG + merge (asyncio.gather)            ║
║      ESCALATE    → out of scope, no hallucination                   ║
║      │                                                               ║
║      ▼                                                               ║
║  RAGPipeline                                                         ║
║      Voyage AI embeddings (voyage-3-lite, 1024 dims)                ║
║      MongoDB Atlas Vector Search (cosine, source pre-filter)        ║
║      OpenRouter LLM generation (temperature=0.0 for determinism)    ║
║      │                                                               ║
║      ▼                                                               ║
║  Evaluation + Drift Detection                                        ║
║      RAGAS faithfulness (LLM-as-judge pattern)                      ║
║      EvalStatus: PASS / FAIL / SKIP (nan → SKIP, not FAIL)         ║
║      MongoDB: persist eval results for drift analysis               ║
║      DriftMonitor: 24h rolling avg vs 7-day baseline               ║
║      Alert: > 15% relative drop                                     ║
╚══════════════════════════════════════════════════════════════════════╝
                                │
╔══════════════════════════════════════════════════════════════════════╗
║                    PLATFORM PRIMITIVES LAYER                         ║
║                                                                      ║
║  ExtractionPrimitive   → document → structured fields               ║
║                           per-field confidence + SourceCitation     ║
║                           (document_id, section_path, page_number)  ║
║                                                                      ║
║  ClassificationPrimitive → content → category                       ║
║                             AUTONOMOUS / FLAG_REVIEW / REQUIRE_HUMAN ║
║                                                                      ║
║  VerificationPrimitive → ExtractionResult → PASS / FAIL / REVIEW   ║
║                           field-level verdicts, worst-case propagation║
╚══════════════════════════════════════════════════════════════════════╝
                                │
╔══════════════════════════════════════════════════════════════════════╗
║                       OBSERVABILITY LAYER                            ║
║                                                                      ║
║  OpenTelemetry → Dash0 (OTLP HTTP, EU West)                        ║
║  5 step-level spans:                                                 ║
║      rag.query (root) → agent.route → agent.router                  ║
║                       → rag.retrieval → rag.generation              ║
║  Attributes: query, session_id, router.action, latency_ms per step  ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## What This System Solves

### The Staleness Problem (Perception Layer)

The most underappreciated failure mode in production RAG: **context goes stale**.

Documentation pages return 404. Content changes but embedded chunks are still the old version. New pages are added to the docs site but never ingested. The system keeps answering confidently from outdated context.

In a compliance or financial context, this isn't quality degradation — it's a liability.

**The solution:** a scheduled Perception layer that runs URL health checks via content hash comparison, stages changed content through an LLM-as-judge evaluation gate, and creates GitHub PRs with full evaluation evidence before anything touches the live corpus. Nothing goes live without human approval.

### The Trust Problem (Reasoning Layer)

AI answers are only as trustworthy as their grounding. The `QueryRouter` agent routes queries before retrieval — so vague queries get reformulated, multi-part questions get decomposed into parallel retrievals, and out-of-scope queries escalate gracefully rather than hallucinating.

Every answer includes source citations. Every evaluation result is persisted to MongoDB. Drift is monitored continuously.

### The Platform Problem (Primitives Layer)

Product teams shouldn't be writing custom LangChain code or managing their own embedding models. The three primitives abstract that complexity:

- **ExtractionPrimitive** — document → structured fields with per-field confidence and `SourceCitation` (document_id, section_path, page_number) for full audit traceability
- **ClassificationPrimitive** — structured input → category with `ReviewDecision` (AUTONOMOUS / FLAG_REVIEW / REQUIRE_HUMAN)
- **VerificationPrimitive** — extraction result → field-level PASS / FAIL / REVIEW, worst-case decision propagation

Confidence thresholds are constructor parameters — product decisions, not engineering constants.

---

## Architectural Decisions

### Why agentic routing before retrieval?

Naive RAG embeds the user's query and runs vector search. The problem: retrieval quality depends entirely on query clarity.

- `"How does it work?"` — too vague to retrieve anything useful
- `"How do I set up auth and also configure webhooks?"` — needs two separate retrievals

The `QueryRouter` uses structured LLM output (JSON → Pydantic) to classify queries into four actions before touching the vector store. If the LLM returns malformed JSON, the router falls back to RETRIEVE rather than crashing. **Fail-safe, not fail-fast.**

Source filtering (routing Dash0 queries to the Dash0 corpus only) runs as a MongoDB pre-filter before ANN search — improving both precision and latency.

### Why semantic chunking?

We tried naive fixed-size chunking first. It failed for technical documentation.

Fixed-size chunking splits on token count. For docs with code examples and structured headers, it produces chunks like `"run the following command:"` with no command — retrieved as top results, useless in context.

Semantic chunking splits on Markdown structure: headers first, then paragraphs, treating code blocks as atomic units.

**Benchmark (20 queries, all three corpora):**
| Strategy | Avg top-1 similarity | Irrelevant retrieval rate |
|---|---|---|
| Naive | 0.71 | 34% |
| Semantic | 0.87 | 20% |

### Why content hashing for corpus freshness?

Timestamps lie. CDN caching means a page's `Last-Modified` header can be hours or days behind its actual content. We compute SHA-256 of the normalised parsed Markdown content — cosmetic HTML changes (whitespace, attribute ordering) don't trigger false positives. Content changes do.

### Why LLM-as-judge before the PR?

The eval gate runs before anything becomes a PR candidate:

1. **Quality** — is this content informative and well-structured? Low-density marketing copy fails.
2. **Relevance** — is this content relevant to the corpus's purpose? Off-topic pages fail.
3. **Regression** — does this contradict or duplicate existing content? Conflicting information fails.

Each dimension is scored 0.0–1.0. The overall verdict is determined by the minimum score. APPROVE → PR. REJECT → logged, not surfaced. REVIEW → SKIP (human decides). **Infrastructure failure is SKIP, not FAIL** — same principle as RAGAS nan handling.

### Why RAGAS faithfulness only?

Three metrics, one used in production:

| Metric | What it detects | Why excluded/included |
|---|---|---|
| `faithfulness` ✅ | Hallucination — answer not grounded in context | LLM-as-judge only; works with any LLM |
| `answer_relevancy` ❌ | Answer doesn't address the question | Requires `embed_query` — unavailable via OpenRouter |
| `context_precision` ❌ | Wrong chunks retrieved | Requires ground-truth reference answers |

Shipping one reliable metric is more valuable than shipping three metrics where two silently return garbage. In production, `answer_relevancy` would use a dedicated Voyage AI embeddings model.

### Why relative drift threshold (15%)?

A system at 0.60 faithfulness dropping to 0.51 is more concerning than one at 0.90 dropping to 0.81, even though both drop by 0.09.

Relative change catches meaningful degradation at any quality level. Absolute thresholds miss degradation in already-mediocre systems and over-alert in high-performing ones.

### Why SourceCitation carries page_number?

For web-sourced documentation (current corpus), `page_number` is `None` — web pages have no pages.

For PDF-based financial documents (Reap's compliance docs, board resolutions, KYB documents), `page_number` is set at ingestion from the PDF parser. The citation flows through retrieval all the way to the API response — turning AI generation into an **auditable compliance log** where every extracted field can be traced to its source clause and page.

The metadata contract at ingestion time determines what's available downstream. That's why chunk metadata design is a product decision, not just an engineering one.

### Why SKIP not FAIL for infrastructure failures?

`EvalStatus.SKIP` means the evaluation infrastructure couldn't complete — it is not a quality signal.

`EvalStatus.FAIL` means the content failed quality evaluation.

Collapsing these produces misleading quality metrics. If you're monitoring faithfulness pass rate and evaluation infrastructure starts failing (LLM judge timeout, malformed response), your pass rate drops — not because quality degraded, but because evals aren't completing. You chase a ghost.

---

## What Failed and What We Changed

### Naive chunking (Attempt 1)
Retrieval quality was poor for code-heavy documentation. Chunks like `"run the following command:"` (with no command) were being returned as top results. **Fix:** semantic chunking on Markdown headers, code blocks as atomic units.

### No source filtering (Attempt 2)
MongoDB questions were returning Dash0 results — both mention "observability" and "performance." Precision was low. **Fix:** three separate corpora with MongoDB Atlas pre-filter. Router infers source from query context.

### Synchronous DECOMPOSE (Attempt 3)
A 3-part question took 3x as long as a single question. **Fix:** `asyncio.gather` for parallel sub-query retrieval. Now takes ~same time as a single query.

### Vector index dimension mismatch (Attempt 4)
`voyage-3-lite` produces 1024-dimensional embeddings. Our initial Atlas index was configured for 512 dims — every insertion silently failed validation. **Fix:** recreated index with `numDimensions: 1024`.

### OpenRouter free model deprecations (Attempts 5–7)
`kimi-k2.6:free` deprecated. `llama-3.1-8b-instruct:free` deprecated. **Fix:** use `openrouter/free` permanent router slug that auto-selects from available free models.

### RAGAS nan on long answers (Attempt 8)
RAGAS LLM judge was hitting token limits on long answers and returning `nan`. We were treating `nan` as `FAIL`. **Fix:** `math.isnan()` guard → `EvalStatus.SKIP`. Evaluation failure is not a quality signal.

### Pydantic rejecting OPENAI_API_KEY (Attempt 9)
RAGAS reads `OPENAI_API_KEY` directly from the environment. Pydantic Settings was rejecting it as an unknown field. **Fix:** `extra="ignore"` in `model_config`.

Full failure log: `docs/BUILD_LOG.md`

---

## What We'd Build Next

**Scheduled perception runs** — currently the `CorpusHealthMonitor` is invoked manually. The right production deployment is a Cloud Scheduler job (daily at 02:00 UTC). The architecture is ready; the scheduler is the last wire to connect.

**Sitemap crawling** — the current Perception layer checks a predefined URL catalogue. True discovery (sitemap parsing, nav link extraction) would catch entirely new documentation pages that were never in the catalogue.

**PR → live promotion automation** — currently, a human merges the PR and the chunks stay in the pending collection. A GitHub Action on merge would read the machine-readable JSON metadata from the PR body and promote the staged chunks to the live collection automatically.

**Re-ranking with a cross-encoder** — current retrieval uses cosine similarity (fast). A cross-encoder re-ranks the top-20 with a more expensive model for the top-5. Would improve precision at the cost of ~300ms latency.

**Fine-tuned embedding model** — `voyage-3-lite` is general-purpose. Fine-tuning on technical documentation (question-answer pairs from MongoDB/Dash0/Reap) would improve domain-specific retrieval. Requires labelled data we don't have yet.

---

## Setup

```bash
# Prerequisites
# MongoDB Atlas cluster with Vector Search index (see index definition below)
# OpenRouter API key: https://openrouter.ai
# Voyage AI API key: https://www.voyageai.com
# Dash0 account (optional, for OTel traces): https://www.dash0.com
# GitHub PAT (optional, for PR drafting): github.com/settings/tokens

git clone https://github.com/cobih/ai-builder
cd ai-builder
pip install -e ".[dev]"

cp .env.example .env
# Edit .env with your credentials
```

**MongoDB Atlas Vector Search index definition:**
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "source"
    }
  ]
}
```

```bash
# Ingest all corpora
ai-builder ingest --source mongodb --source dash0 --source reap

# Query
ai-builder query "How does Atlas Vector Search handle approximate nearest neighbour search?"
ai-builder query "What is Dash0 Agent0?" --evaluate
ai-builder query "How do I authenticate with the Reap API?" --source reap

# Quality dashboard
ai-builder dashboard
```

---

## Tests

```bash
# Unit tests — no API keys required, all fast
pytest tests/unit/ -v
# 75 passing (22 core, 19 primitives, 37 perception)
# 3 router tests require live API keys (pre-existing)

# Integration tests — requires live MongoDB + API keys
pytest tests/integration/ -v -m integration
```

---

## Project Structure

```
ai-builder/
├── src/
│   ├── settings.py              # Pydantic BaseSettings — typed env config
│   ├── models.py                # Domain models — all pipeline contracts
│   ├── cli.py                   # Typer CLI — ingest | query | dashboard
│   ├── primitives.py            # Platform AI primitives (Extraction, Classification, Verification)
│   ├── agents/
│   │   ├── router.py            # QueryRouter — LLM + structured Pydantic output
│   │   └── pipeline.py          # RAGPipeline — route → retrieve → generate
│   ├── corpus/
│   │   └── loader.py            # Document ingestion — naive vs semantic chunking
│   ├── evaluation/
│   │   └── evaluator.py         # RAGAS evaluation + DriftMonitor
│   ├── observability/
│   │   └── telemetry.py         # OpenTelemetry — OTLP export to Dash0
│   └── perception/
│       ├── health_monitor.py    # CorpusHealthMonitor — URL health + content hashing
│       ├── staging_pipeline.py  # StagingPipeline — ingest + LLM-as-judge eval gate
│       └── pr_drafter.py        # PRDrafter — GitHub PRs with evaluation evidence
├── tests/
│   └── unit/
│       ├── test_core.py         # 22 tests — models, chunking, evaluation, drift
│       ├── test_primitives.py   # 19 tests — Extraction, Classification, Verification
│       └── test_perception.py   # 37 tests — health checks, eval gate, PR drafting
├── docs/
│   ├── BUILD_LOG.md             # 12 documented failures with fixes
│   └── chunking_benchmark.md   # Naive vs semantic benchmark results
├── .env.example
└── pyproject.toml               # Modern Python packaging with hatchling
```

---

## Modern Python Patterns

| Pattern | Location | Why |
|---|---|---|
| `async/await` throughout | `pipeline.py`, `evaluator.py`, `perception/` | IO-bound; parallel sub-queries and health checks |
| `TYPE_CHECKING` lazy imports | `primitives.py`, `perception/` | Avoids pulling LangChain at module load time in tests |
| Pydantic v2 `BaseModel` | `models.py`, `primitives.py` | Runtime validation at every boundary |
| `StrEnum` | `models.py`, `primitives.py`, `perception/` | Serialises cleanly to MongoDB and JSON |
| `BaseSettings` | `settings.py` | Environment-first; fails fast with clear errors |
| `@field_validator` | `models.py` | Co-located validation logic |
| Structured LLM output | `router.py`, `staging_pipeline.py` | JSON → Pydantic; type-safe decisions |
| Lazy singleton init | `pipeline.py` | One connection pool per process, not per request |
| `asyncio.gather` | `pipeline.py`, `health_monitor.py` | Parallel retrieval and concurrent URL checks |
| `asyncio.Semaphore` | `health_monitor.py` | Bounded concurrency for URL health checks |
| `tenacity` retry | `loader.py` | Resilient HTTP with exponential backoff |
| Content hash comparison | `health_monitor.py` | Normalised SHA-256 — cosmetic changes don't trigger false positives |
| SKIP vs FAIL distinction | `evaluator.py`, `primitives.py`, `staging_pipeline.py` | Infrastructure failure ≠ quality failure |


