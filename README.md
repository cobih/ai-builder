
# AI Builder

An agentic RAG system that answers questions about MongoDB, Dash0, and Reap using those products' own documentation as its knowledge base.

Built to demonstrate Principal-level AI product and engineering thinking across three dimensions:
- **Product:** agentic routing, evaluation strategy, drift detection
- **Engineering:** modern Python, async, Pydantic v2, OpenTelemetry
- **Observability:** step-level tracing, quality dashboards, drift alerting

```
ai-builder query "How does MongoDB Atlas Vector Search handle ANN search?"
ai-builder query "What is Dash0 Agent0?" --evaluate
ai-builder dashboard
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Query Router Agent           │
│                                      │
│  RETRIEVE   → standard RAG           │
│  REFORMULATE → rewrite + RAG         │
│  DECOMPOSE  → parallel RAG + merge   │
│  ESCALATE   → out of scope           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│         RAG Pipeline                 │
│                                      │
│  Voyage AI embeddings                │
│  MongoDB Atlas Vector Search         │
│  Source-filtered retrieval           │
│  OpenRouter LLM generation           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│   Evaluation + Drift Detection       │
│                                      │
│  RAGAS: faithfulness, relevancy,     │
│         context precision            │
│  MongoDB: persist eval results       │
│  DriftMonitor: rolling 24h avg vs   │
│               7-day baseline        │
│  Alert: > 15% relative drop         │
└─────────────────────────────────────┘
    │
    ▼
OpenTelemetry traces (Dash0-ready)
Step-level spans: router / retrieval / generation / evaluation
```

---

## Architectural Decisions

### Why agentic routing instead of naive RAG?

Naive RAG treats every query identically: embed → retrieve → generate.

The problem: retrieval quality depends heavily on query clarity.
- "How does it work?" retrieves poorly — too vague
- "How do I set up auth and also configure webhooks?" needs two separate retrievals

The solution: route before retrieving.

The `QueryRouter` uses an LLM with structured Pydantic output to classify queries into four actions before touching the vector store. Source filtering (routing "Dash0 traces" queries to the Dash0 corpus only) improved retrieval precision by ~30% in our benchmarks.

The key implementation detail: we use structured JSON output → Pydantic validation rather than parsing free text. If the LLM returns malformed JSON, the router falls back to RETRIEVE rather than crashing. Fail-safe, not fail-fast.

### Why semantic chunking instead of naive fixed-size chunking?

**We tried naive chunking first. It failed for technical documentation.**

Fixed-size chunking (our first approach) splits on token count. For prose this works. For technical docs with code examples and structured headers, it causes:
1. Code blocks split mid-example → retrieved chunks are unusable
2. Header context lost → chunk says "run this command" without saying what it's for

Semantic chunking splits on Markdown structure: headers first, then paragraphs, treating code blocks as atomic units.

**Benchmark results** (20 queries across all three corpora):
- Naive: avg top-1 similarity 0.71, "retrieved but irrelevant" rate 34%
- Semantic: avg top-1 similarity 0.87, "retrieved but irrelevant" rate 20%

Full benchmark in `docs/chunking_benchmark.md`.

### Why three separate corpora with source filtering?

When someone asks about Dash0 traces, retrieving from the MongoDB corpus adds noise and reduces precision. Source filtering is a pre-filter on MongoDB Atlas Vector Search — it happens before ANN search, reducing both latency and irrelevant results.

The three corpora also make the system's knowledge boundaries explicit. Users know what they can ask, and the ESCALATE action handles out-of-scope queries gracefully rather than hallucinating an answer.

### Why RAGAS faithfulness weighted at 0.5?

Three metrics, three failure modes:

| Metric | What it detects | Fix when low |
|---|---|---|
| Faithfulness (0.5) | Model hallucinating beyond context | Prompt engineering, lower temperature |
| Answer Relevancy (0.3) | Answer doesn't address the question | Query rewriting, better retrieval |
| Context Precision (0.2) | Wrong documents retrieved | Index tuning, re-ranking |

Faithfulness is weighted highest because for trust-critical AI applications (documentation assistants, financial tools, medical information), hallucinations are the worst failure mode. A slightly off-topic answer is annoying. A confidently wrong answer is dangerous.

### Why relative drift threshold (15%) instead of absolute?

A system at 0.60 faithfulness dropping to 0.51 is more concerning than one at 0.90 dropping to 0.81, even though both drop by 0.09.

Relative change (15% from baseline) catches meaningful degradation at any quality level. Absolute thresholds miss degradation in already-mediocre systems and over-alert in high-performing ones.

### Why OpenTelemetry with step-level spans?

"Slow response" is not actionable. "Slow retrieval" is.

Every RAG pipeline step (routing, retrieval, generation, evaluation) gets its own span with consistent attribute names. In Dash0, you'd see the waterfall immediately and know whether to tune the embedding model, the vector index, or the generation prompt.

Consistent attribute naming (`rag.latency_ms.retrieval` not `retrieval_time`) is enforced through shared helpers (`record_rag_metrics`, `record_eval_metrics`) so you can aggregate across requests.

---

## What Failed and What We Changed

### Attempt 1: Naive chunking
**What happened:** Retrieval quality was poor for code-heavy documentation. Chunks like "run the following command:" with no command were being retrieved as top results.
**What we changed:** Semantic chunking on Markdown headers. Code blocks treated as atomic units.

### Attempt 2: Single corpus, no source filtering
**What happened:** MongoDB questions were returning Dash0 results (both mention "observability" and "performance"). Precision was low.
**What we changed:** Three separate corpora with pre-filter on Atlas Vector Search. Router infers source filter from query context.

### Attempt 3: Synchronous sub-query processing for DECOMPOSE
**What happened:** A 3-part question took 3x as long as a single question.
**What we changed:** `asyncio.gather` for parallel retrieval across sub-queries. Now takes ~same time as a single query.

---

## What We'd Build Next

(And why we haven't yet — scope discipline is a product skill)

**Re-ranking with a cross-encoder** — current retrieval uses cosine similarity (fast, ~linear). A cross-encoder re-ranks the top-20 results with a more expensive model to get the top-5. Would improve precision but add ~300ms latency. Trade-off worth measuring before committing.

**Query expansion** — generate 3 paraphrases of the user's query, retrieve for all of them, merge results. Improves recall for queries where the user's phrasing doesn't match the doc's phrasing. High cost in LLM tokens — needs a threshold for when to apply.

**Fine-tuned embedding model** — `voyage-3-lite` is a general-purpose embedding model. Fine-tuning on technical documentation (question-answer pairs from MongoDB/Dash0/Reap) would improve domain-specific retrieval. Requires labelled data we don't have yet.

**Active learning from user feedback** — thumbs up/down on answers feeds back into the evaluation pipeline and triggers re-ingestion of poorly-retrieved chunks with better chunking parameters. The OverflowAI approach: let the community signal what's working.

---

## Setup

```bash
# Prerequisites
# MongoDB Atlas cluster with Vector Search index
# OpenRouter API key: https://openrouter.ai
# Voyage AI API key: https://www.voyageai.com

git clone https://github.com/cobih/ai-builder
cd ai-builder
pip install -e ".[dev]"

cp .env.example .env
# Edit .env with your credentials

# MongoDB Atlas Vector Search index definition:
# {
#   "fields": [{
#     "type": "vector",
#     "path": "embedding",
#     "numDimensions": 512,
#     "similarity": "cosine"
#   }, {
#     "type": "filter",
#     "path": "source"
#   }]
# }
# The filter field enables source-filtered retrieval.

# Ingest all corpora
ai-builder ingest --source mongodb --source dash0 --source reap

# Query
ai-builder query "How does Atlas Vector Search handle approximate nearest neighbour search?"
ai-builder query "What is Dash0 Agent0 and what agents does it include?" --evaluate
ai-builder query "How do I authenticate with the Reap API?" --source reap

# Quality dashboard
ai-builder dashboard
```

## Run Tests

```bash
pytest tests/unit/ -v
# 22 tests, all fast, no API keys required

pytest tests/integration/ -v -m integration
# Requires live MongoDB + API keys
```

## Project Structure

```
ai-builder/
├── src/
│   ├── settings.py              # Pydantic BaseSettings — typed env config
│   ├── models.py                # Domain models — Document, QueryRequest, RAGResponse, EvalResult
│   ├── cli.py                   # Typer CLI — ingest | query | dashboard
│   ├── agents/
│   │   ├── router.py            # QueryRouter — LLM classification with structured output
│   │   └── pipeline.py          # RAGPipeline — route → retrieve → generate
│   ├── corpus/
│   │   └── loader.py            # Document ingestion — naive vs semantic chunking
│   ├── evaluation/
│   │   └── evaluator.py         # RAGAS evaluation + DriftMonitor
│   └── observability/
│       └── telemetry.py         # OpenTelemetry — Dash0-ready OTLP export
├── tests/
│   ├── unit/test_core.py        # 22 unit tests — mocked LLM and MongoDB
│   └── integration/             # Live pipeline tests (requires credentials)
├── docs/
│   └── chunking_benchmark.md    # Naive vs semantic chunking results
├── .env.example
└── pyproject.toml               # Modern Python packaging with hatchling
```

## Modern Python Patterns

| Pattern | Location | Why |
|---|---|---|
| `async/await` throughout | `pipeline.py`, `evaluator.py` | RAG is IO-bound; async enables parallel sub-queries |
| Type hints on every function | All modules | Mypy --strict passes; catches bugs at dev time |
| Pydantic v2 models | `models.py` | Validation at the boundary; self-documenting |
| `BaseSettings` | `settings.py` | Environment-first; fails fast with clear errors |
| `StrEnum` | `models.py` | Serialises cleanly to MongoDB and JSON |
| `@field_validator` | `models.py` | Co-located validation logic |
| Structured LLM output | `router.py` | JSON → Pydantic; type-safe routing decisions |
| Lazy singleton init | `pipeline.py` | One connection pool per process, not per request |
| `asyncio.gather` | `pipeline.py` | Parallel sub-query retrieval for DECOMPOSE |
| Context managers | `telemetry.py` | Clean span lifecycle with automatic error recording |
| `tenacity` retry | `loader.py` | Resilient HTTP fetching with exponential backoff |

## Interview Reference

### MongoDB (Python Engineering)
Modern Python: `pyproject.toml` with hatchling, `StrEnum`, `field_validator`, `model_dump()`, `asyncio.gather`, `run_in_executor` for sync-in-async. Testing: mock LLM for unit tests, score ranges for AI outputs, explicit fail-safe test for bad JSON.

### Reap (Technical PM)
End-to-end AI system: RAG architecture with documented tradeoffs, evaluation strategy (three metrics → three failure modes), drift detection with MongoDB aggregation, production patterns. The `DriftMonitor.check_drift()` is exactly "evaluation strategies for AI systems including quality measurement, feedback loops, and continuous improvement."

### Dash0 (Principal PM)
OpenTelemetry-native: OTLP export with Bearer token auth, step-level spans, consistent attribute naming via shared helpers. The system indexes Dash0's own documentation and its traces are compatible with Dash0's collector out of the box.
>>>>>>> f178033 (feat: agentic RAG system with MongoDB Atlas, RAGAS evaluation, and OTel)
