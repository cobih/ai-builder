"""
Query Router Agent — the core of the agentic RAG system.

This is what separates a naive RAG pipeline from an agentic one.

NAIVE RAG:  query → embed → retrieve → generate
AGENTIC RAG: query → DECIDE → (route) → embed → retrieve → generate

The router classifies every incoming query and decides:
- RETRIEVE:     standard RAG — query is clear and in-scope
- REFORMULATE:  query is ambiguous — rewrite it for better retrieval
- DECOMPOSE:    multi-part question — split into parallel sub-queries
- ESCALATE:     out of scope — tell the user honestly

Why does this matter at Principal level?

The routing decision is a product decision, not an engineering one.
What counts as "ambiguous"? What's "out of scope"? What threshold
triggers decomposition? These are quality judgements that PMs own.

The source_filter field is particularly important:
When someone asks "how does Dash0 handle traces", routing to the
Dash0 corpus only — not all three corpora — dramatically improves
retrieval precision. This is the naive vs thoughtful distinction
that separates good RAG from great RAG.
"""

import json

from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.models import RouterAction, RouterDecision
from src.observability.telemetry import record_router_metrics, traced_span
from src.settings import settings

# System prompt for the router LLM
# Design decision: we give the router a clear taxonomy and examples
# rather than asking it to infer the rules. Explicit > implicit for
# classification tasks where consistency matters.
ROUTER_SYSTEM_PROMPT = """You are a query router for a RAG system that answers questions about:
- MongoDB (Atlas Vector Search, developer tools, AI integrations)
- Dash0 (OpenTelemetry-native observability, metrics, traces, logs, Agent0)
- Reap (payment processing platform, APIs, webhooks)

Your job is to classify each user query and return a JSON object.

CLASSIFICATION RULES:

RETRIEVE — use when:
  - The query is clear and specific
  - It's about one of the three products above
  - A direct vector search would likely find a good answer

REFORMULATE — use when:
  - The query is vague or uses unclear terminology
  - It's a follow-up that lacks context
  - Rephrasing would significantly improve retrieval
  - Include your rewrite in reformulated_query

DECOMPOSE — use when:
  - The query contains multiple distinct questions (joined by "and", "also", etc.)
  - Answering well requires retrieving from multiple topic areas
  - Include the sub-queries in sub_queries (2-4 items)

ESCALATE — use when:
  - The query is about something unrelated to MongoDB, Dash0, or Reap
  - The query asks for personal advice, opinions, or real-time data
  - Include the reason in escalation_reason

SOURCE FILTER RULES:
- "mongodb" — query mentions Atlas, vector search, MongoDB, PyMongo, Compass
- "dash0" — query mentions Dash0, Agent0, OTel/OpenTelemetry, traces, metrics, Prometheus
- "reap" — query mentions Reap, payments, cards, webhooks, transactions
- "all" — query spans multiple products or is ambiguous about which product

Return ONLY valid JSON matching this exact schema:
{
  "action": "retrieve" | "reformulate" | "decompose" | "escalate",
  "reasoning": "brief explanation of your decision",
  "source_filter": "mongodb" | "dash0" | "reap" | "all",
  "reformulated_query": null | "rewritten query string",
  "sub_queries": null | ["query 1", "query 2"],
  "escalation_reason": null | "reason string"
}"""


class QueryRouter:
    """
    LLM-powered query router with structured Pydantic output.

    Using structured output (JSON → Pydantic) rather than parsing
    free text means routing decisions are type-safe and testable.
    If the LLM returns malformed JSON, we catch it and fall back
    to RETRIEVE rather than crashing — fail safe, not fail fast.
    """

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=settings.openrouter_router_model,
            temperature=0.0,  # deterministic routing decisions
            max_tokens=512,   # routing response is short
        )

    async def route(self, query: str) -> RouterDecision:
        """
        Classify a query and return a routing decision.

        Falls back to RETRIEVE on any error — we'd rather give a
        potentially suboptimal answer than fail the request.
        """
        with traced_span(
            "agent.router",
            {"query": query, "model": settings.openrouter_router_model},
        ) as span:
            try:
                messages = [
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ]

                response = await self._llm.ainvoke(messages)
                content = response.content

                # Parse JSON → Pydantic
                data = json.loads(str(content))
                decision = RouterDecision(**data)

                record_router_metrics(
                    span,
                    action=decision.action.value,
                    source_filter=decision.source_filter,
                    was_reformulated=decision.action == RouterAction.REFORMULATE,
                    was_decomposed=decision.action == RouterAction.DECOMPOSE,
                    sub_query_count=len(decision.sub_queries or []),
                )

                return decision

            except (json.JSONDecodeError, ValidationError, Exception) as e:
                # Fail safe: default to retrieve
                span.set_attribute("router.fallback", True)
                span.set_attribute("router.error", str(e))

                return RouterDecision(
                    action=RouterAction.RETRIEVE,
                    reasoning=f"Router error — defaulting to RETRIEVE: {e}",
                    source_filter="all",
                )
