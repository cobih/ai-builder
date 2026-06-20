"""
Agentic RAG pipeline — router → retrieve → generate.

The pipeline handles all four router actions:

RETRIEVE:    standard RAG path
REFORMULATE: rewrite query, then standard RAG
DECOMPOSE:   run parallel RAG chains, merge answers
ESCALATE:    return a polite "out of scope" message

Performance design decisions:

1. DECOMPOSE runs sub-queries in parallel (asyncio.gather)
   A 3-part question takes ~same time as 1 question, not 3x.

2. Connection pooling via lazy singleton
   One MongoDB client, reused across requests.
   Rebuilding the client per request would be ~200ms overhead.

3. temperature=0.0 for generation
   Deterministic outputs make RAGAS evaluation consistent.
   The same query should score the same on two evaluation runs.
   Non-zero temperature adds variance that masks real quality changes.
"""

import asyncio
import time

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from pymongo import MongoClient
from pymongo.collection import Collection

from src.agents.router import QueryRouter
from src.models import (
    CorpusSource,
    QueryRequest,
    RAGResponse,
    RetrievedContext,
    RouterAction,
    RouterDecision,
)
from src.observability.telemetry import record_rag_metrics, traced_span
from src.settings import settings

RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful technical assistant with expertise in MongoDB, Dash0, and Reap.

Answer the question using ONLY the provided context. If the context doesn't contain
enough information to answer fully, say so clearly and explain what's missing.

Do not use knowledge outside the provided context.
Always cite which product or documentation section your answer comes from.

Context:
{context}

Question: {question}

Answer:""")

MERGE_PROMPT = ChatPromptTemplate.from_template("""You are synthesising answers to a multi-part question.

Original question: {original_question}

Partial answers to sub-questions:
{partial_answers}

Synthesise these into a single coherent, well-structured answer.
Preserve all technical details. Remove redundancy.""")


class RAGPipeline:
    """
    Agentic RAG pipeline with MongoDB Atlas Vector Search.

    Handles routing, retrieval, generation, and streaming.
    All components are lazily initialised and reused across requests.
    """

    def __init__(self) -> None:
        self._client: MongoClient | None = None
        self._collection: Collection | None = None
        self._vector_store: MongoDBAtlasVectorSearch | None = None
        self._embeddings: VoyageAIEmbeddings | None = None
        self._llm: ChatOpenAI | None = None
        self._router = QueryRouter()

    # ── Lazy initialisers ─────────────────────────────────────────────

    def _get_embeddings(self) -> VoyageAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = VoyageAIEmbeddings(
                voyage_api_key=settings.voyage_api_key,
                model=settings.voyage_embed_model,
            )
        return self._embeddings

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                model=settings.openrouter_model,
                temperature=0.0,
            )
        return self._llm

    def _get_collection(self) -> Collection:
        if self._collection is None:
            if self._client is None:
                self._client = MongoClient(
                    settings.mongodb_uri,
                    maxPoolSize=10,
                    serverSelectionTimeoutMS=5000,
                )
            self._collection = self._client[settings.mongodb_database][
                settings.mongodb_collection
            ]
        return self._collection

    def _get_vector_store(self) -> MongoDBAtlasVectorSearch:
        if self._vector_store is None:
            self._vector_store = MongoDBAtlasVectorSearch(
                collection=self._get_collection(),
                embedding=self._get_embeddings(),
                index_name=settings.mongodb_index_name,
                text_key="content",
                embedding_key="embedding",
            )
        return self._vector_store

    # ── Ingest ────────────────────────────────────────────────────────

    async def ingest_documents(
        self,
        texts: list[str],
        metadatas: list[dict],  # type: ignore[type-arg]
    ) -> int:
        """Embed and store documents. Returns chunk count."""
        from langchain_core.documents import Document as LCDoc

        with traced_span("rag.ingest", {"document_count": len(texts)}):
            docs = [
                LCDoc(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadatas)
            ]
            vs = self._get_vector_store()
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: vs.add_documents(docs)
            )
            return len(docs)

    # ── Retrieval ─────────────────────────────────────────────────────

    async def _retrieve(
        self,
        query: str,
        top_k: int,
        source_filter: str = "all",
    ) -> list[RetrievedContext]:
        """
        Retrieve relevant chunks from MongoDB Atlas Vector Search.

        When source_filter is set, we add a pre-filter to the vector search
        so only chunks from the relevant corpus are considered.
        This is a MongoDB Atlas feature — pre-filtering happens before
        ANN search, reducing both latency and irrelevant results.
        """
        with traced_span(
            "rag.retrieval",
            {"query": query, "top_k": top_k, "source_filter": source_filter},
        ) as span:
            vs = self._get_vector_store()

            # Build pre-filter for corpus-specific search
            search_kwargs: dict = {"k": top_k}
            if source_filter != "all":
                search_kwargs["pre_filter"] = {
                    "source": {"$eq": source_filter}
                }

            raw = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: vs.similarity_search_with_score(
                    query, **search_kwargs
                ),
            )

            span.set_attribute("retrieval.results_count", len(raw))

            return [
                RetrievedContext(
                    document_id=str(i),
                    content=doc.page_content,
                    source=CorpusSource(
                        doc.metadata.get("source", "mongodb")
                    ),
                    url=doc.metadata.get("url", ""),
                    title=doc.metadata.get("section", ""),
                    similarity_score=float(score),
                )
                for i, (doc, score) in enumerate(raw)
            ]

    # ── Generation ────────────────────────────────────────────────────

    async def _generate(
        self,
        query: str,
        contexts: list[RetrievedContext],
    ) -> str:
        """Generate an answer grounded in retrieved context."""
        with traced_span(
            "rag.generation",
            {"model": settings.openrouter_model, "context_count": len(contexts)},
        ):
            context_text = "\n\n---\n\n".join(
                f"[{c.source.value.upper()} — {c.title or c.url}]\n{c.content}"
                for c in contexts
            )

            chain = (
                {
                    "context": lambda _: context_text,
                    "question": RunnablePassthrough(),
                }
                | RAG_PROMPT
                | self._get_llm()
                | StrOutputParser()
            )

            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: chain.invoke(query)
            )

    # ── Main query entrypoint ─────────────────────────────────────────

    async def query(self, request: QueryRequest) -> RAGResponse:
        """
        Route, retrieve, and generate — with full telemetry.

        Handles all four router actions and returns a structured response
        with timing breakdowns for observability.
        """
        top_k = request.top_k or settings.retrieval_top_k
        total_start = time.perf_counter()

        with traced_span(
            "rag.query",
            {"query": request.query, "session_id": str(request.session_id)},
        ) as root_span:

            # Step 1: Route
            with traced_span("agent.route"):
                decision = await self._router.route(request.query)

            root_span.set_attribute("router.action", decision.action.value)

            # Handle ESCALATE immediately
            if decision.action == RouterAction.ESCALATE:
                return RAGResponse(
                    query=request.query,
                    answer=(
                        f"I can only answer questions about MongoDB, Dash0, and Reap. "
                        f"{decision.escalation_reason or 'This question appears to be out of scope.'}"
                    ),
                    contexts=[],
                    session_id=request.session_id,
                    router_decision=decision,
                    model=settings.openrouter_model,
                    latency_ms=0.0,
                    retrieval_latency_ms=0.0,
                    generation_latency_ms=0.0,
                )

            # Determine effective query and source filter
            effective_query = (
                decision.reformulated_query
                if decision.action == RouterAction.REFORMULATE
                and decision.reformulated_query
                else request.query
            )
            source_filter = (
                request.source_filter.value
                if request.source_filter
                else decision.source_filter
            )

            # Step 2: Retrieve
            retrieval_start = time.perf_counter()

            if decision.action == RouterAction.DECOMPOSE and decision.sub_queries:
                # Parallel retrieval for sub-queries
                all_contexts = await asyncio.gather(*[
                    self._retrieve(sq, top_k, source_filter)
                    for sq in decision.sub_queries
                ])
                # Deduplicate by document_id
                seen: set[str] = set()
                contexts: list[RetrievedContext] = []
                for batch in all_contexts:
                    for ctx in batch:
                        if ctx.document_id not in seen:
                            seen.add(ctx.document_id)
                            contexts.append(ctx)
            else:
                contexts = await self._retrieve(
                    effective_query, top_k, source_filter
                )

            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

            # Step 3: Generate
            generation_start = time.perf_counter()

            if decision.action == RouterAction.DECOMPOSE and decision.sub_queries:
                # Generate answers for each sub-query, then merge
                partial = await asyncio.gather(*[
                    self._generate(sq, contexts)
                    for sq in decision.sub_queries
                ])
                partial_text = "\n\n".join(
                    f"Sub-question: {sq}\nAnswer: {ans}"
                    for sq, ans in zip(decision.sub_queries, partial)
                )
                merge_chain = (
                    MERGE_PROMPT | self._get_llm() | StrOutputParser()
                )
                answer = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: merge_chain.invoke({
                        "original_question": request.query,
                        "partial_answers": partial_text,
                    }),
                )
            else:
                answer = await self._generate(effective_query, contexts)

            generation_latency_ms = (time.perf_counter() - generation_start) * 1000
            total_latency_ms = (time.perf_counter() - total_start) * 1000

            corpus_sources = [c.source.value for c in contexts]
            record_rag_metrics(
                root_span,
                total_latency_ms=total_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                context_count=len(contexts),
                answer_length=len(answer),
                corpus_sources=list(set(corpus_sources)),
            )

            return RAGResponse(
                query=request.query,
                answer=answer,
                contexts=contexts,
                session_id=request.session_id,
                router_decision=decision,
                model=settings.openrouter_model,
                latency_ms=total_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
            )

    async def stream_query(self, request: QueryRequest):  # type: ignore[return]
        """
        Streaming variant — yields tokens as generated.

        Why streaming matters: perceived latency < actual latency.
        First token at 300ms feels faster than complete answer at 3s.
        """
        decision = await self._router.route(request.query)

        if decision.action == RouterAction.ESCALATE:
            yield "I can only answer questions about MongoDB, Dash0, and Reap."
            return

        query = decision.reformulated_query or request.query
        source_filter = decision.source_filter
        contexts = await self._retrieve(
            query,
            request.top_k or settings.retrieval_top_k,
            source_filter,
        )

        context_text = "\n\n---\n\n".join(
            f"[{c.source.value.upper()}]\n{c.content}" for c in contexts
        )

        chain = (
            {"context": lambda _: context_text, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self._get_llm()
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
