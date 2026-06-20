"""
CLI for the AI Builder RAG system.

Three commands:
  ingest    — fetch docs and load into MongoDB Atlas
  query     — ask a question with full agent routing
  dashboard — show current quality metrics and drift alerts
"""

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from src.models import CorpusSource, QueryRequest
from src.observability.telemetry import setup_telemetry

app = typer.Typer(
    name="ai-builder",
    help="Agentic RAG over MongoDB, Dash0, and Reap documentation.",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    sources: Annotated[
        list[str],
        typer.Option(
            "--source", "-s",
            help="Corpus to ingest: mongodb | dash0 | reap (repeat for multiple)",
        ),
    ] = ["mongodb", "dash0", "reap"],
    strategy: Annotated[
        str,
        typer.Option(help="Chunking strategy: semantic | naive"),
    ] = "semantic",
) -> None:
    """Fetch documentation and load into MongoDB Atlas Vector Search."""
    from src.corpus.loader import load_corpus
    from src.agents.pipeline import RAGPipeline
    from src.models import ChunkStrategy

    setup_telemetry()
    pipeline = RAGPipeline()

    source_enums = [CorpusSource(s) for s in sources]
    chunk_strategy = ChunkStrategy(strategy)

    async def run() -> None:
        texts, metadatas = [], []
        async for doc in load_corpus(sources=source_enums, strategy=chunk_strategy):
            texts.append(doc.content)
            metadatas.append({
                "source": doc.source.value,
                "url": doc.url,
                "title": doc.title,
                "strategy": doc.chunk_strategy.value,
                "chunk_index": doc.chunk_index,
                **doc.metadata,
            })
            if len(texts) >= 50:  # batch every 50 docs
                count = await pipeline.ingest_documents(texts, metadatas)
                rprint(f"[green]✓[/green] Stored {count} chunks")
                texts.clear()
                metadatas.clear()

        if texts:
            count = await pipeline.ingest_documents(texts, metadatas)
            rprint(f"[green]✓[/green] Stored {count} chunks")

        rprint("\n[bold green]Ingestion complete.[/bold green]")
        pipeline.close()

    asyncio.run(run())


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Your question")],
    stream: Annotated[bool, typer.Option(help="Stream the response")] = False,
    evaluate: Annotated[bool, typer.Option(help="Run RAGAS evaluation")] = False,
    source: Annotated[
        str | None,
        typer.Option(help="Filter corpus: mongodb | dash0 | reap"),
    ] = None,
) -> None:
    """Ask a question with full agentic routing."""
    from src.agents.pipeline import RAGPipeline
    from src.evaluation.evaluator import RAGEvaluator

    setup_telemetry()
    pipeline = RAGPipeline()

    async def run() -> None:
        source_filter = CorpusSource(source) if source else None
        request = QueryRequest(query=question, source_filter=source_filter)

        if stream:
            console.print("\n[bold]Answer:[/bold]")
            async for token in pipeline.stream_query(request):
                print(token, end="", flush=True)
            print()
        else:
            with console.status("Thinking..."):
                response = await pipeline.query(request)

            # Display routing decision
            action = response.router_decision.action.value.upper()
            action_color = {
                "RETRIEVE": "green",
                "REFORMULATE": "yellow",
                "DECOMPOSE": "blue",
                "ESCALATE": "red",
            }.get(action, "white")

            console.print(f"\n[bold]Router:[/bold] [{action_color}]{action}[/{action_color}]")
            console.print(f"[dim]Reasoning: {response.router_decision.reasoning}[/dim]")

            if response.router_decision.reformulated_query:
                console.print(f"[dim]Reformulated: {response.router_decision.reformulated_query}[/dim]")

            console.print(Panel(
                response.answer,
                title="[bold]Answer[/bold]",
                border_style="green",
            ))

            # Show sources
            if response.contexts:
                table = Table(title="Sources", show_header=True)
                table.add_column("Corpus", style="cyan")
                table.add_column("Score", style="green")
                table.add_column("Title/URL")
                for ctx in response.contexts[:3]:
                    table.add_row(
                        ctx.source.value,
                        f"{ctx.similarity_score:.3f}",
                        ctx.title or ctx.url[:60],
                    )
                console.print(table)

            console.print(
                f"\n[dim]Total: {response.latency_ms:.0f}ms | "
                f"Retrieval: {response.retrieval_latency_ms:.0f}ms | "
                f"Generation: {response.generation_latency_ms:.0f}ms[/dim]"
            )

            if evaluate:
                with console.status("Evaluating with RAGAS..."):
                    evaluator = RAGEvaluator()
                    result = await evaluator.evaluate_response(response, persist=True)
                    evaluator.close()

                status_color = "green" if result.status.value == "pass" else "red"
                console.print(f"\n[bold]Evaluation:[/bold]")
                console.print(f"  Faithfulness:      {result.faithfulness:.3f}")
                console.print(f"  Answer Relevancy:  {result.answer_relevancy:.3f}")
                console.print(f"  Context Precision: {result.context_precision:.3f}")
                console.print(f"  Overall Score:     {result.overall_score:.3f}")
                console.print(f"  Status: [{status_color}]{result.status.value.upper()}[/{status_color}]")

        pipeline.close()

    asyncio.run(run())


@app.command()
def dashboard() -> None:
    """Show quality metrics and active drift alerts."""
    from src.evaluation.evaluator import DriftMonitor

    monitor = DriftMonitor()
    dash = monitor.get_quality_dashboard()
    monitor.close()

    console.print(Panel(
        f"[bold]AI Builder Quality Dashboard[/bold]\n"
        f"Generated: {dash.generated_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Total evaluations (24h): {dash.total_evaluations}\n"
        f"Overall pass rate: {dash.overall_pass_rate:.1%}\n"
        f"Trend: {dash.trend}",
        border_style="blue",
    ))

    if dash.per_corpus:
        table = Table(title="Per-Corpus Quality (24h)", show_header=True)
        table.add_column("Corpus", style="cyan")
        table.add_column("Faithfulness", style="green")
        table.add_column("Relevancy", style="green")
        table.add_column("Precision", style="green")
        for corpus, stats in dash.per_corpus.items():
            table.add_row(
                corpus,
                f"{stats.get('faithfulness', 0):.3f}",
                f"{stats.get('answer_relevancy', 0):.3f}",
                f"{stats.get('context_precision', 0):.3f}",
            )
        console.print(table)

    if dash.active_alerts:
        console.print("\n[bold red]🚨 Active Drift Alerts:[/bold red]")
        for alert in dash.active_alerts:
            console.print(f"  [red]• {alert.message}[/red]")
    else:
        console.print("\n[green]✓ No drift alerts active[/green]")


if __name__ == "__main__":
    app()
