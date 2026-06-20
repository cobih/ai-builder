"""
RAGAS evaluation with MongoDB persistence and drift detection.

Three-layer evaluation architecture:

Layer 1: Per-response evaluation (RAGEvaluator)
  Run RAGAS after every query (or on a sample).
  Store results in MongoDB with timestamp + corpus source.

Layer 2: Drift detection (DriftMonitor)
  Query the eval_results collection to compute rolling averages.
  Compare current 24h window against 7-day baseline.
  Fire DriftAlert when any metric drops > 15%.

Layer 3: Quality dashboard (get_quality_dashboard)
  Aggregates state across all corpora for a single-screen view.
  This is what you'd display in Dash0 or Grafana.

Why store eval results in MongoDB?
  Same database as the vectors — one connection, one query surface.
  You can join eval scores against document metadata to see which
  chunks are being retrieved when quality drops.
  That's the debugging workflow: alert → dashboard → trace → chunk.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from datasets import Dataset
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from src.models import (
    DriftAlert,
    EvalResult,
    EvalStatus,
    QualityDashboard,
    RAGResponse,
)
from src.observability.telemetry import record_eval_metrics, traced_span
from src.settings import settings


class RAGEvaluator:
    """
    Evaluates RAG responses using RAGAS and persists results to MongoDB.
    """

    def __init__(self) -> None:
        self._metrics = [faithfulness, answer_relevancy, context_precision]
        self._client: MongoClient | None = None
        self._collection: Collection | None = None

    def _get_collection(self) -> Collection:
        if self._collection is None:
            if self._client is None:
                self._client = MongoClient(settings.mongodb_uri)
            db = self._client[settings.mongodb_database]
            self._collection = db[settings.mongodb_eval_collection]
            # Index for efficient time-range queries
            self._collection.create_index(
                [("evaluated_at", ASCENDING), ("corpus_source", ASCENDING)]
            )
        return self._collection

    async def evaluate_response(
        self,
        response: RAGResponse,
        persist: bool = True,
    ) -> EvalResult:
        """
        Run RAGAS on a single response and optionally persist to MongoDB.

        persist=True in production — we want a permanent quality record.
        persist=False in tests — no database needed for unit tests.
        """
        corpus_source = (
            response.corpus_sources[0].value
            if response.corpus_sources
            else "unknown"
        )

        with traced_span(
            "rag.evaluation",
            {
                "query": response.query,
                "session_id": str(response.session_id),
                "corpus_source": corpus_source,
            },
        ) as span:

            eval_data: dict[str, list[Any]] = {
                "question": [response.query],
                "answer": [response.answer],
                "contexts": [response.context_texts],
            }

            dataset = Dataset.from_dict(eval_data)

            result_df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: evaluate(dataset, metrics=self._metrics).to_pandas(),
            )

            faith = float(result_df["faithfulness"].iloc[0])
            relevancy = float(result_df["answer_relevancy"].iloc[0])
            precision = float(result_df["context_precision"].iloc[0])

            passed = (
                faith >= settings.min_faithfulness_score
                and relevancy >= settings.min_relevancy_score
            )
            status = EvalStatus.PASS if passed else EvalStatus.FAIL

            eval_result = EvalResult(
                session_id=response.session_id,
                query=response.query,
                answer=response.answer,
                corpus_source=corpus_source,
                faithfulness=faith,
                answer_relevancy=relevancy,
                context_precision=precision,
                status=status,
            )

            record_eval_metrics(
                span,
                faithfulness=faith,
                relevancy=relevancy,
                precision=precision,
                status=status.value,
                overall_score=eval_result.overall_score,
            )

            if persist:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._get_collection().insert_one(
                        eval_result.to_mongo()
                    ),
                )

            return eval_result

    async def evaluate_batch(
        self,
        responses: list[RAGResponse],
        persist: bool = True,
    ) -> list[EvalResult]:
        """Evaluate multiple responses in parallel."""
        return await asyncio.gather(*[
            self.evaluate_response(r, persist=persist) for r in responses
        ])

    def summarise(self, results: list[EvalResult]) -> dict[str, float]:
        if not results:
            return {}
        return {
            "pass_rate": sum(1 for r in results if r.status == EvalStatus.PASS) / len(results),
            "avg_faithfulness": sum(r.faithfulness for r in results) / len(results),
            "avg_relevancy": sum(r.answer_relevancy for r in results) / len(results),
            "avg_precision": sum(r.context_precision for r in results) / len(results),
            "avg_overall": sum(r.overall_score for r in results) / len(results),
            "total": float(len(results)),
        }

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None


class DriftMonitor:
    """
    Detects quality drift by comparing rolling window against baseline.

    Design: uses MongoDB aggregation pipeline to compute stats efficiently.
    No need to pull all records into Python — let the database do the math.

    Alert logic:
    - Compute 24h rolling average for each metric per corpus
    - Compute 7-day baseline for each metric per corpus
    - If rolling_avg < baseline * (1 - threshold): fire alert

    Why relative threshold (15%) not absolute?
    A system starting at 0.6 faithfulness dropping to 0.51 is more
    concerning than one at 0.95 dropping to 0.86, even though the
    absolute drops are similar. Relative change is the right signal.
    """

    def __init__(self) -> None:
        self._client: MongoClient | None = None
        self._collection: Collection | None = None

    def _get_collection(self) -> Collection:
        if self._collection is None:
            if self._client is None:
                self._client = MongoClient(settings.mongodb_uri)
            self._collection = self._client[settings.mongodb_database][
                settings.mongodb_eval_collection
            ]
        return self._collection

    def _compute_window_avg(
        self,
        corpus_source: str,
        hours: int,
    ) -> dict[str, float] | None:
        """Compute average metrics for a time window using MongoDB aggregation."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()

        pipeline = [
            {
                "$match": {
                    "corpus_source": corpus_source,
                    "evaluated_at": {"$gte": cutoff_str},
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_faithfulness": {"$avg": "$faithfulness"},
                    "avg_relevancy": {"$avg": "$answer_relevancy"},
                    "avg_precision": {"$avg": "$context_precision"},
                    "count": {"$sum": 1},
                }
            },
        ]

        result = list(self._get_collection().aggregate(pipeline))
        if not result or result[0]["count"] < 3:
            # Not enough data for reliable stats
            return None

        return {
            "faithfulness": result[0]["avg_faithfulness"],
            "answer_relevancy": result[0]["avg_relevancy"],
            "context_precision": result[0]["avg_precision"],
        }

    def check_drift(self, corpus_source: str = "all") -> list[DriftAlert]:
        """
        Compare current 24h window against 7-day baseline.
        Returns list of DriftAlert — empty if no drift detected.
        """
        sources = (
            ["mongodb", "dash0", "reap"]
            if corpus_source == "all"
            else [corpus_source]
        )

        alerts: list[DriftAlert] = []

        for source in sources:
            baseline = self._compute_window_avg(
                source, hours=settings.drift_baseline_days * 24
            )
            current = self._compute_window_avg(
                source, hours=settings.drift_window_hours
            )

            if not baseline or not current:
                continue

            for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
                base_val = baseline[metric]
                curr_val = current[metric]

                if base_val == 0:
                    continue

                drop = (base_val - curr_val) / base_val
                if drop > settings.drift_alert_threshold:
                    alert = DriftAlert(
                        metric=metric,
                        corpus_source=source,
                        baseline_value=base_val,
                        current_value=curr_val,
                        drop_fraction=drop,
                        window_hours=settings.drift_window_hours,
                    )
                    alerts.append(alert)
                    print(f"🚨 {alert.message}")

        return alerts

    def get_quality_dashboard(self) -> QualityDashboard:
        """
        Aggregate quality state across all corpora.
        Returns a structured dashboard ready for display or alerting.
        """
        per_corpus: dict[str, dict[str, float]] = {}
        total_evals = 0
        total_pass = 0

        for source in ["mongodb", "dash0", "reap"]:
            stats = self._compute_window_avg(source, hours=24)
            if stats:
                per_corpus[source] = stats
                # Count pass/fail
                col = self._get_collection()
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(hours=24)
                ).isoformat()
                total = col.count_documents({
                    "corpus_source": source,
                    "evaluated_at": {"$gte": cutoff},
                })
                passed = col.count_documents({
                    "corpus_source": source,
                    "evaluated_at": {"$gte": cutoff},
                    "status": "pass",
                })
                total_evals += total
                total_pass += passed

        alerts = self.check_drift()

        # Determine trend from last 3 days vs today
        recent = self._compute_window_avg("mongodb", hours=24)
        older = self._compute_window_avg("mongodb", hours=72)
        trend = "stable"
        if recent and older:
            delta = recent["faithfulness"] - older["faithfulness"]
            if delta > 0.05:
                trend = "improving"
            elif delta < -0.05:
                trend = "degrading"

        return QualityDashboard(
            total_evaluations=total_evals,
            overall_pass_rate=total_pass / total_evals if total_evals else 0.0,
            per_corpus=per_corpus,
            active_alerts=alerts,
            trend=trend,
        )

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
