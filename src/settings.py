"""
Application settings — typed, validated, environment-first.

Design decision: every configurable value lives here.
No magic strings scattered through the codebase.
If a required value is missing, the app fails at startup
with a clear error — not halfway through a request.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── MongoDB ───────────────────────────────────────────────
    mongodb_uri: str = Field(description="MongoDB Atlas connection string")
    mongodb_database: str = Field(default="ai_builder")
    mongodb_collection: str = Field(default="documents")
    mongodb_index_name: str = Field(default="vector_index")
    mongodb_eval_collection: str = Field(default="eval_results")

    # ── OpenRouter ────────────────────────────────────────────
    openrouter_api_key: str = Field(description="OpenRouter API key")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_model: str = Field(default="meta-llama/llama-3.2-3b-instruct")
    openrouter_router_model: str = Field(default="meta-llama/llama-3.2-3b-instruct")

    # ── Voyage AI ─────────────────────────────────────────────
    voyage_api_key: str = Field(description="Voyage AI API key")
    voyage_embed_model: str = Field(default="voyage-3-lite")

    # ── RAG parameters ────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=64, ge=0)

    # ── Evaluation ────────────────────────────────────────────
    min_faithfulness_score: float = Field(default=0.7, ge=0.0, le=1.0)
    min_relevancy_score: float = Field(default=0.6, ge=0.0, le=1.0)
    drift_alert_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Alert when metric drops this fraction from baseline",
    )
    drift_window_hours: int = Field(default=24)
    drift_baseline_days: int = Field(default=7)

    # ── OpenTelemetry ─────────────────────────────────────────
    otel_enabled: bool = Field(default=False)
    otel_endpoint: str = Field(default="http://localhost:4317")
    otel_service_name: str = Field(default="ai-builder")
    dash0_auth_token: str = Field(default="")


settings = Settings()
