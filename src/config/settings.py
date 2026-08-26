"""Central configuration.

Every tunable constant in RAGuard lives here so that an experiment is a change
to `.env`, not a change to code. That matters for the evaluation CI: a metric
regression must be traceable to a single configuration diff.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    database_url: str = "postgresql://raguard:raguard@localhost:5433/raguard"
    vector_dimension: int = 1024  # BGE-M3 dense output width.
    # Bound API readiness and query waits during a managed-database outage.
    # The pool retries connection creation in the background for longer than a
    # single request is allowed to wait.
    db_pool_timeout_s: float = Field(default=10.0, ge=1.0, le=60.0)
    db_connect_timeout_s: int = Field(default=10, ge=1, le=60)
    db_reconnect_timeout_s: float = Field(default=30.0, ge=5.0, le=300.0)

    # --- Runtime environment ---
    runtime_environment: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias=AliasChoices("RAGUARD_ENVIRONMENT", "runtime_environment"),
    )
    model_cache_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_HOME", "model_cache_dir"),
    )

    # --- LLM provider ---
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    google_api_key: str | None = None
    # Verified against a live key on 2026-08-16. The previous defaults
    # (gemini-2.5-flash / gemini-2.5-flash-lite) now return 404 "no longer
    # available to new users", so the system could not generate out of the box.
    #
    # Pinned rather than a floating `-latest` alias: evaluation baselines are
    # only comparable across runs if the model is fixed. `gemini-flash-latest`
    # was rejected as a default for a second reason — it currently resolves to a
    # model with a 5 requests/minute free-tier quota, which the retry loop and
    # the per-claim judge exhaust immediately.
    gemini_model: str = "gemini-3.1-flash-lite"
    # Deliberately a different, stable model from the generator. A floating
    # `-latest` alias makes a regression report change without a code or config
    # change, which defeats reproducible evaluation. Rebaseline deliberately
    # whenever this explicit model ID changes.
    gemini_judge_model: str = "gemini-3.5-flash-lite"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    # Provider-agnostic model override. Empty means "use the provider default",
    # so switching provider does not require editing a model ID.
    llm_model: str = ""
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_output_tokens: int = 1024
    llm_request_timeout_s: int = Field(default=60, ge=1)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # --- Local models ---
    runtime_profile: Literal["full", "local_compact"] = Field(
        default="full",
        validation_alias=AliasChoices("RAGUARD_RUNTIME_PROFILE", "runtime_profile"),
    )
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Used only when the primary reranker cannot be loaded. 22 M parameters
    # against the primary's 568 M, so it stays usable on CPU-only machines.
    reranker_fallback_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_device: str = "cpu"
    # Empty inherits MODEL_DEVICE. Keeping the reranker separate lets an API
    # process retain embeddings on CPU while benchmarking the cross-encoder on
    # an available GPU with its own memory budget.
    reranker_device: str = ""

    # --- Ingestion ---
    data_dir: Path = Path("data/policies")
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Evaluation ---
    # Configurable so a larger dataset can be evaluated without a code change.
    golden_dataset_path: Path = Path("src/evaluation/golden_dataset.json")

    # --- Retrieval ---
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60

    # --- Reranking ---
    reranker_enabled: bool = True
    # Candidates handed to the cross-encoder. Cost is linear in this number.
    rerank_candidate_top_k: int = Field(default=20, ge=1)
    reranker_batch_size: int = Field(default=16, ge=1)
    reranker_max_length: int = Field(default=512, ge=64)

    # --- Deduplication ---
    dedup_enabled: bool = True
    dedup_near_duplicate_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    dedup_adjacent_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    # Maximum consecutive chunks from one source. Set to 0 to disable the cap.
    # Measured: a cap of 3 removed distinct sections without improving any
    # metric on the current corpus, where documents are 3 to 4 chunks long.
    dedup_max_adjacent_run: int = Field(default=5, ge=0)

    # --- Self-healing (legacy imperative pipeline) ---
    retrieval_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    abstain_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    max_healing_attempts: int = Field(default=2, ge=0, le=5)
    query_rewrite_variants: int = Field(default=3, ge=1, le=6)
    citation_support_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    # --- Self-healing graph (Phase F) ---
    # Specification defaults. These gate the evidence decision together with
    # the structured grader; neither signal decides alone.
    evidence_top_score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    evidence_min_relevant_chunks: int = Field(default=2, ge=1)
    evidence_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    graph_max_retries: int = Field(default=2, ge=0, le=5)
    # One regeneration after a failed citation check, then abstain.
    graph_max_regenerations: int = Field(default=1, ge=0, le=3)
    graph_use_llm: bool = True

    # --- Citation verification (Phase G) ---
    # "entailment" adds semantic checking; "deterministic" keeps the Phase F
    # lexical verifier, which needs no provider.
    verifier_backend: Literal["entailment", "deterministic"] = "entailment"

    # --- Service ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    # Comma-separated origins allowed to call the API from a browser. The
    # default covers a local Streamlit; "*" is accepted but must be a
    # deliberate choice, not a default.
    cors_allow_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    # Empty disables operational endpoints rather than leaving them public.
    # Store this outside source control and rotate it like any other secret.
    admin_api_key: str = Field(default="", repr=False)
    # Process-local protection for expensive requests. A gateway must enforce
    # equivalent tenant/IP limits when more than one API instance is deployed.
    query_max_concurrency: int = Field(default=4, ge=1, le=128)
    query_rate_limit_per_minute: int = Field(default=30, ge=1, le=10_000)
    # `redis` makes the guard atomic across API replicas. It fails closed if
    # Redis is unavailable; `local` is for a one-process development setup.
    admission_backend: Literal["local", "redis"] = "local"
    admission_redis_url: str = "redis://localhost:6379/0"
    admission_redis_namespace: str = "raguard:admission"
    # Must exceed the longest permitted request so a healthy worker keeps its
    # concurrency slot; expiry still recovers a slot after a worker crash.
    admission_lease_seconds: int = Field(default=300, ge=30, le=3_600)
    # Empty keeps local trace context only. Set a collector endpoint, for
    # example http://otel-collector:4318/v1/traces, to export OTLP spans.
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "raguard-api"

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def absolute_data_dir(self) -> Path:
        path = self.data_dir
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def absolute_golden_dataset_path(self) -> Path:
        path = self.golden_dataset_path
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    @property
    def resolved_reranker_device(self) -> str:
        """Configured cross-encoder device, falling back to MODEL_DEVICE."""
        return self.reranker_device.strip() or self.model_device

    @property
    def resolved_reranker_model(self) -> str:
        """Return the profile-selected cross-encoder model."""
        if self.runtime_profile == "local_compact":
            return self.reranker_fallback_model
        return self.reranker_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Call `get_settings.cache_clear()` in tests that patch env."""
    return Settings()
