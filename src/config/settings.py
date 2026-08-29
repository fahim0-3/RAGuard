"""Central configuration.

Every tunable constant in RAGuard lives here so that an experiment is a change
to `.env`, not a change to code. That matters for the evaluation CI: a metric
regression must be traceable to a single configuration diff.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
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
    # Optional direct/admin connection used only for extension and schema DDL.
    # Leave empty to reuse DATABASE_URL (the normal local/direct setup). Keeping
    # this separate lets a future pooled runtime URL avoid migration traffic.
    database_admin_url: str = Field(default="", repr=False)
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
    llm_provider: Literal["gemini", "groq", "ollama"] = "gemini"
    # `static` preserves the historical LLM_PROVIDER-only selection. Dynamic
    # routing chooses once per graph using explicit workload/privacy settings.
    llm_routing_mode: Literal["static", "dynamic"] = "static"
    # Dynamic mode only: keeps all model calls local and deliberately prevents
    # a hosted fallback from sending private data outside the deployment.
    llm_routing_local_only: bool = False
    # Dynamic mode only: select Groq at graph entry for an explicitly strict
    # structured-output workload. Evaluation sets the same preference itself.
    llm_routing_strict_structured_output: bool = False
    google_api_key: str | None = None
    groq_api_key: str | None = Field(default=None, repr=False)
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
    # Groq is optional. Its default model supports strict JSON-schema output,
    # which is used by every structured LLM step in the self-healing graph.
    groq_model: str = "openai/gpt-oss-20b"
    groq_judge_model: str = "openai/gpt-oss-20b"
    # ChatGroq retries transient 429/5xx responses with exponential backoff.
    # An active graph budget passes zero, so hidden retries never exceed the
    # request-level call allowance.
    groq_max_retries: int = Field(default=2, ge=0, le=5)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    # Provider-agnostic model override. Empty means "use the provider default",
    # so switching provider does not require editing a model ID.
    llm_model: str = ""
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_output_tokens: int = 1024
    llm_request_timeout_s: int = Field(default=60, ge=1)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # --- Embeddings and local models ---
    # `gemini` keeps all embedding inference hosted, so it needs no PyTorch or
    # Hugging Face model cache. Documents must be re-ingested after switching.
    embedding_provider: Literal["local", "gemini"] = "local"
    gemini_embedding_model: str = "gemini-embedding-001"
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
    # `auto` prefers CUDA when the installed PyTorch runtime exposes it and
    # otherwise stays on CPU. Keeping this separate lets embeddings remain on
    # CPU while the cross-encoder uses an available GPU. Set an explicit value
    # such as `cpu` or `cuda:0` to override detection.
    reranker_device: str = "auto"

    # --- Hosted reranker (explicit opt-in only) ---
    # `local` is the privacy-preserving default.  There is intentionally no
    # automatic provider selection: setting an API key must never by itself
    # cause policy passages to leave this deployment.
    reranker_provider: Literal["local", "voyage"] = "local"
    reranker_remote_allowed: bool = False
    voyage_api_key: str | None = Field(default=None, repr=False)
    voyage_rerank_model: str = "rerank-2.5-lite"
    hosted_rerank_timeout_seconds: float = Field(default=3.0, ge=0.1, le=60.0)
    hosted_rerank_max_retries: int = Field(default=1, ge=0, le=5)
    hosted_rerank_top_k: int = Field(default=5, ge=1, le=1000)
    hosted_rerank_max_candidates: int = Field(default=20, ge=1, le=1000)
    # A named, evaluated mapping from a provider's scores to RAGuard's
    # confidence thresholds. `unverified` means hosted scores may order chunks
    # but can never be placed in `normalised_rerank_score`.
    reranker_confidence_profile: str = "unverified"
    reranker_fallback_provider: Literal["local", "rrf"] = "local"

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
    # Zero selects a device-aware default: 32 on CUDA, 16 on CPU. An explicit
    # positive value remains an operator override for capacity tuning.
    reranker_batch_size: int = Field(default=0, ge=0)
    reranker_max_length: int = Field(default=512, ge=64)
    # A single resident cross-encoder is deliberately not driven concurrently
    # by default: this avoids GPU memory spikes and CPU thread oversubscription
    # under the API's multi-query admission limit. It never changes ranking.
    reranker_max_concurrency: int = Field(default=1, ge=1, le=16)
    # Zero leaves PyTorch's process-wide thread setting unchanged. Set a
    # positive value only after a CPU benchmark on the target host.
    reranker_cpu_threads: int = Field(default=0, ge=0, le=128)
    # Execute one synthetic pair during background warm-up so tokenizer and
    # kernel initialization do not inflate the first real query.
    reranker_warmup_inference: bool = True

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
    # One wall-clock and provider-call budget spans grading, rewriting,
    # generation, verification, and every graph retry.
    graph_request_timeout_s: int = Field(default=150, ge=5, le=3_600)
    graph_llm_call_limit: int = Field(default=8, ge=1, le=100)
    # The baseline preserves the eight-call evaluation contract. The free
    # hosted pilot profile deliberately caps an individual graph at four calls
    # without changing its topology or retry configuration.
    llm_execution_profile: Literal["baseline", "free_hosted_pilot"] = "baseline"

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

    @model_validator(mode="after")
    def _request_budget_fits_admission_lease(self) -> Settings:
        if self.graph_request_timeout_s >= self.admission_lease_seconds:
            raise ValueError("graph_request_timeout_s must be lower than admission_lease_seconds")
        if self.embedding_provider == "gemini" and self.reranker_provider == "local":
            # Hosted embeddings remove the local sentence-transformers stack.
            # Keep reranking hosted-only as well rather than silently pulling a
            # cross-encoder into a supposedly model-free runtime.
            self.reranker_enabled = False
        return self

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def schema_database_url(self) -> str:
        """Direct connection used for extension and schema administration."""
        return self.database_admin_url.strip() or self.database_url

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
        """Resolve the cross-encoder device without requiring CUDA at import time."""
        requested = self.reranker_device.strip().lower()
        if requested != "auto":
            return requested or self.model_device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def resolved_reranker_batch_size(self) -> int:
        """Return the explicit batch size or a conservative device-aware default."""
        if self.reranker_batch_size:
            return self.reranker_batch_size
        return 32 if self.resolved_reranker_device.startswith("cuda") else 16

    @property
    def resolved_reranker_model(self) -> str:
        """Return the profile-selected cross-encoder model."""
        if self.runtime_profile == "local_compact":
            return self.reranker_fallback_model
        return self.reranker_model

    @property
    def effective_graph_llm_call_limit(self) -> int:
        """Provider-call allowance selected by the explicit execution profile."""
        if self.llm_execution_profile == "free_hosted_pilot":
            return min(self.graph_llm_call_limit, 4)
        return self.graph_llm_call_limit


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Call `get_settings.cache_clear()` in tests that patch env."""
    return Settings()
