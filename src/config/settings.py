"""Central configuration.

Every tunable constant in RAGuard lives here so that an experiment is a change
to `.env`, not a change to code. That matters for the evaluation CI: a metric
regression must be traceable to a single configuration diff.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
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
    vector_dimension: int = Field(default=1024, ge=1, le=16_384)  # BGE-M3 dense output width.
    # Bound API readiness and query waits during a managed-database outage.
    # The pool retries connection creation in the background for longer than a
    # single request is allowed to wait.
    db_pool_timeout_s: float = Field(default=10.0, ge=1.0, le=60.0)
    db_connect_timeout_s: int = Field(default=10, ge=1, le=60)
    db_reconnect_timeout_s: float = Field(default=30.0, ge=5.0, le=300.0)
    # Keep database work bounded by the request budget even when a query plan,
    # network path, or managed database becomes unhealthy.
    db_statement_timeout_s: float = Field(default=30.0, ge=0.1, le=300.0)
    db_pool_min_size: int = Field(default=1, ge=1, le=128)
    db_pool_max_size: int = Field(default=4, ge=1, le=128)
    # A pooled connection returned within this many seconds is reused
    # without a validation query. Against a remote managed database each
    # validation is a full network round trip (measured: ~280 ms, the same
    # as the vector query itself), so validating a connection used a moment
    # ago doubled every dense search. Older connections are still checked,
    # which is what catches the ones a serverless database closed while
    # idle; a read that still meets a dead socket is retried once on a
    # validated pool. 240 s stays under Neon's default 5-minute compute
    # suspend. 0 restores validation on every checkout.
    db_checkout_validation_idle_s: float = Field(default=240.0, ge=0.0, le=3_600.0)
    # `memory` serves dense search from vectors held in the API process,
    # removing the database round trip from every query (pgvector stays the
    # source of truth and the fallback). `pgvector` queries the database
    # per request, as before.
    dense_index_backend: Literal["memory", "pgvector"] = "memory"
    # How often the in-memory indexes check the table for changes made by
    # ingestion in another process, and rebuild if it changed. 0 disables
    # the check; POST /admin/reindex still rebuilds on demand.
    corpus_refresh_interval_s: float = Field(default=60.0, ge=0.0, le=86_400.0)

    # --- Runtime environment ---
    runtime_environment: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias=AliasChoices("RAGUARD_ENVIRONMENT", "runtime_environment"),
    )
    model_cache_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_HOME", "model_cache_dir"),
    )
    # Once models have been downloaded successfully, avoid Hugging Face cache
    # validation traffic on every API restart. A missing cache then fails
    # readiness clearly instead of silently downloading during startup.
    local_model_offline: bool = False

    # --- LLM provider ---
    llm_provider: Literal["gemini", "groq", "openrouter", "ollama"] = "gemini"
    # `static` preserves the historical LLM_PROVIDER-only selection. Dynamic
    # routing chooses once per graph using explicit workload/privacy settings.
    llm_routing_mode: Literal["static", "dynamic"] = "static"
    # Dynamic mode only: keeps all model calls local and deliberately prevents
    # a hosted fallback from sending private data outside the deployment.
    llm_routing_local_only: bool = False
    # Dynamic mode only: allow local Ollama as the last resort after every
    # hosted provider has failed. Off by default for interactive serving:
    # measured on CPU, that path took 31-40 s per request and still
    # abstained, so a hosted outage now returns a fast provider_error
    # instead. Local-first operation is unaffected: LLM_ROUTING_LOCAL_ONLY
    # or LLM_PROVIDER=ollama still route every call to Ollama.
    llm_routing_local_fallback: bool = False
    # Dynamic mode only: select Groq at graph entry for an explicitly strict
    # structured-output workload. Evaluation sets the same preference itself.
    llm_routing_strict_structured_output: bool = False
    google_api_key: str | None = None
    groq_api_key: str | None = Field(default=None, repr=False)
    openrouter_api_key: str | None = Field(default=None, repr=False)
    # Model IDs are pinned so latency and evaluation comparisons are meaningful.
    gemini_model: str = "gemini-3.8-flash"
    gemini_judge_model: str = "gemini-3.8-flash"
    # Groq is first in the dynamic route because its native structured output
    # is the most reliable fit for RAGuard's answer and grading schemas.
    groq_model: str = "openai/gpt-oss-120b"
    groq_judge_model: str = "openai/gpt-oss-120b"
    # GPT-OSS can reject valid requests while enforcing a provider-side schema.
    # Prompt-guided JSON remains validated by the parser and avoids that 400.
    groq_native_structured_output: bool = False
    # GPT-OSS reasoning depth. Unset keeps the provider default (medium).
    # Reasoning tokens are billed against Groq's per-minute token budget and
    # emitted before the first character of the answer, so they cost both
    # latency and rate-limit headroom. Judge and generator are separate
    # because a yes/no grading verdict needs far less deliberation than a
    # cited multi-sentence answer. Evaluate before lowering either.
    groq_reasoning_effort: Literal["low", "medium", "high"] | None = None
    groq_judge_reasoning_effort: Literal["low", "medium", "high"] | None = None
    # OpenRouter is an OpenAI-compatible, optional dynamic fallback. Free
    # models use prompt-plus-parser structured output, never native strict mode.
    openrouter_model: str = "google/gemma-4-26b-a4b-it:free"
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
    # Reasoning models spend this budget before they emit a single character of
    # the answer: GPT-OSS used 613 of 845 output tokens on reasoning for a
    # two-passage question. At 1024 the JSON answer was truncated mid-object,
    # the parser rejected it, and the request failed over to every remaining
    # provider. The ceiling covers reasoning plus a complete structured answer.
    llm_max_output_tokens: int = Field(default=2048, ge=1, le=16_384)
    llm_request_timeout_s: int = Field(default=60, ge=1)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # --- Embeddings and local models ---
    # `gemini` keeps all embedding inference hosted, so it needs no PyTorch or
    # Hugging Face model cache. Documents must be re-ingested after switching.
    embedding_provider: Literal["local", "gemini"] = "local"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_request_timeout_s: float = Field(default=30.0, ge=0.1, le=300.0)
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
    reranker_provider: Literal["local", "voyage", "cohere"] = "local"
    # A second hosted reranker, tried when the first is rate limited, cooling
    # down or unconfigured. `none` keeps the single-provider behaviour and
    # falls straight through to RERANKER_FALLBACK_PROVIDER. Voyage's free
    # tier allows 3 requests a minute, which one abstaining question can
    # spend on its own, so a second hosted provider keeps hosted-quality
    # ordering rather than dropping to the local model.
    reranker_hosted_fallback: Literal["none", "voyage", "cohere"] = "none"
    cohere_api_key: str | None = Field(default=None, repr=False)
    # Cohere's current general-purpose reranker. English-only variants exist
    # but bring no benefit here: the corpus is short English policy text and
    # the hosted model decides order only, never the confidence scores.
    cohere_rerank_model: str = "rerank-v3.5"
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
    # Score the Voyage candidate pool locally while the Voyage request is in
    # flight, instead of scoring Voyage's picks after it returns. The scores
    # are identical (a per-pair sigmoid, independent of batch company); only
    # the ~200 ms local pass moves off the critical path.
    reranker_overlap_local_scoring: bool = True
    # After a Voyage timeout, 5xx or 429, go straight to the local reranker
    # for a short cooldown instead of paying the hosted timeout (3 s per
    # attempt, measured at 6.4 s with one retry) on every request.
    hosted_rerank_cooldown_enabled: bool = True

    # --- Ingestion ---
    data_dir: Path = Path("data/policies")
    chunk_size: int = Field(default=800, ge=1, le=100_000)
    chunk_overlap: int = Field(default=120, ge=0, le=99_999)

    # --- Evaluation ---
    # Configurable so a larger dataset can be evaluated without a code change.
    golden_dataset_path: Path = Path("src/evaluation/golden_dataset.json")

    # --- Retrieval ---
    dense_top_k: int = Field(default=20, ge=1, le=10_000)
    sparse_top_k: int = Field(default=20, ge=1, le=10_000)
    fusion_top_k: int = Field(default=20, ge=1, le=10_000)
    rerank_top_k: int = Field(default=5, ge=1, le=10_000)
    rrf_k: int = Field(default=60, ge=1, le=10_000)

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
    # Time a request may wait for the resident cross-encoder.  This protects
    # the graph deadline when several requests arrive at once.
    reranker_queue_timeout_s: float = Field(default=15.0, ge=0.01, le=300.0)
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
    # Start answer generation at the same moment as evidence grading, and
    # use it only if the grader then finds the evidence sufficient. The
    # answer still passes the grader and citation verification before it
    # can be shown; only the wait between them is removed. First attempt
    # only, and only when the deterministic evidence gate already passes,
    # so an abstention costs at most one discarded generation.
    graph_speculative_generation: bool = True
    # The compact contract is the serving default for low-latency policy Q&A.
    # The condition-aware contract remains available for evaluation or cases
    # that require a full branch-by-branch decision record.
    # A top reranker score at or below this means no retrieved passage has
    # anything to do with the question, and the model grader is skipped: it
    # cannot make such evidence sufficient. Measured on this corpus: 1.3e-05
    # for "do you sell gaming laptops" and 1.4e-04 for "what is your
    # price-match policy", against 0.98 for a question the corpus answers.
    # Two orders of magnitude below `evidence_top_score_threshold`, so a
    # merely weak match still gets a model verdict and its account of what
    # is missing. 0 disables the skip.
    evidence_irrelevant_score_ceiling: float = Field(default=0.001, ge=0.0, le=0.1)
    evidence_grading_mode: Literal["simple", "condition_aware"] = "simple"
    # Evidence grading sits in front of every answer, and a slow provider there
    # delays the whole request before a single token is generated. Bound each
    # grading call well below `llm_request_timeout_s` so a provider that hangs
    # costs seconds rather than the full request budget. A provider failover
    # inherits this bound instead of resetting to the general limit.
    evidence_grading_timeout_s: float = Field(default=15.0, ge=1.0, le=120.0)
    # One wall-clock and provider-call budget spans grading, rewriting,
    # generation, verification, and every graph retry.
    graph_request_timeout_s: int = Field(default=150, ge=5, le=3_600)
    graph_llm_call_limit: int = Field(default=8, ge=1, le=100)
    # The baseline preserves the eight-call evaluation contract. The free
    # hosted pilot profile deliberately caps an individual graph at four calls
    # without changing its topology or retry configuration.
    llm_execution_profile: Literal["baseline", "free_hosted_pilot"] = "baseline"
    # Skip a provider for a short cooldown after a 429, 5xx, timeout or
    # rejected key, instead of paying its failure latency on every request.
    # Applies to dynamic routing of normal traffic only; evaluation keeps a
    # fixed route so its results stay comparable.
    llm_provider_cooldown_enabled: bool = True
    # Count tokens spent against each provider's published limits and route
    # to the next provider before a call would exceed them. A refusal costs
    # a round trip and a failover; the limit is knowable in advance, so the
    # last request that fits is served and the next goes elsewhere.
    llm_token_budget_enabled: bool = True
    # Groq's free tier, per model. Lower these to match a paid tier's
    # published figures, or set 0 to stop counting that window.
    groq_tokens_per_minute: int = Field(default=8_000, ge=0)
    groq_tokens_per_day: int = Field(default=200_000, ge=0)
    # What one call is assumed to cost before it is made. Charged against
    # the remaining budget so a call is only started when it fits. Measured
    # on this corpus: grading about 1,050 tokens, generation about 1,900,
    # verification about 600. The default covers the largest of these.
    llm_estimated_tokens_per_call: int = Field(default=2_000, ge=1)

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

    @field_validator("groq_reasoning_effort", "groq_judge_reasoning_effort", mode="before")
    @classmethod
    def _blank_effort_means_provider_default(cls, value: object) -> object:
        """`GROQ_REASONING_EFFORT=` in a copied template means unset, not invalid."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _request_budget_fits_admission_lease(self) -> Settings:
        if self.graph_request_timeout_s >= self.admission_lease_seconds:
            raise ValueError("graph_request_timeout_s must be lower than admission_lease_seconds")
        if self.db_pool_min_size > self.db_pool_max_size:
            raise ValueError("db_pool_min_size must not exceed db_pool_max_size")
        if self.db_pool_max_size > self.query_max_concurrency:
            raise ValueError("db_pool_max_size must not exceed query_max_concurrency")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be lower than chunk_size")
        if self.fusion_top_k > self.dense_top_k + self.sparse_top_k:
            raise ValueError("fusion_top_k cannot exceed the dense and sparse candidate total")
        if self.rerank_candidate_top_k > self.fusion_top_k:
            raise ValueError("rerank_candidate_top_k must not exceed fusion_top_k")
        if self.rerank_top_k > self.rerank_candidate_top_k:
            raise ValueError("rerank_top_k must not exceed rerank_candidate_top_k")
        if self.db_statement_timeout_s > self.graph_request_timeout_s:
            raise ValueError("db_statement_timeout_s must not exceed graph_request_timeout_s")
        if self.embedding_request_timeout_s > self.graph_request_timeout_s:
            raise ValueError("embedding_request_timeout_s must not exceed graph_request_timeout_s")
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
