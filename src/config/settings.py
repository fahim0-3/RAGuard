"""Central configuration.

Every tunable constant in RAGuard lives here so that an experiment is a change
to `.env`, not a change to code. That matters for the evaluation CI: a metric
regression must be traceable to a single configuration diff.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
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

    # --- LLM provider ---
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_judge_model: str = "gemini-2.5-flash-lite"
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
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Used only when the primary reranker cannot be loaded. 22 M parameters
    # against the primary's 568 M, so it stays usable on CPU-only machines.
    reranker_fallback_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_device: str = "cpu"

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Call `get_settings.cache_clear()` in tests that patch env."""
    return Settings()
