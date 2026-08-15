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
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 1024

    # --- Local models ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    model_device: str = "cpu"

    # --- Ingestion ---
    data_dir: Path = Path("data/policies")
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval ---
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60

    # --- Self-healing ---
    retrieval_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    abstain_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    max_healing_attempts: int = Field(default=2, ge=0, le=5)
    query_rewrite_variants: int = Field(default=3, ge=1, le=6)
    citation_support_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    # --- Service ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    @property
    def absolute_data_dir(self) -> Path:
        path = self.data_dir
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor. Call `get_settings.cache_clear()` in tests that patch env."""
    return Settings()
