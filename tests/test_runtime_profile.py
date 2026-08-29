"""Runtime profiles select models without mutating reproducibility settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.reranking.cross_encoder import CrossEncoderReranker


def test_full_profile_preserves_the_production_reranker():
    settings = Settings(_env_file=None, RAGUARD_RUNTIME_PROFILE="full")

    assert settings.resolved_reranker_model == settings.reranker_model


def test_local_compact_profile_selects_the_fallback_model():
    settings = Settings(_env_file=None, RAGUARD_RUNTIME_PROFILE="local_compact")

    assert settings.resolved_reranker_model == settings.reranker_fallback_model


def test_unknown_runtime_profile_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, RAGUARD_RUNTIME_PROFILE="typo")


def test_free_hosted_pilot_caps_only_the_effective_graph_call_limit():
    settings = Settings(
        _env_file=None,
        graph_llm_call_limit=8,
        llm_execution_profile="free_hosted_pilot",
    )

    assert settings.graph_llm_call_limit == 8
    assert settings.effective_graph_llm_call_limit == 4


def test_baseline_keeps_the_configured_graph_call_limit():
    settings = Settings(_env_file=None, graph_llm_call_limit=8, llm_execution_profile="baseline")

    assert settings.effective_graph_llm_call_limit == 8


def test_reranker_uses_the_profile_resolved_model(monkeypatch):
    settings = Settings(_env_file=None, RAGUARD_RUNTIME_PROFILE="local_compact")
    monkeypatch.setattr("src.reranking.cross_encoder.get_settings", lambda: settings)

    reranker = CrossEncoderReranker()

    assert reranker.model_name == settings.reranker_fallback_model


def test_auto_batch_size_is_device_aware_without_changing_cpu_defaults(monkeypatch):
    settings = Settings(_env_file=None, reranker_batch_size=0)
    monkeypatch.setattr("src.reranking.cross_encoder.get_settings", lambda: settings)

    cpu_reranker = CrossEncoderReranker(model=object(), device="cpu")
    cuda_reranker = CrossEncoderReranker(model=object(), device="cuda")

    assert cpu_reranker.batch_size == 16
    assert cuda_reranker.batch_size == 32


def test_warmup_exposes_loaded_and_failed_states():
    class Working(CrossEncoderReranker):
        def _load(self, model_name):
            return object()

    class Broken(CrossEncoderReranker):
        def _load(self, model_name):
            raise OSError("download failed")

    working = Working(model_name="primary", fallback_model_name="")
    assert working.is_model_loaded is False
    assert working.warmup() is True
    assert working.is_model_loaded is True
    assert working.loaded_model_name == "primary"

    broken = Broken(model_name="primary", fallback_model_name="")
    assert broken.warmup() is False
    assert broken.is_model_loaded is False
    assert "OSError" in (broken.load_error or "")
