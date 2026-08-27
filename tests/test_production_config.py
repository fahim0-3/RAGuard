"""Static production preflight must fail closed without exposing secrets."""

from __future__ import annotations

import pytest

from src.config import (
    ProductionConfigurationError,
    Settings,
    enforce_production_configuration,
    enforce_production_runtime_storage,
    validate_production_settings,
)


def production_settings(**overrides) -> Settings:
    base = Settings(
        _env_file=None,
        RAGUARD_ENVIRONMENT="production",
        database_url="postgresql://user:secret@db.example.net/raguard?sslmode=require",
        google_api_key="g" * 32,
        admin_api_key="a" * 48,
        cors_allow_origins="https://app.example.net",
        api_base_url="https://api.example.net",
        api_host="0.0.0.0",
        admission_backend="redis",
        admission_redis_url="rediss://redis.example.net:6379/0",
        otel_exporter_otlp_endpoint="https://otel.example.net/v1/traces",
        model_cache_dir="/models",
    )
    return base.model_copy(update=overrides)


def test_valid_production_configuration_passes():
    report = validate_production_settings(production_settings())

    assert report.ok is True
    assert report.errors == ()


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"database_url": "postgresql://u:p@localhost/db?sslmode=require"}, "database_url_local"),
        ({"database_url": "postgresql://u:p@db.example.net/db"}, "database_tls_required"),
        (
            {"database_admin_url": "postgresql://u:p@admin.example.net/db"},
            "database_admin_tls_required",
        ),
        (
            {"database_admin_url": "postgresql://u:p@localhost/db?sslmode=require"},
            "database_admin_url_local",
        ),
        ({"google_api_key": "your-key"}, "google_api_key_missing"),
        (
            {"llm_provider": "ollama", "ollama_base_url": "http://localhost:11434"},
            "ollama_url_invalid",
        ),
        ({"admin_api_key": "short"}, "admin_api_key_weak"),
        ({"cors_allow_origins": "*"}, "cors_origins_unsafe"),
        ({"cors_allow_origins": "http://app.example.net"}, "cors_origin_not_https"),
        ({"api_base_url": "http://localhost:8000"}, "api_base_url_not_https"),
        ({"api_host": "127.0.0.1"}, "api_host_not_public"),
        ({"model_cache_dir": None}, "model_cache_unconfigured"),
        ({"admission_redis_url": "redis://localhost:6379/0"}, "redis_url_invalid"),
    ],
)
def test_unsafe_production_configuration_is_rejected(overrides, expected_code):
    report = validate_production_settings(production_settings(**overrides))

    assert expected_code in {issue.code for issue in report.errors}


def test_production_exception_contains_codes_but_not_secrets():
    settings = production_settings(admin_api_key="private-secret", google_api_key="private-provider")

    with pytest.raises(ProductionConfigurationError) as raised:
        enforce_production_configuration(settings)

    message = str(raised.value)
    assert "admin_api_key_weak" in message
    assert "google_api_key_missing" in message
    assert "private-secret" not in message
    assert "private-provider" not in message


def test_development_does_not_fail_startup_on_production_defaults():
    settings = Settings(_env_file=None, RAGUARD_ENVIRONMENT="development")

    report = enforce_production_configuration(settings)

    assert report.errors


def test_production_runtime_requires_the_attached_model_directory(tmp_path):
    settings = production_settings(model_cache_dir=tmp_path / "not-mounted")

    with pytest.raises(ProductionConfigurationError, match="model_cache_missing"):
        enforce_production_runtime_storage(settings)


def test_production_runtime_accepts_a_writable_model_directory(tmp_path):
    enforce_production_runtime_storage(production_settings(model_cache_dir=tmp_path))


def test_development_runtime_does_not_require_remote_model_storage():
    settings = Settings(_env_file=None, RAGUARD_ENVIRONMENT="development")

    enforce_production_runtime_storage(settings)


def test_api_startup_enforces_the_production_contract(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    invalid = production_settings(admin_api_key="weak")
    monkeypatch.setattr("api.main.get_settings", lambda: invalid)

    with pytest.raises(ProductionConfigurationError), TestClient(app):
        pass
