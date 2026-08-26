"""The production preflight CLI reports safe status and exit codes."""

from __future__ import annotations

from scripts import production_preflight


def production_settings():
    from src.config import Settings

    return Settings(
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


def test_preflight_requires_production_environment(monkeypatch):
    from src.config import Settings

    monkeypatch.setattr(
        production_preflight,
        "get_settings",
        lambda: Settings(_env_file=None, RAGUARD_ENVIRONMENT="development"),
    )

    assert production_preflight.main([]) == 2


def test_preflight_dependency_failure_has_distinct_exit_code(monkeypatch):
    monkeypatch.setattr(production_preflight, "get_settings", production_settings)
    monkeypatch.setattr(
        production_preflight,
        "_check_database",
        lambda: (False, "database unavailable (ConnectionTimeout)"),
    )

    assert production_preflight.main(["--check-database"]) == 3


def test_preflight_passes_without_optional_network_checks(monkeypatch):
    monkeypatch.setattr(production_preflight, "get_settings", production_settings)

    assert production_preflight.main([]) == 0
