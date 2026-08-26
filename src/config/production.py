"""Production configuration checks that never expose secret values."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.config.settings import Settings

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "db", "postgres"}
PLACEHOLDER_FRAGMENTS = {"example", "change-me", "your-", "placeholder"}


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    errors: tuple[PreflightIssue, ...] = ()
    warnings: tuple[PreflightIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


class ProductionConfigurationError(RuntimeError):
    """Static production configuration is unsafe or incomplete."""


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(fragment in lowered for fragment in PLACEHOLDER_FRAGMENTS)


def validate_production_settings(settings: Settings) -> PreflightReport:
    """Return safe, deterministic production errors and recommendations."""
    errors: list[PreflightIssue] = []
    warnings: list[PreflightIssue] = []

    database = urlparse(settings.database_url)
    if database.scheme not in {"postgres", "postgresql"} or not database.hostname:
        errors.append(PreflightIssue("database_url_invalid", "DATABASE_URL must be PostgreSQL."))
    else:
        if database.hostname.lower() in LOCAL_HOSTS:
            errors.append(
                PreflightIssue(
                    "database_url_local",
                    "Production DATABASE_URL must point to a managed or remote database.",
                )
            )
        sslmode = parse_qs(database.query).get("sslmode", [""])[0].lower()
        if sslmode not in {"require", "verify-ca", "verify-full"}:
            errors.append(
                PreflightIssue(
                    "database_tls_required",
                    "Production DATABASE_URL must enforce TLS with sslmode.",
                )
            )

    if settings.llm_provider == "gemini":
        key = settings.google_api_key or ""
        if len(key) < 20 or _is_placeholder(key):
            errors.append(
                PreflightIssue("google_api_key_missing", "A non-placeholder provider key is required.")
            )
    else:
        ollama_url = urlparse(settings.ollama_base_url)
        if (
            ollama_url.scheme != "https"
            or not ollama_url.hostname
            or ollama_url.hostname in LOCAL_HOSTS
        ):
            errors.append(
                PreflightIssue(
                    "ollama_url_invalid",
                    "Production Ollama must use a remote HTTPS endpoint.",
                )
            )

    if len(settings.admin_api_key) < 32 or _is_placeholder(settings.admin_api_key):
        errors.append(
            PreflightIssue(
                "admin_api_key_weak",
                "ADMIN_API_KEY must be a non-placeholder value of at least 32 characters.",
            )
        )

    origins = settings.cors_allow_origins_list
    if not origins or "*" in origins:
        errors.append(
            PreflightIssue("cors_origins_unsafe", "Production CORS origins must be explicit.")
        )
    else:
        for origin in origins:
            parsed = urlparse(origin)
            if parsed.scheme != "https" or not parsed.hostname or parsed.hostname in LOCAL_HOSTS:
                errors.append(
                    PreflightIssue(
                        "cors_origin_not_https",
                        "Every production CORS origin must be a remote HTTPS origin.",
                    )
                )
                break

    api_url = urlparse(settings.api_base_url)
    if api_url.scheme != "https" or not api_url.hostname or api_url.hostname in LOCAL_HOSTS:
        errors.append(
            PreflightIssue(
                "api_base_url_not_https",
                "Production API_BASE_URL must be the public HTTPS API URL.",
            )
        )

    if settings.api_host != "0.0.0.0":
        errors.append(
            PreflightIssue("api_host_not_public", "Production API_HOST must bind to 0.0.0.0.")
        )

    if settings.model_cache_dir is None or not settings.model_cache_dir.anchor:
        errors.append(
            PreflightIssue(
                "model_cache_unconfigured",
                "HF_HOME must be an absolute path backed by remote persistent storage.",
            )
        )

    if settings.admission_backend != "redis":
        warnings.append(
            PreflightIssue(
                "admission_not_distributed",
                "Local admission control is safe only for a single API replica.",
            )
        )
    else:
        redis_url = urlparse(settings.admission_redis_url)
        if (
            redis_url.scheme not in {"redis", "rediss"}
            or not redis_url.hostname
            or redis_url.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            errors.append(
                PreflightIssue(
                    "redis_url_invalid",
                    "Distributed admission requires a remote Redis URL.",
                )
            )
        elif redis_url.scheme != "rediss":
            warnings.append(
                PreflightIssue(
                    "redis_tls_disabled",
                    "Use rediss:// unless Redis is on a trusted private network.",
                )
            )
    if not settings.otel_exporter_otlp_endpoint:
        warnings.append(
            PreflightIssue(
                "tracing_export_disabled",
                "No OTLP collector is configured; traces remain process-local.",
            )
        )
    if settings.runtime_profile == "local_compact":
        warnings.append(
            PreflightIssue(
                "compact_reranker_selected",
                "The compact reranker reduces storage but must be evaluated before promotion.",
            )
        )

    return PreflightReport(errors=tuple(errors), warnings=tuple(warnings))


def enforce_production_configuration(settings: Settings) -> PreflightReport:
    """Fail startup on static production errors, returning warnings otherwise."""
    report = validate_production_settings(settings)
    if settings.runtime_environment == "production" and report.errors:
        codes = ", ".join(issue.code for issue in report.errors)
        raise ProductionConfigurationError(f"production preflight failed: {codes}")
    return report


def enforce_production_runtime_storage(settings: Settings) -> None:
    """Fail production startup when the attached model cache is not writable.

    Static preflight deliberately does not create or write to the configured
    path because provider build/pre-deploy jobs cannot see runtime disks. This
    check runs inside the service after the disk is mounted and before model
    warm-up starts.
    """
    if settings.runtime_environment != "production":
        return

    cache_dir = Path(settings.model_cache_dir) if settings.model_cache_dir else None
    if cache_dir is None or not cache_dir.is_dir():
        raise ProductionConfigurationError(
            "production runtime check failed: model_cache_missing"
        )
    if not os.access(cache_dir, os.W_OK):
        raise ProductionConfigurationError(
            "production runtime check failed: model_cache_not_writable"
        )
