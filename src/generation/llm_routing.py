"""Deterministic per-run LLM provider routing and bounded failover.

Routing never reads a customer query and never calls a model. Dynamic mode is
selected entirely from explicit configuration and the caller's workload context.
The mutable route is held in a context variable, making one provider selection
stick across every graph node until a classified provider failure advances it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from src.config.settings import Settings

Provider = Literal["gemini", "groq", "ollama"]
Workload = Literal["normal", "evaluation"]

__all__ = [
    "LLMRoute",
    "Provider",
    "Workload",
    "advance_route",
    "current_provider",
    "current_route",
    "current_workload",
    "is_retryable_provider_error",
    "route_context",
    "select_route",
    "workload_context",
]


@dataclass(slots=True)
class LLMRoute:
    """A provider choice and the deterministic candidates remaining after it."""

    mode: str
    workload: Workload
    candidates: tuple[Provider, ...]
    index: int = 0
    fallback_reasons: list[str] = field(default_factory=list)

    @property
    def provider(self) -> Provider:
        return self.candidates[self.index]

    @property
    def can_fallback(self) -> bool:
        return self.mode == "dynamic" and self.index + 1 < len(self.candidates)

    def snapshot(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "routing_mode": self.mode,
            "workload": self.workload,
            "fallbacks": list(self.fallback_reasons),
        }


_ROUTE: ContextVar[LLMRoute | None] = ContextVar("raguard_llm_route", default=None)
_WORKLOAD: ContextVar[Workload] = ContextVar("raguard_llm_workload", default="normal")


def current_route() -> LLMRoute | None:
    return _ROUTE.get()


def current_workload() -> Workload:
    return _WORKLOAD.get()


def current_provider(settings: Settings) -> Provider:
    """Resolve the route provider, creating no persistent route outside a run."""
    route = current_route()
    return route.provider if route is not None else select_route(settings).provider


def _configured_providers(settings: Settings) -> tuple[Provider, ...]:
    providers: list[Provider] = []
    if settings.google_api_key:
        providers.append("gemini")
    if settings.groq_api_key:
        providers.append("groq")
    # Ollama has no credential to preflight. A connection failure is handled as
    # the final failover result rather than mistakenly treating local mode as
    # unavailable without attempting it.
    providers.append("ollama")
    return tuple(providers)


def select_route(settings: Settings, workload: Workload | None = None) -> LLMRoute:
    """Select a provider using only settings and a caller-declared workload."""
    workload = workload or current_workload()
    if settings.llm_routing_mode == "static":
        return LLMRoute("static", workload, (settings.llm_provider,))

    available = set(_configured_providers(settings))
    if settings.llm_routing_local_only:
        return LLMRoute("dynamic", workload, ("ollama",))

    strict = settings.llm_routing_strict_structured_output or workload == "evaluation"
    preferred: tuple[Provider, ...] = (
        ("groq", "gemini", "ollama") if strict else ("gemini", "groq", "ollama")
    )
    candidates = tuple(provider for provider in preferred if provider in available)
    # Ollama is always present in `available`; this defensive fallback keeps the
    # returned route non-empty if provider configuration grows in the future.
    return LLMRoute("dynamic", workload, candidates or ("ollama",))


@contextmanager
def route_context(settings: Settings, workload: Workload | None = None) -> Iterator[LLMRoute]:
    """Bind one deterministic route for the complete graph invocation."""
    route = select_route(settings, workload)
    token = _ROUTE.set(route)
    try:
        yield route
    finally:
        _ROUTE.reset(token)


@contextmanager
def workload_context(workload: Workload) -> Iterator[None]:
    """Mark a caller-owned workload without exposing it to customer input."""
    token = _WORKLOAD.set(workload)
    try:
        yield
    finally:
        _WORKLOAD.reset(token)


def _nested_provider_error_code(exc: Exception) -> str | None:
    """Read a structured provider code without treating arbitrary 400s alike."""
    body = getattr(exc, "body", None)
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    if not isinstance(error, Mapping):
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None


def is_retryable_provider_error(exc: Exception) -> str | None:
    """Return a safe failure category that permits deterministic failover."""
    # Groq can reject a model-produced native strict-schema response with this
    # code. It is a provider execution failure, not proof that the user request
    # is invalid. Deliberately recognise the exact structured code only: other
    # HTTP 400s remain fail-closed and never trigger a provider switch.
    if _nested_provider_error_code(exc) == "json_validate_failed":
        return "structured_output_failure"

    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status == 429:
        return "rate_limited"
    if isinstance(exc, TimeoutError):
        return "timeout"

    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "timeout" in name or "timed out" in message or "timeout" in message:
        return "timeout"
    if status == 429 or "429" in message or "rate limit" in message:
        return "rate_limited"
    if (
        "providererror" in name
        or "unavailable" in message
        or "connection" in message
        or "transport" in message
        or "not set" in message
    ):
        return "provider_unavailable"
    return None


def advance_route(exc: Exception) -> bool:
    """Move to the next configured dynamic candidate after a retryable failure."""
    route = current_route()
    reason = is_retryable_provider_error(exc)
    if route is None or reason is None or not route.can_fallback:
        return False
    previous = route.provider
    route.index += 1
    route.fallback_reasons.append(f"{previous}:{reason}")
    return True
