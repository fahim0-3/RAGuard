"""Deterministic per-run LLM provider routing and bounded failover.

Routing never reads a customer query and never calls a model. Dynamic mode is
selected entirely from explicit configuration and the caller's workload context.
The mutable route is held in a context variable, making one provider selection
stick across every graph node until a classified provider failure advances it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from src.generation.rate_limit import budget_status

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from src.config.settings import Settings

Provider = Literal["gemini", "groq", "openrouter", "ollama"]
Workload = Literal["normal", "evaluation"]

__all__ = [
    "PROVIDER_HEALTH",
    "LLMRoute",
    "ProviderHealth",
    "Provider",
    "Workload",
    "advance_route",
    "current_provider",
    "current_route",
    "current_workload",
    "is_retryable_provider_error",
    "record_provider_success",
    "retry_after_seconds",
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
    #: Providers passed over because they failed moments ago, as
    #: `provider:category`. Kept apart from `fallback_reasons`, which records
    #: only attempts that were actually made and failed.
    skipped: list[str] = field(default_factory=list)
    #: Whether this route consults and updates the circuit breaker. Decided
    #: once at selection, so a disabled breaker is off for the whole request.
    breaker_enabled: bool = False

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
            "skipped": list(self.skipped),
        }


_ROUTE: ContextVar[LLMRoute | None] = ContextVar("raguard_llm_route", default=None)
#: A request's route is shared by calls that run concurrently (evidence
#: grading and the speculative draft). Advancing it is check-then-act.
_ROUTE_LOCK = threading.Lock()
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
    if settings.openrouter_api_key:
        providers.append("openrouter")
    # Ollama has no credential to preflight. A connection failure is handled as
    # the final failover result rather than mistakenly treating local mode as
    # unavailable without attempting it.
    providers.append("ollama")
    return tuple(providers)


#: How long a provider is passed over after each kind of failure.
#:
#: Measured on this deployment, a Gemini 503 cost about 3 s and an OpenRouter
#: 429 about 2.5 s, and both recurred on every request that reached them. With
#: no memory of the previous failure each request paid the full price again,
#: which is how one failed generation grew into a 30-second request.
#:
#: Structured-output failures are deliberately absent. They depend on the
#: prompt and the answer length, not on whether the provider is healthy, and
#: benching a working provider over one malformed answer would push every later
#: request onto a slower fallback.
_COOLDOWN_S: dict[str, float] = {
    # A complete request can already take 20 to 40 seconds after a hosted
    # provider starts rejecting calls.  The benchmark deliberately spaces
    # requests by 45 seconds, so a one-minute cooldown expired before the
    # next request even began.  Ten minutes avoids repeatedly probing a
    # provider whose free-tier quota is known to be exhausted while retaining
    # a bounded recovery window when the provider does not supply Retry-After.
    "rate_limited": 600.0,
    "provider_unavailable": 120.0,
    "timeout": 30.0,
    # A rejected key does not fix itself between requests.
    "unauthorized": 600.0,
}

#: Upper bound on a provider-supplied retry hint. A daily token quota can
#: report hours; beyond this the provider is simply retried and benched again.
_MAX_RETRY_AFTER_S = 1_800.0


def retry_after_seconds(exc: Exception) -> float | None:
    """The provider's own `Retry-After` hint, when its error carries one.

    Groq's per-day token quota answers 429 with a wait measured in minutes. A
    fixed 60-second cooldown then re-tried a provider that had already said it
    would refuse, and every such retry cost a full round trip. Only the header
    is read; the error body, which can echo request content, is never parsed.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
    except Exception:  # noqa: BLE001 - header containers vary by SDK
        return None
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


class ProviderHealth:
    """Process-wide memory of recent provider failures: a simple circuit breaker.

    A provider that failed for an operational reason is skipped until its
    cooldown expires; the first request after expiry tries it again, and a
    success clears it at once. There is no half-open probing and no failure
    counting, because one 429 or 503 is already the signal: the next request
    seconds later would almost certainly repeat it.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._until: dict[str, tuple[float, str]] = {}

    def record_failure(
        self, provider: str, category: str, retry_after_s: float | None = None
    ) -> None:
        cooldown = _COOLDOWN_S.get(category)
        if cooldown is None:
            return
        if retry_after_s is not None:
            # Honour a longer provider hint; never shorten the floor with it.
            cooldown = max(cooldown, min(retry_after_s, _MAX_RETRY_AFTER_S))
        with self._lock:
            self._until[provider] = (self._clock() + cooldown, category)

    def record_success(self, provider: str) -> None:
        with self._lock:
            self._until.pop(provider, None)

    def cooling(self, provider: str) -> str | None:
        """The failure category if `provider` is still cooling down, else None."""
        with self._lock:
            entry = self._until.get(provider)
            if entry is None:
                return None
            until, category = entry
            if self._clock() >= until:
                del self._until[provider]
                return None
            return category

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Seconds remaining per cooling provider. Operational data only."""
        now = self._clock()
        with self._lock:
            return {
                provider: {"category": category, "remaining_s": round(until - now, 1)}
                for provider, (until, category) in self._until.items()
                if until > now
            }

    def remaining_s(self, provider: str) -> float:
        """Seconds until the provider's cooldown ends; 0 when it is healthy."""
        with self._lock:
            entry = self._until.get(provider)
        return max(0.0, entry[0] - self._clock()) if entry else 0.0

    def reset(self) -> None:
        with self._lock:
            self._until.clear()


PROVIDER_HEALTH = ProviderHealth()


def _estimated_call_tokens(settings: Settings) -> int:
    """What one provider call is assumed to cost before it is made."""
    return int(getattr(settings, "llm_estimated_tokens_per_call", 2_000) or 2_000)


def _breaker_applies(settings: Settings, workload: Workload) -> bool:
    """Evaluation runs keep a fixed route, so their results stay comparable."""
    enabled = bool(getattr(settings, "llm_provider_cooldown_enabled", True))
    return enabled and workload == "normal"


def select_route(settings: Settings, workload: Workload | None = None) -> LLMRoute:
    """Select a provider using only settings and a caller-declared workload."""
    workload = workload or current_workload()
    if settings.llm_routing_mode == "static":
        return LLMRoute("static", workload, (settings.llm_provider,))

    available = set(_configured_providers(settings))
    if settings.llm_routing_local_only:
        return LLMRoute("dynamic", workload, ("ollama",))

    # Groq is first for every hosted workload: its native strict JSON-schema
    # path is the fastest reliable match for RAGuard's answer, grading, and
    # verification contracts. Gemini is the independent managed fallback;
    # OpenRouter is last among hosted providers because its configured free
    # model may vary in structured-output fidelity.
    hosted: tuple[Provider, ...] = ("groq", "gemini", "openrouter")
    candidates = tuple(provider for provider in hosted if provider in available)
    # Ollama joins a hosted route only when explicitly allowed; otherwise an
    # outage of every hosted provider fails fast instead of spending tens of
    # seconds on a CPU model. A deployment with no hosted credential at all is
    # running locally by construction, so Ollama is its route.
    if not candidates or getattr(settings, "llm_routing_local_fallback", False):
        candidates = (*candidates, "ollama")

    skipped: list[str] = []
    breaker = _breaker_applies(settings, workload)
    if breaker:
        healthy: list[Provider] = []
        for provider in candidates:
            # A provider that failed moments ago, or whose published token
            # budget the next call would exhaust, is passed over before it
            # can refuse the request.
            category = PROVIDER_HEALTH.cooling(provider) or budget_status(
                provider, settings, _estimated_call_tokens(settings)
            )
            if category is None:
                healthy.append(provider)
            else:
                skipped.append(f"{provider}:{category}")
        # Every provider cooling at once is an outage. Walking the whole route
        # again would re-pay every known failure; refusing outright would never
        # notice recovery. Probe only the provider due back soonest, so each
        # request costs at most one attempt until something recovers.
        if not healthy:
            probe = min(candidates, key=PROVIDER_HEALTH.remaining_s)
            healthy = [probe]
            skipped = [entry for entry in skipped if not entry.startswith(f"{probe}:")]
        candidates = tuple(healthy)

    return LLMRoute("dynamic", workload, candidates, skipped=skipped, breaker_enabled=breaker)


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

    # Prompt-guided structured output is parsed and validated locally. A
    # malformed object from one provider is equivalent to a provider-side
    # schema rejection, so it may advance the explicit fallback route.
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "validationerror" in name or "outputparser" in name or "output parser" in message:
        return "structured_output_failure"

    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    # Google rejects a bad API key with HTTP 400 INVALID_ARGUMENT rather than
    # 401, so status alone left a rejected Gemini key unclassified: the route
    # stopped there instead of moving to the next provider.
    if "api key not valid" in message or "api_key_invalid" in message:
        return "unauthorized"
    # Credentials are provider-specific. A rejected OpenRouter key must not
    # prevent the independent providers later in the configured route from
    # answering the request.
    if status in {401, 403}:
        return "unauthorized"
    if status == 429:
        return "rate_limited"
    # A 5xx is the provider's own failure. Matching it by status rather than by
    # message wording keeps a 503 classified when its text changes.
    if isinstance(status, int) and 500 <= status <= 599:
        return "provider_unavailable"
    if isinstance(exc, TimeoutError):
        return "timeout"

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


def advance_route(exc: Exception, failed_provider: str | None = None) -> bool:
    """Move to the next configured dynamic candidate after a retryable failure.

    `failed_provider` is the provider the failing call actually used. Two
    calls of one request can fail on the same provider at nearly the same
    moment; the first moves the route on, and without this the second
    would be charged to the provider the route had just moved to, benching
    a healthy provider under someone else's error and skipping it.
    """
    route = current_route()
    reason = is_retryable_provider_error(exc)
    if route is None or reason is None:
        return False
    with _ROUTE_LOCK:
        return _advance_locked(route, reason, exc, failed_provider)


def _advance_locked(
    route: LLMRoute, reason: str, exc: Exception, failed_provider: str | None
) -> bool:
    if failed_provider is not None and failed_provider != route.provider:
        # The route already moved past this provider for a concurrent call.
        # Record its health, but do not advance again: retry on the
        # provider the route now points at.
        if route.mode == "dynamic" and route.breaker_enabled:
            PROVIDER_HEALTH.record_failure(failed_provider, reason, retry_after_seconds(exc))
        return route.mode == "dynamic"
    previous = route.provider
    breaker = route.mode == "dynamic" and route.breaker_enabled
    if breaker:
        # Recorded even for the last candidate: otherwise the final provider's
        # failure is forgotten and every later request pays it again.
        PROVIDER_HEALTH.record_failure(previous, reason, retry_after_seconds(exc))
    if not route.can_fallback:
        return False
    route.fallback_reasons.append(f"{previous}:{reason}")
    route.index += 1
    # Another request may have benched a provider after this route was chosen.
    # Skip it too, but never the last candidate: a final attempt beats none.
    while breaker and route.can_fallback:
        category = PROVIDER_HEALTH.cooling(route.provider)
        if category is None:
            break
        route.skipped.append(f"{route.provider}:{category}")
        route.index += 1
    return True


def record_provider_success(provider: str) -> None:
    """Clear a provider's cooldown as soon as it answers."""
    PROVIDER_HEALTH.record_success(provider)
