"""Client-side accounting of each provider's published token limits.

A free tier refuses a request that exceeds its budget, and that refusal is
expensive: a round trip, a failover, and on a bad day a cascade through every
remaining provider. The budget is knowable in advance, so this module keeps a
local tally of what has been spent and lets routing pick a provider that still
has room, instead of discovering the limit by being rejected.

The tally is per API process. That is exact for a single-process deployment and
an underestimate when several processes share one key, which is the safe
direction: a shared key exhausts sooner than any one process believes, and the
circuit breaker still catches the 429 that results.

Windows are rolling rather than aligned to a clock boundary. Groq reports its
minute limit as "try again in 3.2s", which is rolling behaviour, and a rolling
day is the conservative reading of a daily cap.

Nothing here reads a prompt, an answer, or a key: only token counts and times.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "PROVIDER_BUDGETS",
    "TokenBudget",
    "budget_snapshot",
    "budget_status",
    "configure_budgets",
    "record_usage",
    "reset_budgets",
]

_MINUTE_S = 60.0
_DAY_S = 86_400.0


@dataclass(frozen=True)
class BudgetLimits:
    """Published limits for one provider. Zero means "no limit known"."""

    tokens_per_minute: int = 0
    tokens_per_day: int = 0


class TokenBudget:
    """A rolling tally of tokens spent against one provider's limits."""

    def __init__(self, limits: BudgetLimits, clock: Callable[[], float] = time.monotonic) -> None:
        self.limits = limits
        self._clock = clock
        self._lock = threading.Lock()
        self._events: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - _DAY_S
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _spent(self, now: float, window_s: float) -> int:
        cutoff = now - window_s
        return sum(tokens for at, tokens in self._events if at >= cutoff)

    def record(self, tokens: int) -> None:
        """Add the tokens one completed call actually cost."""
        if tokens <= 0:
            return
        with self._lock:
            now = self._clock()
            self._prune(now)
            self._events.append((now, int(tokens)))

    def exceeded_by(self, estimated_tokens: int) -> str | None:
        """Which window a call of this size would exhaust, or None if it fits.

        The estimate is charged against the remaining budget before the call,
        so the last request that would fit is allowed and the one after it is
        routed elsewhere. An unknown limit (zero) never blocks.
        """
        with self._lock:
            now = self._clock()
            self._prune(now)
            if self.limits.tokens_per_day and (
                self._spent(now, _DAY_S) + estimated_tokens > self.limits.tokens_per_day
            ):
                return "token_budget_day"
            if self.limits.tokens_per_minute and (
                self._spent(now, _MINUTE_S) + estimated_tokens > self.limits.tokens_per_minute
            ):
                return "token_budget_minute"
        return None

    def snapshot(self) -> dict[str, int]:
        """Remaining headroom in each window. Operational counters only."""
        with self._lock:
            now = self._clock()
            self._prune(now)
            minute_spent = self._spent(now, _MINUTE_S)
            day_spent = self._spent(now, _DAY_S)
        return {
            "tokens_per_minute": self.limits.tokens_per_minute,
            "tokens_used_last_minute": minute_spent,
            "tokens_remaining_this_minute": max(0, self.limits.tokens_per_minute - minute_spent)
            if self.limits.tokens_per_minute
            else -1,
            "tokens_per_day": self.limits.tokens_per_day,
            "tokens_used_last_day": day_spent,
            "tokens_remaining_today": max(0, self.limits.tokens_per_day - day_spent)
            if self.limits.tokens_per_day
            else -1,
        }

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


#: Per-provider budgets, rebuilt whenever settings change.
PROVIDER_BUDGETS: dict[str, TokenBudget] = {}
_registry_lock = threading.Lock()
_configured_for: tuple | None = None


def _limits_from(settings: object) -> dict[str, BudgetLimits]:
    """Read published limits from settings. Only Groq publishes token limits
    RAGuard can count reliably; the others are left unlimited here and remain
    covered by the circuit breaker when they do refuse."""
    return {
        "groq": BudgetLimits(
            tokens_per_minute=int(getattr(settings, "groq_tokens_per_minute", 0) or 0),
            tokens_per_day=int(getattr(settings, "groq_tokens_per_day", 0) or 0),
        )
    }


def configure_budgets(settings: object) -> None:
    """Build the registry once per distinct limit configuration."""
    global _configured_for
    limits = _limits_from(settings)
    signature = tuple(
        sorted(
            (name, value.tokens_per_minute, value.tokens_per_day) for name, value in limits.items()
        )
    )
    with _registry_lock:
        if _configured_for == signature and PROVIDER_BUDGETS:
            return
        for name, value in limits.items():
            existing = PROVIDER_BUDGETS.get(name)
            if existing is None:
                PROVIDER_BUDGETS[name] = TokenBudget(value)
            else:
                # Keep the tally; only the ceiling changed.
                existing.limits = value
        _configured_for = signature


def budget_status(provider: str, settings: object, estimated_tokens: int) -> str | None:
    """Why `provider` should be passed over right now, or None to use it."""
    if not getattr(settings, "llm_token_budget_enabled", True):
        return None
    configure_budgets(settings)
    budget = PROVIDER_BUDGETS.get(provider)
    if budget is None:
        return None
    return budget.exceeded_by(estimated_tokens)


def record_usage(provider: str, tokens: int) -> None:
    """Charge a completed call to its provider.

    The registry is built on demand rather than assumed: routing normally
    builds it first, but a call made outside a routed request must still be
    counted, or the budget would under-report and let a later request through
    that the provider then refuses.
    """
    if tokens <= 0:
        return
    budget = PROVIDER_BUDGETS.get(provider)
    if budget is None:
        from src.config import get_settings

        configure_budgets(get_settings())
        budget = PROVIDER_BUDGETS.get(provider)
    if budget is not None:
        budget.record(tokens)


def budget_snapshot() -> dict[str, dict[str, int]]:
    return {name: budget.snapshot() for name, budget in PROVIDER_BUDGETS.items()}


def reset_budgets() -> None:
    """Clear every tally. Used by tests."""
    global _configured_for
    with _registry_lock:
        for budget in PROVIDER_BUDGETS.values():
            budget.reset()
        PROVIDER_BUDGETS.clear()
        _configured_for = None
