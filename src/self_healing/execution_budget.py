"""One deadline and one provider-call allowance for an entire graph request.

Provider SDK retries are disabled while this budget is active. Retries hidden
inside one model invocation cannot be counted or bounded by the graph, so the
self-healing topology owns every retry instead.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock

__all__ = [
    "ExecutionBudget",
    "ExecutionBudgetExceeded",
    "LLMCallPermit",
    "current_budget",
    "ensure_time_remaining",
    "request_budget",
    "reserve_llm_call",
]


class ExecutionBudgetExceeded(RuntimeError):
    """Raised before a provider call that would exceed the request budget."""

    def __init__(self, reason: str, stage: str) -> None:
        self.reason = reason
        self.stage = stage
        super().__init__(f"request budget exhausted: {reason} at {stage}")


@dataclass(frozen=True, slots=True)
class LLMCallPermit:
    """Bounded provider settings for one admitted model invocation."""

    timeout_s: float
    max_retries: int


@dataclass(slots=True)
class ExecutionBudget:
    timeout_s: float
    max_llm_calls: int
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    llm_calls_used: int = 0
    exhausted_reason: str = ""
    exhausted_stage: str = ""
    started_monotonic: float = field(init=False, repr=False)
    deadline_monotonic: float = field(init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.max_llm_calls <= 0:
            raise ValueError("max_llm_calls must be positive")
        self.started_monotonic = self.clock()
        self.deadline_monotonic = self.started_monotonic + float(self.timeout_s)

    def _exhaust(self, reason: str, stage: str) -> None:
        if not self.exhausted_reason:
            self.exhausted_reason = reason
            self.exhausted_stage = stage
        raise ExecutionBudgetExceeded(self.exhausted_reason, self.exhausted_stage)

    def ensure_time_remaining(self, stage: str) -> None:
        """Fail at a graph boundary once the monotonic deadline has passed."""
        with self._lock:
            if self.deadline_monotonic - self.clock() <= 0:
                self._exhaust("deadline", stage)

    def reserve_llm_call(self, stage: str, default_timeout_s: float) -> LLMCallPermit:
        """Atomically admit one real provider call and bound its timeout."""
        with self._lock:
            remaining = self.deadline_monotonic - self.clock()
            if remaining <= 0:
                self._exhaust("deadline", stage)
            if self.llm_calls_used >= self.max_llm_calls:
                self._exhaust("llm_call_limit", stage)

            self.llm_calls_used += 1
            # The provider receives no more time than remains for the request.
            # A tiny positive floor satisfies clients that reject zero timeout.
            timeout_s = max(0.001, min(float(default_timeout_s), remaining))
            return LLMCallPermit(timeout_s=timeout_s, max_retries=0)

    def snapshot(self) -> dict[str, object]:
        elapsed_ms = max(0.0, (self.clock() - self.started_monotonic) * 1000.0)
        return {
            "request_timeout_s": float(self.timeout_s),
            "llm_call_limit": self.max_llm_calls,
            "llm_calls_used": self.llm_calls_used,
            "budget_exhausted": bool(self.exhausted_reason),
            "budget_exhaustion_reason": self.exhausted_reason,
            "budget_exhaustion_stage": self.exhausted_stage,
            "budget_elapsed_ms": round(elapsed_ms, 1),
        }


_CURRENT_BUDGET: ContextVar[ExecutionBudget | None] = ContextVar(
    "raguard_request_budget", default=None
)


def current_budget() -> ExecutionBudget | None:
    return _CURRENT_BUDGET.get()


@contextmanager
def request_budget(budget: ExecutionBudget) -> Iterator[ExecutionBudget]:
    token = _CURRENT_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _CURRENT_BUDGET.reset(token)


def reserve_llm_call(
    stage: str,
    *,
    default_timeout_s: float,
    default_max_retries: int = 0,
) -> LLMCallPermit:
    """Reserve against the active request, or preserve standalone defaults."""
    budget = current_budget()
    if budget is None:
        return LLMCallPermit(
            timeout_s=float(default_timeout_s),
            max_retries=int(default_max_retries),
        )
    return budget.reserve_llm_call(stage, default_timeout_s)


def ensure_time_remaining(stage: str) -> None:
    budget = current_budget()
    if budget is not None:
        budget.ensure_time_remaining(stage)
