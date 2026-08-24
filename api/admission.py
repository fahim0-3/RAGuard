"""Admission control for expensive public query work.

The local implementation supports a one-process developer setup. Production
instances use :class:`RedisQueryAdmission`: an atomic Lua script applies a
shared sliding rate window and leased concurrency slots across API replicas.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Protocol

from redis import Redis
from redis.exceptions import RedisError

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

_ACQUIRE_SCRIPT = """
local now = redis.call('TIME')
local now_ms = now[1] * 1000 + math.floor(now[2] / 1000)
local cutoff = now_ms - 60000
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', cutoff)
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now_ms)

if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[2]) then
  return 1
end
if redis.call('ZCARD', KEYS[2]) >= tonumber(ARGV[3]) then
  return 2
end

redis.call('ZADD', KEYS[1], now_ms, ARGV[1])
redis.call('PEXPIRE', KEYS[1], 60000)
redis.call('ZADD', KEYS[2], now_ms + tonumber(ARGV[4]), ARGV[1])
redis.call('PEXPIRE', KEYS[2], tonumber(ARGV[4]))
return 0
"""

_RELEASE_SCRIPT = "return redis.call('ZREM', KEYS[1], ARGV[1])"


@dataclass(frozen=True)
class AdmissionLease:
    """A successful admission's unique release token, or a rejection reason."""

    token: str | None = None
    reason: str | None = None


class AdmissionController(Protocol):
    def acquire(
        self,
        client_ip: str,
        *,
        max_concurrency: int,
        requests_per_minute: int,
        now: float | None = None,
    ) -> AdmissionLease: ...

    def release(self, lease: AdmissionLease | None = None) -> None: ...

    def reset(self) -> None: ...


class QueryAdmission:
    """Thread-safe local concurrency and sliding-window rate guard."""

    def __init__(self) -> None:
        self._active: set[str] = set()
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def acquire(
        self,
        client_ip: str,
        *,
        max_concurrency: int,
        requests_per_minute: int,
        now: float | None = None,
    ) -> AdmissionLease:
        """Create a local lease, returning a bounded public rejection reason."""
        now = time.monotonic() if now is None else now
        cutoff = now - 60.0

        with self._lock:
            window = self._requests[client_ip]
            while window and window[0] <= cutoff:
                window.popleft()

            if len(window) >= requests_per_minute:
                return AdmissionLease(reason="rate_limited")
            if len(self._active) >= max_concurrency:
                return AdmissionLease(reason="busy")

            token = secrets.token_urlsafe(24)
            window.append(now)
            self._active.add(token)
            return AdmissionLease(token=token)

    def try_acquire(
        self,
        client_ip: str,
        *,
        max_concurrency: int,
        requests_per_minute: int,
        now: float | None = None,
    ) -> str | None:
        """Compatibility helper for callers that only need a rejection reason."""
        return self.acquire(
            client_ip,
            max_concurrency=max_concurrency,
            requests_per_minute=requests_per_minute,
            now=now,
        ).reason

    def release(self, lease: AdmissionLease | None = None) -> None:
        with self._lock:
            if lease and lease.token:
                self._active.discard(lease.token)
            elif self._active:
                # Backward-compatible no-argument cleanup for local callers.
                self._active.pop()

    def reset(self) -> None:
        """Clear local counters for an explicit application/test reset."""
        with self._lock:
            self._active.clear()
            self._requests.clear()


class RedisQueryAdmission:
    """Atomic shared admission control with expiring concurrency leases.

    The Redis server owns time, so clock drift between API workers cannot
    bypass a window. A lease expiry recovers capacity after a worker crash;
    configure it longer than the maximum allowed request duration.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        namespace: str,
        lease_seconds: int,
        client: Redis | None = None,
    ) -> None:
        self._namespace = namespace.rstrip(":")
        # Keep both Lua keys in one hash slot when Redis Cluster is used.
        cluster_tag = hashlib.sha256(self._namespace.encode("utf-8")).hexdigest()[:16]
        self._key_prefix = f"{self._namespace}:{{{cluster_tag}}}"
        self._lease_ms = lease_seconds * 1_000
        self._client = client or Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
            health_check_interval=30,
        )

    def acquire(
        self,
        client_ip: str,
        *,
        max_concurrency: int,
        requests_per_minute: int,
        now: float | None = None,  # noqa: ARG002 - Redis supplies authoritative time.
    ) -> AdmissionLease:
        token = secrets.token_urlsafe(24)
        peer_hash = hashlib.sha256(client_ip.encode("utf-8")).hexdigest()
        rate_key = f"{self._key_prefix}:rate:{peer_hash}"
        active_key = f"{self._key_prefix}:active"
        try:
            result = int(
                self._client.eval(
                    _ACQUIRE_SCRIPT,
                    2,
                    rate_key,
                    active_key,
                    token,
                    max_concurrency,
                    requests_per_minute,
                    self._lease_ms,
                )
            )
        except RedisError:
            logger.exception("Redis admission backend unavailable")
            return AdmissionLease(reason="admission_backend_unavailable")

        if result == 1:
            return AdmissionLease(reason="rate_limited")
        if result == 2:
            return AdmissionLease(reason="busy")
        return AdmissionLease(token=token)

    def release(self, lease: AdmissionLease | None = None) -> None:
        if not lease or not lease.token:
            return
        try:
            self._client.eval(_RELEASE_SCRIPT, 1, f"{self._key_prefix}:active", lease.token)
        except RedisError:
            # The lease expires automatically; do not turn a completed query
            # into a client error merely because cleanup could not run.
            logger.warning("Redis admission lease release failed")

    def reset(self) -> None:
        """No-op: production reset must never delete shared limiter state."""


def build_admission(settings: Settings) -> AdmissionController:
    """Build the configured backend once during application import/startup."""
    if settings.admission_backend == "redis":
        return RedisQueryAdmission(
            settings.admission_redis_url,
            namespace=settings.admission_redis_namespace,
            lease_seconds=settings.admission_lease_seconds,
        )
    return QueryAdmission()


query_admission = build_admission(get_settings())
