"""Unit coverage for local and Redis-backed query admission guards."""

from redis.exceptions import RedisError

from api.admission import AdmissionLease, QueryAdmission, RedisQueryAdmission, build_admission
from src.config import Settings


def test_admission_limits_concurrent_work():
    guard = QueryAdmission()

    assert guard.try_acquire("127.0.0.1", max_concurrency=1, requests_per_minute=10) is None
    assert guard.try_acquire("127.0.0.2", max_concurrency=1, requests_per_minute=10) == "busy"

    guard.release()
    assert guard.try_acquire("127.0.0.2", max_concurrency=1, requests_per_minute=10) is None


def test_admission_limits_each_peer_within_a_minute():
    guard = QueryAdmission()

    assert guard.try_acquire("127.0.0.1", max_concurrency=3, requests_per_minute=2, now=100.0) is None
    guard.release()
    assert guard.try_acquire("127.0.0.1", max_concurrency=3, requests_per_minute=2, now=101.0) is None
    guard.release()
    assert guard.try_acquire("127.0.0.1", max_concurrency=3, requests_per_minute=2, now=102.0) == "rate_limited"
    assert guard.try_acquire("127.0.0.2", max_concurrency=3, requests_per_minute=2, now=102.0) is None


class FakeRedis:
    def __init__(self, *results: int) -> None:
        self.results = list(results)
        self.calls: list[tuple[object, ...]] = []

    def eval(self, *args):
        self.calls.append(args)
        return self.results.pop(0)


def test_redis_admission_uses_one_atomic_lease_and_hashed_peer_key():
    redis = FakeRedis(0, 1)
    guard = RedisQueryAdmission(
        "redis://unused",
        namespace="test:admission",
        lease_seconds=300,
        client=redis,  # type: ignore[arg-type]
    )

    lease = guard.acquire("203.0.113.7", max_concurrency=4, requests_per_minute=30)
    guard.release(lease)

    assert lease.token
    acquire_call, release_call = redis.calls
    assert acquire_call[1] == 2
    assert "203.0.113.7" not in str(acquire_call)
    assert acquire_call[2].startswith("test:admission:{")
    assert ":rate:" in acquire_call[2]
    assert acquire_call[3].startswith("test:admission:{")
    assert acquire_call[3].endswith(":active")
    assert release_call[1:] == (1, acquire_call[3], lease.token)


def test_redis_admission_returns_safe_rejection_codes():
    redis = FakeRedis(1, 2)
    guard = RedisQueryAdmission("redis://unused", namespace="test", lease_seconds=300, client=redis)  # type: ignore[arg-type]

    assert guard.acquire("127.0.0.1", max_concurrency=1, requests_per_minute=1).reason == "rate_limited"
    assert guard.acquire("127.0.0.2", max_concurrency=1, requests_per_minute=1).reason == "busy"


def test_redis_admission_fails_closed_when_the_backend_is_unavailable():
    class UnavailableRedis:
        def eval(self, *args):
            raise RedisError("connection refused")

    guard = RedisQueryAdmission(
        "redis://unused", namespace="test", lease_seconds=300, client=UnavailableRedis()  # type: ignore[arg-type]
    )

    assert guard.acquire("127.0.0.1", max_concurrency=1, requests_per_minute=1) == AdmissionLease(
        reason="admission_backend_unavailable"
    )


def test_build_admission_selects_redis_without_connecting():
    controller = build_admission(Settings(_env_file=None, admission_backend="redis"))  # type: ignore[call-arg]

    assert isinstance(controller, RedisQueryAdmission)
