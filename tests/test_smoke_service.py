"""Post-deployment smoke checks remain safe and deterministic."""

from __future__ import annotations

from scripts import smoke_service


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("unexpected error response")

    def json(self):
        return self._payload


class FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get(self, url):
        if url.endswith("/health"):
            return FakeResponse(200, {"status": "ok"})
        if url.endswith("/config"):
            return FakeResponse(
                200,
                {
                    "runtime_environment": "production",
                    "runtime_profile": "local_compact",
                },
            )
        return FakeResponse(
            200,
            {
                "status": "ready",
                "checks": {
                    "database": {"chunks_indexed": 22},
                    "embedding_model": {"status": "loaded"},
                    "reranker_model": {"status": "loaded"},
                },
            },
        )


def test_remote_smoke_requires_https():
    assert smoke_service.run_smoke("http://api.example.net", deadline_s=1) == 2


def test_smoke_accepts_a_ready_service(monkeypatch):
    monkeypatch.setattr(smoke_service.httpx, "Client", lambda **kwargs: FakeClient())

    result = smoke_service.run_smoke(
        "https://api.example.net",
        deadline_s=1,
        expected_profile="local_compact",
    )

    assert result == 0


def test_smoke_rejects_an_unexpected_runtime_profile(monkeypatch):
    monkeypatch.setattr(smoke_service.httpx, "Client", lambda **kwargs: FakeClient())

    result = smoke_service.run_smoke(
        "https://api.example.net",
        deadline_s=1,
        expected_profile="full",
    )

    assert result == 3
