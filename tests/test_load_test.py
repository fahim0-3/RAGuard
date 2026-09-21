from __future__ import annotations

import pytest

from scripts.load_test import Sample, _is_loopback, summarise


def test_load_tool_rejects_non_loopback_addresses():
    assert _is_loopback("http://127.0.0.1:8000") is True
    assert _is_loopback("http://localhost:8000") is True
    assert _is_loopback("https://api.example.com") is False


def test_load_summary_has_stable_aggregate_metrics_only():
    summary = summarise(
        [
            Sample(100.0, 200, "answer"),
            Sample(300.0, 503, ""),
            Sample(200.0, 200, "abstain"),
        ]
    )

    assert summary["requests"] == 3
    assert summary["successful_http_responses"] == 2
    assert summary["http_statuses"] == {"200": 2, "503": 1}
    assert summary["latency_ms"] == {"mean": 200.0, "p50": 200.0, "p95": 300.0, "max": 300.0}


@pytest.mark.parametrize("url", ["", "ftp://localhost", "http://example.com"])
def test_loopback_parser_rejects_invalid_or_remote_urls(url):
    assert _is_loopback(url) is False
