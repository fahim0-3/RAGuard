"""Tests for safe, actionable managed-database connection diagnostics."""

from __future__ import annotations

from scripts.check_remote_db import connection_failure_message


def test_permission_denied_socket_error_is_classified_without_endpoint_details():
    message = connection_failure_message(
        "connection failed: connection to server at \"203.0.113.7\", port 5432 failed: "
        "Permission denied (0x0000271D/10013)"
    )

    assert "outbound TCP" in message
    assert "203.0.113.7" not in message
    assert "5432" in message


def test_timeout_is_distinguished_from_network_policy_blocking():
    message = connection_failure_message("connection timed out")

    assert "timed out" in message.lower()
    assert "blocked by local network policy" not in message


def test_unknown_error_stays_secret_safe_and_actionable():
    message = connection_failure_message("password=not-for-output host=private.example")

    assert message == "Database connection failed. Run the read-only check from an environment with database access."
