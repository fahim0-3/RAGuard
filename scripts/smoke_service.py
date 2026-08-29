"""Post-deployment liveness, configuration, and readiness smoke check."""

from __future__ import annotations

import argparse
import sys
import time
from urllib.parse import urlparse

import httpx


def run_smoke(
    base_url: str,
    *,
    deadline_s: float,
    expected_profile: str | None = None,
    allow_http: bool = False,
) -> int:
    base_url = base_url.rstrip("/")
    parsed = urlparse(base_url)
    if not parsed.hostname or (parsed.scheme != "https" and not allow_http):
        print("ERROR base URL must use HTTPS (or pass --allow-http for local checks).")
        return 2

    try:
        with httpx.Client(timeout=10, follow_redirects=False) as client:
            health = client.get(f"{base_url}/health")
            health.raise_for_status()
            if health.json().get("status") != "ok":
                raise ValueError("unexpected health contract")

            config = client.get(f"{base_url}/config")
            config.raise_for_status()
            configuration = config.json()
            profile = configuration.get("runtime_profile")
            environment = configuration.get("runtime_environment")
            if not allow_http and environment != "production":
                print("ERROR the remote service is not running in production mode.")
                return 6
            if expected_profile and profile != expected_profile:
                print(f"ERROR runtime profile is {profile!r}; expected {expected_profile!r}.")
                return 3

            deadline = time.monotonic() + deadline_s
            last_status = "unknown"
            while time.monotonic() < deadline:
                response = client.get(f"{base_url}/ready")
                payload = response.json()
                last_status = payload.get("status", "unknown")
                if response.status_code == 200 and last_status == "ready":
                    checks = payload.get("checks", {})
                    print(
                        "Smoke check passed: "
                        f"environment={environment}, profile={profile}, "
                        f"chunks={checks.get('database', {}).get('chunks_indexed')}, "
                        f"embedding={checks.get('embedding_model', {}).get('status')}, "
                        f"reranker={checks.get('reranker_model', {}).get('status')}"
                    )
                    return 0
                time.sleep(2)
    except (httpx.HTTPError, ValueError) as exc:
        print(f"ERROR service smoke check failed ({type(exc).__name__}).")
        return 4

    print(f"ERROR service did not become ready before the deadline (status={last_status}).")
    return 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--deadline-s", type=float, default=300)
    parser.add_argument("--expected-profile", choices=("full", "local_compact"))
    parser.add_argument("--allow-http", action="store_true")
    args = parser.parse_args(argv)
    return run_smoke(
        args.base_url,
        deadline_s=args.deadline_s,
        expected_profile=args.expected_profile,
        allow_http=args.allow_http,
    )


if __name__ == "__main__":
    sys.exit(main())
