"""Dependency-free container liveness probe."""

from __future__ import annotations

import os
import urllib.request


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    path = os.getenv("HEALTHCHECK_PATH", "/health")
    if not path.startswith("/"):
        raise SystemExit(1)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as response:
        if response.status != 200:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
