"""Diagnose every configured LLM provider through RAGuard's own provider stack.

For each provider the script runs the real evidence-grading structured-output
call (the same `build_json_chain` path the graph uses) in static routing mode,
then reports whether it worked, how long it took, and how a failure is
classified: quota, authentication, invalid model, rate limit, timeout,
availability, or unsupported structured output.

It never prints a key, a prompt, a provider message body, or policy text; only
status codes, provider error codes, categories and timings.

Usage:
    python scripts/diagnose_providers.py            # every configured provider
    python scripts/diagnose_providers.py --only groq,gemini
    python scripts/diagnose_providers.py --no-raw   # skip provider account probes
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: One tiny, non-sensitive passage; the point is the structured contract.
QUESTION = "How long do card refunds take?"
PASSAGE = "Credit and debit cards: 5 to 7 business days."


@dataclass
class Probe:
    provider: str
    model: str
    ok: bool
    seconds: float
    structured: bool = False
    category: str = ""
    status: str = ""
    code: str = ""
    diagnosis: str = ""


def _error_facts(exc: Exception) -> tuple[str, str]:
    """Status and provider error code, without the message body."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None) or ""
    body = getattr(exc, "body", None)
    code = ""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or error.get("status") or error.get("type") or "")
    return str(status), code


def _diagnose(category: str, status: str, code: str, exc_name: str) -> str:
    if category == "unauthorized" or status in {"401", "403"}:
        return "authentication (key rejected or lacks access)"
    if status == "404" or "not_found" in code.lower():
        return "invalid model id"
    if category == "rate_limited" or status == "429":
        return "rate limit or quota (see account probe below)"
    if category == "timeout":
        return "timeout"
    if category == "structured_output_failure":
        return "unsupported or malformed structured output"
    if category == "provider_unavailable" or status.startswith("5"):
        return "provider availability (5xx or connection)"
    return f"other ({exc_name})"


def probe(provider: str, model_env: dict[str, str]) -> Probe:
    for key, value in {"LLM_ROUTING_MODE": "static", "LLM_PROVIDER": provider, **model_env}.items():
        os.environ[key] = value

    from src.config import get_settings
    from src.generation import llm_factory
    from src.generation.llm_routing import is_retryable_provider_error
    from src.self_healing.evidence_grader import _build_grader_chain

    get_settings.cache_clear()
    llm_factory.reset_model_cache()
    model = llm_factory.model_name_for("judge", provider)

    started = time.perf_counter()
    try:
        raw = _build_grader_chain(timeout_s=20, max_retries=0).invoke(
            {"question": QUESTION, "context": f"[1] {PASSAGE}"}
        )
    except Exception as exc:  # noqa: BLE001 - every failure is the finding
        category = is_retryable_provider_error(exc) or "not_retryable"
        status, code = _error_facts(exc)
        return Probe(
            provider,
            model,
            False,
            round(time.perf_counter() - started, 2),
            category=category,
            status=status,
            code=code,
            diagnosis=_diagnose(category, status, code, type(exc).__name__),
        )
    structured = isinstance(raw, dict) and {"relevant", "sufficient"} <= set(raw)
    return Probe(
        provider,
        model,
        True,
        round(time.perf_counter() - started, 2),
        structured=structured,
        diagnosis="healthy" if structured else "answered, but not the grading schema",
    )


def gemini_account_probe() -> list[str]:
    """Is the configured model listed, and do sibling models answer?"""
    import httpx

    from src.config import get_settings

    settings = get_settings()
    if not settings.google_api_key:
        return ["gemini: no key configured"]
    base = "https://generativelanguage.googleapis.com/v1beta"
    lines: list[str] = []
    listed = httpx.get(f"{base}/models", params={"key": settings.google_api_key}, timeout=20)
    models = listed.json().get("models", [])
    names = [m.get("name", "").removeprefix("models/") for m in models]
    flash = sorted(
        m.get("name", "").removeprefix("models/")
        for m in models
        if "flash" in m.get("name", "")
        and "generateContent" in (m.get("supportedGenerationMethods") or [])
    )
    lines.append(
        f"gemini: models endpoint HTTP {listed.status_code}; configured model listed: "
        f"{settings.gemini_model in names}"
    )
    candidates = [settings.gemini_model] + [n for n in flash if n != settings.gemini_model][:4]
    for name in candidates:
        started = time.perf_counter()
        response = httpx.post(
            f"{base}/models/{name}:generateContent",
            params={"key": settings.google_api_key},
            json={"contents": [{"parts": [{"text": "Reply with ok."}]}]},
            timeout=30,
        )
        status = (
            response.json().get("error", {}).get("status", "")
            if response.status_code != 200
            else "OK"
        )
        lines.append(
            f"gemini: {name:<32} HTTP {response.status_code} {status} "
            f"({time.perf_counter() - started:.2f}s)"
        )
    return lines


def openrouter_account_probe() -> list[str]:
    """Account limits, and whether a 429 is the account's or the upstream model's."""
    import httpx

    from src.config import get_settings

    settings = get_settings()
    if not settings.openrouter_api_key:
        return ["openrouter: no key configured"]
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
    lines: list[str] = []
    key = httpx.get("https://openrouter.ai/api/v1/key", headers=headers, timeout=20)
    if key.status_code == 200:
        data = key.json().get("data", {})
        lines.append(
            "openrouter: key valid; free tier: "
            f"{data.get('is_free_tier')}; limit: {data.get('limit')}; "
            f"usage: {data.get('usage')}; rate_limit: {data.get('rate_limit')}"
        )
    else:
        lines.append(f"openrouter: key endpoint HTTP {key.status_code}")
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json={
            "model": settings.openrouter_model,
            "messages": [{"role": "user", "content": "Reply with ok."}],
            "max_tokens": 8,
        },
        timeout=30,
    )
    detail = ""
    if response.status_code == 429:
        text = response.text.lower()
        if "free-models-per-day" in text or "per-day" in text:
            detail = "account daily free-model quota"
        elif "upstream" in text or "temporarily rate-limited" in text:
            detail = "upstream provider rate limit on this free model"
        else:
            detail = "rate limited"
    lines.append(
        f"openrouter: {settings.openrouter_model} HTTP {response.status_code} {detail}".rstrip()
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="")
    parser.add_argument("--no-raw", action="store_true")
    args = parser.parse_args()

    from src.config import get_settings

    settings = get_settings()
    plan: list[tuple[str, dict[str, str]]] = []
    if settings.groq_api_key:
        for model in dict.fromkeys(
            [settings.groq_judge_model, "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
        ):
            plan.append(("groq", {"GROQ_JUDGE_MODEL": model}))
    if settings.google_api_key:
        plan.append(("gemini", {}))
    if settings.openrouter_api_key:
        plan.append(("openrouter", {}))
    plan.append(("ollama", {}))
    if args.only:
        wanted = set(args.only.split(","))
        plan = [item for item in plan if item[0] in wanted]

    results = [probe(provider, env) for provider, env in plan]
    print(f"{'provider':<11} {'model':<36} {'ok':<3} {'secs':>6}  diagnosis")
    for r in results:
        detail = f" [category={r.category} status={r.status} code={r.code}]" if not r.ok else ""
        print(
            f"{r.provider:<11} {r.model:<36} {'yes' if r.ok else 'no':<3} {r.seconds:>6.2f}  {r.diagnosis}{detail}"
        )

    if not args.no_raw:
        wanted = set(args.only.split(",")) if args.only else {"gemini", "openrouter"}
        print()
        if "gemini" in wanted:
            for line in gemini_account_probe():
                print(line)
        if "openrouter" in wanted:
            for line in openrouter_account_probe():
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
