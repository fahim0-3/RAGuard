"""Provider factory for chat models.

Nothing else in RAGuard imports a provider SDK directly. Switching from Gemini
to a local Ollama model is a change to `LLM_PROVIDER` in `.env`, not a code
change, which keeps the "same pipeline, different backend" comparison honest
for the evaluation chapter. Answer-generation business logic in `answer_chain`
never names a provider.

Three logical roles are exposed so that cost and determinism can be tuned
independently:

- `generator`  answers the user, at the configured temperature (default 0)
- `rewriter`   produces query variants, warmer so the variants differ
- `judge`      scores metrics in nightly CI, cheapest model available

Failure policy: importing this module must never fail, and constructing a model
must never leak a key. Missing credentials raise `LLMProviderError` with an
instruction, at call time, not at import time — the deterministic test suite
imports this module constantly and must not need a key to do so.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

from src.config import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "LLMProviderError",
    "Role",
    "get_chat_model",
    "provider_config",
    "reset_model_cache",
]

Role = Literal["generator", "rewriter", "judge"]

#: Added to the configured base temperature. Rewriting needs diversity; a
#: generator and a judge must be reproducible.
_ROLE_TEMPERATURE_OFFSET: dict[str, float] = {
    "generator": 0.0,
    "rewriter": 0.3,
    "judge": 0.0,
}


class LLMProviderError(RuntimeError):
    """Raised when the configured provider cannot be constructed."""


def _temperature_for(role: Role) -> float:
    settings = get_settings()
    base = float(settings.llm_temperature)
    return max(0.0, min(2.0, base + _ROLE_TEMPERATURE_OFFSET.get(role, 0.0)))


def model_name_for(role: Role) -> str:
    """Resolve the model ID for a role, honouring the provider-agnostic override."""
    settings = get_settings()
    if settings.llm_provider == "ollama":
        return settings.llm_model or settings.ollama_model
    if role == "judge":
        return settings.gemini_judge_model
    return settings.llm_model or settings.gemini_model


def _build_gemini(
    role: Role,
    *,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> Any:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LLMProviderError(
            "langchain-google-genai is not installed. Run: pip install langchain-google-genai"
        ) from exc

    settings = get_settings()
    if not settings.google_api_key:
        # Deliberately no value interpolation anywhere near the key.
        raise LLMProviderError(
            "GOOGLE_API_KEY is not set. Obtain a key from Google AI Studio and add "
            "it to .env, or set LLM_PROVIDER=ollama to run fully offline."
        )

    return ChatGoogleGenerativeAI(
        model=model_name_for(role),
        google_api_key=settings.google_api_key,
        temperature=_temperature_for(role),
        max_output_tokens=settings.llm_max_output_tokens,
        timeout=timeout_s if timeout_s is not None else settings.llm_request_timeout_s,
        max_retries=max_retries if max_retries is not None else settings.llm_max_retries,
    )


def _build_ollama(
    role: Role,
    *,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> Any:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LLMProviderError(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from exc

    settings = get_settings()
    effective_timeout = (
        timeout_s if timeout_s is not None else settings.llm_request_timeout_s
    )
    return ChatOllama(
        model=model_name_for(role),
        base_url=settings.ollama_base_url,
        temperature=_temperature_for(role),
        num_predict=settings.llm_max_output_tokens,
        # ChatOllama forwards these values to its sync and async httpx clients.
        client_kwargs={"timeout": effective_timeout},
    )


_BUILDERS = {"gemini": _build_gemini, "ollama": _build_ollama}


@lru_cache(maxsize=32)
def get_chat_model(
    role: Role = "generator",
    *,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> Any:
    """Return a role model, optionally bounded by a request-level permit."""
    settings = get_settings()
    builder = _BUILDERS.get(settings.llm_provider)
    if builder is None:
        raise LLMProviderError(
            f"Unknown LLM_PROVIDER {settings.llm_provider!r}; "
            f"expected one of {sorted(_BUILDERS)}"
        )
    logger.info("Building %s model for role=%s", settings.llm_provider, role)
    return builder(role, timeout_s=timeout_s, max_retries=max_retries)


def provider_config(role: Role = "generator") -> dict[str, Any]:
    """Configuration block for reports. Never contains the key itself."""
    settings = get_settings()
    return {
        "provider": settings.llm_provider,
        "model": model_name_for(role),
        "role": role,
        "temperature": _temperature_for(role),
        "max_output_tokens": settings.llm_max_output_tokens,
        "timeout_s": settings.llm_request_timeout_s,
        "max_retries": settings.llm_max_retries,
        "credentials_present": bool(settings.google_api_key)
        if settings.llm_provider == "gemini"
        else True,
    }


def reset_model_cache() -> None:
    """Clear cached models. Call after changing settings in tests."""
    get_chat_model.cache_clear()
