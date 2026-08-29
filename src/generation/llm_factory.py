"""Provider factory for chat models.

Nothing else in RAGuard imports a provider SDK directly. Switching among Gemini,
Groq, and local Ollama is a change to `LLM_PROVIDER` in `.env`, not a code
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
from src.generation.llm_routing import advance_route, current_provider

logger = logging.getLogger(__name__)

__all__ = [
    "LLMProviderError",
    "Role",
    "get_chat_model",
    "build_json_chain",
    "get_structured_chat_model",
    "provider_config",
    "reset_model_cache",
    "uses_native_structured_output",
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


def _selected_provider() -> str:
    """Resolve the configured provider without requiring route settings in test doubles."""
    settings = get_settings()
    if getattr(settings, "llm_routing_mode", "static") == "static":
        return settings.llm_provider
    return current_provider(settings)


def model_name_for(role: Role, provider: str | None = None) -> str:
    """Resolve the model ID for a role.

    ``LLM_MODEL`` is a legacy static-provider override.  A model identifier is
    generally provider-specific (for example, a Groq model name is not a
    Gemini model name), so it must not follow a dynamic route into a fallback
    provider.  Dynamic routing always uses the selected provider's configured
    model, preserving a usable bounded fallback chain.
    """
    settings = get_settings()
    provider = provider or _selected_provider()
    model_override = settings.llm_model if settings.llm_routing_mode == "static" else ""
    if provider == "ollama":
        return model_override or settings.ollama_model
    if provider == "groq":
        if role == "judge":
            return settings.groq_judge_model
        return model_override or settings.groq_model
    if provider == "openrouter":
        return settings.openrouter_model
    if role == "judge":
        return settings.gemini_judge_model
    return model_override or settings.gemini_model


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
    effective_timeout = timeout_s if timeout_s is not None else settings.llm_request_timeout_s
    return ChatOllama(
        model=model_name_for(role),
        base_url=settings.ollama_base_url,
        temperature=_temperature_for(role),
        num_predict=settings.llm_max_output_tokens,
        # ChatOllama forwards these values to its sync and async httpx clients.
        client_kwargs={"timeout": effective_timeout},
    )


def _build_groq(
    role: Role,
    *,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> Any:
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LLMProviderError(
            "langchain-groq is not installed. Run: pip install langchain-groq"
        ) from exc

    settings = get_settings()
    if not settings.groq_api_key:
        raise LLMProviderError(
            "GROQ_API_KEY is not set. Create a Groq API key, or select Gemini or Ollama."
        )

    # ChatGroq applies bounded exponential backoff to its retryable provider
    # failures (including 429). A graph request passes max_retries=0 through
    # its execution budget, keeping every actual provider request visible to
    # the graph's call allowance.
    retries = settings.groq_max_retries if max_retries is None else max_retries
    retries = min(retries, settings.groq_max_retries)
    return ChatGroq(
        model=model_name_for(role),
        api_key=settings.groq_api_key,
        temperature=_temperature_for(role),
        max_tokens=settings.llm_max_output_tokens,
        timeout=timeout_s if timeout_s is not None else settings.llm_request_timeout_s,
        max_retries=retries,
    )


def _build_openrouter(
    role: Role,
    *,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> Any:
    """Build OpenRouter through LangChain's OpenAI-compatible client."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise LLMProviderError(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        ) from exc

    settings = get_settings()
    if not settings.openrouter_api_key:
        raise LLMProviderError(
            "OPENROUTER_API_KEY is not set. Create an OpenRouter key, or select Gemini, Groq, or Ollama."
        )

    return ChatOpenAI(
        model=model_name_for(role),
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=_temperature_for(role),
        max_tokens=settings.llm_max_output_tokens,
        timeout=timeout_s if timeout_s is not None else settings.llm_request_timeout_s,
        max_retries=max_retries if max_retries is not None else settings.llm_max_retries,
    )


_BUILDERS = {
    "gemini": _build_gemini,
    "groq": _build_groq,
    "openrouter": _build_openrouter,
    "ollama": _build_ollama,
}


@lru_cache(maxsize=96)
def _get_chat_model(
    provider: str,
    role: Role = "generator",
    *,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> Any:
    builder = _BUILDERS.get(provider)
    if builder is None:
        raise LLMProviderError(
            f"Unknown LLM_PROVIDER {provider!r}; expected one of {sorted(_BUILDERS)}"
        )
    logger.info("Building %s model for role=%s", provider, role)
    return builder(role, timeout_s=timeout_s, max_retries=max_retries)


def get_chat_model(
    role: Role = "generator",
    *,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> Any:
    """Return a model for the route selected for this invocation."""
    return _get_chat_model(
        _selected_provider(), role, timeout_s=timeout_s, max_retries=max_retries
    )


def uses_native_structured_output() -> bool:
    """Whether the selected provider is bound to a JSON schema by this factory."""
    return _selected_provider() == "groq"


def get_structured_chat_model(
    role: Role,
    schema: dict[str, Any],
    *,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> Any:
    """Return a schema-bound Groq model, otherwise the ordinary chat model.

    Gemini, OpenRouter, and Ollama retain prompt-plus-parser behavior. Groq's
    GPT-OSS models support strict native JSON schemas, so use constrained
    decoding when Groq is explicitly selected instead of trusting a prompt to
    produce parseable JSON.
    """
    model = get_chat_model(role, timeout_s=timeout_s, max_retries=max_retries)
    if not uses_native_structured_output():
        return model
    return model.with_structured_output(schema, method="json_schema", strict=True)


def build_json_chain(
    prompt: Any,
    role: Role,
    schema: dict[str, Any],
    *,
    timeout_s: float | None = None,
    max_retries: int | None = None,
) -> Any:
    """Build a JSON chain with one budget-accounted dynamic-provider failover.

    The first model call has already been reserved by the graph node. A
    fallback reserves a second permit before re-invoking the complete chain, so
    it cannot silently exceed the graph's execution allowance.
    """
    from langchain_core.runnables import RunnableLambda

    def build_once(call_timeout_s: float | None, call_max_retries: int | None) -> Any:
        model = get_structured_chat_model(
            role,
            schema,
            timeout_s=call_timeout_s,
            max_retries=call_max_retries,
        )
        chain = prompt | model
        if uses_native_structured_output():
            return chain
        from langchain_core.output_parsers import JsonOutputParser

        return chain | JsonOutputParser()

    def invoke_with_failover(value: Any) -> Any:
        call_timeout_s = timeout_s
        call_max_retries = max_retries
        while True:
            try:
                return build_once(call_timeout_s, call_max_retries).invoke(value)
            except Exception as exc:  # noqa: BLE001 - providers expose varied exception types
                if not advance_route(exc):
                    raise
                # The original node spent its own permit. A provider switch is
                # another real inference attempt and must claim another slot.
                from src.config import get_settings as get_runtime_settings
                from src.self_healing.execution_budget import reserve_llm_call

                settings = get_runtime_settings()
                permit = reserve_llm_call(
                    "provider_fallback",
                    default_timeout_s=settings.llm_request_timeout_s,
                    default_max_retries=settings.llm_max_retries,
                )
                call_timeout_s = permit.timeout_s
                call_max_retries = permit.max_retries

    return RunnableLambda(invoke_with_failover)


def provider_config(role: Role = "generator") -> dict[str, Any]:
    """Configuration block for reports. Never contains the key itself."""
    settings = get_settings()
    provider = _selected_provider()
    return {
        "provider": provider,
        "model": model_name_for(role, provider),
        "role": role,
        "temperature": _temperature_for(role),
        "max_output_tokens": settings.llm_max_output_tokens,
        "timeout_s": settings.llm_request_timeout_s,
        "max_retries": settings.llm_max_retries,
        "credentials_present": (
            bool(settings.google_api_key)
            if provider == "gemini"
            else bool(settings.groq_api_key)
            if provider == "groq"
            else bool(settings.openrouter_api_key)
            if provider == "openrouter"
            else True
        ),
    }


def reset_model_cache() -> None:
    """Clear cached models. Call after changing settings in tests."""
    _get_chat_model.cache_clear()
