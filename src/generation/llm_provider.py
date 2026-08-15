"""Provider abstraction for chat models.

Nothing else in RAGuard imports a provider SDK directly. Switching from Gemini
to a local Ollama model is a change to `LLM_PROVIDER` in `.env`, not a code
change, which keeps the "same pipeline, different backend" comparison honest
for the evaluation chapter.

Three logical roles are exposed so that cost and determinism can be tuned
independently:

- `generator`  answers the user, temperature 0
- `rewriter`   produces query variants, slightly warmer for diversity
- `judge`      scores Ragas metrics in nightly CI, cheapest model available
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel

from src.config import get_settings

logger = logging.getLogger(__name__)

Role = Literal["generator", "rewriter", "judge"]

_ROLE_TEMPERATURE: dict[Role, float] = {
    "generator": 0.0,
    "rewriter": 0.3,  # Diversity is the point of rewriting; zero collapses variants.
    "judge": 0.0,
}


class LLMProviderError(RuntimeError):
    """Raised when the configured provider cannot be constructed."""


def _build_gemini(role: Role) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:  # pragma: no cover
        raise LLMProviderError(
            "langchain-google-genai is not installed. Run: pip install langchain-google-genai"
        ) from exc

    settings = get_settings()
    if not settings.google_api_key:
        raise LLMProviderError(
            "GOOGLE_API_KEY is not set. Obtain a key from Google AI Studio and add "
            "it to .env, or set LLM_PROVIDER=ollama to run fully offline."
        )

    model = settings.gemini_judge_model if role == "judge" else settings.gemini_model
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        temperature=_ROLE_TEMPERATURE[role],
        max_output_tokens=settings.llm_max_output_tokens,
    )


def _build_ollama(role: Role) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover
        raise LLMProviderError(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from exc

    settings = get_settings()
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=_ROLE_TEMPERATURE[role],
        num_predict=settings.llm_max_output_tokens,
    )


_BUILDERS = {"gemini": _build_gemini, "ollama": _build_ollama}


@lru_cache(maxsize=8)
def get_chat_model(role: Role = "generator") -> BaseChatModel:
    """Return the chat model for a role. Cached per (provider, role)."""
    settings = get_settings()
    builder = _BUILDERS.get(settings.llm_provider)
    if builder is None:
        raise LLMProviderError(
            f"Unknown LLM_PROVIDER {settings.llm_provider!r}; expected one of {sorted(_BUILDERS)}"
        )
    logger.info("Building %s model for role=%s", settings.llm_provider, role)
    return builder(role)


def reset_model_cache() -> None:
    """Clear cached models. Call after changing settings in tests."""
    get_chat_model.cache_clear()
