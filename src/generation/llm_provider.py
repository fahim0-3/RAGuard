"""Backwards-compatible alias for :mod:`src.generation.llm_factory`.

The factory moved to `llm_factory.py` in Phase E. Four modules outside
generation already import `get_chat_model` from this path, so the name is kept
as a re-export rather than updated at every call site: renaming a module is not
worth a diff across the self-healing and evaluation layers.

New code should import from `src.generation.llm_factory`.
"""

from __future__ import annotations

from src.generation.llm_factory import (
    LLMProviderError,
    Role,
    get_chat_model,
    model_name_for,
    provider_config,
    reset_model_cache,
)

__all__ = [
    "LLMProviderError",
    "Role",
    "get_chat_model",
    "model_name_for",
    "provider_config",
    "reset_model_cache",
]
