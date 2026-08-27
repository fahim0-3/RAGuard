"""Dense embeddings from either local BGE-M3 or the Gemini API.

The model is roughly 2.2 GB on disk and several seconds to load, so it is
cached process-wide. Vectors are L2-normalised at encode time, which lets the
database use cosine distance (`<=>`) and lets us treat `1 - distance` as a
similarity in [0, 1].
"""

from __future__ import annotations

import logging
import math
import threading
from typing import TYPE_CHECKING

from src.config import get_settings

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_lock = threading.Lock()

#: Set when a warm-up attempt failed, so readiness can distinguish "still
#: loading" from "will never load". Written under `_lock`.
_load_error: str | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the shared BGE-M3 model, loading it on first use."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                settings = get_settings()
                logger.info(
                    "Loading embedding model %s on %s",
                    settings.embedding_model,
                    settings.model_device,
                )
                try:
                    _model = SentenceTransformer(
                        settings.embedding_model,
                        device=settings.model_device,
                    )
                except Exception as exc:
                    # Remember why, then re-raise for the caller. A background
                    # warm-up has nobody to raise to, and readiness needs the
                    # reason to report "failed" instead of "still loading".
                    global _load_error
                    _load_error = f"{type(exc).__name__}: {exc}"
                    logger.exception("Embedding model failed to load")
                    raise
                _load_error = None
    return _model


def embed_texts(texts: list[str], batch_size: int = 8) -> list[list[float]]:
    """Embed a batch of passages."""
    if not texts:
        return []
    settings = get_settings()
    if settings.embedding_provider == "gemini":
        return _embed_gemini(texts, task_type="RETRIEVAL_DOCUMENT")
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 32,
    )
    return [vector.tolist() for vector in vectors]


def embed_query(query: str) -> list[float]:
    """Embed a single query.

    BGE-M3 is trained without an instruction prefix, so queries and passages
    use the identical encoder path. Do not add a prefix here; doing so shifts
    the query vector away from the indexed passages.
    """
    if get_settings().embedding_provider == "gemini":
        return _embed_gemini([query], task_type="RETRIEVAL_QUERY")[0]
    return embed_texts([query])[0]


def _embed_gemini(texts: list[str], *, task_type: str) -> list[list[float]]:
    """Call Gemini embeddings without retaining text or vectors beyond the request."""
    from google import genai
    from google.genai import types

    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for Gemini embeddings")
    response = genai.Client(api_key=settings.google_api_key).models.embed_content(
        model=settings.gemini_embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.vector_dimension,
        ),
    )
    vectors = [list(item.values) for item in response.embeddings or []]
    if len(vectors) != len(texts) or any(len(vector) != settings.vector_dimension for vector in vectors):
        raise RuntimeError("Gemini embedding response has an unexpected dimension")
    return [_normalise(vector) for vector in vectors]


def _normalise(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise RuntimeError("Gemini embedding response contained a zero vector")
    return [value / magnitude for value in vector]


def embedding_dimension() -> int:
    if get_settings().embedding_provider == "gemini":
        return get_settings().vector_dimension
    return get_embedding_model().get_sentence_embedding_dimension()


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------
#
# The loader above is lazy by design and is left that way: ingestion, the
# evaluation CLI, and the test tiers all rely on importing this module without
# paying a 2.2 GB download. What was missing is a way for a *service* to load
# the model on its own terms and report whether it is usable, so that the first
# user query does not sit inside the download.


def is_model_loaded() -> bool:
    """Whether the model is resident. Never triggers a load.

    Deliberately does not take `_lock`: a readiness probe must not block behind
    an in-flight load, and reading one reference is atomic under the GIL.
    """
    settings = get_settings()
    return bool(settings.google_api_key) if settings.embedding_provider == "gemini" else _model is not None


def model_load_error() -> str | None:
    """Why the last warm-up failed, or None."""
    settings = get_settings()
    if settings.embedding_provider == "gemini" and not settings.google_api_key:
        return "ConfigurationError: GOOGLE_API_KEY is required for Gemini embeddings"
    return _load_error


def warmup_embedding_model() -> bool:
    """Load the model now, returning success instead of raising.

    Intended for a service start-up path. Safe to call from several threads and
    safe to call repeatedly: it delegates to `get_embedding_model()`, whose
    double-checked lock means the second caller waits for the first rather than
    starting a second download.
    """
    if get_settings().embedding_provider == "gemini":
        # Authentication is checked by readiness; do not make a billable probe.
        return bool(get_settings().google_api_key)
    try:
        get_embedding_model()
    except Exception:
        # `get_embedding_model` has already logged and recorded the reason.
        return False
    logger.info("Embedding model ready")
    return True
