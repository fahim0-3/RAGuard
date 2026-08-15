"""BGE-M3 dense embedding loader.

The model is roughly 2.2 GB on disk and several seconds to load, so it is
cached process-wide. Vectors are L2-normalised at encode time, which lets the
database use cosine distance (`<=>`) and lets us treat `1 - distance` as a
similarity in [0, 1].
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from src.config import get_settings

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_lock = threading.Lock()


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
                _model = SentenceTransformer(
                    settings.embedding_model,
                    device=settings.model_device,
                )
    return _model


def embed_texts(texts: list[str], batch_size: int = 8) -> list[list[float]]:
    """Embed a batch of passages."""
    if not texts:
        return []
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
    return embed_texts([query])[0]


def embedding_dimension() -> int:
    return get_embedding_model().get_sentence_embedding_dimension()
