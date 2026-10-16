"""In-process embedding.

Loaded once at startup, ~80 MB resident, ~5 ms per encode on CPU. The model is
loaded eagerly rather than lazily: a lazy load makes the FIRST request after
every restart pay a 2-3 s penalty, and it lands in your p99 rather than in
startup where it belongs.
"""

from __future__ import annotations

import numpy as np

from ..config import EMBED_DIM, EMBED_MODEL

_model = None


def load() -> None:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)


def encode(text: str) -> np.ndarray:
    """L2-normalised float32 vector.

    Normalising here means cosine distance and inner product agree, and pgvector's
    `<=>` operator returns a value we can convert to similarity with `1 - d`
    without worrying about magnitude.
    """
    if _model is None:
        load()
    v = _model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    v = np.asarray(v, dtype=np.float32)
    assert v.shape == (EMBED_DIM,), f"expected {EMBED_DIM}-d, got {v.shape}"
    return v
