"""Real text embedder backed by sentence-transformers (optional dependency)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import TextEmbedder, normalize


class SentenceTransformerEmbedder(TextEmbedder):
    """Wraps a ``sentence-transformers`` model.

    Install with ``pip install -e ".[text]"``. Falls back is *not* automatic —
    the factory in :mod:`hybridrag.embed` chooses this only when a non-"hash"
    model id is configured.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                'Install it with: pip install -e ".[text]"'
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return normalize(np.asarray(vecs, dtype=np.float32))
