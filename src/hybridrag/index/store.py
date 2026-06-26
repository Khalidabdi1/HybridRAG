"""Vector store with a numpy backend (default) and optional FAISS acceleration.

A store holds one modality's vectors plus a parallel list of metadata records.
It supports cosine search (vectors are stored L2-normalized, so cosine reduces
to a dot product) and JSON+npy persistence.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np

from ..embed.base import normalize


class VectorStore:
    def __init__(self, dim: int, use_faiss: bool = False) -> None:
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._meta: List[Dict[str, Any]] = []
        self._use_faiss = use_faiss
        self._faiss_index = None
        if use_faiss:
            self._init_faiss()

    # ---- construction ----
    def _init_faiss(self) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                'faiss is required for use_faiss=True. Install with: pip install -e ".[faiss]"'
            ) from exc
        self._faiss = faiss
        self._faiss_index = faiss.IndexFlatIP(self.dim)

    def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        if len(metas) == 0:
            return
        vectors = normalize(np.asarray(vectors, dtype=np.float32))
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        if len(metas) != vectors.shape[0]:
            raise ValueError("vectors and metas length mismatch")
        self._vectors = np.vstack([self._vectors, vectors]) if len(self._meta) else vectors
        self._meta.extend(metas)
        if self._faiss_index is not None:
            self._faiss_index.add(vectors)

    # ---- mutation ----
    def delete_where(self, predicate: Callable[[Dict[str, Any]], bool]) -> int:
        """Remove every record whose metadata matches ``predicate``.

        Returns the number of records removed. The FAISS index (if any) is
        rebuilt from the surviving vectors, since ``IndexFlatIP`` has no
        in-place removal.
        """
        if not self._meta:
            return 0
        keep = [i for i, m in enumerate(self._meta) if not predicate(m)]
        removed = len(self._meta) - len(keep)
        if removed == 0:
            return 0
        self._vectors = (
            self._vectors[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        )
        self._meta = [self._meta[i] for i in keep]
        if self._faiss_index is not None:
            self._faiss_index = self._faiss.IndexFlatIP(self.dim)
            if len(self._vectors):
                self._faiss_index.add(self._vectors)
        return removed

    def delete_doc(self, doc_id: str) -> int:
        """Remove every record belonging to ``doc_id``. Returns the count."""
        return self.delete_where(lambda m: m.get("doc_id") == doc_id)

    def doc_ids(self) -> Set[str]:
        """The set of distinct ``doc_id`` values currently stored."""
        return {m.get("doc_id") for m in self._meta if m.get("doc_id") is not None}

    # ---- query ----
    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[float, Dict[str, Any]]]:
        if len(self._meta) == 0:
            return []
        query = normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
        k = min(top_k, len(self._meta))
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(query, k)
            scores, idxs = scores[0], idxs[0]
        else:
            sims = (self._vectors @ query.T).ravel()
            idxs = np.argpartition(-sims, k - 1)[:k]
            idxs = idxs[np.argsort(-sims[idxs])]
            scores = sims[idxs]
        return [(float(s), self._meta[int(i)]) for s, i in zip(scores, idxs) if int(i) >= 0]

    def __len__(self) -> int:
        return len(self._meta)

    # ---- persistence ----
    def save(self, directory: str, name: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, f"{name}.vectors.npy"), self._vectors)
        with open(os.path.join(directory, f"{name}.meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"dim": self.dim, "meta": self._meta}, fh)

    @classmethod
    def load(cls, directory: str, name: str, use_faiss: bool = False) -> "VectorStore":
        with open(os.path.join(directory, f"{name}.meta.json"), "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        store = cls(dim=payload["dim"], use_faiss=use_faiss)
        vectors = np.load(os.path.join(directory, f"{name}.vectors.npy"))
        store.add(vectors, payload["meta"])
        return store

    @classmethod
    def exists(cls, directory: str, name: str) -> bool:
        return os.path.exists(os.path.join(directory, f"{name}.meta.json"))
