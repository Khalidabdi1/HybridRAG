"""Embedder interfaces and a dependency-free hashing fallback.

The fallback encoders let HybridRAG run end to end with nothing but numpy.
They are *not* semantically strong — they exist so the pipeline, fusion, and
API are testable and demonstrable offline. Swap in real encoders
(``sentence-transformers`` for text, a VLM for vision) via the factory
functions in :mod:`hybridrag.embed`.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Iterator, List, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero rows are left as zeros."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def iter_batches(items: Sequence, batch_size: int) -> Iterator[List]:
    """Yield successive ``batch_size``-sized slices of ``items``.

    A non-positive ``batch_size`` yields the whole sequence as one batch (no
    batching). Used to bound how many tiles/queries a GPU encoder holds at once.
    """
    seq = list(items)
    if batch_size <= 0:
        if seq:
            yield seq
        return
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


class TextEmbedder(ABC):
    """Encodes text strings into a fixed-size vector space."""

    dim: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32, L2-normalized matrix."""


class VisionEmbedder(ABC):
    """Encodes image tiles (paths) into a fixed-size vector space."""

    dim: int

    @abstractmethod
    def encode(self, image_paths: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32, L2-normalized matrix."""

    @abstractmethod
    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text queries into the *same* space as the image tiles.

        VLM embedding models are trained so that a text query and a relevant
        image land close together; this method exposes that text-side encoder.
        """


class BatchedVisionEmbedder(VisionEmbedder):
    """A :class:`VisionEmbedder` that encodes in fixed-size mini-batches.

    Real VLM encoders must bound how many tiles/queries live on the GPU at
    once, or they OOM on large pages/corpora. Subclasses implement
    :meth:`_encode_images` / :meth:`_encode_texts` over a *single* batch
    (returning raw, un-normalized features); this base slices the input into
    ``batch_size`` chunks, encodes each, concatenates in input order, and
    L2-normalizes once at the end. Batched output is therefore identical to a
    single hypothetical call — order and values are preserved.
    """

    batch_size: int = 16

    def encode(self, image_paths: Sequence[str]) -> np.ndarray:
        return self._run_batched(self._encode_images, image_paths)

    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        return self._run_batched(self._encode_texts, texts)

    def _run_batched(self, fn, items: Sequence) -> np.ndarray:
        seq = list(items)
        if not seq:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [np.asarray(fn(batch), dtype=np.float32)
                 for batch in iter_batches(seq, self.batch_size)]
        return normalize(np.concatenate(parts, axis=0))

    @abstractmethod
    def _encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        """Encode one batch of image paths into raw ``(len(batch), dim)`` features."""

    @abstractmethod
    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode one batch of query strings into raw ``(len(batch), dim)`` features."""


def _hash_vector(token: str, dim: int) -> np.ndarray:
    """Deterministically map a token to a unit-ish vector via SHA-256 bytes."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    # Repeat the 32-byte digest to fill `dim`, then center around zero.
    raw = (digest * ((dim // len(digest)) + 1))[:dim]
    vec = np.frombuffer(bytes(raw), dtype=np.uint8).astype(np.float32)
    return vec - 127.5


class HashTextEmbedder(TextEmbedder):
    """Bag-of-hashed-tokens text encoder. Deterministic, offline, no deps."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _TOKEN_RE.findall((text or "").lower())
            if not tokens:
                continue
            acc = np.zeros(self.dim, dtype=np.float32)
            for tok in tokens:
                acc += _hash_vector(tok, self.dim)
            out[i] = acc / len(tokens)
        return normalize(out)


class HashVisionEmbedder(VisionEmbedder):
    """Deterministic offline stand-in for a real VLM image encoder.

    It hashes lightweight image statistics (file path + size + a coarse byte
    histogram) so that identical tiles map to identical vectors and text
    queries can be projected into the same space through shared token hashing.
    This is purely a scaffold to keep the pipeline runnable without a GPU.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _image_vector(self, path: str) -> np.ndarray:
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            data = path.encode("utf-8")
        hist = np.zeros(256, dtype=np.float32)
        if data:
            counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
            hist = counts.astype(np.float32)
        # Fold the 256-bin histogram into `dim` and mix with a path hash so
        # tiles from different docs stay distinguishable.
        reps = (self.dim // 256) + 1
        folded = np.tile(hist, reps)[: self.dim]
        folded = folded + _hash_vector(path, self.dim) * 0.01
        return folded

    def encode(self, image_paths: Sequence[str]) -> np.ndarray:
        out = np.stack([self._image_vector(p) for p in image_paths]) if image_paths \
            else np.zeros((0, self.dim), dtype=np.float32)
        return normalize(out)

    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        # Project queries through the same hashing used for paths so that a
        # query mentioning a tile's source can align with it. Weak by design.
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _TOKEN_RE.findall((text or "").lower())
            if not tokens:
                continue
            acc = np.zeros(self.dim, dtype=np.float32)
            for tok in tokens:
                acc += _hash_vector(tok, self.dim)
            out[i] = acc / len(tokens)
        return normalize(out)
