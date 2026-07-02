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
from typing import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero rows are left as zeros."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


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
    """A :class:`VisionEmbedder` that runs its model in fixed-size mini-batches.

    Real vision/VLM encoders must bound how many tiles they push through the
    model in a single forward pass — a large corpus (or a large ingest buffer)
    handed to ``encode()`` in one call would otherwise blow up GPU memory. This
    base splits any input into ``encode_batch_size`` chunks, delegates each chunk
    to :meth:`_encode_images` / :meth:`_encode_texts`, and concatenates the
    results **in input order**, so batching is transparent to callers and the
    output is identical to a single (hypothetical) forward pass.

    Subclasses implement the two per-batch hooks; this class owns the looping,
    the empty-input handling, and the final concatenation. The looping logic is
    dependency-free and unit-tested with a numpy-only fake backend.
    """

    #: model-level micro-batch size; each hook receives at most this many items.
    encode_batch_size: int = 16

    def _encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        """Encode a single mini-batch of image paths → ``(len, dim)`` float32."""
        raise NotImplementedError

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a single mini-batch of query texts → ``(len, dim)`` float32."""
        raise NotImplementedError

    @staticmethod
    def _batches(items: Sequence[str], size: int):
        size = max(1, int(size))
        for start in range(0, len(items), size):
            yield items[start : start + size]

    def _run_batched(self, items: Sequence[str], fn) -> np.ndarray:
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [fn(list(batch)) for batch in self._batches(items, self.encode_batch_size)]
        return np.concatenate(parts, axis=0).astype(np.float32)

    def encode(self, image_paths: Sequence[str]) -> np.ndarray:
        return self._run_batched(image_paths, self._encode_images)

    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        return self._run_batched(texts, self._encode_texts)


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
