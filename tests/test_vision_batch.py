"""Tests for the batched vision-embedder machinery and backend selection.

These run on numpy alone: they exercise the batching/order/normalization logic
in `BatchedVisionEmbedder` through a fake subclass (no torch/transformers), and
the `vision_backend` resolution used by the factory. The real
`VLMVisionEmbedder` / `QwenVLEmbedder` wrappers require the optional `vision`
extra and are not imported here.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybridrag.config import HybridConfig
from hybridrag.embed import _resolve_vision_backend
from hybridrag.embed.base import BatchedVisionEmbedder, iter_batches


def test_iter_batches_splits_evenly_and_remainder():
    assert list(iter_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(iter_batches([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


def test_iter_batches_empty_and_nonpositive():
    assert list(iter_batches([], 4)) == []
    # A non-positive batch size means "one batch, no splitting".
    assert list(iter_batches([1, 2, 3], 0)) == [[1, 2, 3]]
    assert list(iter_batches([1, 2, 3], -1)) == [[1, 2, 3]]
    assert list(iter_batches([], 0)) == []


class _RecordingEmbedder(BatchedVisionEmbedder):
    """Fake VLM: encodes each item to a deterministic per-index row and records
    the batch sizes it was handed, so we can assert batching behaviour."""

    def __init__(self, dim: int = 4, batch_size: int = 3) -> None:
        self.dim = dim
        self.batch_size = batch_size
        self.image_batches: list[int] = []
        self.text_batches: list[int] = []

    def _row(self, key: str) -> np.ndarray:
        # Un-normalized, distinct per key so we can check order + normalization.
        n = float(len(key))
        return np.arange(self.dim, dtype=np.float32) + n

    def _encode_images(self, image_paths):
        self.image_batches.append(len(image_paths))
        return np.stack([self._row(p) for p in image_paths])

    def _encode_texts(self, texts):
        self.text_batches.append(len(texts))
        return np.stack([self._row(t) for t in texts])


def test_batched_encode_chunks_and_preserves_order():
    emb = _RecordingEmbedder(dim=4, batch_size=3)
    paths = [f"tile_{i}" for i in range(7)]

    out = emb.encode(paths)

    # 7 items at batch_size 3 → batches of 3, 3, 1.
    assert emb.image_batches == [3, 3, 1]
    assert out.shape == (7, 4)
    # Rows are L2-normalized...
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(7), atol=1e-5)
    # ...and in input order: batched == a single un-chunked call, normalized.
    single = emb._row("tile_3")
    np.testing.assert_allclose(out[3], single / np.linalg.norm(single), atol=1e-5)


def test_batched_encode_query_uses_text_path():
    emb = _RecordingEmbedder(dim=4, batch_size=2)
    out = emb.encode_query(["a", "bb", "ccc"])
    assert emb.text_batches == [2, 1]
    assert out.shape == (3, 4)


def test_batched_empty_inputs_return_zero_rows():
    emb = _RecordingEmbedder(dim=5, batch_size=3)
    assert emb.encode([]).shape == (0, 5)
    assert emb.encode_query([]).shape == (0, 5)
    # No batch callbacks fire for empty input.
    assert emb.image_batches == [] and emb.text_batches == []


@pytest.mark.parametrize(
    "model,backend,expected",
    [
        ("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", "auto", "qwen"),
        ("some/Qwen-VL-embed", "auto", "qwen"),
        ("openai/clip-vit-base-patch32", "auto", "clip"),
        ("google/siglip-base", "auto", "clip"),
        # Explicit backend overrides the name heuristic in both directions.
        ("openai/clip-vit-base-patch32", "qwen", "qwen"),
        ("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", "clip", "clip"),
    ],
)
def test_resolve_vision_backend(model, backend, expected):
    cfg = HybridConfig(vision_model=model, vision_backend=backend)
    assert _resolve_vision_backend(cfg) == expected
