"""Tests for the Qwen-VL adapter and the batched vision embedder base.

The real Qwen-VL / CLIP models need torch + transformers, which the numpy-only
CI suite does not install. So we test the two things that *are* dependency-free
and where the real bugs live: the mini-batching loop (order, batch sizes, empty
input, equivalence to a single pass) via a fake backend, plus factory routing
and config round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybridrag.config import HybridConfig
from hybridrag.embed import BatchedVisionEmbedder, build_vision_embedder


class _FakeBatched(BatchedVisionEmbedder):
    """Numpy-only backend that records the size of every mini-batch it sees."""

    def __init__(self, dim: int = 8, encode_batch_size: int = 4) -> None:
        self.dim = dim
        self.encode_batch_size = encode_batch_size
        self.image_batches: list[int] = []
        self.text_batches: list[int] = []

    def _vec(self, key: str) -> np.ndarray:
        # deterministic, distinct per key so we can check ordering
        h = abs(hash(key)) % 9973
        return np.full(self.dim, float(h), dtype=np.float32)

    def _encode_images(self, image_paths):
        self.image_batches.append(len(image_paths))
        return np.stack([self._vec(p) for p in image_paths])

    def _encode_texts(self, texts):
        self.text_batches.append(len(texts))
        return np.stack([self._vec(t) for t in texts])


def test_empty_inputs_return_zero_by_dim():
    emb = _FakeBatched(dim=8)
    assert emb.encode([]).shape == (0, 8)
    assert emb.encode_query([]).shape == (0, 8)
    assert emb.image_batches == [] and emb.text_batches == []


def test_micro_batches_respect_encode_batch_size():
    emb = _FakeBatched(encode_batch_size=4)
    paths = [f"tile-{i}.png" for i in range(10)]
    emb.encode(paths)
    # 10 items at batch 4 -> [4, 4, 2], never larger than the cap
    assert emb.image_batches == [4, 4, 2]
    assert max(emb.image_batches) <= 4


def test_batched_output_matches_single_pass_and_order():
    emb = _FakeBatched(dim=8, encode_batch_size=3)
    paths = [f"p{i}" for i in range(7)]
    got = emb.encode(paths)
    # reference: encode each individually (a "batch of one" each), same vectors
    ref = np.stack([emb._vec(p) for p in paths]).astype(np.float32)
    assert got.shape == (7, 8)
    np.testing.assert_array_equal(got, ref)


def test_query_path_is_batched_too():
    emb = _FakeBatched(encode_batch_size=2)
    emb.encode_query(["a", "b", "c", "d", "e"])
    assert emb.text_batches == [2, 2, 1]


def test_batch_size_of_one_is_valid():
    emb = _FakeBatched(encode_batch_size=1)
    emb.encode(["x", "y", "z"])
    assert emb.image_batches == [1, 1, 1]


def test_zero_or_negative_batch_size_is_clamped():
    emb = _FakeBatched(encode_batch_size=0)
    # clamped to 1 internally rather than looping forever / erroring
    emb.encode(["x", "y"])
    assert emb.image_batches == [1, 1]


def test_factory_routes_qwen_id_to_qwen_adapter():
    cfg = HybridConfig(vision_model="Qwen/Qwen2-VL-2B-Instruct")
    try:
        emb = build_vision_embedder(cfg)
    except ImportError as exc:
        # torch/transformers absent (CI): the message must name the Qwen adapter,
        # proving the factory routed to it rather than the CLIP wrapper.
        assert "QwenVLVisionEmbedder" in str(exc)
    else:  # pragma: no cover - only when torch is installed
        from hybridrag.embed.qwen_vl import QwenVLVisionEmbedder

        assert isinstance(emb, QwenVLVisionEmbedder)


def test_factory_routes_non_qwen_id_to_clip_wrapper():
    cfg = HybridConfig(vision_model="openai/clip-vit-base-patch32")
    try:
        emb = build_vision_embedder(cfg)
    except ImportError as exc:
        assert "VLMVisionEmbedder" in str(exc)
    else:  # pragma: no cover - only when torch is installed
        from hybridrag.embed.vision import VLMVisionEmbedder

        assert isinstance(emb, VLMVisionEmbedder)


def test_hash_model_still_default_and_no_deps():
    emb = build_vision_embedder(HybridConfig())
    vecs = emb.encode_query(["hello world"])
    assert vecs.shape == (1, emb.dim)


def test_new_config_fields_round_trip(tmp_path):
    cfg = HybridConfig(
        vision_model="Qwen/Qwen2-VL-7B-Instruct",
        vision_encode_batch_size=32,
        vision_precision="bf16",
        vision_device="cuda",
    )
    path = tmp_path / "cfg.json"
    cfg.to_file(str(path))
    loaded = HybridConfig.from_file(str(path))
    assert loaded.vision_encode_batch_size == 32
    assert loaded.vision_precision == "bf16"
    assert loaded.vision_device == "cuda"
    assert loaded.vision_model == "Qwen/Qwen2-VL-7B-Instruct"


def test_qwen_adapter_raises_clean_import_error_without_torch():
    from hybridrag.embed.qwen_vl import QwenVLVisionEmbedder

    try:
        QwenVLVisionEmbedder()
    except ImportError as exc:
        assert "pip install" in str(exc)
    else:  # pragma: no cover - only when torch is installed
        pytest.skip("torch is installed; import-error path not exercised")
