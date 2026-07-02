"""Embedder factories.

``build_text_embedder`` / ``build_vision_embedder`` return the deterministic
hashing fallback when the configured model is ``"hash"`` (the default), and the
real model wrapper otherwise. This keeps the core importable with only numpy.
"""

from __future__ import annotations

from ..config import HybridConfig
from .base import (
    BatchedVisionEmbedder,
    HashTextEmbedder,
    HashVisionEmbedder,
    TextEmbedder,
    VisionEmbedder,
    normalize,
)


def build_text_embedder(config: HybridConfig) -> TextEmbedder:
    if config.text_model == "hash":
        return HashTextEmbedder(dim=config.text_dim)
    from .text import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_name=config.text_model)


def build_vision_embedder(config: HybridConfig) -> VisionEmbedder:
    if config.vision_model == "hash":
        return HashVisionEmbedder(dim=config.vision_dim)
    batch = config.vision_encode_batch_size
    if "qwen" in config.vision_model.lower():
        from .qwen_vl import QwenVLVisionEmbedder

        return QwenVLVisionEmbedder(
            model_name=config.vision_model,
            device=config.vision_device,
            precision=config.vision_precision,
            encode_batch_size=batch,
        )
    from .vision import VLMVisionEmbedder

    return VLMVisionEmbedder(
        model_name=config.vision_model,
        device=config.vision_device,
        encode_batch_size=batch,
    )


__all__ = [
    "TextEmbedder",
    "VisionEmbedder",
    "BatchedVisionEmbedder",
    "HashTextEmbedder",
    "HashVisionEmbedder",
    "build_text_embedder",
    "build_vision_embedder",
    "normalize",
]
