"""Embedder factories.

``build_text_embedder`` / ``build_vision_embedder`` return the deterministic
hashing fallback when the configured model is ``"hash"`` (the default), and the
real model wrapper otherwise. This keeps the core importable with only numpy.
"""

from __future__ import annotations

from ..config import HybridConfig
from .base import (
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
    from .vision import VLMVisionEmbedder

    return VLMVisionEmbedder(model_name=config.vision_model)


__all__ = [
    "TextEmbedder",
    "VisionEmbedder",
    "HashTextEmbedder",
    "HashVisionEmbedder",
    "build_text_embedder",
    "build_vision_embedder",
    "normalize",
]
