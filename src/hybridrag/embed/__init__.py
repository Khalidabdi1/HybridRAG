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
    iter_batches,
    normalize,
)


def _resolve_vision_backend(config: HybridConfig) -> str:
    """Pick the real-VLM wrapper for a non-"hash" vision model."""
    backend = (config.vision_backend or "auto").lower()
    if backend in {"clip", "qwen"}:
        return backend
    name = config.vision_model.lower()
    return "qwen" if ("qwen" in name or "gme" in name) else "clip"


def build_text_embedder(config: HybridConfig) -> TextEmbedder:
    if config.text_model == "hash":
        return HashTextEmbedder(dim=config.text_dim)
    from .text import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_name=config.text_model)


def build_vision_embedder(config: HybridConfig) -> VisionEmbedder:
    if config.vision_model == "hash":
        return HashVisionEmbedder(dim=config.vision_dim)
    if _resolve_vision_backend(config) == "qwen":
        from .vision import QwenVLEmbedder

        return QwenVLEmbedder(
            model_name=config.vision_model,
            batch_size=config.vision_batch_size,
            query_instruction=config.vision_query_instruction,
        )
    from .vision import VLMVisionEmbedder

    return VLMVisionEmbedder(
        model_name=config.vision_model, batch_size=config.vision_batch_size
    )


__all__ = [
    "TextEmbedder",
    "VisionEmbedder",
    "BatchedVisionEmbedder",
    "HashTextEmbedder",
    "HashVisionEmbedder",
    "build_text_embedder",
    "build_vision_embedder",
    "iter_batches",
    "normalize",
]
