"""Real vision embedder backed by a VLM embedding model (optional dependency).

The default target is a Qwen-VL style multimodal *embedding* model that maps
both image tiles and text queries into a shared space — the same family used by
pixel-only systems. This is an optional, GPU-friendly path; the core pipeline
runs without it via the hashing fallback in :mod:`hybridrag.embed.base`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import VisionEmbedder, normalize


class VLMVisionEmbedder(VisionEmbedder):
    """Wraps a HuggingFace multimodal embedding model.

    Install with ``pip install -e ".[vision]"``. The exact API differs between
    model families; this wrapper targets models exposing ``get_image_features``
    / ``get_text_features`` (CLIP-style) which covers a broad range including
    SigLIP and many VLM embedding heads. For bespoke VLMs, subclass and
    override :meth:`encode` / :meth:`encode_query`.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto") -> None:
        try:
            import torch  # type: ignore
            from PIL import Image  # noqa: F401  (used in encode)  # type: ignore
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "torch, transformers and pillow are required for VLMVisionEmbedder. "
                'Install them with: pip install -e ".[vision]"'
            ) from exc

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cpu"
            )
        self._device = device
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device).eval()
        # Probe output dim with a tiny dummy text pass.
        with torch.no_grad():
            feats = self._model.get_text_features(
                **self._processor(text=["probe"], return_tensors="pt", padding=True).to(device)
            )
        self.dim = int(feats.shape[-1])

    def encode(self, image_paths: Sequence[str]) -> np.ndarray:
        if not image_paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        from PIL import Image  # type: ignore

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return normalize(feats.cpu().numpy().astype(np.float32))

    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with self._torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return normalize(feats.cpu().numpy().astype(np.float32))
