"""Real vision embedder backed by a CLIP/SigLIP-style model (optional dependency).

For the Qwen-VL family use :class:`hybridrag.embed.qwen_vl.QwenVLVisionEmbedder`.
This wrapper targets models exposing ``get_image_features`` / ``get_text_features``
(CLIP, SigLIP, and many VLM embedding heads). Both real embedders share the
:class:`~hybridrag.embed.base.BatchedVisionEmbedder` base, so a large ``encode()``
call is transparently split into GPU-memory-bounded mini-batches. The core
pipeline still runs without any of this via the hashing fallback.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import BatchedVisionEmbedder, normalize


class VLMVisionEmbedder(BatchedVisionEmbedder):
    """Wraps a HuggingFace CLIP/SigLIP-style multimodal embedding model.

    Install with ``pip install -e ".[vision]"``. For bespoke VLMs, subclass and
    override :meth:`_encode_images` / :meth:`_encode_texts`.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        device: str = "auto",
        encode_batch_size: int = 16,
    ) -> None:
        try:
            import torch  # type: ignore
            from PIL import Image  # noqa: F401  (used in _encode_images)  # type: ignore
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "torch, transformers and pillow are required for VLMVisionEmbedder. "
                'Install them with: pip install -e ".[vision]"'
            ) from exc

        self._torch = torch
        self.encode_batch_size = max(1, int(encode_batch_size))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cpu"
            )
        self._device = device
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device).eval()
        # Probe output dim with a tiny dummy text pass.
        self.dim = int(self._encode_texts(["probe"]).shape[-1])

    def _encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        from PIL import Image  # type: ignore

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return normalize(feats.cpu().numpy().astype(np.float32))

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with self._torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return normalize(feats.cpu().numpy().astype(np.float32))
