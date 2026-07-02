"""Qwen-VL embedding adapter with batched GPU inference (optional dependency).

Pixel/Hybrid RAG's vision path stands or falls on the VLM that maps image tiles
and text queries into a shared space. This adapter targets the **Qwen-VL family**
(Qwen2-VL / Qwen2.5-VL and their embedding heads) — the models the project brief
calls out as the reference VLMs — while degrading gracefully to any CLIP/SigLIP
style model that exposes ``get_image_features`` / ``get_text_features``.

Two things matter for a real deployment, and both are handled here:

* **Batched inference.** :class:`QwenVLVisionEmbedder` subclasses
  :class:`~hybridrag.embed.base.BatchedVisionEmbedder`, so no matter how many
  tiles arrive in one ``encode()`` call (the ingest layer buffers hundreds), the
  model only ever sees ``vision_encode_batch_size`` at a time. Peak GPU memory is
  bounded by the micro-batch, not the corpus.
* **Precision & device.** Auto-selects cuda/mps/cpu and runs fp16/bf16 autocast
  on CUDA when asked, halving memory and roughly doubling throughput.

Everything torch-specific lives behind an opt-in import; the core package still
runs on numpy alone via the hashing fallback.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import BatchedVisionEmbedder, normalize


class QwenVLVisionEmbedder(BatchedVisionEmbedder):
    """Batched Qwen-VL (or CLIP/SigLIP) image+text embedder.

    Install with ``pip install -e ".[vision]"``. The embedding API differs across
    checkpoints, so this wrapper probes for the cleanest one available:

    1. ``get_image_features`` / ``get_text_features`` — CLIP/SigLIP and several
       Qwen embedding heads. Used directly when present.
    2. Otherwise the base model is run and its ``last_hidden_state`` is
       **attention-mask mean-pooled** — the standard way to pull a single vector
       out of a Qwen2-VL backbone that has no dedicated pooling head.

    Vectors are L2-normalized so inner-product search equals cosine similarity,
    matching the fallback encoders and the vector store's assumptions.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        *,
        device: str = "auto",
        precision: str = "auto",
        encode_batch_size: int = 16,
    ) -> None:
        try:
            import torch  # type: ignore
            from PIL import Image  # noqa: F401  (used in _encode_images)  # type: ignore
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "torch, transformers and pillow are required for QwenVLVisionEmbedder. "
                'Install them with: pip install -e ".[vision]"'
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.encode_batch_size = max(1, int(encode_batch_size))

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device
        self._autocast_dtype = self._resolve_dtype(precision, device)

        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device).eval()
        self._has_feature_heads = hasattr(self._model, "get_image_features") and hasattr(
            self._model, "get_text_features"
        )
        # Probe output dim once with a tiny text pass.
        probe = self._encode_texts(["probe"])
        self.dim = int(probe.shape[-1])

    def _resolve_dtype(self, precision: str, device: str):
        torch = self._torch
        if device != "cuda" or precision in ("fp32", "none"):
            return None
        if precision == "fp16":
            return torch.float16
        if precision == "bf16":
            return torch.bfloat16
        # "auto": prefer bf16 where supported, else fp16.
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16

    # -- torch execution helpers -------------------------------------------------
    def _autocast(self):
        torch = self._torch
        if self._autocast_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
        return torch.no_grad()  # no_grad is applied outside; this is a harmless nesting

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        # attention_mask: (b, t) -> (b, t, 1); average over unmasked tokens.
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _to_numpy(self, feats) -> np.ndarray:
        return normalize(feats.float().cpu().numpy().astype(np.float32))

    # -- BatchedVisionEmbedder hooks --------------------------------------------
    def _encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        from PIL import Image  # type: ignore

        torch = self._torch
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with torch.no_grad(), self._autocast():
            if self._has_feature_heads:
                feats = self._model.get_image_features(**inputs)
            else:
                out = self._model(**inputs)
                mask = inputs.get("attention_mask")
                if mask is None:
                    feats = out.last_hidden_state.mean(dim=1)
                else:
                    feats = self._mean_pool(out.last_hidden_state, mask)
        return self._to_numpy(feats)

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with torch.no_grad(), self._autocast():
            if self._has_feature_heads:
                feats = self._model.get_text_features(**inputs)
            else:
                out = self._model(**inputs)
                mask = inputs.get("attention_mask")
                if mask is None:
                    feats = out.last_hidden_state.mean(dim=1)
                else:
                    feats = self._mean_pool(out.last_hidden_state, mask)
        return self._to_numpy(feats)
