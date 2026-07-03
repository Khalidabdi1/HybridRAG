"""Real vision embedders backed by VLM embedding models (optional dependency).

Two backends are provided, both batched so they bound GPU memory on large
pages/corpora:

* :class:`VLMVisionEmbedder` — CLIP-style dual-head models exposing
  ``get_image_features`` / ``get_text_features`` (CLIP, SigLIP, and many VLM
  embedding heads).
* :class:`QwenVLEmbedder` — Qwen2-VL-based multimodal embedding models of the
  GME family (e.g. ``Alibaba-NLP/gme-Qwen2-VL-2B-Instruct``) that expose
  ``get_image_embeddings`` / ``get_text_embeddings`` and align an
  instruction-prefixed text query with a page image in one shared space. This
  is the retrieval backbone pixel-only systems rely on, now first-class here.

Both are optional, GPU-friendly paths; the core pipeline runs without them via
the hashing fallback in :mod:`hybridrag.embed.base`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import BatchedVisionEmbedder


def _resolve_device(torch, device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class VLMVisionEmbedder(BatchedVisionEmbedder):
    """Wraps a CLIP-style HuggingFace multimodal embedding model.

    Install with ``pip install -e ".[vision]"``. Targets models exposing
    ``get_image_features`` / ``get_text_features``, which covers CLIP, SigLIP,
    and many VLM embedding heads. Encoding runs in ``batch_size`` chunks so a
    long page (many tiles) or a large query set never has to fit on the GPU at
    once. For bespoke VLMs, subclass and override the ``_encode_*`` methods.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        batch_size: int = 16,
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
        self._device = _resolve_device(torch, device)
        self.batch_size = int(batch_size)
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device).eval()
        # Probe output dim with a tiny dummy text pass.
        with torch.no_grad():
            feats = self._model.get_text_features(
                **self._processor(text=["probe"], return_tensors="pt", padding=True).to(self._device)
            )
        self.dim = int(feats.shape[-1])

    def _encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        from PIL import Image  # type: ignore

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return feats.cpu().numpy().astype(np.float32)

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with self._torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return feats.cpu().numpy().astype(np.float32)


class QwenVLEmbedder(BatchedVisionEmbedder):
    """Qwen2-VL-based multimodal embedding model (GME family).

    Targets models such as ``Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`` that expose
    ``get_image_embeddings`` / ``get_text_embeddings`` and map a page image and
    a text query into one shared space — the exact retrieval backbone the
    pixel-only systems HybridRAG competes with are built on. Text queries are
    instruction-prefixed (the convention these models are trained with), and
    tiles/queries are encoded in ``batch_size`` GPU mini-batches.

    Install with ``pip install -e ".[vision]"`` (plus the model's own
    ``trust_remote_code`` weights, fetched on first use). Different GME
    checkpoints expose slightly different signatures; this wrapper adapts to the
    common ``images=`` / ``texts=`` + ``instruction=`` shape and falls back to
    positional calls when needed.
    """

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        device: str = "auto",
        batch_size: int = 8,
        query_instruction: str = "Find a document page that answers the query.",
    ) -> None:
        try:
            import torch  # type: ignore
            from PIL import Image  # noqa: F401  (used in _encode_images)  # type: ignore
            from transformers import AutoModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "torch, transformers and pillow are required for QwenVLEmbedder. "
                'Install them with: pip install -e ".[vision]"'
            ) from exc

        self._torch = torch
        self._device = _resolve_device(torch, device)
        self.batch_size = int(batch_size)
        self.query_instruction = query_instruction
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = (
            AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype)
            .to(self._device)
            .eval()
        )
        # Probe output dim with a tiny dummy text pass.
        probe = self._encode_texts(["probe"])
        self.dim = int(np.asarray(probe).shape[-1])

    def _as_numpy(self, out) -> np.ndarray:
        if hasattr(out, "cpu"):  # a torch.Tensor
            out = out.cpu().numpy()
        return np.atleast_2d(np.asarray(out, dtype=np.float32))

    def _encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        from PIL import Image  # type: ignore

        images = [Image.open(p).convert("RGB") for p in image_paths]
        with self._torch.no_grad():
            try:
                out = self._model.get_image_embeddings(images=images)
            except TypeError:  # pragma: no cover - signature variance across checkpoints
                out = self._model.get_image_embeddings(images)
        return self._as_numpy(out)

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        with self._torch.no_grad():
            try:
                out = self._model.get_text_embeddings(
                    texts=texts, instruction=self.query_instruction
                )
            except TypeError:  # pragma: no cover - signature variance across checkpoints
                out = self._model.get_text_embeddings(texts)
        return self._as_numpy(out)
