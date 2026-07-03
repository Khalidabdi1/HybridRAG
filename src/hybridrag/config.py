"""Configuration for the HybridRAG pipeline.

Every knob has a sensible default so that ``HybridConfig()`` produces a working
system out of the box. Values can be overridden in code or loaded from a JSON
file via :meth:`HybridConfig.from_file`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class HybridConfig:
    # ---- storage ----
    storage_dir: str = ".hybridrag"

    # ---- text pipeline ----
    text_model: str = "hash"  # "hash" (fallback) or a sentence-transformers id
    chunk_size: int = 800  # characters per chunk
    chunk_overlap: int = 120  # character overlap between chunks
    text_dim: int = 384  # embedding dim for the fallback hashing encoder

    # ---- vision pipeline ----
    vision_model: str = "hash"  # "hash" (fallback) or a transformers VLM id
    # Which real-VLM wrapper to build for a non-"hash" model. "auto" picks
    # QwenVLEmbedder (GME / Qwen2-VL family) when the model id mentions "qwen"
    # or "gme", else the CLIP-style VLMVisionEmbedder; force with "clip"/"qwen".
    vision_backend: str = "auto"  # "auto" | "clip" | "qwen"
    vision_batch_size: int = 16  # tiles/queries per GPU mini-batch on the real encoders
    # Instruction prepended to text queries for GME/Qwen2-VL embedders.
    vision_query_instruction: str = "Find a document page that answers the query."
    tile_height: int = 512  # px height of each tile
    tile_overlap: int = 64  # px overlap between vertical tiles
    vision_dim: int = 512  # embedding dim for the fallback hashing encoder

    # ---- rendering ----
    viewport_width: int = 1280
    render_scale: float = 1.0

    # ---- selective pixel indexing ----
    # Decide whether a page is worth the (expensive) vision path. "always" keeps
    # the classic Pixel RAG behaviour, "never" is text-only, and "auto" scores
    # each page's visual richness (tables/charts/figures) and indexes pixels only
    # when it clears the threshold — saving storage and GPU on text-native pages.
    pixel_selection: str = "auto"  # "auto" | "always" | "never"
    pixel_selection_threshold: float = 0.35  # richness score needed to index pixels
    pixel_selection_text_weight: float = 0.7  # how much pre-render text structure counts

    # ---- retrieval / fusion ----
    top_k: int = 10
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    text_weight: float = 1.0
    vision_weight: float = 1.0
    enable_router: bool = True  # let the router adjust per-query weights

    # ---- reranking ----
    enable_rerank: bool = False  # rerank the fused candidate set before returning
    rerank_model: str = "lexical"  # "lexical" (BM25, no deps), "none", or a CrossEncoder id
    rerank_top_n: int = 30  # how many fused candidates to rerank (fetch this many first)
    rerank_blend: float = 0.5  # weight on the reranker score vs the fusion score [0..1]

    # ---- answer synthesis (the "final readout") ----
    # Compose a grounded, cited answer from the fused hits. "extractive" (default,
    # no deps) selects the best sentences from retrieved chunks; "none" disables
    # synthesis (search only). A real LLM/VLM reader is passed in code.
    reader_model: str = "extractive"  # "extractive" (no deps) | "none"
    answer_max_sentences: int = 3  # sentences the extractive reader stitches together
    answer_top_k: int = 5  # how many fused hits to feed the reader

    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "HybridConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["extra"] = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs)

    def to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **kwargs: Any) -> "HybridConfig":
        data = self.to_dict()
        data.update(kwargs)
        return HybridConfig(**data)
