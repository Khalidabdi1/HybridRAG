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
    tile_height: int = 512  # px height of each tile
    tile_overlap: int = 64  # px overlap between vertical tiles
    vision_dim: int = 512  # embedding dim for the fallback hashing encoder

    # ---- rendering ----
    viewport_width: int = 1280
    render_scale: float = 1.0

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
