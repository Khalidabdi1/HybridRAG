"""Core data types shared across the HybridRAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Modality(str, Enum):
    """A retrieval modality."""

    TEXT = "text"
    VISION = "vision"


@dataclass
class Chunk:
    """A piece of extracted text that can be embedded and retrieved."""

    id: str
    doc_id: str
    text: str
    page: int = 0
    order: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tile:
    """A rendered image tile (a slice of a page screenshot)."""

    id: str
    doc_id: str
    page: int = 0
    row: int = 0
    col: int = 0
    # Path to the tile image on disk (PNG). May be None in metadata-only mode.
    image_path: Optional[str] = None
    bbox: Optional[List[int]] = None  # [x0, y0, x1, y1] in page pixels
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A source document (web page, PDF, or image) plus its derived units."""

    id: str
    source: str
    title: str = ""
    chunks: List[Chunk] = field(default_factory=list)
    tiles: List[Tile] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single fused search hit returned to the caller."""

    doc_id: str
    score: float
    modality: Modality
    # The underlying unit id (chunk id or tile id).
    unit_id: str = ""
    text: str = ""
    image_path: Optional[str] = None
    page: int = 0
    # Per-modality contributions, useful for debugging fusion.
    components: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "modality": self.modality.value,
            "unit_id": self.unit_id,
            "text": self.text,
            "image_path": self.image_path,
            "page": self.page,
            "components": {k: round(v, 6) for k, v in self.components.items()},
            "meta": self.meta,
        }
