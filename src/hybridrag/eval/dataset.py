"""Evaluation datasets: a small typed container plus loaders.

A dataset is a corpus of documents (each carrying text and/or an image tile)
together with a set of queries and their relevance judgements (``qrels``). The
same dataset is indexed once and then queried under each modality so that
text-only, vision-only, and hybrid retrieval are compared on equal footing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..config import HybridConfig
from ..engine import HybridRAG
from ..types import Tile


@dataclass
class EvalDocument:
    doc_id: str
    text: str = ""
    title: str = ""
    image_path: Optional[str] = None
    page: int = 0

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())

    @property
    def has_image(self) -> bool:
        return bool(self.image_path)


@dataclass
class EvalQuery:
    query: str
    # Either ["docA", "docB"] (binary) or {"docA": 2.0, "docB": 1.0} (graded).
    relevant: Union[List[str], Dict[str, float]]
    id: str = ""


@dataclass
class EvalDataset:
    name: str
    documents: List[EvalDocument] = field(default_factory=list)
    queries: List[EvalQuery] = field(default_factory=list)

    @property
    def n_text_docs(self) -> int:
        return sum(1 for d in self.documents if d.has_text)

    @property
    def n_image_docs(self) -> int:
        return sum(1 for d in self.documents if d.has_image)

    # ---- loaders ----
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalDataset":
        docs = [
            EvalDocument(
                doc_id=str(d["doc_id"]),
                text=d.get("text", ""),
                title=d.get("title", ""),
                image_path=d.get("image_path"),
                page=int(d.get("page", 0)),
            )
            for d in data.get("documents", [])
        ]
        queries = [
            EvalQuery(query=q["query"], relevant=q["relevant"], id=str(q.get("id", "")))
            for q in data.get("queries", [])
        ]
        return cls(name=data.get("name", "dataset"), documents=docs, queries=queries)

    @classmethod
    def from_file(cls, path: str) -> "EvalDataset":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "text": d.text,
                    "title": d.title,
                    "image_path": d.image_path,
                    "page": d.page,
                }
                for d in self.documents
            ],
            "queries": [
                {"id": q.id, "query": q.query, "relevant": q.relevant} for q in self.queries
            ],
        }


def build_engine(dataset: EvalDataset, config: Optional[HybridConfig] = None) -> HybridRAG:
    """Index a dataset into a fresh :class:`HybridRAG` engine.

    Text documents are chunked and added to the text store; documents carrying
    an ``image_path`` are added to the vision store as a single tile. A document
    may contribute to both modalities.
    """
    engine = HybridRAG(config or HybridConfig())
    tiles: List[Tile] = []
    for doc in dataset.documents:
        if doc.has_text:
            engine.add_text(doc.doc_id, doc.text, title=doc.title, page=doc.page)
        if doc.has_image:
            tiles.append(
                Tile(
                    id=f"{doc.doc_id}:tile",
                    doc_id=doc.doc_id,
                    image_path=doc.image_path,
                    page=doc.page,
                )
            )
    if tiles:
        engine.add_tiles(tiles)
    return engine
