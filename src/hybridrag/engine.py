"""The HybridRAG engine: ingest documents into two modalities and search both.

This is the public entry point most users want::

    from hybridrag import HybridConfig
    from hybridrag.engine import HybridRAG

    rag = HybridRAG(HybridConfig())
    rag.add_text("doc1", "Some long document text ...", title="Demo")
    results = rag.search("what does the document say?")

It keeps two :class:`VectorStore` instances — one for text chunks, one for image
tiles — and fuses their results at query time. Real model backends are opt-in
via config (``text_model`` / ``vision_model``); the default hashing encoders let
everything run with only numpy.
"""

from __future__ import annotations

import os
from typing import List, Optional

from .config import HybridConfig
from .embed import build_text_embedder, build_vision_embedder
from .index import VectorStore
from .pipeline.extract import chunk_text, html_to_text
from .retrieve.fusion import reciprocal_rank_fusion
from .retrieve.router import route
from .types import Chunk, Modality, SearchResult, Tile


class HybridRAG:
    def __init__(self, config: Optional[HybridConfig] = None) -> None:
        self.config = config or HybridConfig()
        self.text_embedder = build_text_embedder(self.config)
        self.vision_embedder = build_vision_embedder(self.config)
        self.text_store = VectorStore(dim=self.text_embedder.dim)
        self.vision_store = VectorStore(dim=self.vision_embedder.dim)

    # ------------------------------------------------------------------ ingest
    def add_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.text_embedder.encode([c.text for c in chunks])
        metas = [
            {
                "unit_id": c.id,
                "doc_id": c.doc_id,
                "text": c.text,
                "page": c.page,
                "order": c.order,
                "meta": c.meta,
            }
            for c in chunks
        ]
        self.text_store.add(vectors, metas)
        return len(chunks)

    def add_tiles(self, tiles: List[Tile]) -> int:
        if not tiles:
            return 0
        paths = [t.image_path or t.id for t in tiles]
        vectors = self.vision_embedder.encode(paths)
        metas = [
            {
                "unit_id": t.id,
                "doc_id": t.doc_id,
                "image_path": t.image_path,
                "page": t.page,
                "row": t.row,
                "col": t.col,
                "meta": t.meta,
            }
            for t in tiles
        ]
        self.vision_store.add(vectors, metas)
        return len(tiles)

    def add_text(self, doc_id: str, text: str, title: str = "", page: int = 0) -> int:
        """Chunk and index raw text. Returns the number of chunks added."""
        chunks = chunk_text(
            text,
            doc_id=doc_id,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            page=page,
        )
        for c in chunks:
            if title:
                c.meta["title"] = title
        return self.add_chunks(chunks)

    def add_html(self, doc_id: str, html: str, title: str = "") -> int:
        return self.add_text(doc_id, html_to_text(html), title=title)

    # ------------------------------------------------------------ update/delete
    def delete(self, doc_id: str) -> dict:
        """Remove all text chunks and image tiles belonging to ``doc_id``.

        Returns the number of units removed per modality. This is the cheap
        incremental path that avoids re-indexing the whole corpus when a single
        document is removed or about to be replaced.
        """
        removed_text = self.text_store.delete_doc(doc_id)
        removed_vision = self.vision_store.delete_doc(doc_id)
        return {"text_removed": removed_text, "vision_removed": removed_vision}

    def upsert_text(self, doc_id: str, text: str, title: str = "", page: int = 0) -> dict:
        """Replace a document's text: delete any existing chunks, then re-add.

        Use this when a source document changed — only this ``doc_id`` is
        re-embedded, the rest of the index is untouched.
        """
        removed = self.delete(doc_id)
        added = self.add_text(doc_id, text, title=title, page=page)
        return {"doc_id": doc_id, "chunks_added": added, **removed}

    def upsert_html(self, doc_id: str, html: str, title: str = "") -> dict:
        """HTML counterpart of :meth:`upsert_text`."""
        return self.upsert_text(doc_id, html_to_text(html), title=title)

    def doc_ids(self) -> set:
        """Every distinct ``doc_id`` present in either modality."""
        return self.text_store.doc_ids() | self.vision_store.doc_ids()

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        modality: Optional[Modality] = None,
    ) -> List[SearchResult]:
        """Search both modalities and fuse. ``modality`` forces a single one."""
        top_k = top_k or self.config.top_k

        if self.config.enable_router and modality is None:
            decision = route(query, self.config.text_weight, self.config.vision_weight)
            weights = decision.weights()
        else:
            weights = {
                Modality.TEXT: self.config.text_weight,
                Modality.VISION: self.config.vision_weight,
            }
        if modality is not None:
            for m in list(weights):
                if m != modality:
                    weights[m] = 0.0

        # Over-fetch per modality so fusion has material to work with.
        fetch = max(top_k * 3, top_k)
        ranked = {}

        if weights.get(Modality.TEXT, 0.0) > 0 and len(self.text_store):
            qv = self.text_embedder.encode([query])
            hits = self.text_store.search(qv, top_k=fetch)
            ranked[Modality.TEXT] = [
                SearchResult(
                    doc_id=m["doc_id"],
                    score=s,
                    modality=Modality.TEXT,
                    unit_id=m["unit_id"],
                    text=m.get("text", ""),
                    page=m.get("page", 0),
                    meta=m.get("meta", {}),
                )
                for s, m in hits
            ]

        if weights.get(Modality.VISION, 0.0) > 0 and len(self.vision_store):
            qv = self.vision_embedder.encode_query([query])
            hits = self.vision_store.search(qv, top_k=fetch)
            ranked[Modality.VISION] = [
                SearchResult(
                    doc_id=m["doc_id"],
                    score=s,
                    modality=Modality.VISION,
                    unit_id=m["unit_id"],
                    image_path=m.get("image_path"),
                    page=m.get("page", 0),
                    meta=m.get("meta", {}),
                )
                for s, m in hits
            ]

        return reciprocal_rank_fusion(
            ranked, weights, rrf_k=self.config.rrf_k, top_k=top_k
        )

    # ------------------------------------------------------------- persistence
    def save(self, directory: Optional[str] = None) -> str:
        directory = directory or self.config.storage_dir
        os.makedirs(directory, exist_ok=True)
        self.config.to_file(os.path.join(directory, "config.json"))
        self.text_store.save(directory, "text")
        self.vision_store.save(directory, "vision")
        return directory

    @classmethod
    def load(cls, directory: str) -> "HybridRAG":
        config = HybridConfig.from_file(os.path.join(directory, "config.json"))
        engine = cls(config)
        if VectorStore.exists(directory, "text"):
            engine.text_store = VectorStore.load(directory, "text")
        if VectorStore.exists(directory, "vision"):
            engine.vision_store = VectorStore.load(directory, "vision")
        return engine

    def stats(self) -> dict:
        return {
            "documents": len(self.doc_ids()),
            "text_units": len(self.text_store),
            "vision_units": len(self.vision_store),
            "text_model": self.config.text_model,
            "vision_model": self.config.vision_model,
            "text_dim": self.text_embedder.dim,
            "vision_dim": self.vision_embedder.dim,
        }
