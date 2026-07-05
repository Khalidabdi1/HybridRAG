"""Batched, parallel ingestion for large corpora.

Indexing is Pixel/Hybrid RAG's slowest phase (disadvantage #2 from the project
motivation): every page is extracted, chunked, optionally rendered + tiled, and
then embedded. The naive path in :class:`~hybridrag.engine.HybridRAG` embeds one
document at a time — ``add_text`` calls ``encode()`` once per document — which
wastes a real encoder's throughput: a GPU model amortizes almost all of its cost
over the batch, so 10,000 one-chunk calls are dramatically slower than a handful
of 128-chunk calls.

:class:`BatchIngestor` fixes both halves of that cost:

* **Batched embedding.** Prepared chunks/tiles are *buffered across documents*
  and flushed to the vector store in fixed-size batches, so each ``encode()``
  call sees ``batch_size`` units at once. This is the exact shape a batched GPU
  encoder wants, and it also cuts per-call overhead on the numpy fallback.
* **Parallel preparation.** The CPU/IO-bound prep of each document — HTML→text,
  chunking, tiling metadata, and the selective-pixel richness decision — runs on
  a small thread pool, so the next document is being prepared while the current
  batch is being embedded. Store mutation stays single-threaded and in input
  order, so the result is byte-for-byte identical to serial ingestion.

Everything here runs on numpy alone; the batching simply changes *when* the
existing, tested :meth:`HybridRAG.add_chunks` / :meth:`HybridRAG.add_tiles`
encode calls happen, never their metadata.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..pipeline.extract import chunk_text, html_to_text
from ..types import Chunk, Tile


@dataclass
class IngestDoc:
    """One document to ingest.

    Supply text directly (``text``) or as ``html`` (converted with
    :func:`html_to_text`). ``image_paths`` are already-rendered page/tile PNGs to
    push through the vision modality — rendering itself (Playwright/PyMuPDF) is a
    separate, optional step, so the ingestor stays dependency-free and testable.
    """

    doc_id: str
    text: Optional[str] = None
    html: Optional[str] = None
    title: str = ""
    page: int = 0
    image_paths: Optional[List[str]] = None
    apply_selection: bool = True  # honour the selective-pixel policy for image_paths
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestDoc":
        doc_id = data.get("doc_id") or data.get("id")
        if not doc_id:
            raise ValueError("each document needs a `doc_id`")
        return cls(
            doc_id=str(doc_id),
            text=data.get("text"),
            html=data.get("html"),
            title=data.get("title", ""),
            page=int(data.get("page", 0)),
            image_paths=data.get("image_paths") or data.get("images"),
            apply_selection=bool(data.get("apply_selection", True)),
            meta=data.get("meta") or {},
        )


@dataclass
class IngestStats:
    """What a batch ingest run produced."""

    documents: int = 0
    chunks_added: int = 0
    tiles_added: int = 0
    tiles_deduplicated: int = 0  # tiles dropped as content-duplicates before embedding
    vision_skipped: int = 0  # docs whose images were skipped by pixel selection
    text_batches: int = 0
    vision_batches: int = 0
    seconds: float = 0.0

    @property
    def docs_per_second(self) -> float:
        return self.documents / self.seconds if self.seconds > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks_added": self.chunks_added,
            "tiles_added": self.tiles_added,
            "tiles_deduplicated": self.tiles_deduplicated,
            "vision_skipped": self.vision_skipped,
            "text_batches": self.text_batches,
            "vision_batches": self.vision_batches,
            "seconds": round(self.seconds, 4),
            "docs_per_second": round(self.docs_per_second, 2),
        }


@dataclass
class _Prepared:
    """Result of preparing one document (pure, thread-safe)."""

    doc_id: str
    chunks: List[Chunk]
    tiles: List[Tile]
    index_pixels: bool
    reason: str = ""


class BatchIngestor:
    """Buffer prepared units across documents and embed them in batches.

    Typical use::

        ingestor = BatchIngestor(engine, batch_size=128, max_workers=4)
        stats = ingestor.ingest(docs)   # docs: Iterable[IngestDoc | dict]
        engine.save()

    ``batch_size`` controls how many units accumulate before an ``encode()``
    flush (text and vision buffers flush independently). ``max_workers`` > 1
    prepares documents on a thread pool while batches embed; the store is still
    written on a single thread in input order.
    """

    def __init__(
        self,
        engine: "Any",
        batch_size: int = 128,
        vision_batch_size: Optional[int] = None,
        max_workers: int = 1,
        upsert: bool = False,
        on_progress: Optional[Callable[[IngestStats], None]] = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.engine = engine
        self.batch_size = batch_size
        self.vision_batch_size = vision_batch_size or batch_size
        self.max_workers = max(1, int(max_workers))
        self.upsert = upsert
        self.on_progress = on_progress

    # ------------------------------------------------------------------ prepare
    def _prepare(self, doc: IngestDoc) -> _Prepared:
        """Extract, chunk, tile, and score one document. Pure — no store writes."""
        text = doc.text
        if text is None and doc.html is not None:
            text = html_to_text(doc.html)
        chunks: List[Chunk] = []
        if text:
            chunks = chunk_text(
                text,
                doc_id=doc.doc_id,
                chunk_size=self.engine.config.chunk_size,
                overlap=self.engine.config.chunk_overlap,
                page=doc.page,
            )
            for c in chunks:
                if doc.title:
                    c.meta["title"] = doc.title
                if doc.meta:
                    c.meta.update(doc.meta)

        tiles: List[Tile] = []
        index_pixels = False
        reason = ""
        if doc.image_paths:
            index_pixels = True
            if doc.apply_selection:
                decision = self.engine.should_index_pixels(
                    text=text, image=doc.image_paths[0]
                )
                index_pixels = decision.index_pixels
                reason = decision.reason
            if index_pixels:
                for row, path in enumerate(doc.image_paths):
                    tiles.append(
                        Tile(
                            id=f"{doc.doc_id}_p{doc.page}_t{row}",
                            doc_id=doc.doc_id,
                            page=doc.page,
                            row=row,
                            col=0,
                            image_path=path,
                            meta=dict(doc.meta),
                        )
                    )
        return _Prepared(doc.doc_id, chunks, tiles, index_pixels, reason)

    # -------------------------------------------------------------------- flush
    def _flush_text(self, buf: List[Chunk], stats: IngestStats) -> None:
        if not buf:
            return
        stats.chunks_added += self.engine.add_chunks(buf)
        stats.text_batches += 1
        buf.clear()

    def _flush_tiles(self, buf: List[Tile], stats: IngestStats) -> None:
        if not buf:
            return
        stats.tiles_added += self.engine.add_tiles(buf)
        last = getattr(self.engine, "last_dedup", None)
        if last is not None:
            stats.tiles_deduplicated += last.duplicate_tiles
        stats.vision_batches += 1
        buf.clear()

    # ------------------------------------------------------------------- ingest
    def ingest(self, docs: Iterable[Any]) -> IngestStats:
        """Ingest an iterable of :class:`IngestDoc` (or dicts). Batches internally.

        Returns an :class:`IngestStats`. Does *not* persist — call
        ``engine.save()`` afterwards (once, not per document, is the point).
        """
        stats = IngestStats()
        start = time.perf_counter()
        text_buf: List[Chunk] = []
        tile_buf: List[Tile] = []
        deleted: set = set()

        def consume(prepared: _Prepared, had_images: bool) -> None:
            if self.upsert and prepared.doc_id not in deleted:
                self.engine.delete(prepared.doc_id)
                deleted.add(prepared.doc_id)
            if prepared.chunks:
                text_buf.extend(prepared.chunks)
                if len(text_buf) >= self.batch_size:
                    self._flush_text(text_buf, stats)
            if prepared.tiles:
                tile_buf.extend(prepared.tiles)
                if len(tile_buf) >= self.vision_batch_size:
                    self._flush_tiles(tile_buf, stats)
            if had_images and not prepared.index_pixels:
                stats.vision_skipped += 1
            stats.documents += 1
            if self.on_progress is not None:
                self.on_progress(stats)

        norm = (d if isinstance(d, IngestDoc) else IngestDoc.from_dict(d) for d in docs)

        if self.max_workers == 1:
            for doc in norm:
                had_images = bool(doc.image_paths)
                consume(self._prepare(doc), had_images)
        else:
            # Bounded look-ahead: prepare up to `window` docs concurrently while
            # consuming (embedding) in submission order. Keeps memory bounded and
            # store writes deterministic.
            window = self.max_workers * 2
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                pending: deque = deque()
                for doc in norm:
                    pending.append((pool.submit(self._prepare, doc), bool(doc.image_paths)))
                    if len(pending) >= window:
                        fut, had_images = pending.popleft()
                        consume(fut.result(), had_images)
                while pending:
                    fut, had_images = pending.popleft()
                    consume(fut.result(), had_images)

        self._flush_text(text_buf, stats)
        self._flush_tiles(tile_buf, stats)
        stats.seconds = time.perf_counter() - start
        return stats
