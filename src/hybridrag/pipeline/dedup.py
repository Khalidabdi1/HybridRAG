"""Tile-level deduplication — cut vision storage and VLM inference.

Pixel/Hybrid RAG's storage bill (disadvantage #1 from the project motivation) is
dominated by the vision modality: one screenshot per page, several tiles per
page, and an embedding per tile. But a large fraction of those tiles are
*byte-for-byte identical* — the running header, footer, logo, sidebar, or blank
margin that repeats on every page of a document. A 200-page report with a fixed
header band tiles into ~200 copies of the same header tile, and embedding +
storing each one is pure waste.

:class:`TileDeduplicator` collapses identical tiles to a single representative
*before* they are embedded and stored:

* **VLM inference saved** (disadvantage #4) — each unique tile is embedded once,
  not once per occurrence. On a real GPU encoder this is the dominant indexing
  cost, so dedup directly attacks slow, expensive indexing.
* **Vector + artifact storage saved** (disadvantage #1) — one stored record and
  one embedding per unique tile. Every other occurrence is recorded as metadata
  on the representative (``occurrences``), so citations still resolve to every
  page the tile appears on.

Scope
-----
Dedup is scoped **per ``doc_id``** by default (``scope="doc"``): only tiles from
the same document are collapsed together. This is where the repetition
overwhelmingly lives (a document's own header on every one of its pages) and it
keeps delete-by-``doc_id`` correct — a representative is only ever shared within
one document, so removing that document removes it cleanly, never orphaning a
tile another document still points at. ``scope="global"`` collapses across
documents for the maximum saving; use it when you never delete individual docs.

Matching
--------
Exact (byte) hashing is the default (``method="exact"``) and needs no
dependencies — it folds re-used files that are literally the same bytes. Tiles
with no readable image file (metadata-only tiles) fall back to hashing their
``image_path``/``id`` so they always stay distinct. Perceptual matching
(``method="ahash"``) folds tiles whose 8x8 average-hash is identical — the same
image re-encoded to a different PNG, a colour-profile change, a 1px crop — and
needs Pillow; it degrades gracefully to exact hashing when Pillow or the file is
unavailable.

Everything here is pure and deterministic; the deduplicator never touches the
store or the encoder, it only decides which tiles are worth embedding.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..types import Tile

_READ_CHUNK = 1 << 20  # 1 MiB


def exact_hash(tile: Tile) -> str:
    """SHA-256 of the tile's image bytes.

    Falls back to hashing the ``image_path`` (or ``id``) when the file cannot be
    read, so path-less / metadata-only tiles remain distinct and are never
    collapsed by accident.
    """
    path = tile.image_path
    if path:
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(_READ_CHUNK), b""):
                    h.update(block)
            return "sha256:" + h.hexdigest()
        except OSError:
            pass
    key = tile.image_path or tile.id
    return "id:" + hashlib.sha256(key.encode("utf-8")).hexdigest()


def average_hash(tile: Tile, size: int = 8) -> Optional[str]:
    """Perceptual 8x8 average-hash of the tile image, or ``None`` if unavailable.

    Grayscales and downsamples to ``size``x``size``, then emits one bit per pixel
    (>= mean). Two tiles with the same average-hash are visually identical up to
    minor re-encoding; returning ``None`` (no Pillow / unreadable file) signals
    the caller to fall back to :func:`exact_hash`.
    """
    path = tile.image_path
    if not path or not os.path.exists(path):
        return None
    try:
        from PIL import Image  # optional (render extra)
    except ImportError:  # pragma: no cover - exercised only without Pillow
        return None
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((size, size))
            pixels = list(small.tobytes())  # mode "L" -> one byte per pixel
    except OSError:
        return None
    if not pixels:
        return None
    avg = sum(pixels) / len(pixels)
    bits = 0
    for p in pixels:
        bits = (bits << 1) | (1 if p >= avg else 0)
    width = (size * size + 3) // 4
    return "ahash:" + format(bits, "0{}x".format(width))


@dataclass
class DedupStats:
    """What a deduplication pass found."""

    input_tiles: int = 0
    unique_tiles: int = 0
    duplicate_tiles: int = 0
    groups_with_duplicates: int = 0
    bytes_saved: int = 0  # estimated raw-artifact bytes NOT stored (duplicate files)
    method: str = "exact"
    scope: str = "doc"

    @property
    def dedup_ratio(self) -> float:
        """Fraction of input tiles that were duplicates (0.0 .. 1.0)."""
        return self.duplicate_tiles / self.input_tiles if self.input_tiles else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tiles": self.input_tiles,
            "unique_tiles": self.unique_tiles,
            "duplicate_tiles": self.duplicate_tiles,
            "groups_with_duplicates": self.groups_with_duplicates,
            "dedup_ratio": round(self.dedup_ratio, 4),
            "bytes_saved": self.bytes_saved,
            "method": self.method,
            "scope": self.scope,
        }


def _occurrence(tile: Tile) -> Dict[str, Any]:
    return {
        "unit_id": tile.id,
        "doc_id": tile.doc_id,
        "page": tile.page,
        "row": tile.row,
        "col": tile.col,
        "image_path": tile.image_path,
    }


def _clone_with_occurrences(rep: Tile, members: List[Tile]) -> Tile:
    """Copy ``rep`` (never mutate the caller's tile) and annotate occurrences."""
    meta = dict(rep.meta)
    meta["occurrences"] = [_occurrence(t) for t in members]
    meta["duplicate_count"] = len(members)
    return Tile(
        id=rep.id,
        doc_id=rep.doc_id,
        page=rep.page,
        row=rep.row,
        col=rep.col,
        image_path=rep.image_path,
        bbox=rep.bbox,
        meta=meta,
    )


class TileDeduplicator:
    """Collapse content-identical tiles to a single embedded/stored representative.

    ``method``
        ``"exact"`` (default) — SHA-256 of the tile bytes. ``"ahash"`` —
        perceptual 8x8 average-hash (needs Pillow; falls back to exact when a
        file can't be opened).
    ``scope``
        ``"doc"`` (default) — only collapse tiles that share a ``doc_id`` (keeps
        delete-by-doc correct). ``"global"`` — collapse across all documents.
    ``tile_bytes``
        Assumed on-disk size of one tile PNG, used only to *estimate*
        ``bytes_saved`` in the stats. Matches ``CostModel.tile_bytes``.
    """

    def __init__(
        self,
        method: str = "exact",
        scope: str = "doc",
        tile_bytes: int = 200_000,
    ) -> None:
        if method not in ("exact", "ahash"):
            raise ValueError("method must be 'exact' or 'ahash'")
        if scope not in ("doc", "global"):
            raise ValueError("scope must be 'doc' or 'global'")
        self.method = method
        self.scope = scope
        self.tile_bytes = int(tile_bytes)

    def content_key(self, tile: Tile) -> str:
        if self.method == "ahash":
            key = average_hash(tile)
            if key is not None:
                return key
        return exact_hash(tile)

    def _group_key(self, tile: Tile) -> Tuple[str, str]:
        scope_key = tile.doc_id if self.scope == "doc" else "*"
        return (scope_key, self.content_key(tile))

    def deduplicate(self, tiles: List[Tile]) -> Tuple[List[Tile], DedupStats]:
        """Return ``(unique_tiles, stats)``.

        The returned list preserves first-seen order. A group of size 1 yields
        the *original* tile untouched; a group of size > 1 yields a fresh copy of
        the first tile carrying ``meta["occurrences"]`` and
        ``meta["duplicate_count"]`` for the whole group.
        """
        stats = DedupStats(
            input_tiles=len(tiles), method=self.method, scope=self.scope
        )
        if not tiles:
            return [], stats

        order: List[Tuple[str, str]] = []
        groups: Dict[Tuple[str, str], List[Tile]] = {}
        for tile in tiles:
            gkey = self._group_key(tile)
            bucket = groups.get(gkey)
            if bucket is None:
                groups[gkey] = [tile]
                order.append(gkey)
            else:
                bucket.append(tile)

        unique: List[Tile] = []
        for gkey in order:
            members = groups[gkey]
            if len(members) == 1:
                unique.append(members[0])
                continue
            stats.groups_with_duplicates += 1
            stats.bytes_saved += (len(members) - 1) * self.tile_bytes
            unique.append(_clone_with_occurrences(members[0], members))

        stats.unique_tiles = len(unique)
        stats.duplicate_tiles = stats.input_tiles - stats.unique_tiles
        return unique, stats


def build_deduplicator(config: Any) -> Optional[TileDeduplicator]:
    """Construct the deduplicator a :class:`HybridConfig` asks for, or ``None``.

    ``tile_dedup == "off"`` disables dedup entirely (returns ``None``); ``"doc"``
    / ``"global"`` select the scope. ``tile_dedup_method`` picks exact vs
    perceptual matching.
    """
    mode = getattr(config, "tile_dedup", "doc")
    if mode in (None, "off", "none", False):
        return None
    method = getattr(config, "tile_dedup_method", "exact")
    tile_bytes = 200_000
    extra = getattr(config, "extra", None)
    if isinstance(extra, dict) and "tile_bytes" in extra:
        tile_bytes = int(extra["tile_bytes"])
    return TileDeduplicator(method=method, scope=mode, tile_bytes=tile_bytes)


# --------------------------------------------------------------------------- #
# Directory helpers — turn rendered tile files into Tiles for offline analysis.
# --------------------------------------------------------------------------- #

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _doc_id_from_filename(name: str) -> str:
    """Infer a doc_id from a tile filename produced by the render pipeline.

    Tiles are named ``{doc_id}_p{page}_t{row}.png`` (see
    :func:`hybridrag.pipeline.render.tile_image`), so the doc_id is the prefix
    before ``_p``. Falls back to the whole stem when the pattern is absent.
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    marker = stem.rfind("_p")
    return stem[:marker] if marker > 0 else stem


def tiles_from_paths(paths: List[str]) -> List[Tile]:
    """Build :class:`Tile` objects from image paths, inferring ``doc_id``.

    Useful for a dry-run dedup analysis over already-rendered tiles without
    re-plumbing them through ingestion.
    """
    tiles: List[Tile] = []
    for i, path in enumerate(paths):
        tiles.append(
            Tile(
                id=os.path.splitext(os.path.basename(path))[0] or f"tile{i}",
                doc_id=_doc_id_from_filename(path),
                image_path=path,
            )
        )
    return tiles


def scan_tiles(directory: str) -> List[Tile]:
    """Recursively collect image files under ``directory`` as :class:`Tile`.

    ``doc_id`` is inferred from each filename; a per-directory grouping is not
    imposed so this works for the flat ``<storage>/tiles`` layout the render
    pipeline writes.
    """
    paths: List[str] = []
    for root, _dirs, files in os.walk(directory):
        for fn in sorted(files):
            if fn.lower().endswith(_IMAGE_EXTS):
                paths.append(os.path.join(root, fn))
    paths.sort()
    return tiles_from_paths(paths)
