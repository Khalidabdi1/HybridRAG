"""Tests for tile-level deduplication (`hybridrag.pipeline.dedup`).

These run on numpy alone. Byte-exact dedup is exercised with real (tiny) files
written directly; the perceptual `ahash` path is guarded by Pillow like the rest
of the vision suite.
"""

from __future__ import annotations

import pytest

from hybridrag.config import HybridConfig
from hybridrag.engine import HybridRAG
from hybridrag.pipeline.dedup import (
    DedupStats,
    TileDeduplicator,
    build_deduplicator,
    exact_hash,
    scan_tiles,
    tiles_from_paths,
)
from hybridrag.pipeline.ingest import IngestStats
from hybridrag.types import Modality, Tile


def _write(path, data: bytes):
    with open(path, "wb") as fh:
        fh.write(data)
    return str(path)


def _tile(id, doc_id, path=None, page=0, row=0):
    return Tile(id=id, doc_id=doc_id, page=page, row=row, image_path=path)


# ------------------------------------------------------------------ core dedup

def test_identical_files_collapse_within_doc(tmp_path):
    header = _write(tmp_path / "h.png", b"HEADER-BYTES")
    body0 = _write(tmp_path / "b0.png", b"BODY-0")
    body1 = _write(tmp_path / "b1.png", b"BODY-1")
    # Same doc, header appears on both pages plus two distinct bodies.
    tiles = [
        _tile("d_p0_t0", "d", header, page=0, row=0),
        _tile("d_p0_t1", "d", body0, page=0, row=1),
        _tile("d_p1_t0", "d", header, page=1, row=0),
        _tile("d_p1_t1", "d", body1, page=1, row=1),
    ]
    unique, stats = TileDeduplicator().deduplicate(tiles)
    assert len(unique) == 3
    assert stats.input_tiles == 4
    assert stats.unique_tiles == 3
    assert stats.duplicate_tiles == 1
    assert stats.groups_with_duplicates == 1
    assert stats.dedup_ratio == pytest.approx(0.25)
    # The header representative records both occurrences.
    header_rep = next(t for t in unique if t.image_path == header)
    occ = header_rep.meta["occurrences"]
    assert header_rep.meta["duplicate_count"] == 2
    assert {o["page"] for o in occ} == {0, 1}


def test_scope_doc_does_not_cross_documents(tmp_path):
    logo = _write(tmp_path / "logo.png", b"SHARED-LOGO")
    tiles = [_tile("a_p0_t0", "a", logo), _tile("b_p0_t0", "b", logo)]
    unique_doc, stats_doc = TileDeduplicator(scope="doc").deduplicate(tiles)
    assert len(unique_doc) == 2  # different docs -> kept separate
    assert stats_doc.duplicate_tiles == 0

    unique_global, stats_global = TileDeduplicator(scope="global").deduplicate(tiles)
    assert len(unique_global) == 1  # global scope collapses across docs
    assert stats_global.duplicate_tiles == 1


def test_pathless_tiles_stay_distinct():
    # No image files -> hash falls back to id/path, so nothing collapses.
    tiles = [_tile(f"t{i}", "d", None, row=i) for i in range(3)]
    unique, stats = TileDeduplicator().deduplicate(tiles)
    assert len(unique) == 3
    assert stats.duplicate_tiles == 0


def test_single_tile_passes_through_untouched(tmp_path):
    p = _write(tmp_path / "x.png", b"ONE")
    original = _tile("d_p0_t0", "d", p)
    unique, stats = TileDeduplicator().deduplicate([original])
    assert unique[0] is original  # identity: no clone for singleton groups
    assert "occurrences" not in original.meta
    assert stats.unique_tiles == 1


def test_first_seen_order_preserved(tmp_path):
    a = _write(tmp_path / "a.png", b"A")
    b = _write(tmp_path / "b.png", b"B")
    tiles = [
        _tile("d_p0_t0", "d", a, page=0),
        _tile("d_p1_t0", "d", b, page=1),
        _tile("d_p2_t0", "d", a, page=2),  # dup of first
    ]
    unique, _ = TileDeduplicator().deduplicate(tiles)
    assert [t.image_path for t in unique] == [a, b]


def test_empty_input():
    unique, stats = TileDeduplicator().deduplicate([])
    assert unique == []
    assert stats.input_tiles == 0
    assert stats.dedup_ratio == 0.0


def test_bytes_saved_estimate(tmp_path):
    p = _write(tmp_path / "p.png", b"REPEAT")
    tiles = [_tile(f"d_p{i}_t0", "d", p, page=i) for i in range(5)]
    _unique, stats = TileDeduplicator(tile_bytes=1000).deduplicate(tiles)
    assert stats.duplicate_tiles == 4
    assert stats.bytes_saved == 4000  # (5 - 1) copies * 1000 bytes


def test_invalid_options():
    with pytest.raises(ValueError):
        TileDeduplicator(method="bogus")
    with pytest.raises(ValueError):
        TileDeduplicator(scope="bogus")


def test_exact_hash_reads_file_bytes(tmp_path):
    p1 = _write(tmp_path / "one.png", b"SAME")
    p2 = _write(tmp_path / "two.png", b"SAME")  # different name, same bytes
    assert exact_hash(_tile("t1", "d", p1)) == exact_hash(_tile("t2", "d", p2))
    p3 = _write(tmp_path / "three.png", b"DIFFERENT")
    assert exact_hash(_tile("t3", "d", p3)) != exact_hash(_tile("t1", "d", p1))


def test_ahash_folds_reencoded_but_byte_different_copies(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    import numpy as np

    arr = np.random.RandomState(0).randint(0, 255, (40, 40, 3), dtype=np.uint8)
    png = tmp_path / "a.png"
    bmp = tmp_path / "a.bmp"  # same pixels, different container -> different bytes
    Image.fromarray(arr).save(png)
    Image.fromarray(arr).save(bmp)
    t_png = _tile("d_p0_t0", "d", str(png), page=0)
    t_bmp = _tile("d_p1_t0", "d", str(bmp), page=1)

    # Exact byte hashing sees them as distinct...
    _u_exact, s_exact = TileDeduplicator(method="exact").deduplicate([t_png, t_bmp])
    assert s_exact.duplicate_tiles == 0
    # ...but perceptual average-hash folds them together.
    _u_ahash, s_ahash = TileDeduplicator(method="ahash").deduplicate([t_png, t_bmp])
    assert s_ahash.duplicate_tiles == 1


# ------------------------------------------------------------------- factory

def test_build_deduplicator_from_config():
    assert build_deduplicator(HybridConfig(tile_dedup="off")) is None
    d = build_deduplicator(HybridConfig(tile_dedup="global", tile_dedup_method="exact"))
    assert isinstance(d, TileDeduplicator)
    assert d.scope == "global"


def test_stats_to_dict():
    d = DedupStats(input_tiles=10, unique_tiles=4, duplicate_tiles=6).to_dict()
    assert d["dedup_ratio"] == 0.6
    assert d["input_tiles"] == 10


# ------------------------------------------------------- directory / filename

def test_doc_id_inferred_from_tile_filename(tmp_path):
    p = _write(tmp_path / "report2024_p3_t1.png", b"X")
    tiles = tiles_from_paths([p])
    assert tiles[0].doc_id == "report2024"


def test_scan_tiles_walks_directory(tmp_path):
    sub = tmp_path / "tiles"
    sub.mkdir()
    _write(sub / "d_p0_t0.png", b"A")
    _write(sub / "d_p1_t0.png", b"A")  # duplicate content
    _write(sub / "notes.txt", b"ignore me")
    tiles = scan_tiles(str(sub))
    assert len(tiles) == 2  # .txt ignored
    _unique, stats = TileDeduplicator().deduplicate(tiles)
    assert stats.duplicate_tiles == 1


# --------------------------------------------------------------- engine wiring

def test_engine_dedup_reduces_vision_store(tmp_path):
    header = _write(tmp_path / "hdr.png", b"HDR")
    body = _write(tmp_path / "bdy.png", b"BDY")
    rag = HybridRAG(HybridConfig())  # tile_dedup defaults to "doc"
    tiles = [
        _tile("d_p0_t0", "d", header, page=0, row=0),
        _tile("d_p0_t1", "d", body, page=0, row=1),
        _tile("d_p1_t0", "d", header, page=1, row=0),
    ]
    added = rag.add_tiles(tiles)
    assert added == 2  # header collapsed
    assert len(rag.vision_store) == 2
    assert rag.last_dedup.duplicate_tiles == 1
    stats = rag.stats()
    assert stats["tiles_deduplicated"] == 1
    assert stats["tile_dedup"] == "doc"


def test_engine_dedup_off_keeps_all(tmp_path):
    header = _write(tmp_path / "hdr.png", b"HDR")
    rag = HybridRAG(HybridConfig(tile_dedup="off"))
    tiles = [_tile("d_p0_t0", "d", header, page=0), _tile("d_p1_t0", "d", header, page=1)]
    assert rag.add_tiles(tiles) == 2
    assert len(rag.vision_store) == 2
    assert rag.last_dedup is None


def test_dedup_survives_delete_by_doc(tmp_path):
    header = _write(tmp_path / "hdr.png", b"HDR")
    rag = HybridRAG(HybridConfig())
    rag.add_tiles([_tile("d_p0_t0", "d", header, page=0), _tile("d_p1_t0", "d", header, page=1)])
    assert len(rag.vision_store) == 1  # collapsed within doc "d"
    removed = rag.delete("d")
    assert removed["vision_removed"] == 1
    assert len(rag.vision_store) == 0  # doc-scoped dedup -> clean delete


def test_batch_ingest_reports_deduplicated(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    import numpy as np

    # One header tile repeated as separate files with identical bytes.
    def png(path, val):
        Image.fromarray(np.full((32, 32, 3), val, dtype=np.uint8)).save(path)
        return str(path)

    h0 = png(tmp_path / "h0.png", 100)
    h1 = png(tmp_path / "h1.png", 100)  # same pixels/bytes as h0
    rag = HybridRAG(HybridConfig(pixel_selection="always"))
    stats = rag.add_documents(
        [{"doc_id": "d", "text": "t", "image_paths": [h0, h1]}], batch_size=8
    )
    assert isinstance(stats, IngestStats)
    assert stats.tiles_added == 1
    assert stats.tiles_deduplicated == 1


def test_dedup_does_not_break_vision_search(tmp_path):
    header = _write(tmp_path / "hdr.png", b"HDR-CONTENT")
    body = _write(tmp_path / "bdy.png", b"BDY-CONTENT")
    rag = HybridRAG(HybridConfig())
    rag.add_tiles([
        _tile("d_p0_t0", "d", header, page=0),
        _tile("d_p0_t1", "d", body, page=0),
        _tile("d_p1_t0", "d", header, page=1),
    ])
    res = rag.search("hdr content", modality=Modality.VISION, top_k=2)
    assert res and all(r.doc_id == "d" for r in res)
