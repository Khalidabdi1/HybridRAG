"""Tests for batched, parallel ingestion (`hybridrag.pipeline.ingest`).

These run on numpy alone — no models, no browser. Image docs use tiny PNGs
written with Pillow (an install-time dep of the render extra, present in CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from hybridrag.config import HybridConfig
from hybridrag.engine import HybridRAG
from hybridrag.pipeline.ingest import BatchIngestor, IngestDoc, IngestStats
from hybridrag.types import Modality


def _engine(**overrides):
    return HybridRAG(HybridConfig(**overrides))


def _docs(n=30):
    return [
        {"doc_id": f"d{i}", "text": f"Document {i} about topic {i % 3}. " * 20}
        for i in range(n)
    ]


def test_ingest_indexes_all_documents():
    rag = _engine()
    stats = rag.add_documents(_docs(30), batch_size=8)
    assert isinstance(stats, IngestStats)
    assert stats.documents == 30
    assert stats.chunks_added == len(rag.text_store)
    assert len(rag.doc_ids()) == 30
    assert stats.text_batches >= 1


def test_batching_flushes_in_batches_not_per_doc():
    rag = _engine()
    # 20 docs, each a single chunk, batch_size 5 -> exactly 4 encode batches.
    docs = [{"doc_id": f"d{i}", "text": f"short doc {i}"} for i in range(20)]
    stats = rag.add_documents(docs, batch_size=5)
    assert stats.chunks_added == 20
    assert stats.text_batches == 4  # not 20 — the whole point


def test_parallel_matches_serial_exactly():
    docs = _docs(40)
    serial = _engine()
    serial.add_documents(docs, batch_size=8, max_workers=1)
    parallel = _engine()
    parallel.add_documents(docs, batch_size=8, max_workers=4)
    assert np.allclose(serial.text_store._vectors, parallel.text_store._vectors)
    assert serial.text_store._meta == parallel.text_store._meta


def test_ingest_matches_add_text_baseline():
    docs = _docs(10)
    batched = _engine()
    batched.add_documents(docs, batch_size=4)
    baseline = _engine()
    for d in docs:
        baseline.add_text(d["doc_id"], d["text"])
    assert np.allclose(baseline.text_store._vectors, batched.text_store._vectors)
    assert baseline.text_store._meta == batched.text_store._meta


def test_search_after_batch_ingest():
    rag = _engine()
    rag.add_documents(
        [
            {"doc_id": "animals", "text": "Lions tigers and bears roam the forest. " * 20},
            {"doc_id": "finance", "text": "Quarterly revenue and profit margins grew. " * 20},
        ],
        batch_size=4,
    )
    res = rag.search("revenue and profit", top_k=3)
    assert res[0].doc_id == "finance"


def test_html_documents_are_extracted():
    rag = _engine()
    stats = rag.add_documents(
        [{"doc_id": "h1", "html": "<html><body><p>Hello <b>world</b> content</p></body></html>"}],
        batch_size=4,
    )
    assert stats.chunks_added >= 1
    res = rag.search("hello world content", top_k=1)
    assert res and res[0].doc_id == "h1"


def test_title_and_meta_carried_onto_chunks():
    rag = _engine()
    rag.add_documents(
        [{"doc_id": "d", "text": "some content here. " * 20, "title": "My Title",
          "meta": {"source": "unit-test"}}],
        batch_size=4,
    )
    res = rag.search("content", top_k=1)
    assert res[0].meta.get("title") == "My Title"
    assert res[0].meta.get("source") == "unit-test"


def test_upsert_replaces_document():
    rag = _engine()
    rag.add_documents([{"doc_id": "d0", "text": "original content about cats " * 10}], batch_size=4)
    before = len(rag.doc_ids())
    rag.add_documents([{"doc_id": "d0", "text": "brand new content about zebras"}],
                      batch_size=4, upsert=True)
    assert len(rag.doc_ids()) == before  # replaced, not duplicated
    res = rag.search("zebras", top_k=1)
    assert res and res[0].doc_id == "d0"
    assert "cats" not in res[0].text


def test_progress_callback_is_called_per_doc():
    rag = _engine()
    seen = []
    rag.add_documents(_docs(5), batch_size=2, on_progress=lambda s: seen.append(s.documents))
    assert seen == [1, 2, 3, 4, 5]


def test_empty_and_textless_docs_are_safe():
    rag = _engine()
    stats = rag.add_documents(
        [{"doc_id": "empty", "text": ""}, {"doc_id": "d1", "text": "real content here " * 10}],
        batch_size=4,
    )
    assert stats.documents == 2
    assert "d1" in rag.doc_ids()


def test_ingest_doc_dataclass_accepted_directly():
    rag = _engine()
    docs = [IngestDoc(doc_id="x", text="content about turtles " * 10)]
    stats = rag.add_documents(docs, batch_size=4)
    assert stats.documents == 1
    assert "x" in rag.doc_ids()


def test_ingest_doc_from_dict_requires_doc_id():
    with pytest.raises(ValueError):
        IngestDoc.from_dict({"text": "no id"})


def test_bad_batch_size_rejected():
    with pytest.raises(ValueError):
        BatchIngestor(_engine(), batch_size=0)


def test_stats_to_dict_and_throughput():
    stats = IngestStats(documents=10, chunks_added=20, seconds=2.0)
    d = stats.to_dict()
    assert d["documents"] == 10 and d["chunks_added"] == 20
    assert d["docs_per_second"] == 5.0
    assert IngestStats(documents=1, seconds=0.0).docs_per_second == 0.0


# ------------------------------------------------------------------ vision path

@pytest.fixture()
def rich_png(tmp_path):
    from PIL import Image

    arr = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    arr[::20, :, :] = 0  # ruled lines -> reads as tabular/rich
    path = tmp_path / "rich.png"
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.fixture()
def plain_png(tmp_path):
    from PIL import Image

    path = tmp_path / "plain.png"
    Image.fromarray(np.full((400, 600, 3), 250, dtype=np.uint8)).save(path)
    return str(path)


def test_rich_image_is_indexed_into_vision(rich_png):
    rag = _engine(pixel_selection="always")
    stats = rag.add_documents(
        [{"doc_id": "tbl", "text": "revenue table", "image_paths": [rich_png]}],
        batch_size=4,
    )
    assert stats.tiles_added == 1
    assert len(rag.vision_store) == 1
    res = rag.search("revenue table", modality=Modality.VISION, top_k=1)
    assert res and res[0].doc_id == "tbl"


def test_selection_skips_text_native_page(plain_png):
    rag = _engine(pixel_selection="auto")
    stats = rag.add_documents(
        [{"doc_id": "prose", "text": "Just prose about nothing much. " * 40,
          "image_paths": [plain_png]}],
        batch_size=4,
    )
    assert stats.tiles_added == 0
    assert stats.vision_skipped == 1
    assert len(rag.vision_store) == 0


def test_selection_override_via_apply_selection(plain_png):
    rag = _engine(pixel_selection="auto")
    stats = rag.add_documents(
        [{"doc_id": "prose", "text": "prose " * 40, "image_paths": [plain_png],
          "apply_selection": False}],
        batch_size=4,
    )
    assert stats.tiles_added == 1  # forced in despite low richness
