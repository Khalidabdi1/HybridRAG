from hybridrag.config import HybridConfig
from hybridrag.engine import HybridRAG
from hybridrag.types import Modality, Tile


def _engine():
    return HybridRAG(HybridConfig())


def test_add_text_and_search():
    rag = _engine()
    n = rag.add_text("doc1", "The quick brown fox jumps over the lazy dog. " * 40, title="Foxes")
    assert n >= 1
    results = rag.search("quick brown fox", top_k=5)
    assert results
    assert results[0].doc_id == "doc1"


def test_search_separates_docs():
    rag = _engine()
    rag.add_text("animals", "Lions tigers and bears live in the wild forests. " * 30)
    rag.add_text("finance", "Quarterly revenue and profit margins grew this year. " * 30)
    res = rag.search("revenue and profit", top_k=3)
    assert res[0].doc_id == "finance"


def test_force_modality_text_only():
    rag = _engine()
    rag.add_text("d", "hello world content here. " * 20)
    rag.add_tiles([Tile(id="t1", doc_id="d", image_path=None)])
    res = rag.search("hello world", modality=Modality.TEXT)
    assert all(r.modality == Modality.TEXT for r in res)


def test_save_and_load(tmp_path):
    rag = _engine()
    rag.add_text("d", "persisted document content for round trip. " * 20)
    rag.save(str(tmp_path))
    loaded = HybridRAG.load(str(tmp_path))
    assert loaded.stats()["text_units"] == rag.stats()["text_units"]
    res = loaded.search("persisted document", top_k=3)
    assert res and res[0].doc_id == "d"


def test_tiles_indexed():
    rag = _engine()
    tiles = [Tile(id=f"t{i}", doc_id="img_doc", page=0, row=i) for i in range(3)]
    assert rag.add_tiles(tiles) == 3
    assert rag.stats()["vision_units"] == 3
