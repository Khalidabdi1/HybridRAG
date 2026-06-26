import numpy as np

from hybridrag.index import VectorStore


def test_add_and_search():
    store = VectorStore(dim=4)
    store.add(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
        [{"unit_id": "a", "doc_id": "a"}, {"unit_id": "b", "doc_id": "b"},
         {"unit_id": "c", "doc_id": "c"}],
    )
    hits = store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=2)
    assert hits[0][1]["doc_id"] == "a"
    assert hits[0][0] > hits[1][0]


def test_empty_search():
    assert VectorStore(dim=4).search(np.ones(4, dtype=np.float32)) == []


def test_save_and_load(tmp_path):
    store = VectorStore(dim=3)
    store.add(np.eye(3, dtype=np.float32), [{"unit_id": str(i), "doc_id": str(i)} for i in range(3)])
    store.save(str(tmp_path), "text")
    assert VectorStore.exists(str(tmp_path), "text")
    loaded = VectorStore.load(str(tmp_path), "text")
    assert len(loaded) == 3
    hits = loaded.search(np.array([0, 1, 0], dtype=np.float32), top_k=1)
    assert hits[0][1]["doc_id"] == "1"


def test_delete_doc_removes_records():
    store = VectorStore(dim=3)
    store.add(
        np.eye(3, dtype=np.float32),
        [{"unit_id": "a1", "doc_id": "a"}, {"unit_id": "a2", "doc_id": "a"},
         {"unit_id": "b1", "doc_id": "b"}],
    )
    removed = store.delete_doc("a")
    assert removed == 2
    assert len(store) == 1
    assert store.doc_ids() == {"b"}
    # surviving vector is still searchable and correctly aligned with its meta
    hits = store.search(np.array([0, 0, 1], dtype=np.float32), top_k=1)
    assert hits[0][1]["unit_id"] == "b1"


def test_delete_missing_doc_is_noop():
    store = VectorStore(dim=2)
    store.add(np.eye(2, dtype=np.float32), [{"doc_id": "x"}, {"doc_id": "y"}])
    assert store.delete_doc("absent") == 0
    assert len(store) == 2


def test_delete_all_then_add_again():
    store = VectorStore(dim=2)
    store.add(np.eye(2, dtype=np.float32), [{"doc_id": "x"}, {"doc_id": "x"}])
    assert store.delete_doc("x") == 2
    assert len(store) == 0
    # store remains usable after being emptied
    store.add(np.array([[1, 0]], dtype=np.float32), [{"doc_id": "z"}])
    assert len(store) == 1
    assert store.doc_ids() == {"z"}


def test_dim_mismatch_raises():
    store = VectorStore(dim=4)
    try:
        store.add(np.ones((1, 3), dtype=np.float32), [{"doc_id": "x"}])
    except ValueError:
        return
    raise AssertionError("expected ValueError on dim mismatch")
