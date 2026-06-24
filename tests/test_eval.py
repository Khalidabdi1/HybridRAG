from hybridrag.config import HybridConfig
from hybridrag.eval import build_engine, evaluate, sample_dataset
from hybridrag.eval.dataset import EvalDataset


def test_sample_dataset_shape():
    ds = sample_dataset()
    assert ds.n_text_docs == len(ds.documents)
    assert len(ds.queries) >= 8
    # every query's gold doc exists in the corpus
    ids = {d.doc_id for d in ds.documents}
    for q in ds.queries:
        for doc in q.relevant:
            assert doc in ids


def test_evaluate_text_baseline_is_strong():
    ds = sample_dataset()
    engine = build_engine(ds)
    report = evaluate(engine, ds.queries, ks=(1, 5, 10), dataset_name=ds.name)
    text = next(m for m in report.modes if m.mode == "text")
    # lexical baseline should comfortably find most gold docs in the top 10
    assert text.recall[10] >= 0.75
    assert text.mrr > 0.5
    assert text.indexed_units == len(ds.documents)


def test_vision_mode_empty_is_noted_not_dropped():
    ds = sample_dataset()
    report = evaluate(build_engine(ds), ds.queries, modes=("text", "vision", "hybrid"))
    vision = next(m for m in report.modes if m.mode == "vision")
    assert vision.indexed_units == 0
    assert vision.recall[max(report.ks)] == 0.0
    assert "skipped" in vision.note


def test_report_table_and_dict_roundtrip():
    ds = sample_dataset()
    report = evaluate(build_engine(ds), ds.queries, dataset_name=ds.name)
    table = report.table()
    assert "recall" in table and "hybrid" in table
    d = report.to_dict()
    assert d["dataset"] == ds.name
    assert {m["mode"] for m in d["modes"]} == {"text", "vision", "hybrid"}
    assert d["index"]["text_units"] == len(ds.documents)


def test_dataset_json_roundtrip():
    ds = sample_dataset()
    restored = EvalDataset.from_dict(ds.to_dict())
    assert restored.name == ds.name
    assert len(restored.documents) == len(ds.documents)
    assert restored.queries[0].query == ds.queries[0].query


def test_vision_docs_route_to_vision_store():
    ds = EvalDataset(
        name="img",
        documents=[EvalDocument_with_image()],
        queries=[],
    )
    engine = build_engine(ds, HybridConfig())
    assert len(engine.vision_store) == 1


def EvalDocument_with_image():
    from hybridrag.eval.dataset import EvalDocument

    # image_path need not exist on disk: the hashing vision encoder falls back
    # to hashing the path string when the file is unreadable.
    return EvalDocument(doc_id="img1", image_path="/nonexistent/tile.png")
