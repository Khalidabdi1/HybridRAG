"""Tests for the cost & storage model (hybridrag.eval.cost)."""

from hybridrag.config import HybridConfig
from hybridrag.engine import HybridRAG
from hybridrag.eval import CostModel, estimate_cost, sample_dataset
from hybridrag.eval.dataset import build_engine
from hybridrag.eval.cost import _distinct_pages, _human_bytes
from hybridrag.types import Tile


def _engine_with_both():
    """A small engine carrying text chunks and vision tiles across two pages."""
    eng = HybridRAG(HybridConfig())
    eng.add_text("doc1", "alpha beta gamma " * 80, title="Doc One")
    eng.add_tiles([
        Tile(id="doc1:p0:t0", doc_id="doc1", image_path="/x/doc1_p0_t0.png", page=0),
        Tile(id="doc1:p0:t1", doc_id="doc1", image_path="/x/doc1_p0_t1.png", page=0),
        Tile(id="doc1:p1:t0", doc_id="doc1", image_path="/x/doc1_p1_t0.png", page=1),
    ])
    return eng


def test_cost_model_to_dict_roundtrips_fields():
    cm = CostModel()
    d = cm.to_dict()
    assert "storage_usd_per_gb_month" in d
    # constructing from the dict yields an equal model
    assert CostModel(**d).to_dict() == d


def test_measured_vector_bytes_are_exact():
    eng = _engine_with_both()
    report = estimate_cost(eng)
    t_units = len(eng.text_store)
    v_units = len(eng.vision_store)
    text = report.mode("text")
    vision = report.mode("vision")
    assert text.vector_bytes == t_units * eng.text_embedder.dim * 4
    assert vision.vector_bytes == v_units * eng.vision_embedder.dim * 4


def test_hybrid_is_the_sum_of_text_and_vision():
    report = estimate_cost(_engine_with_both())
    t, v, h = report.mode("text"), report.mode("vision"), report.mode("hybrid")
    assert h.vector_bytes == t.vector_bytes + v.vector_bytes
    assert h.artifact_bytes == t.artifact_bytes + v.artifact_bytes
    assert abs(h.index_usd - (t.index_usd + v.index_usd)) < 1e-12
    assert abs(h.query_usd - (t.query_usd + v.query_usd)) < 1e-12


def test_vision_is_more_expensive_than_text_everywhere():
    report = estimate_cost(_engine_with_both())
    cm = report.cost_model
    t, v = report.mode("text"), report.mode("vision")
    # storage, index and query are all dominated by the pixel modality
    assert v.storage_bytes > t.storage_bytes
    assert v.index_usd > t.index_usd
    assert v.query_usd > t.query_usd
    assert v.storage_usd_per_month(cm) > t.storage_usd_per_month(cm)


def test_screenshot_attributed_per_distinct_page():
    eng = _engine_with_both()  # tiles span 2 distinct pages
    report = estimate_cost(eng)
    cm = report.cost_model
    v = report.mode("vision")
    pages = _distinct_pages(eng.vision_store)
    assert pages == 2
    expected_raw = pages * cm.screenshot_bytes_per_page + len(eng.vision_store) * cm.tile_bytes
    assert v.artifact_bytes == expected_raw


def test_projection_scales_linearly_and_vision_dwarfs_text():
    report = estimate_cost(_engine_with_both())
    p1 = report.project(1_000_000)
    p10 = report.project(10_000_000)
    # linear in corpus size
    assert p10["vision"]["total_bytes"] == 10 * p1["vision"]["total_bytes"]
    # the whole point: pixels cost vastly more to store at scale
    assert p1["vision"]["total_bytes"] > 10 * p1["text"]["total_bytes"]
    assert p1["hybrid"]["total_bytes"] == p1["text"]["total_bytes"] + p1["vision"]["total_bytes"]
    assert p10["vision"]["total_tb"] > 0


def test_cost_model_overrides_change_the_numbers():
    eng = _engine_with_both()
    cheap = estimate_cost(eng).mode("vision").storage_usd_per_month(CostModel())
    pricey_cm = CostModel(storage_usd_per_gb_month=0.23)  # 10x
    pricey = estimate_cost(eng, cost_model=pricey_cm).mode("vision").storage_usd_per_month(pricey_cm)
    assert abs(pricey - cheap * 10) < 1e-9


def test_tables_render_without_error():
    report = estimate_cost(sample_dataset_engine())
    assert "Cost model" in report.table()
    proj = report.projection_table(10_000_000)
    assert "Storage projection" in proj
    assert "10,000,000" in proj


def test_human_bytes_formatting():
    assert _human_bytes(0) == "0 B"
    assert _human_bytes(1536).endswith("KB")
    assert _human_bytes(5 * 1024 ** 4).endswith("TB")


def sample_dataset_engine():
    return build_engine(sample_dataset())


def test_mcp_cost_tool():
    from hybridrag.mcp_server import call_tool

    eng = _engine_with_both()
    out = call_tool("hybridrag_cost", {"project_pages": 5_000_000}, engine=eng)
    assert {"modes", "measured", "cost_model", "projection"} <= set(out)
    assert out["projection"]["pages"] == 5_000_000
    # pixels cost more to store at scale than text
    assert out["projection"]["vision"]["total_bytes"] > out["projection"]["text"]["total_bytes"]
