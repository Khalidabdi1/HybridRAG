"""Tests for selective pixel indexing (numpy only — no Pillow needed)."""

import numpy as np

from hybridrag.config import HybridConfig
from hybridrag.engine import HybridRAG
from hybridrag.mcp_server import call_tool
from hybridrag.pipeline.select import (
    PixelSelector,
    score_image_richness,
    score_text_richness,
)


# ----------------------------------------------------------------- synthetic pages
def _blank_page(h=1024, w=800):
    """A near-white page of plain prose: mostly white with a little sparse ink."""
    img = np.ones((h, w, 3), dtype="float32")
    rng = np.random.RandomState(0)
    # scatter a little dark "text" ink — sparse, never full-width.
    for _ in range(40):
        y = rng.randint(0, h - 4)
        x = rng.randint(0, w - 120)
        img[y : y + 2, x : x + 100] = 0.1
    return img


def _table_page(h=1024, w=800):
    """A page with ruled horizontal/vertical lines — a table."""
    img = np.ones((h, w, 3), dtype="float32")
    for y in range(80, h - 80, 60):  # horizontal rules
        img[y : y + 2, 40 : w - 40] = 0.0
    for x in range(40, w - 40, 120):  # vertical rules
        img[80 : h - 80, x : x + 2] = 0.0
    return img


def _chart_page(h=1024, w=800):
    """A colourful figure region — a chart."""
    img = np.ones((h, w, 3), dtype="float32")
    img[200:700, 100:700, 0] = 0.9  # a big saturated red/blue block
    img[200:700, 100:700, 1] = 0.2
    img[200:700, 100:700, 2] = 0.3
    return img


# ----------------------------------------------------------------- image richness
def test_blank_page_is_not_rich():
    assert score_image_richness(_blank_page()) < 0.2


def test_table_page_is_richer_than_blank():
    assert score_image_richness(_table_page()) > score_image_richness(_blank_page())


def test_chart_page_is_rich():
    assert score_image_richness(_chart_page()) > 0.35


def test_image_signals_breakdown():
    _, sig = score_image_richness(_table_page(), return_signals=True)
    assert set(sig) == {"lines", "color", "midtone"}
    assert sig["lines"] > 0.0


# ----------------------------------------------------------------- text richness
def test_prose_is_not_rich():
    prose = "The history of Rome spans many centuries of prose. " * 40
    assert score_text_richness(text=prose) < 0.3


def test_html_table_is_rich():
    html = "<html><body><table>" + "<tr><td>1</td><td>2</td></tr>" * 5 + "</table></body></html>"
    assert score_text_richness(html=html) > 0.5


def test_numeric_table_text_is_rich():
    rows = "\n".join("Item %d   %d.00   %d   %d%%" % (i, i * 10, i, i) for i in range(20))
    assert score_text_richness(text=rows) > score_text_richness(text="just some words here")


def test_code_stays_text():
    code = "def foo(x):\n    return x + 1\n\nclass Bar:\n    pass\n" * 10
    assert score_text_richness(text=code) < 0.4


# ----------------------------------------------------------------- policy
def test_policy_always_and_never():
    always = PixelSelector(mode="always")
    never = PixelSelector(mode="never")
    assert always.decide(text="anything").index_pixels is True
    assert never.decide(image=_chart_page()).index_pixels is False


def test_auto_skips_prose_indexes_table():
    sel = PixelSelector(mode="auto", threshold=0.35)
    assert sel.decide(image=_blank_page()).index_pixels is False
    assert sel.decide(image=_table_page()).index_pixels is True


def test_text_signal_can_trigger_render_pre_image():
    sel = PixelSelector(mode="auto", threshold=0.35)
    html = "<table>" + "<tr><td>a</td><td>b</td></tr>" * 6 + "</table>"
    # No image yet — a strong HTML table should still trigger the vision path.
    assert sel.decide(html=html).index_pixels is True


def test_no_signals_defaults_to_index():
    sel = PixelSelector(mode="auto")
    assert sel.decide().index_pixels is True


# ----------------------------------------------------------------- engine + MCP
def test_engine_should_index_pixels():
    rag = HybridRAG(HybridConfig(pixel_selection="auto"))
    assert rag.should_index_pixels(image=_chart_page()).index_pixels is True
    assert rag.should_index_pixels(text="plain prose " * 50).index_pixels is False


def test_engine_add_tiles_if_rich_skips():
    from hybridrag.types import Tile

    rag = HybridRAG(HybridConfig(pixel_selection="auto"))
    tiles = [Tile(id="t0", doc_id="d", page=0, image_path=None)]
    out = rag.add_tiles_if_rich(tiles, text="plain prose paragraph " * 50)
    assert out["added"] == 0
    assert out["decision"].index_pixels is False
    assert len(rag.vision_store) == 0


def test_config_roundtrip_includes_selection(tmp_path):
    cfg = HybridConfig(pixel_selection="never", pixel_selection_threshold=0.5)
    p = tmp_path / "config.json"
    cfg.to_file(str(p))
    loaded = HybridConfig.from_file(str(p))
    assert loaded.pixel_selection == "never"
    assert loaded.pixel_selection_threshold == 0.5


def test_mcp_richness_tool():
    out = call_tool("hybridrag_richness", {"html": "<table><tr><td>1</td></tr></table>", "mode": "auto"})
    assert "index_pixels" in out
    assert "score" in out
    assert out["mode"] == "auto"
