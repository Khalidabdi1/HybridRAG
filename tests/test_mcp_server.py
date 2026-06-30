"""Tests for the MCP server tool layer.

These exercise the dependency-light handler logic (no `mcp` package needed),
which is what the MCP wiring delegates to.
"""

from __future__ import annotations

import pytest

from hybridrag.config import HybridConfig
from hybridrag.engine import HybridRAG
from hybridrag.mcp_server import TOOL_SPECS, call_tool, load_engine


@pytest.fixture()
def engine():
    eng = HybridRAG(HybridConfig())
    eng.add_text("doc1", "The quarterly revenue table shows growth across regions.", title="Q1")
    eng.add_text("doc2", "A JSON config file with logging levels and handlers.", title="Cfg")
    return eng


def test_tool_specs_are_well_formed():
    names = {t["name"] for t in TOOL_SPECS}
    assert {"hybridrag_search", "hybridrag_add_text", "hybridrag_add_html", "hybridrag_stats"} <= names
    for spec in TOOL_SPECS:
        assert spec["description"]
        assert spec["inputSchema"]["type"] == "object"


def test_search_returns_ranked_results(engine):
    out = call_tool("hybridrag_search", {"query": "revenue table", "top_k": 3}, engine=engine)
    assert out["query"] == "revenue table"
    assert out["count"] >= 1
    assert out["results"][0]["doc_id"] in {"doc1", "doc2"}
    # every result is JSON-serialisable with the expected shape
    assert "score" in out["results"][0]
    assert "modality" in out["results"][0]


def test_search_requires_query(engine):
    with pytest.raises(ValueError):
        call_tool("hybridrag_search", {}, engine=engine)


def test_answer_returns_grounded_cited_answer(engine):
    out = call_tool("hybridrag_answer", {"query": "quarterly revenue growth"}, engine=engine)
    assert out["query"] == "quarterly revenue growth"
    assert out["text"]
    assert out["reader"] == "extractive"
    assert out["citations"]
    assert out["citations"][0]["doc_id"] in {"doc1", "doc2"}


def test_answer_requires_query(engine):
    with pytest.raises(ValueError):
        call_tool("hybridrag_answer", {}, engine=engine)


def test_search_force_modality(engine):
    out = call_tool("hybridrag_search", {"query": "logging", "modality": "text"}, engine=engine)
    assert all(r["modality"] == "text" for r in out["results"])


def test_add_text_persists(tmp_path):
    storage = str(tmp_path / "idx")
    eng = HybridRAG(HybridConfig(storage_dir=storage))
    out = call_tool(
        "hybridrag_add_text",
        {"doc_id": "d1", "text": "hello world " * 50, "title": "T"},
        engine=eng,
    )
    assert out["chunks_added"] >= 1
    # reloading the persisted index should see the data
    reloaded = load_engine(storage)
    assert reloaded.stats()["text_units"] >= 1


def test_add_html_extracts_and_indexes(tmp_path):
    eng = HybridRAG(HybridConfig(storage_dir=str(tmp_path / "idx")))
    out = call_tool(
        "hybridrag_add_html",
        {"doc_id": "h1", "html": "<h1>Title</h1><p>Body text here.</p>"},
        engine=eng,
    )
    assert out["chunks_added"] >= 1


def test_stats_tool(engine):
    out = call_tool("hybridrag_stats", {}, engine=engine)
    assert out["text_units"] >= 1
    assert "text_model" in out
    assert "storage" in out


def test_delete_tool_removes_doc(engine):
    out = call_tool("hybridrag_delete", {"doc_id": "doc1"}, engine=engine)
    assert out["doc_id"] == "doc1"
    assert out["text_removed"] >= 1
    listed = call_tool("hybridrag_list_docs", {}, engine=engine)
    assert "doc1" not in listed["doc_ids"]
    assert "doc2" in listed["doc_ids"]


def test_delete_tool_requires_doc_id(engine):
    with pytest.raises(ValueError):
        call_tool("hybridrag_delete", {}, engine=engine)


def test_update_text_tool_replaces(engine):
    out = call_tool(
        "hybridrag_update_text",
        {"doc_id": "doc1", "text": "completely rewritten body about turbines. " * 10},
        engine=engine,
    )
    assert out["doc_id"] == "doc1"
    assert out["text_removed"] >= 1
    assert out["chunks_added"] >= 1
    res = call_tool("hybridrag_search", {"query": "turbines", "modality": "text"}, engine=engine)
    assert res["results"] and res["results"][0]["doc_id"] == "doc1"


def test_list_docs_tool(engine):
    out = call_tool("hybridrag_list_docs", {}, engine=engine)
    assert out["count"] == 2
    assert set(out["doc_ids"]) == {"doc1", "doc2"}


def test_unknown_tool_raises(engine):
    with pytest.raises(KeyError):
        call_tool("nope", {}, engine=engine)
