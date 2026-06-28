from hybridrag.config import HybridConfig
from hybridrag.engine import HybridRAG
from hybridrag.retrieve.rerank import (
    LexicalReranker,
    build_reranker,
    rerank_results,
)
from hybridrag.types import Modality, SearchResult


def _r(doc, score, text="", modality=Modality.TEXT):
    return SearchResult(
        doc_id=doc, score=score, modality=modality, unit_id=doc + "_u", text=text
    )


# --------------------------------------------------------------- LexicalReranker

def test_bm25_prefers_documents_matching_query_terms():
    rr = LexicalReranker()
    docs = [
        "the cat sat on the mat in the sun",
        "quarterly revenue grew on strong cloud demand",
        "a dog ran across the green field",
    ]
    scores = rr.score("cloud revenue growth", docs)
    assert scores[1] == max(scores)
    assert scores[1] > 0.0


def test_bm25_zero_when_no_query_terms_or_no_overlap():
    rr = LexicalReranker()
    assert rr.score("", ["anything"]) == [0.0]
    assert rr.score("zzz qqq", ["completely different words here"]) == [0.0]


def test_bm25_rare_term_outweighs_common_term():
    rr = LexicalReranker()
    # "the" appears everywhere (low IDF); "photosynthesis" is rare (high IDF).
    docs = [
        "the the the the the",
        "the leaf performs photosynthesis using the light",
        "the the the the the",
    ]
    scores = rr.score("the photosynthesis", docs)
    assert scores[1] == max(scores)


# --------------------------------------------------------------- build_reranker

def test_build_reranker_off_by_default():
    assert build_reranker(HybridConfig()) is None


def test_build_reranker_lexical_when_enabled():
    cfg = HybridConfig(enable_rerank=True, rerank_model="lexical")
    rr = build_reranker(cfg)
    assert isinstance(rr, LexicalReranker)


def test_build_reranker_none_model_disables():
    cfg = HybridConfig(enable_rerank=True, rerank_model="none")
    assert build_reranker(cfg) is None


# --------------------------------------------------------------- rerank_results

def test_rerank_reorders_by_relevance_to_query():
    # Fusion ranked the off-topic doc first; reranking should fix that.
    results = [
        _r("off", score=1.0, text="unrelated football scores and weather"),
        _r("hit", score=0.9, text="annual revenue and profit margin analysis"),
    ]
    rr = LexicalReranker()
    out = rerank_results(rr, "revenue profit margin", results, blend=1.0)
    assert out[0].doc_id == "hit"
    assert "rerank" in out[0].components


def test_rerank_blend_zero_preserves_fusion_order():
    results = [
        _r("a", score=1.0, text="alpha"),
        _r("b", score=0.5, text="revenue revenue revenue"),
    ]
    out = rerank_results(LexicalReranker(), "revenue", results, blend=0.0)
    assert [r.doc_id for r in out] == ["a", "b"]


def test_rerank_keeps_image_only_candidate_by_fusion_standing():
    # An image-only tile (no text) should not be dumped to the bottom.
    results = [
        _r("img", score=1.0, text="", modality=Modality.VISION),
        _r("txt", score=0.2, text="nothing relevant here"),
    ]
    out = rerank_results(LexicalReranker(), "revenue", results, blend=0.5)
    assert out[0].doc_id == "img"


def test_rerank_top_k_truncates():
    results = [_r(f"d{i}", score=1.0 - i * 0.1, text=f"doc {i}") for i in range(6)]
    out = rerank_results(LexicalReranker(), "doc", results, blend=0.5, top_k=3)
    assert len(out) == 3


# --------------------------------------------------------------- engine wiring

def test_engine_search_rerank_param_overrides_config():
    cfg = HybridConfig(rerank_blend=1.0)  # rerank off by default, pure-rerank order
    rag = HybridRAG(cfg)
    rag.add_text("d1", "the company revenue growth was strong this fiscal year", title="A")
    rag.add_text("d2", "a recipe for chocolate chip cookies", title="B")
    # Forcing rerank on for a single query should still work end to end.
    results = rag.search("revenue growth", rerank=True)
    assert results
    assert results[0].doc_id == "d1"
    assert "rerank" in results[0].components


def test_engine_config_enable_rerank_persists_reranker():
    cfg = HybridConfig(enable_rerank=True)
    rag = HybridRAG(cfg)
    assert rag.reranker is not None
    rag.add_text("d1", "machine learning model training pipeline", title="A")
    out = rag.search("machine learning training")
    assert out and "rerank" in out[0].components
