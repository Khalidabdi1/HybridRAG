from hybridrag.retrieve.fusion import reciprocal_rank_fusion
from hybridrag.types import Modality, SearchResult


def _r(doc, modality, unit):
    return SearchResult(doc_id=doc, score=0.0, modality=modality, unit_id=unit)


def test_doc_found_by_both_modalities_ranks_first():
    ranked = {
        Modality.TEXT: [_r("A", Modality.TEXT, "a1"), _r("B", Modality.TEXT, "b1")],
        Modality.VISION: [_r("A", Modality.VISION, "a_img"), _r("C", Modality.VISION, "c_img")],
    }
    weights = {Modality.TEXT: 1.0, Modality.VISION: 1.0}
    fused = reciprocal_rank_fusion(ranked, weights, rrf_k=60, top_k=10)
    assert fused[0].doc_id == "A"
    # A has contributions from both modalities
    assert "text" in fused[0].components and "vision" in fused[0].components


def test_zero_weight_modality_ignored():
    ranked = {
        Modality.TEXT: [_r("A", Modality.TEXT, "a1")],
        Modality.VISION: [_r("B", Modality.VISION, "b_img")],
    }
    weights = {Modality.TEXT: 1.0, Modality.VISION: 0.0}
    fused = reciprocal_rank_fusion(ranked, weights)
    docs = {r.doc_id for r in fused}
    assert docs == {"A"}


def test_top_k_truncation():
    ranked = {Modality.TEXT: [_r(f"D{i}", Modality.TEXT, f"u{i}") for i in range(20)]}
    fused = reciprocal_rank_fusion(ranked, {Modality.TEXT: 1.0}, top_k=5)
    assert len(fused) == 5
