import pytest

from hybridrag.retrieve.fusion import (
    calibrated_fusion,
    fuse,
    reciprocal_rank_fusion,
)
from hybridrag.types import Modality, SearchResult


def _r(doc, modality, unit, score=0.0):
    return SearchResult(doc_id=doc, score=score, modality=modality, unit_id=unit)


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


# --------------------------------------------------------------- calibrated

def test_calibrated_reinforces_agreement():
    ranked = {
        Modality.TEXT: [
            _r("A", Modality.TEXT, "a1", 0.9),
            _r("B", Modality.TEXT, "b1", 0.4),
        ],
        Modality.VISION: [
            _r("A", Modality.VISION, "a_img", 0.8),
            _r("C", Modality.VISION, "c_img", 0.3),
        ],
    }
    weights = {Modality.TEXT: 1.0, Modality.VISION: 1.0}
    fused = calibrated_fusion(ranked, weights, top_k=10, norm="minmax")
    assert fused[0].doc_id == "A"
    assert "text" in fused[0].components and "vision" in fused[0].components


def test_calibrated_preserves_confidence_gap():
    # In one modality, the top hit is far more confident than the runner-up.
    # RRF flattens the gap (rank 0 vs 1); calibrated fusion keeps it, so the
    # confident doc leads a doc that RRF would rank almost equally.
    ranked = {
        Modality.TEXT: [
            _r("A", Modality.TEXT, "a1", 0.99),
            _r("B", Modality.TEXT, "b1", 0.20),
        ],
    }
    weights = {Modality.TEXT: 1.0, Modality.VISION: 1.0}
    fused = calibrated_fusion(ranked, weights, norm="minmax")
    scores = {r.doc_id: r.score for r in fused}
    assert scores["A"] > scores["B"]
    # Min-max puts the top at 1.0 and the bottom at 0.0 within this modality.
    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(0.0)


def test_calibrated_zero_weight_modality_ignored():
    ranked = {
        Modality.TEXT: [_r("A", Modality.TEXT, "a1", 0.5)],
        Modality.VISION: [_r("B", Modality.VISION, "b_img", 0.9)],
    }
    fused = calibrated_fusion(ranked, {Modality.TEXT: 1.0, Modality.VISION: 0.0})
    assert {r.doc_id for r in fused} == {"A"}


@pytest.mark.parametrize("norm", ["minmax", "zscore", "softmax"])
def test_calibrated_norms_stay_bounded_and_ordered(norm):
    ranked = {
        Modality.TEXT: [
            _r("A", Modality.TEXT, "a1", 0.9),
            _r("B", Modality.TEXT, "b1", 0.5),
            _r("C", Modality.TEXT, "c1", 0.1),
        ],
    }
    fused = calibrated_fusion(ranked, {Modality.TEXT: 1.0}, norm=norm)
    order = [r.doc_id for r in fused]
    assert order == ["A", "B", "C"]  # score order preserved by every norm
    for r in fused:
        assert 0.0 <= r.score <= 1.0


def test_calibrated_degenerate_scores_do_not_crash():
    # All-equal scores: minmax -> all 1.0, still returns every doc.
    ranked = {
        Modality.TEXT: [
            _r("A", Modality.TEXT, "a1", 0.5),
            _r("B", Modality.TEXT, "b1", 0.5),
        ],
    }
    fused = calibrated_fusion(ranked, {Modality.TEXT: 1.0}, norm="minmax")
    assert {r.doc_id for r in fused} == {"A", "B"}
    assert all(r.score == pytest.approx(1.0) for r in fused)


def test_fuse_dispatch_matches_direct_calls():
    ranked = {
        Modality.TEXT: [_r("A", Modality.TEXT, "a1", 0.9), _r("B", Modality.TEXT, "b1", 0.4)],
        Modality.VISION: [_r("A", Modality.VISION, "a_img", 0.8)],
    }
    weights = {Modality.TEXT: 1.0, Modality.VISION: 1.0}
    assert fuse(ranked, weights, method="rrf", top_k=10, rrf_k=60) == \
        reciprocal_rank_fusion(ranked, weights, rrf_k=60, top_k=10)
    assert fuse(ranked, weights, method="calibrated", top_k=10, norm="minmax") == \
        calibrated_fusion(ranked, weights, top_k=10, norm="minmax")


def test_fuse_rejects_unknown_method():
    with pytest.raises(ValueError):
        fuse({}, {}, method="bogus")
