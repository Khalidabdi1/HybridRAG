import math

from hybridrag.eval import metrics as M


def test_recall_and_precision_basic():
    ranked = ["a", "b", "c", "d"]
    rel = ["a", "c"]
    assert M.recall_at_k(ranked, rel, 4) == 1.0
    assert M.recall_at_k(ranked, rel, 1) == 0.5
    assert M.precision_at_k(ranked, rel, 2) == 0.5  # a relevant, b not
    assert M.precision_at_k(ranked, rel, 4) == 0.5


def test_hit_at_k():
    ranked = ["x", "y", "a"]
    assert M.hit_at_k(ranked, ["a"], 3) == 1.0
    assert M.hit_at_k(ranked, ["a"], 2) == 0.0


def test_reciprocal_rank():
    assert M.reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert M.reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert M.reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_ndcg_perfect_and_graded():
    # perfect ranking -> 1.0
    assert M.ndcg_at_k(["a", "b"], {"a": 1.0, "b": 1.0}, 2) == 1.0
    # graded: putting the higher-gain doc first beats the reverse
    good = M.ndcg_at_k(["a", "b"], {"a": 3.0, "b": 1.0}, 2)
    bad = M.ndcg_at_k(["b", "a"], {"a": 3.0, "b": 1.0}, 2)
    assert good == 1.0
    assert bad < good
    # known value: gain 3 at rank2, gain1 at rank1 vs ideal
    expected = (1.0 / math.log2(2) + 3.0 / math.log2(3)) / (
        3.0 / math.log2(2) + 1.0 / math.log2(3)
    )
    assert abs(bad - expected) < 1e-9


def test_empty_relevance_is_zero():
    assert M.recall_at_k(["a"], [], 1) == 0.0
    assert M.ndcg_at_k(["a"], [], 1) == 0.0


def test_dedup_preserves_order():
    assert M.dedup(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall_counts_distinct_docs_only():
    # duplicate relevant doc in the ranked list shouldn't inflate recall
    ranked = ["a", "a", "a"]
    assert M.recall_at_k(ranked, ["a", "b"], 3) == 0.5
