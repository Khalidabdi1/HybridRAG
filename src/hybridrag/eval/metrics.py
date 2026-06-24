"""Ranking metrics for retrieval evaluation.

All metrics take a ranked list of ``doc_id`` strings (best first) and a set of
relevant documents. Relevance may be a plain sequence of ids (binary relevance)
or a ``{doc_id: gain}`` mapping (graded relevance, used by nDCG). The functions
are pure and dependency-free so they are trivial to unit-test.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Union

Relevance = Union[Mapping[str, float], Sequence[str]]


def gain_map(relevant: Relevance) -> Dict[str, float]:
    """Normalize either form of relevance into a ``{doc_id: gain}`` dict."""
    if isinstance(relevant, Mapping):
        return {str(k): float(v) for k, v in relevant.items()}
    return {str(d): 1.0 for d in relevant}


def dedup(ranked: Sequence[str]) -> List[str]:
    """Drop duplicate doc ids while preserving first-seen order."""
    seen = set()
    out: List[str] = []
    for d in ranked:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def hit_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    """1.0 if any relevant document appears in the top ``k``, else 0.0."""
    gains = gain_map(relevant)
    return 1.0 if any(gains.get(d, 0.0) > 0 for d in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    """Fraction of all relevant documents retrieved within the top ``k``."""
    gains = gain_map(relevant)
    n_rel = sum(1 for g in gains.values() if g > 0)
    if n_rel == 0:
        return 0.0
    found = sum(1 for d in dedup(ranked)[:k] if gains.get(d, 0.0) > 0)
    return found / n_rel


def precision_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    if k <= 0:
        return 0.0
    gains = gain_map(relevant)
    found = sum(1 for d in dedup(ranked)[:k] if gains.get(d, 0.0) > 0)
    return found / k


def reciprocal_rank(ranked: Sequence[str], relevant: Relevance) -> float:
    """1/rank of the first relevant document (0.0 if none retrieved)."""
    gains = gain_map(relevant)
    for i, d in enumerate(dedup(ranked), 1):
        if gains.get(d, 0.0) > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    return sum(
        gains.get(d, 0.0) / math.log2(i + 1)
        for i, d in enumerate(dedup(ranked)[:k], 1)
    )


def ndcg_at_k(ranked: Sequence[str], relevant: Relevance, k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` (graded-relevance aware)."""
    gains = gain_map(relevant)
    dcg = dcg_at_k(ranked, gains, k)
    ideal = sorted((g for g in gains.values() if g > 0), reverse=True)
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal[:k], 1))
    return dcg / idcg if idcg > 0 else 0.0
