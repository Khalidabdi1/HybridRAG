"""Result fusion across modalities.

Two strategies live here, both grouping hits by ``doc_id`` so a document found
by both modalities is reinforced:

* **Reciprocal Rank Fusion (RRF)** — combines ranked lists using only *rank*,
  never the raw scores. Robust when text-encoder and vision-encoder cosine
  similarities are on incomparable scales, but it throws away *confidence*: a
  near-exact top hit and a mediocre one a rank apart contribute almost the same.

* **Score-calibrated fusion** — first normalizes each modality's raw similarity
  scores onto a common ``[0, 1]`` scale (min-max, squashed z-score, or softmax),
  then combines the calibrated scores with the per-modality weights. This keeps
  the *magnitude* of a match: a modality that is very confident about a document
  pushes it up more than a lukewarm one, and the size of the gap between hits
  survives fusion. Useful when the score distributions are informative and
  roughly stationary (e.g. after calibrating a single encoder pair).

``fuse`` dispatches between them so callers pick a method by name.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from ..types import Modality, SearchResult


def reciprocal_rank_fusion(
    ranked: Dict[Modality, List[SearchResult]],
    weights: Dict[Modality, float],
    rrf_k: int = 60,
    top_k: int = 10,
) -> List[SearchResult]:
    """Fuse per-modality ranked lists into one list keyed by document.

    Results are grouped by ``doc_id`` so that a document found by both modalities
    is reinforced. The surviving representative for a doc is the highest-ranked
    unit across modalities, annotated with per-modality contributions.
    """
    contributions: Dict[str, Dict[str, float]] = {}
    best_rank: Dict[str, Tuple[int, SearchResult]] = {}

    for modality, results in ranked.items():
        weight = weights.get(modality, 0.0)
        if weight == 0.0:
            continue
        for rank, res in enumerate(results):
            contrib = weight / (rrf_k + rank)
            doc = res.doc_id
            contributions.setdefault(doc, {})
            contributions[doc][modality.value] = contributions[doc].get(modality.value, 0.0) + contrib
            # Keep the best (lowest) rank's unit as the representative.
            if doc not in best_rank or rank < best_rank[doc][0]:
                best_rank[doc] = (rank, res)

    return _assemble(contributions, {d: rep for d, (_, rep) in best_rank.items()}, top_k)


def _normalize_scores(scores: List[float], norm: str, softmax_temp: float) -> List[float]:
    """Map a modality's raw similarity scores onto a common ``[0, 1]`` scale.

    * ``"minmax"`` — linear rescale to ``[0, 1]``; a degenerate list (all equal)
      maps to ``1.0`` (every hit equally confident).
    * ``"zscore"`` — standardize then squash through a logistic so the result
      stays in ``(0, 1)`` and combines additively; a zero-variance list maps to
      ``0.5``.
    * ``"softmax"`` — temperature-scaled softmax over the list, so the scores
      form a distribution that sharpens as ``softmax_temp`` shrinks.
    """
    n = len(scores)
    if n == 0:
        return []
    if norm == "minmax":
        lo, hi = min(scores), max(scores)
        if hi - lo <= 1e-12:
            return [1.0] * n
        return [(s - lo) / (hi - lo) for s in scores]
    if norm == "zscore":
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(var)
        if std <= 1e-12:
            return [0.5] * n
        return [1.0 / (1.0 + math.exp(-(s - mean) / std)) for s in scores]
    if norm == "softmax":
        temp = softmax_temp if softmax_temp > 1e-6 else 1e-6
        m = max(scores)
        exps = [math.exp((s - m) / temp) for s in scores]
        total = sum(exps)
        if total <= 0:
            return [1.0 / n] * n
        return [e / total for e in exps]
    raise ValueError(f"unknown fusion norm: {norm!r} (want minmax|zscore|softmax)")


def calibrated_fusion(
    ranked: Dict[Modality, List[SearchResult]],
    weights: Dict[Modality, float],
    top_k: int = 10,
    norm: str = "minmax",
    softmax_temp: float = 0.1,
) -> List[SearchResult]:
    """Fuse by combining *calibrated* per-modality scores rather than ranks.

    Each modality's raw scores are normalized to a comparable ``[0, 1]`` scale
    (see :func:`_normalize_scores`), scaled by the modality weight, and summed
    per ``doc_id``. Unlike RRF, the confidence magnitude and the gaps between
    hits survive fusion. The representative unit for a doc is the one with the
    single largest calibrated contribution.
    """
    contributions: Dict[str, Dict[str, float]] = {}
    best: Dict[str, Tuple[float, SearchResult]] = {}

    for modality, results in ranked.items():
        weight = weights.get(modality, 0.0)
        if weight == 0.0 or not results:
            continue
        normed = _normalize_scores([r.score for r in results], norm, softmax_temp)
        for res, ns in zip(results, normed):
            contrib = weight * ns
            doc = res.doc_id
            contributions.setdefault(doc, {})
            contributions[doc][modality.value] = contributions[doc].get(modality.value, 0.0) + contrib
            # Keep the unit with the largest single calibrated contribution.
            if doc not in best or contrib > best[doc][0]:
                best[doc] = (contrib, res)

    return _assemble(contributions, {d: rep for d, (_, rep) in best.items()}, top_k)


def _assemble(
    contributions: Dict[str, Dict[str, float]],
    reps: Dict[str, SearchResult],
    top_k: int,
) -> List[SearchResult]:
    """Build fused results from per-doc contributions + representatives."""
    fused: List[SearchResult] = []
    for doc, comps in contributions.items():
        rep = reps[doc]
        fused.append(
            SearchResult(
                doc_id=doc,
                score=sum(comps.values()),
                modality=rep.modality,
                unit_id=rep.unit_id,
                text=rep.text,
                image_path=rep.image_path,
                page=rep.page,
                components=comps,
                meta=rep.meta,
            )
        )
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused[:top_k]


def fuse(
    ranked: Dict[Modality, List[SearchResult]],
    weights: Dict[Modality, float],
    method: str = "rrf",
    top_k: int = 10,
    rrf_k: int = 60,
    norm: str = "minmax",
    softmax_temp: float = 0.1,
) -> List[SearchResult]:
    """Dispatch to a fusion strategy by name (``"rrf"`` or ``"calibrated"``)."""
    if method == "rrf":
        return reciprocal_rank_fusion(ranked, weights, rrf_k=rrf_k, top_k=top_k)
    if method == "calibrated":
        return calibrated_fusion(
            ranked, weights, top_k=top_k, norm=norm, softmax_temp=softmax_temp
        )
    raise ValueError(f"unknown fusion method: {method!r} (want rrf|calibrated)")
