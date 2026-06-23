"""Result fusion across modalities.

Reciprocal Rank Fusion (RRF) combines ranked lists without needing the score
scales to be comparable — ideal here, since cosine similarities from a text
encoder and a vision encoder are not on the same scale. Each list contributes
``weight / (rrf_k + rank)`` to every item it ranks.
"""

from __future__ import annotations

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
    fused: Dict[str, SearchResult] = {}
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

    for doc, comps in contributions.items():
        _, rep = best_rank[doc]
        total = sum(comps.values())
        fused[doc] = SearchResult(
            doc_id=doc,
            score=total,
            modality=rep.modality,
            unit_id=rep.unit_id,
            text=rep.text,
            image_path=rep.image_path,
            page=rep.page,
            components=comps,
            meta=rep.meta,
        )

    ordered = sorted(fused.values(), key=lambda r: r.score, reverse=True)
    return ordered[:top_k]
