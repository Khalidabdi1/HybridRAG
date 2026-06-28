"""Reranking of the fused candidate set.

Reciprocal Rank Fusion (see :mod:`hybridrag.retrieve.fusion`) is a *score-free*
combiner — it only looks at the rank a unit got from each modality, never at the
query/document content together. That makes it robust across incomparable score
scales, but it also means two documents that a modality ranked #3 and #4 are
treated as nearly identical even when one is a far better answer to *this*
query.

A reranker fixes that by scoring each surviving candidate against the query
*jointly* — the cross-encoder idea. Real cross-encoders (e.g. a
``sentence-transformers`` ``CrossEncoder``) read the query and document together
and are markedly more precise than bi-encoder retrieval; they are also too slow
to run over a whole corpus, which is exactly why they belong here, on the small
fused candidate set.

This module keeps the same pattern as the embedders: a dependency-free default
(:class:`LexicalReranker`, a BM25 scorer over the candidate set) so HybridRAG
reranks meaningfully on numpy alone, and an opt-in :class:`CrossEncoderReranker`
that loads a real model when one is configured.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Optional, Sequence

from ..config import HybridConfig
from ..types import SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class Reranker(ABC):
    """Scores ``(query, document)`` pairs jointly.

    ``score`` returns one relevance value per document. The absolute scale does
    not matter — :func:`rerank_results` min-max normalizes before blending — but
    higher must mean more relevant.
    """

    name: str = "reranker"

    @abstractmethod
    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        ...


class LexicalReranker(Reranker):
    """BM25 scorer over the candidate set, treated as a tiny corpus.

    Unlike the bag-of-hashed-tokens retrieval encoder, BM25 weights rare query
    terms (IDF) and saturates term frequency, so it discriminates *within* a
    small set of already-relevant candidates far better than cosine on hashed
    vectors. It needs no model and is fully deterministic, which makes it a
    genuine quality lift available offline rather than a placeholder.
    """

    name = "lexical"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        q_terms = _tokenize(query)
        if not q_terms or not documents:
            return [0.0] * len(documents)

        doc_tokens = [_tokenize(d) for d in documents]
        doc_len = [len(t) for t in doc_tokens]
        n = len(documents)
        avgdl = (sum(doc_len) / n) or 1.0

        # Document frequency of each term across the candidate set.
        df: Counter = Counter()
        for toks in doc_tokens:
            for term in set(toks):
                df[term] += 1

        q_unique = set(q_terms)
        idf = {
            term: math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            for term in q_unique
            if df.get(term)
        }

        scores: List[float] = []
        for toks, length in zip(doc_tokens, doc_len):
            tf = Counter(toks)
            norm = self.k1 * (1.0 - self.b + self.b * (length / avgdl))
            s = 0.0
            for term, weight in idf.items():
                f = tf.get(term, 0)
                if f:
                    s += weight * (f * (self.k1 + 1.0)) / (f + norm)
            scores.append(s)
        return scores


class CrossEncoderReranker(Reranker):
    """Wraps a ``sentence-transformers`` ``CrossEncoder`` (opt-in, needs deps)."""

    name = "cross-encoder"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # lazy, optional dep

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        if not documents:
            return []
        pairs = [(query, d or "") for d in documents]
        preds = self._model.predict(pairs)
        return [float(x) for x in preds]


def build_reranker(config: HybridConfig) -> Optional[Reranker]:
    """Return the configured reranker, or ``None`` when reranking is off.

    ``rerank_model`` is ``"lexical"`` (default, BM25, no deps), ``"none"`` to
    disable, or a ``sentence-transformers`` CrossEncoder id to load a real model.
    """
    if not config.enable_rerank:
        return None
    model = (config.rerank_model or "lexical").strip()
    if model in ("", "none"):
        return None
    if model == "lexical":
        return LexicalReranker()
    return CrossEncoderReranker(model)


def _candidate_text(result: SearchResult) -> str:
    """Best available text to score a candidate on (tiles may carry only a title)."""
    if result.text:
        return result.text
    meta = result.meta or {}
    return str(meta.get("title") or meta.get("ocr") or "")


def rerank_results(
    reranker: Reranker,
    query: str,
    results: List[SearchResult],
    blend: float = 0.5,
    top_k: Optional[int] = None,
) -> List[SearchResult]:
    """Re-score and reorder fused ``results`` with ``reranker``.

    The final score blends the reranker signal with the existing fusion score,
    both min-max normalized to ``[0, 1]`` so neither dominates by scale::

        combined = blend * rerank_norm + (1 - blend) * fusion_norm

    Candidates with no scorable text (e.g. image-only tiles) keep their fusion
    standing instead of being pushed down by a zero rerank score. The blended
    value is recorded under ``components['rerank']`` for transparency.
    """
    if not results:
        return results
    blend = min(1.0, max(0.0, blend))

    docs = [_candidate_text(r) for r in results]
    raw = reranker.score(query, docs)

    def _minmax(values: List[float]) -> List[float]:
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return [0.0 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    rr_norm = _minmax(raw)
    fusion_norm = _minmax([r.score for r in results])

    rescored: List[SearchResult] = []
    for r, rr, fn, doc in zip(results, rr_norm, fusion_norm, docs):
        # Image-only candidate with nothing to score: defer to fusion standing.
        rr_eff = fn if not doc.strip() else rr
        combined = blend * rr_eff + (1.0 - blend) * fn
        comps = dict(r.components)
        comps["rerank"] = round(rr_eff, 6)
        rescored.append(
            SearchResult(
                doc_id=r.doc_id,
                score=combined,
                modality=r.modality,
                unit_id=r.unit_id,
                text=r.text,
                image_path=r.image_path,
                page=r.page,
                components=comps,
                meta=r.meta,
            )
        )

    rescored.sort(key=lambda r: r.score, reverse=True)
    return rescored[:top_k] if top_k else rescored
