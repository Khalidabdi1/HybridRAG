"""Evaluation harness: compare text-only, vision-only, and hybrid retrieval.

The harness indexes a dataset once, then replays every query under each
requested *mode* and aggregates ranking metrics, latency, and the index
footprint. This is the empirical answer to the central HybridRAG question:
*for this corpus and these encoders, does fusing pixels with text actually
help, and what does it cost?*

Run it on the built-in sample::

    from hybridrag.eval import evaluate, sample_dataset, build_engine

    ds = sample_dataset()
    report = evaluate(build_engine(ds), ds.queries)
    print(report.table())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Sequence

from ..engine import HybridRAG
from ..types import Modality
from . import metrics as M
from .dataset import EvalQuery

# mode name -> forced modality passed to engine.search (None == fused hybrid)
MODE_MODALITY: Dict[str, Optional[Modality]] = {
    "text": Modality.TEXT,
    "vision": Modality.VISION,
    "hybrid": None,
}
DEFAULT_MODES = ("text", "vision", "hybrid")
DEFAULT_KS = (1, 3, 5, 10)


@dataclass
class ModeReport:
    mode: str
    n_queries: int
    indexed_units: int
    recall: Dict[int, float] = field(default_factory=dict)
    precision: Dict[int, float] = field(default_factory=dict)
    ndcg: Dict[int, float] = field(default_factory=dict)
    hit: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    mean_latency_ms: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "n_queries": self.n_queries,
            "indexed_units": self.indexed_units,
            "recall": self.recall,
            "precision": self.precision,
            "ndcg": self.ndcg,
            "hit": self.hit,
            "mrr": round(self.mrr, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "note": self.note,
        }


@dataclass
class EvalReport:
    dataset: str
    ks: List[int]
    modes: List[ModeReport]
    index: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "ks": self.ks,
            "index": self.index,
            "modes": [m.to_dict() for m in self.modes],
        }

    def table(self, primary_k: Optional[int] = None) -> str:
        """Render a compact comparison table for one ``k`` (default: max k)."""
        k = primary_k or max(self.ks)
        rows = [
            ("mode", "recall", "prec", "nDCG", "hit", "MRR", "ms/q", "units"),
        ]
        for m in self.modes:
            rows.append(
                (
                    m.mode,
                    f"{m.recall.get(k, 0.0):.3f}",
                    f"{m.precision.get(k, 0.0):.3f}",
                    f"{m.ndcg.get(k, 0.0):.3f}",
                    f"{m.hit.get(k, 0.0):.3f}",
                    f"{m.mrr:.3f}",
                    f"{m.mean_latency_ms:.2f}",
                    str(m.indexed_units),
                )
            )
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        lines = [
            f"Dataset: {self.dataset}   metrics @k={k}   "
            f"(text_units={self.index.get('text_units')}, "
            f"vision_units={self.index.get('vision_units')})"
        ]

        def fmt(row):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(row))

        lines.append(fmt(rows[0]))
        lines.append("  ".join("-" * w for w in widths))
        for row in rows[1:]:
            lines.append(fmt(row))
        notes = [f"  note[{m.mode}]: {m.note}" for m in self.modes if m.note]
        return "\n".join(lines + notes)


def _index_footprint(engine: HybridRAG) -> Dict[str, object]:
    """Concrete, measurable index size — vectors live in RAM/on disk as float32.

    Raw screenshots and tile PNGs are *not* counted here: they live on disk and
    are the dominant storage cost of pixel RAG, which is exactly the trade-off
    HybridRAG exists to let you measure against text-only indexing.
    """
    t_units = len(engine.text_store)
    v_units = len(engine.vision_store)
    t_dim = engine.text_embedder.dim
    v_dim = engine.vision_embedder.dim
    t_bytes = t_units * t_dim * 4
    v_bytes = v_units * v_dim * 4
    return {
        "text_units": t_units,
        "vision_units": v_units,
        "text_dim": t_dim,
        "vision_dim": v_dim,
        "text_vector_bytes": t_bytes,
        "vision_vector_bytes": v_bytes,
        "total_vector_bytes": t_bytes + v_bytes,
    }


def evaluate(
    engine: HybridRAG,
    queries: Sequence[EvalQuery],
    ks: Sequence[int] = DEFAULT_KS,
    modes: Sequence[str] = DEFAULT_MODES,
    dataset_name: str = "dataset",
) -> EvalReport:
    """Replay ``queries`` against ``engine`` under each mode and aggregate.

    A mode whose underlying store is empty is reported with zeroed metrics and a
    note rather than being silently dropped, so the comparison stays honest.
    """
    ks = sorted({int(k) for k in ks})
    max_k = max(ks)
    mode_reports: List[ModeReport] = []

    for mode in modes:
        if mode not in MODE_MODALITY:
            raise ValueError(f"unknown mode {mode!r}; choose from {sorted(MODE_MODALITY)}")
        modality = MODE_MODALITY[mode]

        if modality is Modality.TEXT:
            units = len(engine.text_store)
        elif modality is Modality.VISION:
            units = len(engine.vision_store)
        else:
            units = len(engine.text_store) + len(engine.vision_store)

        rep = ModeReport(mode=mode, n_queries=len(queries), indexed_units=units)
        if units == 0:
            rep.note = "no indexed units for this modality — skipped"
            for k in ks:
                rep.recall[k] = rep.precision[k] = rep.ndcg[k] = rep.hit[k] = 0.0
            mode_reports.append(rep)
            continue

        ranked_lists: List[List[str]] = []
        latencies: List[float] = []
        for q in queries:
            t0 = time.perf_counter()
            results = engine.search(q.query, top_k=max_k, modality=modality)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            ranked_lists.append(M.dedup([r.doc_id for r in results]))

        rels = [q.relevant for q in queries]
        for k in ks:
            rep.recall[k] = mean(M.recall_at_k(r, rel, k) for r, rel in zip(ranked_lists, rels))
            rep.precision[k] = mean(
                M.precision_at_k(r, rel, k) for r, rel in zip(ranked_lists, rels)
            )
            rep.ndcg[k] = mean(M.ndcg_at_k(r, rel, k) for r, rel in zip(ranked_lists, rels))
            rep.hit[k] = mean(M.hit_at_k(r, rel, k) for r, rel in zip(ranked_lists, rels))
        rep.mrr = mean(M.reciprocal_rank(r, rel) for r, rel in zip(ranked_lists, rels))
        rep.mean_latency_ms = mean(latencies) if latencies else 0.0
        mode_reports.append(rep)

    return EvalReport(
        dataset=dataset_name,
        ks=ks,
        modes=mode_reports,
        index=_index_footprint(engine),
    )
