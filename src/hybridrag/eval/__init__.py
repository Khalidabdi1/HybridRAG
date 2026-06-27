"""Evaluation harness for HybridRAG.

Compare text-only, vision-only, and hybrid retrieval on the same corpus with
standard ranking metrics (recall@k, precision@k, nDCG@k, hit@k, MRR), latency,
and index footprint.
"""

from __future__ import annotations

from .cost import CostModel, CostReport, ModeCost, estimate_cost
from .dataset import EvalDataset, EvalDocument, EvalQuery, build_engine
from .harness import DEFAULT_KS, DEFAULT_MODES, EvalReport, ModeReport, evaluate
from .sample import sample_dataset

__all__ = [
    "EvalDataset",
    "EvalDocument",
    "EvalQuery",
    "build_engine",
    "evaluate",
    "EvalReport",
    "ModeReport",
    "sample_dataset",
    "DEFAULT_KS",
    "DEFAULT_MODES",
    "CostModel",
    "CostReport",
    "ModeCost",
    "estimate_cost",
]
