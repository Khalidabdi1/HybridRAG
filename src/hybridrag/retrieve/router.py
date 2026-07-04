"""Query router: decide how much each modality should contribute per query.

A core thesis of HybridRAG: you should not pay the vision cost for every query.
Code, logs, JSON, and stack traces are better and cheaper served by text search;
questions about charts, tables, diagrams, and layout benefit from pixels. The
router inspects the query (no embedding call, no GPU) and returns per-modality
weights that bias fusion accordingly. It never disables a modality entirely, so
recall is preserved.

Two routers ship, behind one interface (:meth:`Router.route`):

* :class:`HeuristicRouter` — the original hand-tuned keyword rules (:func:`route`).
* :class:`LearnedRouter` — a tiny **logistic-regression** classifier over
  interpretable query features (cue densities, code structure, numeric density,
  length). It runs on numpy alone, ships pre-trained on an embedded seed set so
  it works out of the box, and can be **retrained on your own query logs** with
  :meth:`LearnedRouter.fit` — the point of a learned router over fixed constants.

Pick one with ``config.router_model`` (``"heuristic"`` / ``"learned"``) via
:func:`build_router`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..types import Modality

# Cues that the answer lives in text.
_TEXT_CUES = re.compile(
    r"\b(code|function|class|def|import|stack ?trace|exception|error|log|logs|"
    r"json|yaml|regex|api|endpoint|sql|query|variable|syntax|command|cli|"
    r"config|null|undefined|traceback)\b",
    re.IGNORECASE,
)
# Cues that the answer lives in the visual layout.
_VISION_CUES = re.compile(
    r"\b(chart|charts|graph|graphs|figure|figures|diagram|diagrams|table|tables|"
    r"plot|plots|layout|screenshot|image|images|color|colou?red|axis|legend|"
    r"infographic|map|maps|drawing|photo|picture|visual|appears|looks|shown)\b",
    re.IGNORECASE,
)
# Structural hints that the query *contains* code/data.
_CODEISH = re.compile(r"[{}();=]|->|=>|::|\b0x[0-9a-fA-F]+\b")

# Feature vector layout — order matters and is part of the serialised model.
FEATURE_NAMES: Tuple[str, ...] = (
    "bias",
    "text_cue_density",
    "vision_cue_density",
    "codeish",
    "numeric_density",
    "length",
)


@dataclass
class RouteDecision:
    text_weight: float
    vision_weight: float
    reason: str
    # Learned routers expose the calibrated P(vision-relevant) for the query;
    # heuristic routing leaves this None.
    vision_probability: Optional[float] = None

    def weights(self) -> Dict[Modality, float]:
        return {Modality.TEXT: self.text_weight, Modality.VISION: self.vision_weight}

    def to_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {
            "text_weight": round(self.text_weight, 4),
            "vision_weight": round(self.vision_weight, 4),
            "reason": self.reason,
        }
        if self.vision_probability is not None:
            d["vision_probability"] = round(self.vision_probability, 4)
        return d


def extract_features(query: str) -> np.ndarray:
    """Interpretable, cheap features used by :class:`LearnedRouter`.

    Every feature is a regex/string statistic on the query — no model call — so
    routing stays free. Densities are clamped and scaled to roughly [0, 1] so
    the learned weights are comparable across features.
    """
    tokens = query.split()
    n_tokens = max(len(tokens), 1)
    text_hits = len(_TEXT_CUES.findall(query))
    vision_hits = len(_VISION_CUES.findall(query))
    codeish = 1.0 if _CODEISH.search(query) else 0.0
    numeric_tokens = sum(1 for t in tokens if any(ch.isdigit() for ch in t))
    return np.array(
        [
            1.0,  # bias
            min(text_hits, 4) / 4.0,
            min(vision_hits, 4) / 4.0,
            codeish,
            numeric_tokens / n_tokens,
            min(n_tokens, 20) / 20.0,
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Heuristic router (original behaviour, kept for back-compat and as a baseline)
# --------------------------------------------------------------------------- #

def route(query: str, base_text: float = 1.0, base_vision: float = 1.0) -> RouteDecision:
    """Hand-tuned keyword router. Returns per-modality weights for ``query``.

    The base weights come from config; the router scales them by the strength of
    text vs. vision cues found in the query. This is the original router and the
    baseline the learned router is measured against.
    """
    text_hits = len(_TEXT_CUES.findall(query)) + (1 if _CODEISH.search(query) else 0)
    vision_hits = len(_VISION_CUES.findall(query))

    text_w, vision_w = base_text, base_vision
    reason = "balanced: no strong modality cue"

    if text_hits and not vision_hits:
        # Strong text signal — downweight (don't kill) vision.
        text_w = base_text * (1.0 + 0.5 * min(text_hits, 3))
        vision_w = base_vision * 0.3
        reason = f"text-leaning: {text_hits} text cue(s)"
    elif vision_hits and not text_hits:
        vision_w = base_vision * (1.0 + 0.5 * min(vision_hits, 3))
        text_w = base_text * 0.5
        reason = f"vision-leaning: {vision_hits} vision cue(s)"
    elif vision_hits and text_hits:
        text_w = base_text * (1.0 + 0.25 * text_hits)
        vision_w = base_vision * (1.0 + 0.25 * vision_hits)
        reason = f"both: {text_hits} text / {vision_hits} vision cue(s)"

    return RouteDecision(text_weight=text_w, vision_weight=vision_w, reason=reason)


class HeuristicRouter:
    """Object wrapper around :func:`route` so both routers share an interface."""

    model = "heuristic"

    def route(self, query: str, base_text: float = 1.0, base_vision: float = 1.0) -> RouteDecision:
        return route(query, base_text, base_vision)


# --------------------------------------------------------------------------- #
# Learned router — logistic regression over the features above
# --------------------------------------------------------------------------- #

# A small, human-labelled seed set. `label` is P(vision-relevant): 1.0 = the
# answer lives in the visual layout (charts/tables/figures/design), 0.0 = the
# answer is text (code/logs/JSON/prose), 0.5 = genuinely mixed or neutral. The
# default router is trained on this deterministically, so it works out of the
# box; retrain on your own labelled query logs with LearnedRouter.fit().
DEFAULT_TRAINING_EXAMPLES: Tuple[Tuple[str, float], ...] = (
    # text-native
    ("how do I fix this python stack trace exception", 0.0),
    ("what does this function return when the input is null", 0.0),
    ("parse the json config and validate the schema", 0.0),
    ("grep the logs for the error message", 0.0),
    ("def foo(): return {a: 1}", 0.0),
    ("which sql query joins these two tables by id", 0.05),
    ("explain this regex and the api endpoint it calls", 0.0),
    ("undefined variable in the cli command output", 0.0),
    ("summarise the terms of service paragraph", 0.15),
    ("what is the capital city mentioned in the article", 0.2),
    # vision-native
    ("which chart shows the revenue trend over time", 1.0),
    ("read the value in the third row of the table", 1.0),
    ("what colour is the largest slice of the pie chart", 1.0),
    ("describe the layout of the dashboard screenshot", 1.0),
    ("which bar in the graph is the tallest", 1.0),
    ("what does the legend on the axis say", 0.95),
    ("find the figure that shows the architecture diagram", 0.95),
    ("what is drawn in the infographic on page two", 1.0),
    ("compare the two plots side by side", 0.9),
    ("which region on the map is highlighted", 0.9),
    # mixed / neutral
    ("show the json config and the diagram of the api", 0.5),
    ("tell me about the history of rome", 0.5),
    ("what were the quarterly revenue numbers", 0.55),
    ("does the table in the code comment match the figure", 0.6),
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class LearnedRouter:
    """Logistic-regression query router (numpy only, deterministic training).

    Predicts a calibrated ``P(vision-relevant)`` from :func:`extract_features`
    and maps it to per-modality weights. Unlike the fixed heuristic constants,
    the *combination* of signals is learned from labelled queries and can be
    tuned to a corpus by calling :meth:`fit` on your own query logs.
    """

    weights: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURE_NAMES)))
    lean: float = 0.7  # how hard the probability swings the weights (0 = off)
    model: str = "learned"

    # ------------------------------------------------------------------ train
    def fit(
        self,
        examples: Sequence[Tuple[str, float]],
        iterations: int = 4000,
        lr: float = 0.5,
        l2: float = 1e-3,
    ) -> "LearnedRouter":
        """Fit the classifier on ``(query, vision_label)`` pairs in place.

        ``vision_label`` is a soft target in [0, 1] (1 = vision-relevant). Uses
        full-batch gradient descent on cross-entropy with L2 regularisation,
        initialised at zero — so training is **fully deterministic** (no random
        seed) and the shipped default weights are reproducible.
        """
        if not examples:
            raise ValueError("need at least one training example")
        X = np.stack([extract_features(q) for q, _ in examples])
        y = np.clip(np.array([float(t) for _, t in examples], dtype=np.float64), 0.0, 1.0)
        w = np.zeros(X.shape[1], dtype=np.float64)
        n = X.shape[0]
        for _ in range(iterations):
            p = _sigmoid(X @ w)
            grad = X.T @ (p - y) / n + l2 * w
            grad[0] -= l2 * w[0]  # don't regularise the bias
            w -= lr * grad
        self.weights = w
        return self

    # ------------------------------------------------------------------ apply
    def vision_probability(self, query: str) -> float:
        return float(_sigmoid(extract_features(query) @ self.weights))

    def route(self, query: str, base_text: float = 1.0, base_vision: float = 1.0) -> RouteDecision:
        p = self.vision_probability(query)
        swing = self.lean * (2.0 * p - 1.0)  # (-lean .. +lean)
        vision_w = base_vision * (1.0 + swing)
        text_w = base_text * (1.0 - swing)
        if p >= 0.6:
            reason = f"learned: vision-leaning (p={p:.2f})"
        elif p <= 0.4:
            reason = f"learned: text-leaning (p={p:.2f})"
        else:
            reason = f"learned: balanced (p={p:.2f})"
        return RouteDecision(
            text_weight=text_w,
            vision_weight=vision_w,
            reason=reason,
            vision_probability=p,
        )

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, object]:
        return {
            "model": "learned",
            "feature_names": list(FEATURE_NAMES),
            "weights": [float(x) for x in self.weights],
            "lean": self.lean,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LearnedRouter":
        weights = np.array(data.get("weights", []), dtype=np.float64)
        if weights.shape[0] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} weights, got {weights.shape[0]}"
            )
        return cls(weights=weights, lean=float(data.get("lean", 0.7)))

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "LearnedRouter":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def default(cls, examples: Optional[Sequence[Tuple[str, float]]] = None) -> "LearnedRouter":
        """A router trained on the embedded seed set (or ``examples``)."""
        return cls().fit(list(examples or DEFAULT_TRAINING_EXAMPLES))


def build_router(config) -> object:
    """Build the router named by ``config.router_model``.

    ``"learned"`` loads trained weights from ``config.router_weights_path`` when
    set, otherwise trains the default on the embedded seed set. Anything else
    (``"heuristic"``, ``"none"``) yields the keyword router.
    """
    model = (getattr(config, "router_model", "heuristic") or "heuristic").lower()
    if model == "learned":
        path = getattr(config, "router_weights_path", "") or ""
        if path:
            return LearnedRouter.load(path)
        return LearnedRouter.default()
    return HeuristicRouter()
