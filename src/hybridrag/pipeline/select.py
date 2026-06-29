"""Selective pixel indexing — decide *whether a page is worth the vision path*.

Pixel RAG's two sharpest costs are **storage** (a screenshot, several tiles, and
a tile embedding per page) and **GPU** (the VLM runs at index, rerank, and read).
Both scale with the number of pages you push through the vision pipeline. But not
every page earns it: a page of prose, a code listing, or a JSON dump is served
perfectly well by the text index, and rendering + tiling + embedding it is pure
waste.

This module scores how *visually rich* a page is — does it carry tables, charts,
figures, or non-trivial layout that the text extractor would mangle? — and turns
that into a yes/no decision about indexing pixels. Two signal sources, cheapest
first:

* :func:`score_text_richness` reads the *extracted text or HTML* — available
  **before** you render anything — and looks for tabular structure, embedded
  media tags (``<table>``/``<svg>``/``<canvas>``/``<img>``/``<figure>``), dense
  numeric grids, and column alignment. This lets you skip rendering a page
  entirely.
* :func:`score_image_richness` reads a *rendered page image* (numpy array, PIL
  image, or PNG path) and measures ruled lines (tables), colour saturation
  (charts/figures), and mid-tone density (photos) — the things that survive only
  in pixels.

:class:`PixelSelector` combines whatever signals are available, applies the
policy in :class:`~hybridrag.config.HybridConfig` (``pixel_selection`` =
``"auto"`` / ``"always"`` / ``"never"`` and ``pixel_selection_threshold``), and
returns a :class:`SelectionDecision`.

Everything here runs on **numpy alone**. Pillow is only needed to load an image
from a path; pass a numpy array and there are no optional deps at all, which is
why the scorers are directly unit-testable offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Text / HTML richness (pre-render, cheapest signal)
# ---------------------------------------------------------------------------

# Markup that almost always means "there is something here the text extractor
# cannot faithfully linearise". Weighted by how strongly each implies visual
# content that belongs in the pixel index.
_MEDIA_TAGS = {
    "table": 1.0,
    "svg": 1.0,
    "canvas": 1.0,
    "figure": 0.8,
    "img": 0.6,
    "picture": 0.6,
    "chart": 0.9,
    "math": 0.7,
}
_MEDIA_RE = re.compile(
    r"<\s*(table|svg|canvas|figure|img|picture|math)\b|chart", re.IGNORECASE
)
# Row/cell tags scale a table's "size" — more rows means more structure to lose.
_CELL_RE = re.compile(r"<\s*(tr|td|th|thead|tbody)\b", re.IGNORECASE)
_TAG_RE = re.compile(r"<[a-zA-Z!/]")


def _looks_like_html(text: str) -> bool:
    if not text:
        return False
    sample = text[:4000]
    return len(_TAG_RE.findall(sample)) >= 3


def _html_richness(html: str) -> float:
    """Richness from markup: how much of the page is media/tabular structure."""
    counts: Dict[str, int] = {}
    for m in _MEDIA_RE.finditer(html):
        key = (m.group(1) or "chart").lower()
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return 0.0
    # The *presence* of a strong media tag (table/svg/canvas/chart) already means
    # "render me" — anchor the score there, then let the heaviest tag's weight and
    # the table's cell count lift it toward 1.0.
    base = 0.55 if any(k in counts for k in ("table", "svg", "canvas", "chart")) else 0.35
    heaviest = max(_MEDIA_TAGS.get(k, 0.5) for k in counts)
    cells = len(_CELL_RE.findall(html))
    size = 1.0 - np.exp(-cells / 12.0)  # saturating: a bigger table scores higher
    score = base + 0.3 * heaviest + 0.25 * size
    return float(min(1.0, score))


# A row that looks tabular: 3+ numeric tokens, or pipe-delimited cells, or
# multiple runs of 2+ spaces (column alignment in monospace dumps).
_NUM_TOKEN_RE = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*")
_COL_GAP_RE = re.compile(r"\S {2,}\S")


def _is_tabular_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 4:
        return False
    if stripped.count("|") >= 2:
        return True
    if len(_NUM_TOKEN_RE.findall(stripped)) >= 3:
        return True
    if len(_COL_GAP_RE.findall(line)) >= 2:
        return True
    return False


def _plain_text_richness(text: str) -> float:
    """Richness from raw text layout: tabular rows and numeric density."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    tabular = sum(1 for ln in lines if _is_tabular_line(ln))
    tabular_frac = tabular / len(lines)

    digits = sum(c.isdigit() for c in text)
    alpha = sum(c.isalpha() for c in text)
    digit_ratio = digits / max(digits + alpha, 1)

    # Tabular structure dominates; a high digit ratio nudges it up (price lists,
    # financial tables) but never decides on its own.
    score = 0.85 * min(1.0, tabular_frac * 2.5) + 0.4 * min(1.0, digit_ratio * 3.0)
    return float(min(1.0, score))


def score_text_richness(text: Optional[str] = None, html: Optional[str] = None) -> float:
    """Cheap pre-render richness in ``[0, 1]`` from extracted text and/or HTML.

    Higher means "more likely to contain tables/figures/layout the text index
    would lose". ``html`` (when given, or auto-detected inside ``text``) is the
    stronger signal because structural tags are unambiguous. Returns ``0.0`` when
    there is nothing to score.
    """
    scores: List[float] = []
    if html:
        scores.append(_html_richness(html))
    if text:
        if html is None and _looks_like_html(text):
            scores.append(_html_richness(text))
        scores.append(_plain_text_richness(text))
    if not scores:
        return 0.0
    return float(max(scores))


# ---------------------------------------------------------------------------
# Image richness (post-render, the signal that only pixels carry)
# ---------------------------------------------------------------------------


def _to_arrays(image: Any) -> tuple:
    """Normalise an input image to ``(gray, rgb_or_None)`` float arrays in [0,1].

    Accepts a numpy array (HxW grayscale, or HxWx3/4), a PIL ``Image``, or a path
    to a PNG/JPG (needs Pillow). Returns grayscale always; rgb only when colour
    information is present (so the colour signal is skipped on grayscale pages).
    """
    arr = None
    if isinstance(image, np.ndarray):
        arr = image
    elif isinstance(image, str):
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                'Pillow is required to score an image from a path. '
                'Pass a numpy array instead, or install with: pip install -e ".[render]"'
            ) from exc
        with Image.open(image) as img:
            arr = np.asarray(img.convert("RGB"))
    else:  # assume a PIL image
        arr = np.asarray(image)

    arr = np.asarray(arr)
    if arr.dtype != np.float64 and arr.dtype != np.float32:
        arr = arr.astype("float32")
    if arr.max() > 1.0:
        arr = arr / 255.0

    rgb = None
    if arr.ndim == 3:
        rgb = arr[:, :, :3]
        gray = rgb.mean(axis=2)
    else:
        gray = arr
    return gray, rgb


def _line_score(gray: np.ndarray) -> float:
    """Density of long horizontal/vertical rules — the signature of tables."""
    ink = gray < 0.5  # dark pixels
    h, w = ink.shape
    if h == 0 or w == 0:
        return 0.0
    # A row/column is a "rule" when most of its length is ink.
    row_rules = int((ink.mean(axis=1) > 0.6).sum())
    col_rules = int((ink.mean(axis=0) > 0.6).sum())
    # Normalise by a modest expected count so a handful of rules already scores.
    return float(min(1.0, (row_rules / 8.0) + (col_rules / 12.0)))


def _color_score(rgb: Optional[np.ndarray]) -> float:
    """Fraction of saturated (colourful) pixels — charts/figures, not B/W text."""
    if rgb is None:
        return 0.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = mx - mn  # cheap saturation proxy
    colourful = float((sat > 0.18).mean())
    return float(min(1.0, colourful * 4.0))


def _midtone_score(gray: np.ndarray) -> float:
    """Fraction of mid-grey pixels — photos/renders, vs bimodal black-on-white text."""
    mid = float(((gray > 0.25) & (gray < 0.75)).mean())
    return float(min(1.0, mid * 3.0))


# Weights blend the three visual cues into a single richness score. Lines and
# colour are the strongest "this needs pixels" signals; mid-tone catches photos.
_IMG_WEIGHTS = {"lines": 0.45, "color": 0.4, "midtone": 0.25}


def score_image_richness(image: Any, return_signals: bool = False):
    """Visual richness of a rendered page in ``[0, 1]``.

    Combines ruled-line density (tables), colour saturation (charts/figures), and
    mid-tone density (photos). A blank or plain-prose page scores near ``0``; a
    page with a ruled table or a colour chart scores high. Accepts a numpy array,
    a PIL image, or an image path (the path form needs Pillow).

    With ``return_signals=True`` returns ``(score, signals_dict)`` for debugging.
    """
    gray, rgb = _to_arrays(image)
    signals = {
        "lines": _line_score(gray),
        "color": _color_score(rgb),
        "midtone": _midtone_score(gray),
    }
    score = sum(_IMG_WEIGHTS[k] * v for k, v in signals.items())
    score = float(min(1.0, score))
    if return_signals:
        return score, signals
    return score


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class SelectionDecision:
    """The outcome of deciding whether to index a page's pixels."""

    index_pixels: bool
    score: float
    mode: str
    threshold: float
    signals: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_pixels": self.index_pixels,
            "score": round(self.score, 4),
            "mode": self.mode,
            "threshold": self.threshold,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "reason": self.reason,
        }


class PixelSelector:
    """Apply a :class:`~hybridrag.config.HybridConfig` selection policy.

    ``mode`` is one of:

    * ``"always"`` — index pixels for every page (classic Pixel RAG behaviour).
    * ``"never"`` — text-only; the vision path is disabled.
    * ``"auto"`` — score the page from whatever signals are available and index
      pixels only when the score clears ``threshold``.

    The image signal, when present, dominates (it directly observes the rendered
    page); the text/HTML signal is blended in and, on its own, is enough to
    *trigger* rendering but is weighted below a real image observation.
    """

    def __init__(self, mode: str = "auto", threshold: float = 0.35,
                 text_weight: float = 0.7) -> None:
        self.mode = mode
        self.threshold = threshold
        self.text_weight = text_weight

    @classmethod
    def from_config(cls, config: Any) -> "PixelSelector":
        return cls(
            mode=getattr(config, "pixel_selection", "auto"),
            threshold=getattr(config, "pixel_selection_threshold", 0.35),
            text_weight=getattr(config, "pixel_selection_text_weight", 0.7),
        )

    def decide(
        self,
        text: Optional[str] = None,
        html: Optional[str] = None,
        image: Any = None,
    ) -> SelectionDecision:
        if self.mode == "always":
            return SelectionDecision(True, 1.0, self.mode, self.threshold,
                                     reason="policy=always")
        if self.mode == "never":
            return SelectionDecision(False, 0.0, self.mode, self.threshold,
                                     reason="policy=never")

        signals: Dict[str, float] = {}
        score = 0.0
        have_image = image is not None
        have_text = bool(text) or bool(html)

        if have_image:
            img_score, img_signals = score_image_richness(image, return_signals=True)
            signals.update(img_signals)
            score = img_score
        if have_text:
            txt_score = score_text_richness(text=text, html=html)
            signals["text"] = txt_score
            if have_image:
                # Image is the primary observation; let strong text structure
                # (e.g. an HTML <table>) lift a borderline image score.
                score = max(score, self.text_weight * txt_score,
                            0.6 * score + 0.4 * txt_score)
            else:
                # No render yet — decide purely on the pre-render text signal.
                score = txt_score

        if not have_image and not have_text:
            # Nothing to judge: be conservative and keep pixels (don't silently
            # drop a page we know nothing about).
            return SelectionDecision(True, 1.0, self.mode, self.threshold,
                                     reason="no signals; defaulting to index")

        index = score >= self.threshold
        reason = (
            f"score={score:.3f} {'>=' if index else '<'} threshold={self.threshold:.2f}"
        )
        return SelectionDecision(index, float(score), self.mode, self.threshold,
                                 signals=signals, reason=reason)


def build_selector(config: Any) -> PixelSelector:
    """Construct a :class:`PixelSelector` from a config (mirrors the embedders)."""
    return PixelSelector.from_config(config)
