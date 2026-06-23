"""Lightweight query router.

A core thesis of HybridRAG: you should not pay the vision cost for every query.
Code, logs, JSON, and stack traces are better and cheaper served by text search;
questions about charts, tables, diagrams, and layout benefit from pixels. The
router inspects the query (no model call) and returns per-modality weights that
bias fusion accordingly. It never disables a modality entirely unless the signal
is strong, so recall is preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

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


@dataclass
class RouteDecision:
    text_weight: float
    vision_weight: float
    reason: str

    def weights(self) -> Dict[Modality, float]:
        return {Modality.TEXT: self.text_weight, Modality.VISION: self.vision_weight}


def route(query: str, base_text: float = 1.0, base_vision: float = 1.0) -> RouteDecision:
    """Return per-modality weights for ``query``.

    The base weights come from config; the router scales them by the strength of
    text vs. vision cues found in the query.
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
