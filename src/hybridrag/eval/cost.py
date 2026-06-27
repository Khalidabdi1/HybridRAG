"""Cost & storage model: what does each retrieval modality actually cost?

The headline argument for HybridRAG over a pixels-only system is economic:
storing one screenshot (plus several tile crops *and* their vision embeddings)
per page is far more expensive — in bytes, in indexing GPU, and in $/query — than
storing chunked text and its embeddings. The ranking metrics in
:mod:`hybridrag.eval.harness` tell you whether pixels *help*; this module tells
you what they *cost*, so the trade-off can be made with numbers instead of
intuition.

Everything here is a transparent, dependency-free model:

* **Measured** from the index — vector counts and dimensions are exact.
* **Modelled** from unit prices — raw artifact bytes (screenshots, tiles, stored
  text), embedding compute, rendering, and per-query encoding all come from a
  :class:`CostModel` whose defaults are documented order-of-magnitude figures
  you should override with your own provider's numbers.

Because pixel artifacts dominate at scale, the report also *projects* storage to
an arbitrary corpus size (the user's "10 million pages" question) using a small
per-page structural model, so you can answer "how many terabytes is that?"
before you index a single real page.

Example::

    from hybridrag.eval import estimate_cost, sample_dataset, build_engine
    report = estimate_cost(build_engine(sample_dataset()))
    print(report.table())
    print(report.projection_table(pages=10_000_000))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

_GB = 1024 ** 3
_MB = 1024 ** 2
_TB = 1024 ** 4
_BYTES_PER_FLOAT32 = 4


@dataclass
class CostModel:
    """Unit prices and artifact sizes. Override the defaults with real numbers.

    Defaults are deliberately conservative, public-cloud order-of-magnitude
    figures (mid-2020s): object storage around $0.023/GB-month, a text embedding
    roughly two orders of magnitude cheaper to compute and to query than a VLM
    tile embedding, and a full-page PNG screenshot far larger than the text it
    contains. They exist so the model runs out of the box, not as a price quote.
    """

    # ---- storage ($/GB-month) ----
    storage_usd_per_gb_month: float = 0.023

    # ---- one-time indexing compute ($) ----
    text_embed_usd_per_1k_units: float = 0.0001   # cheap CPU/text-model encode
    vision_embed_usd_per_1k_units: float = 0.02    # VLM tile encode (~200x text)
    render_usd_per_page: float = 0.0003            # Playwright screenshot CPU time

    # ---- per-query encoding ($) ----
    text_query_usd: float = 0.000002               # embed one text query
    vision_query_usd: float = 0.0004               # embed one query with a VLM

    # ---- raw artifact sizes (bytes) ----
    bytes_per_text_unit_raw: int = 1024            # stored chunk text (~1 KB)
    screenshot_bytes_per_page: int = 307_200       # ~300 KB full-page PNG
    tile_bytes: int = 81_920                        # ~80 KB per tile crop

    # ---- per-page corpus structure (for projections) ----
    text_chunks_per_page: float = 3.0              # chunks a page of text yields
    tiles_per_page: float = 4.0                    # tiles a rendered page is cut into

    def to_dict(self) -> Dict[str, float]:
        return {
            "storage_usd_per_gb_month": self.storage_usd_per_gb_month,
            "text_embed_usd_per_1k_units": self.text_embed_usd_per_1k_units,
            "vision_embed_usd_per_1k_units": self.vision_embed_usd_per_1k_units,
            "render_usd_per_page": self.render_usd_per_page,
            "text_query_usd": self.text_query_usd,
            "vision_query_usd": self.vision_query_usd,
            "bytes_per_text_unit_raw": self.bytes_per_text_unit_raw,
            "screenshot_bytes_per_page": self.screenshot_bytes_per_page,
            "tile_bytes": self.tile_bytes,
            "text_chunks_per_page": self.text_chunks_per_page,
            "tiles_per_page": self.tiles_per_page,
        }


@dataclass
class ModeCost:
    """Cost breakdown for a single retrieval mode (text / vision / hybrid)."""

    mode: str
    vector_bytes: int = 0          # embeddings, exact (units * dim * 4)
    artifact_bytes: int = 0        # raw text / screenshots+tiles, modelled
    index_usd: float = 0.0         # one-time embed (+ render) cost
    query_usd: float = 0.0         # cost to serve one query

    @property
    def storage_bytes(self) -> int:
        return self.vector_bytes + self.artifact_bytes

    def storage_usd_per_month(self, cm: CostModel) -> float:
        return self.storage_bytes / _GB * cm.storage_usd_per_gb_month

    def to_dict(self, cm: CostModel) -> dict:
        return {
            "mode": self.mode,
            "vector_bytes": self.vector_bytes,
            "artifact_bytes": self.artifact_bytes,
            "storage_bytes": self.storage_bytes,
            "storage_mb": round(self.storage_bytes / _MB, 4),
            "storage_usd_per_month": round(self.storage_usd_per_month(cm), 6),
            "index_usd": round(self.index_usd, 6),
            "query_usd": round(self.query_usd, 8),
        }


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024.0 or unit == "PB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} PB"


def _usd(x: float) -> str:
    if x == 0:
        return "$0"
    if x < 0.01:
        return f"${x:.2e}"
    if x < 1000:
        return f"${x:,.2f}"
    return f"${x:,.0f}"


@dataclass
class CostReport:
    """Per-mode cost breakdown plus an at-scale storage projection."""

    dataset: str
    cost_model: CostModel
    modes: List[ModeCost] = field(default_factory=list)
    # exact, measured from the index
    text_units: int = 0
    vision_units: int = 0
    text_dim: int = 0
    vision_dim: int = 0

    def mode(self, name: str) -> Optional[ModeCost]:
        for m in self.modes:
            if m.mode == name:
                return m
        return None

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "cost_model": self.cost_model.to_dict(),
            "measured": {
                "text_units": self.text_units,
                "vision_units": self.vision_units,
                "text_dim": self.text_dim,
                "vision_dim": self.vision_dim,
            },
            "modes": [m.to_dict(self.cost_model) for m in self.modes],
        }

    # ----------------------------------------------------------- projection
    def project(self, pages: int) -> Dict[str, dict]:
        """Project storage to ``pages`` pages using the per-page structural model.

        Text mode stores ``text_chunks_per_page`` text vectors + their raw text
        per page. Vision mode stores one screenshot, ``tiles_per_page`` tile
        crops, and ``tiles_per_page`` vision vectors per page. Hybrid stores
        both. This is what turns "10 million pages" into terabytes and dollars.
        """
        cm = self.cost_model
        text_vec = int(round(cm.text_chunks_per_page * (self.text_dim or 0) * _BYTES_PER_FLOAT32))
        text_raw = int(round(cm.text_chunks_per_page * cm.bytes_per_text_unit_raw))
        vis_vec = int(round(cm.tiles_per_page * (self.vision_dim or 0) * _BYTES_PER_FLOAT32))
        vis_raw = int(round(cm.screenshot_bytes_per_page + cm.tiles_per_page * cm.tile_bytes))

        per_page = {
            "text": text_vec + text_raw,
            "vision": vis_vec + vis_raw,
        }
        per_page["hybrid"] = per_page["text"] + per_page["vision"]

        out: Dict[str, dict] = {}
        for name, ppb in per_page.items():
            total = ppb * pages
            out[name] = {
                "bytes_per_page": ppb,
                "total_bytes": total,
                "total_tb": round(total / _TB, 4),
                "storage_usd_per_month": round(total / _GB * cm.storage_usd_per_gb_month, 2),
            }
        return out

    # --------------------------------------------------------------- tables
    def table(self) -> str:
        cm = self.cost_model
        header = ("mode", "vectors", "artifacts", "total", "$/mo (store)", "$/index", "$/query")
        rows = [header]
        for m in self.modes:
            rows.append((
                m.mode,
                _human_bytes(m.vector_bytes),
                _human_bytes(m.artifact_bytes),
                _human_bytes(m.storage_bytes),
                _usd(m.storage_usd_per_month(cm)),
                _usd(m.index_usd),
                _usd(m.query_usd),
            ))
        widths = [max(len(r[i]) for r in rows) for i in range(len(header))]

        def fmt(row):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(row))

        lines = [
            f"Cost model — dataset: {self.dataset}  "
            f"(text_units={self.text_units}, vision_units={self.vision_units})",
            fmt(header),
            "  ".join("-" * w for w in widths),
        ]
        lines += [fmt(r) for r in rows[1:]]
        return "\n".join(lines)

    def projection_table(self, pages: int) -> str:
        proj = self.project(pages)
        header = ("mode", "bytes/page", "total", "$/mo (store)")
        rows = [header]
        for name in ("text", "vision", "hybrid"):
            p = proj[name]
            rows.append((
                name,
                _human_bytes(p["bytes_per_page"]),
                _human_bytes(p["total_bytes"]),
                _usd(p["storage_usd_per_month"]),
            ))
        widths = [max(len(r[i]) for r in rows) for i in range(len(header))]

        def fmt(row):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(row))

        vis = proj["vision"]["total_bytes"]
        txt = proj["text"]["total_bytes"]
        ratio = (vis / txt) if txt else float("inf")
        lines = [
            f"Storage projection — {pages:,} pages",
            fmt(header),
            "  ".join("-" * w for w in widths),
        ]
        lines += [fmt(r) for r in rows[1:]]
        if txt:
            lines.append(f"  vision/text storage ratio: {ratio:.0f}x")
        return "\n".join(lines)


def estimate_cost(
    engine,
    cost_model: Optional[CostModel] = None,
    dataset_name: str = "index",
) -> CostReport:
    """Build a :class:`CostReport` from an indexed :class:`~hybridrag.engine.HybridRAG`.

    Vector counts/dims are read straight off the engine's stores (exact). Raw
    artifact bytes and all dollar figures come from ``cost_model``. The vision
    artifact estimate attributes one screenshot to every distinct page and one
    tile to every vision unit, the same shape a real render pipeline produces.
    """
    cm = cost_model or CostModel()

    t_units = len(engine.text_store)
    v_units = len(engine.vision_store)
    t_dim = engine.text_embedder.dim
    v_dim = engine.vision_embedder.dim

    # distinct rendered pages backing the vision tiles (one screenshot each)
    pages = _distinct_pages(engine.vision_store)

    text_vec = t_units * t_dim * _BYTES_PER_FLOAT32
    text_raw = t_units * cm.bytes_per_text_unit_raw
    vis_vec = v_units * v_dim * _BYTES_PER_FLOAT32
    vis_raw = pages * cm.screenshot_bytes_per_page + v_units * cm.tile_bytes

    text_index = t_units / 1000.0 * cm.text_embed_usd_per_1k_units
    vision_index = (
        v_units / 1000.0 * cm.vision_embed_usd_per_1k_units
        + pages * cm.render_usd_per_page
    )

    text_mode = ModeCost(
        mode="text",
        vector_bytes=text_vec,
        artifact_bytes=text_raw,
        index_usd=text_index,
        query_usd=cm.text_query_usd,
    )
    vision_mode = ModeCost(
        mode="vision",
        vector_bytes=vis_vec,
        artifact_bytes=vis_raw,
        index_usd=vision_index,
        query_usd=cm.vision_query_usd,
    )
    hybrid_mode = ModeCost(
        mode="hybrid",
        vector_bytes=text_vec + vis_vec,
        artifact_bytes=text_raw + vis_raw,
        index_usd=text_index + vision_index,
        query_usd=cm.text_query_usd + cm.vision_query_usd,
    )

    return CostReport(
        dataset=dataset_name,
        cost_model=cm,
        modes=[text_mode, vision_mode, hybrid_mode],
        text_units=t_units,
        vision_units=v_units,
        text_dim=t_dim,
        vision_dim=v_dim,
    )


def _distinct_pages(store) -> int:
    """Count distinct (doc_id, page) pairs backing the vision tiles.

    Each distinct page corresponds to one full-page screenshot. Falls back to
    the unit count if page metadata is unavailable.
    """
    if len(store) == 0:
        return 0
    metas = getattr(store, "_meta", None)
    if not metas:
        return len(store)
    seen = set()
    for m in metas:
        seen.add((m.get("doc_id"), m.get("page", 0)))
    return len(seen) or len(store)
