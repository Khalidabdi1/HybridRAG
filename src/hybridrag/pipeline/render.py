"""Page rendering and tiling.

Rendering a web page to a screenshot needs Playwright; rendering PDF pages needs
PyMuPDF. Both are optional. Tiling slices a tall page image into fixed-height,
overlapping tiles so each tile is a manageable input for a vision encoder.

When Pillow is unavailable we still emit :class:`Tile` metadata (without images)
so downstream code paths and tests work; the hashing vision embedder degrades
gracefully on missing files.
"""

from __future__ import annotations

import os
from typing import List

from ..types import Tile


def tile_image(
    image_path: str,
    doc_id: str,
    out_dir: str,
    page: int = 0,
    tile_height: int = 512,
    overlap: int = 64,
) -> List[Tile]:
    """Slice a page screenshot vertically into overlapping tiles.

    Requires Pillow. If Pillow is missing, raises ImportError — callers that
    only need metadata should build :class:`Tile` objects directly.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            'Pillow is required for tiling. Install with: pip install -e ".[render]"'
        ) from exc

    os.makedirs(out_dir, exist_ok=True)
    tiles: List[Tile] = []
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        step = max(tile_height - overlap, 1)
        row = 0
        y = 0
        while y < height:
            y1 = min(y + tile_height, height)
            crop = img.crop((0, y, width, y1))
            tile_id = f"{doc_id}_p{page}_t{row}"
            tile_path = os.path.join(out_dir, f"{tile_id}.png")
            crop.save(tile_path)
            tiles.append(
                Tile(
                    id=tile_id,
                    doc_id=doc_id,
                    page=page,
                    row=row,
                    col=0,
                    image_path=tile_path,
                    bbox=[0, y, width, y1],
                )
            )
            if y1 >= height:
                break
            y += step
            row += 1
    return tiles


def render_url_to_png(url: str, out_path: str, viewport_width: int = 1280,
                      scale: float = 1.0) -> str:
    """Render a full-page screenshot of ``url`` using Playwright (optional dep)."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            'Playwright is required for URL rendering. Install with: '
            'pip install -e ".[render]" && playwright install chromium'
        ) from exc

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": viewport_width, "height": 900},
            device_scale_factor=scale,
        )
        page.goto(url, wait_until="networkidle")
        page.screenshot(path=out_path, full_page=True)
        browser.close()
    return out_path


def render_pdf_to_pngs(path: str, out_dir: str, scale: float = 1.5) -> List[str]:
    """Render each PDF page to a PNG using PyMuPDF (optional dep)."""
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            'PyMuPDF is required for PDF rendering. Install with: pip install -e ".[render]"'
        ) from exc

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    matrix = fitz.Matrix(scale, scale)
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix)
            out_path = os.path.join(out_dir, f"page_{i}.png")
            pix.save(out_path)
            paths.append(out_path)
    return paths
