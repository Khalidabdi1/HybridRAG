"""Text extraction and chunking.

Extraction is intentionally dependency-light: plain text and HTML work with the
standard library; PDF text uses PyMuPDF when available. Chunking is a
character-window splitter with overlap that tries to break on sentence/whitespace
boundaries so chunks stay readable.
"""

from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from typing import List

from ..types import Chunk

_WS_RE = re.compile(r"[ \t\f\v]+")
_NL_RE = re.compile(r"\n{3,}")


class _TextHTMLParser(HTMLParser):
    """Collect visible text, dropping script/style content."""

    _SKIP = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(html: str) -> str:
    parser = _TextHTMLParser()
    parser.feed(html)
    return clean_text(parser.text())


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 120,
    page: int = 0,
) -> List[Chunk]:
    """Split ``text`` into overlapping character windows on soft boundaries."""
    text = clean_text(text)
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    start = 0
    order = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Try to break on a sentence end, then a newline, then whitespace.
            window = text[start:end]
            for sep in (". ", "\n", " "):
                idx = window.rfind(sep)
                if idx > chunk_size * 0.5:
                    end = start + idx + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            cid = hashlib.sha1(f"{doc_id}:{order}:{piece[:32]}".encode("utf-8")).hexdigest()[:16]
            chunks.append(Chunk(id=cid, doc_id=doc_id, text=piece, page=page, order=order))
            order += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_pdf_text_by_page(path: str) -> List[str]:
    """Return one text string per PDF page using PyMuPDF (optional dep)."""
    try:
        import fitz  # type: ignore  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            'PyMuPDF is required for PDF extraction. Install with: pip install -e ".[render]"'
        ) from exc
    pages: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(clean_text(page.get_text()))
    return pages
