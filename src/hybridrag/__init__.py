"""HybridRAG: fuse text RAG and pixel (vision) RAG into one searchable index.

HybridRAG indexes a document along *two* modalities at once:

* **Text** — extracted text is chunked and embedded with a text encoder.
* **Pixels** — the rendered page is sliced into tiles and embedded with a
  vision encoder (a VLM such as Qwen-VL).

At query time both indexes are searched and the results are fused with
Reciprocal Rank Fusion, optionally guided by a lightweight query router that
decides how much each modality should contribute.

This addresses the practical shortcomings of a pixels-only system (storage
cost, slow indexing, hard updates, GPU pressure, and the cases — code, logs,
JSON — where text search is simply faster and cheaper) while keeping the
visual strength of pixel RAG for tables, charts, and complex layouts.
"""

from .types import Chunk, Document, Modality, SearchResult, Tile
from .config import HybridConfig

__version__ = "0.7.0"

__all__ = [
    "Chunk",
    "Document",
    "Modality",
    "SearchResult",
    "Tile",
    "HybridConfig",
    "__version__",
]
