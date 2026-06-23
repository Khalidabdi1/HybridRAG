"""FastAPI search server for a persisted HybridRAG index.

Run with::

    hybridrag serve --storage .hybridrag --port 8000
    # or: uvicorn hybridrag.serve.api:app  (set HYBRIDRAG_STORAGE env var)

Endpoints:
    GET  /health           -> {"status": "ok", ...stats}
    GET  /search?q=...&k=  -> fused results
    POST /search           -> {"query": ..., "top_k": ..., "modality": ...}
"""

from __future__ import annotations

import os
from typing import Optional

from ..engine import HybridRAG
from ..types import Modality


def create_app(storage_dir: Optional[str] = None):
    try:
        from fastapi import FastAPI, Query
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            'FastAPI is required to serve. Install with: pip install -e ".[serve]"'
        ) from exc

    storage_dir = storage_dir or os.environ.get("HYBRIDRAG_STORAGE", ".hybridrag")
    engine = HybridRAG.load(storage_dir)
    app = FastAPI(title="HybridRAG", version="0.1.0")

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 10
        modality: Optional[str] = None

    def _run(query: str, top_k: int, modality: Optional[str]):
        m = Modality(modality) if modality else None
        results = engine.search(query, top_k=top_k, modality=m)
        return {"query": query, "results": [r.to_dict() for r in results]}

    @app.get("/health")
    def health():
        return {"status": "ok", **engine.stats()}

    @app.get("/search")
    def search_get(q: str = Query(...), k: int = 10, modality: Optional[str] = None):
        return _run(q, k, modality)

    @app.post("/search")
    def search_post(req: SearchRequest):
        return _run(req.query, req.top_k, req.modality)

    return app


# Convenience for `uvicorn hybridrag.serve.api:app`
try:  # pragma: no cover - only when fastapi installed and index exists
    app = create_app()
except Exception:  # noqa: BLE001
    app = None
