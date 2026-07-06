"""Model Context Protocol (MCP) server for HybridRAG.

This exposes a persistent HybridRAG index to MCP clients such as Claude Code,
Claude Desktop, or the Claude Agent SDK, so an assistant can *index* documents
and *search* them mid-conversation.

Design notes
------------
The tool logic lives in plain, dependency-light functions (:func:`call_tool`
and the ``_tool_*`` handlers) that operate on a :class:`HybridRAG` engine and
return JSON-serialisable dicts. They run on numpy alone and are unit-tested
without the ``mcp`` package installed.

The MCP wiring (:func:`main`) is imported lazily so the module stays importable
even when the optional ``mcp`` dependency is absent. Install it with::

    pip install -e ".[mcp]"

and run the server with::

    hybridrag-mcp            # or: python -m hybridrag.mcp_server

The index it serves is read from ``$HYBRIDRAG_STORAGE`` (default
``.hybridrag_index``); it is created on first write and persisted after every
mutating tool call.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from .config import HybridConfig
from .engine import HybridRAG
from .types import Modality

DEFAULT_STORAGE = ".hybridrag_index"


def storage_dir() -> str:
    """The index directory this server reads from / writes to."""
    return os.environ.get("HYBRIDRAG_STORAGE", DEFAULT_STORAGE)


def load_engine(storage: Optional[str] = None) -> HybridRAG:
    """Load the persisted engine, or create a fresh one bound to ``storage``."""
    storage = storage or storage_dir()
    if os.path.exists(os.path.join(storage, "config.json")):
        return HybridRAG.load(storage)
    return HybridRAG(HybridConfig(storage_dir=storage))


# --------------------------------------------------------------------------- #
# Tool schemas — advertised to MCP clients via list_tools.
# --------------------------------------------------------------------------- #

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "hybridrag_search",
        "description": (
            "Search the HybridRAG index across both text and pixel (vision) "
            "modalities and return fused, ranked results. Use this to ground "
            "answers in indexed documents. Set `modality` to 'text' or 'vision' "
            "to force a single modality; omit it to let the router blend both."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (default 5).",
                    "minimum": 1,
                    "maximum": 50,
                },
                "modality": {
                    "type": "string",
                    "enum": ["text", "vision"],
                    "description": "Optional: force a single modality.",
                },
                "rerank": {
                    "type": "boolean",
                    "description": (
                        "Optional: rerank the fused candidates with a BM25/"
                        "cross-encoder pass for higher precision. Defaults to the "
                        "index configuration."
                    ),
                },
                "fusion": {
                    "type": "string",
                    "enum": ["rrf", "calibrated"],
                    "description": (
                        "Optional: fusion strategy. 'rrf' fuses on rank alone; "
                        "'calibrated' normalizes and combines raw scores, keeping "
                        "confidence magnitude. Defaults to the index configuration."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "hybridrag_answer",
        "description": (
            "Retrieve from the HybridRAG index AND synthesize a grounded, cited "
            "answer to a question in one call (the RAG 'final readout'). The "
            "default extractive reader composes the answer only from sentences "
            "that appear in indexed chunks — every claim carries a [n] citation "
            "and nothing is hallucinated. Relevant tables/charts with no readable "
            "text are returned as `visual_evidence`. Prefer this over "
            "`hybridrag_search` when the user asks a question rather than for a "
            "list of passages."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language question."},
                "top_k": {
                    "type": "integer",
                    "description": "How many fused hits to feed the reader (default 5).",
                    "minimum": 1,
                    "maximum": 20,
                },
                "modality": {
                    "type": "string",
                    "enum": ["text", "vision"],
                    "description": "Optional: force a single modality.",
                },
                "rerank": {
                    "type": "boolean",
                    "description": "Optional: rerank fused candidates before synthesis.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "hybridrag_add_text",
        "description": (
            "Chunk and index a raw text document into the text modality. "
            "Returns the number of chunks added. Persists the index."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Stable id for the document."},
                "text": {"type": "string", "description": "The document body."},
                "title": {"type": "string", "description": "Optional title."},
            },
            "required": ["doc_id", "text"],
        },
    },
    {
        "name": "hybridrag_add_html",
        "description": (
            "Extract text from an HTML string, then chunk and index it into the "
            "text modality (no browser/screenshot needed). Persists the index."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Stable id for the document."},
                "html": {"type": "string", "description": "Raw HTML."},
                "title": {"type": "string", "description": "Optional title."},
            },
            "required": ["doc_id", "html"],
        },
    },
    {
        "name": "hybridrag_add_batch",
        "description": (
            "Index MANY text documents in one batched call — the fast path for a "
            "corpus. Pass `documents`: a list of objects, each with a `doc_id` and "
            "`text` (or `html`) plus an optional `title`. Chunks are buffered "
            "across documents and embedded in batches, so this is far cheaper than "
            "one add_text call per document. Set `replace` to upsert (delete any "
            "existing units for each doc_id first). Persists the index."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "description": "Documents to index.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doc_id": {"type": "string"},
                            "text": {"type": "string"},
                            "html": {"type": "string"},
                            "title": {"type": "string"},
                        },
                        "required": ["doc_id"],
                    },
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Units per embedding batch (default 128).",
                    "minimum": 1,
                },
                "replace": {
                    "type": "boolean",
                    "description": "Upsert: delete existing units for each doc_id first.",
                },
            },
            "required": ["documents"],
        },
    },
    {
        "name": "hybridrag_delete",
        "description": (
            "Delete every text chunk and image tile belonging to a document id. "
            "Use this to remove a document from the index. Persists the index."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document id to remove."},
            },
            "required": ["doc_id"],
        },
    },
    {
        "name": "hybridrag_update_text",
        "description": (
            "Replace a document's text in place: delete any existing units for "
            "`doc_id`, then chunk and re-index the new text. Only this document "
            "is re-embedded. Use this when a source document changed. Persists."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document id to replace."},
                "text": {"type": "string", "description": "The new document body."},
                "title": {"type": "string", "description": "Optional title."},
            },
            "required": ["doc_id", "text"],
        },
    },
    {
        "name": "hybridrag_list_docs",
        "description": (
            "List the distinct document ids currently present in the index "
            "(across both text and vision modalities)."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hybridrag_stats",
        "description": (
            "Report index statistics: number of documents, text and vision "
            "units, the configured models, embedding dimensions, and storage "
            "location."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hybridrag_cost",
        "description": (
            "Model the storage and dollar cost of the current index per "
            "modality (text vs vision/pixel vs hybrid): vector + raw-artifact "
            "bytes, one-time indexing $, and $/query. Also projects total "
            "storage and monthly cost to a target corpus size — use this to "
            "answer 'how much would N pages cost to store?' and to quantify why "
            "pixel-only RAG is far more expensive than text or hybrid."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_pages": {
                    "type": "integer",
                    "description": "Corpus size to project storage to (default 1,000,000).",
                    "minimum": 1,
                },
            },
        },
    },
    {
        "name": "hybridrag_route",
        "description": (
            "Explain how the query router would split a query across the text "
            "and pixel (vision) modalities, WITHOUT running a search. Returns the "
            "per-modality weights and, for the learned router, the calibrated "
            "P(vision-relevant). Use this to reason about why a code/log/JSON "
            "query stays text-heavy while a chart/table/layout query pulls in "
            "pixels. Set `model` to 'learned' for the trained classifier or "
            "'heuristic' for the keyword rules (default: the index config)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query to route."},
                "model": {
                    "type": "string",
                    "enum": ["heuristic", "learned"],
                    "description": "Which router to use (default: the index config).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "hybridrag_dedup",
        "description": (
            "Analyse a set of rendered page/tile images for content duplicates "
            "and report how much vision storage and VLM inference tile-level "
            "deduplication would save. Repeated tiles (a document's header/footer/"
            "logo band on every page, blank margins) are byte-identical, so each "
            "unique tile only needs to be embedded and stored once. Pass "
            "`image_paths` (a list of tile PNG paths); `method` 'exact' (byte "
            "hash) or 'ahash' (perceptual, folds re-encoded copies); `scope` "
            "'doc' (collapse within each document, keeps delete-by-doc correct) "
            "or 'global' (collapse across documents). Returns unique vs duplicate "
            "counts and estimated bytes saved. This directly quantifies the "
            "storage argument (disadvantage #1) for pixel-heavy corpora."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tile/page image paths to analyse.",
                },
                "method": {
                    "type": "string",
                    "enum": ["exact", "ahash"],
                    "description": "Byte-exact (default) or perceptual average-hash matching.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["doc", "global"],
                    "description": "Collapse within each doc_id (default) or across all docs.",
                },
            },
            "required": ["image_paths"],
        },
    },
    {
        "name": "hybridrag_richness",
        "description": (
            "Score how visually rich a page is (tables, charts, figures, dense "
            "layout) and decide whether it is worth indexing into the pixel/"
            "vision modality. Pass `text` and/or `html` (the cheap pre-render "
            "signal) to decide before rendering. Use this to explain or simulate "
            "selective pixel indexing — why a code/log/prose page should stay "
            "text-only while a financial-table page earns the vision path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Extracted page text to score."},
                "html": {"type": "string", "description": "Raw HTML to score (stronger signal)."},
                "mode": {
                    "type": "string",
                    "enum": ["auto", "always", "never"],
                    "description": "Selection policy (default: the index config).",
                },
                "threshold": {
                    "type": "number",
                    "description": "Richness score required to index pixels (default: index config).",
                },
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Tool handlers — pure functions over an engine + arguments.
# --------------------------------------------------------------------------- #

def _tool_search(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query", "")
    if not query:
        raise ValueError("`query` is required")
    top_k = int(args.get("top_k") or engine.config.top_k)
    modality = None
    if args.get("modality"):
        modality = Modality(args["modality"])
    rerank = args.get("rerank")
    fusion = args.get("fusion")
    results = engine.search(
        query, top_k=top_k, modality=modality, rerank=rerank, fusion=fusion
    )
    return {
        "query": query,
        "count": len(results),
        "results": [r.to_dict() for r in results],
    }


def _tool_answer(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query", "")
    if not query:
        raise ValueError("`query` is required")
    top_k = int(args.get("top_k") or engine.config.answer_top_k)
    modality = Modality(args["modality"]) if args.get("modality") else None
    ans = engine.answer(query, top_k=top_k, modality=modality, rerank=args.get("rerank"))
    return ans.to_dict()


def _tool_add_text(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = args.get("doc_id")
    text = args.get("text")
    if not doc_id or text is None:
        raise ValueError("`doc_id` and `text` are required")
    n = engine.add_text(doc_id, text, title=args.get("title", ""))
    engine.save()
    return {"doc_id": doc_id, "chunks_added": n, "storage": engine.config.storage_dir}


def _tool_add_html(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = args.get("doc_id")
    html = args.get("html")
    if not doc_id or html is None:
        raise ValueError("`doc_id` and `html` are required")
    n = engine.add_html(doc_id, html, title=args.get("title", ""))
    engine.save()
    return {"doc_id": doc_id, "chunks_added": n, "storage": engine.config.storage_dir}


def _tool_add_batch(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    documents = args.get("documents")
    if not documents or not isinstance(documents, list):
        raise ValueError("`documents` must be a non-empty list")
    for d in documents:
        if not isinstance(d, dict) or not d.get("doc_id"):
            raise ValueError("each document needs a `doc_id`")
        if d.get("text") is None and d.get("html") is None:
            raise ValueError(f"document {d.get('doc_id')!r} needs `text` or `html`")
    batch_size = int(args.get("batch_size") or 128)
    stats = engine.add_documents(
        documents, batch_size=batch_size, upsert=bool(args.get("replace"))
    )
    engine.save()
    out = stats.to_dict()
    out["storage"] = engine.config.storage_dir
    return out


def _tool_delete(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = args.get("doc_id")
    if not doc_id:
        raise ValueError("`doc_id` is required")
    result = engine.delete(doc_id)
    engine.save()
    return {"doc_id": doc_id, "storage": engine.config.storage_dir, **result}


def _tool_update_text(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = args.get("doc_id")
    text = args.get("text")
    if not doc_id or text is None:
        raise ValueError("`doc_id` and `text` are required")
    result = engine.upsert_text(doc_id, text, title=args.get("title", ""))
    engine.save()
    result["storage"] = engine.config.storage_dir
    return result


def _tool_list_docs(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    doc_ids = sorted(engine.doc_ids())
    return {"count": len(doc_ids), "doc_ids": doc_ids}


def _tool_stats(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    stats = engine.stats()
    stats["storage"] = engine.config.storage_dir
    return stats


def _tool_cost(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    from .eval import estimate_cost

    pages = int(args.get("project_pages") or 1_000_000)
    report = estimate_cost(engine, dataset_name=engine.config.storage_dir)
    out = report.to_dict()
    out["projection"] = {"pages": pages, **report.project(pages)}
    return out


def _tool_route(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    from .retrieve.router import HeuristicRouter, LearnedRouter, build_router

    query = args.get("query", "")
    if not query:
        raise ValueError("`query` is required")
    model = args.get("model")
    if model == "learned":
        router = LearnedRouter.default()
    elif model == "heuristic":
        router = HeuristicRouter()
    else:
        router = build_router(engine.config)
    decision = router.route(
        query, engine.config.text_weight, engine.config.vision_weight
    )
    out = decision.to_dict()
    out["router_model"] = getattr(router, "model", "heuristic")
    out["query"] = query
    return out


def _tool_dedup(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    from .pipeline.dedup import TileDeduplicator, tiles_from_paths

    image_paths = args.get("image_paths")
    if not image_paths or not isinstance(image_paths, list):
        raise ValueError("`image_paths` must be a non-empty list")
    dedup = TileDeduplicator(
        method=args.get("method") or "exact",
        scope=args.get("scope") or "doc",
    )
    _unique, stats = dedup.deduplicate(tiles_from_paths([str(p) for p in image_paths]))
    return stats.to_dict()


def _tool_richness(engine: HybridRAG, args: Dict[str, Any]) -> Dict[str, Any]:
    from .pipeline.select import PixelSelector

    text = args.get("text")
    html = args.get("html")
    if not text and not html:
        raise ValueError("at least one of `text` or `html` is required")
    selector = PixelSelector(
        mode=args.get("mode") or engine.config.pixel_selection,
        threshold=(
            float(args["threshold"]) if args.get("threshold") is not None
            else engine.config.pixel_selection_threshold
        ),
        text_weight=engine.config.pixel_selection_text_weight,
    )
    return selector.decide(text=text, html=html).to_dict()


_HANDLERS: Dict[str, Callable[[HybridRAG, Dict[str, Any]], Dict[str, Any]]] = {
    "hybridrag_search": _tool_search,
    "hybridrag_answer": _tool_answer,
    "hybridrag_route": _tool_route,
    "hybridrag_richness": _tool_richness,
    "hybridrag_dedup": _tool_dedup,
    "hybridrag_add_text": _tool_add_text,
    "hybridrag_add_html": _tool_add_html,
    "hybridrag_add_batch": _tool_add_batch,
    "hybridrag_delete": _tool_delete,
    "hybridrag_update_text": _tool_update_text,
    "hybridrag_list_docs": _tool_list_docs,
    "hybridrag_stats": _tool_stats,
    "hybridrag_cost": _tool_cost,
}


def call_tool(name: str, arguments: Dict[str, Any], engine: Optional[HybridRAG] = None) -> Dict[str, Any]:
    """Dispatch a tool call. Loads the persisted engine if one isn't supplied.

    Raises ``KeyError`` for an unknown tool name and ``ValueError`` for bad
    arguments — callers (including the MCP layer) translate these into errors.
    """
    if name not in _HANDLERS:
        raise KeyError(f"unknown tool: {name}")
    engine = engine if engine is not None else load_engine()
    return _HANDLERS[name](engine, arguments or {})


# --------------------------------------------------------------------------- #
# MCP wiring — only touched when the optional `mcp` package is installed.
# --------------------------------------------------------------------------- #

def main() -> int:
    """Run the stdio MCP server. Requires the optional ``mcp`` dependency."""
    try:
        import anyio
        import mcp.types as mcp_types
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError as exc:  # pragma: no cover - exercised only without mcp
        raise SystemExit(
            "The MCP server requires the 'mcp' package. Install it with:\n"
            '    pip install -e ".[mcp]"\n'
            f"(import error: {exc})"
        )

    server: Server = Server("hybridrag")

    @server.list_tools()
    async def list_tools() -> list:
        return [
            mcp_types.Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["inputSchema"],
            )
            for spec in TOOL_SPECS
        ]

    @server.call_tool()
    async def handle_call(name: str, arguments: Dict[str, Any]) -> list:
        try:
            result = call_tool(name, arguments or {})
            text = json.dumps(result, indent=2)
        except Exception as exc:  # surface errors as tool content, not a crash
            text = json.dumps({"error": str(exc), "tool": name})
        return [mcp_types.TextContent(type="text", text=text)]

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    anyio.run(_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
