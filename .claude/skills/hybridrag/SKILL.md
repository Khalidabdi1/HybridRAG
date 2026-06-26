---
name: hybridrag
description: >-
  Index and search documents with HybridRAG — a hybrid text + pixel (vision)
  retrieval engine. Use when the user wants to ground answers in a document
  corpus, build a searchable index, ingest text/HTML/PDFs/web pages, or compare
  text vs vision vs hybrid retrieval. Works over the `hybridrag` MCP server
  (tools prefixed `hybridrag_`) or the `hybridrag` CLI.
---

# HybridRAG

HybridRAG fuses **text RAG** and **pixel (vision) RAG** into one index. Text is
chunked and embedded with a text encoder; rendered pages are sliced into tiles
and embedded with a vision encoder (a VLM such as Qwen-VL). At query time both
indexes are searched and fused with Reciprocal Rank Fusion, guided by a query
router that weights each modality per query. The core runs on **numpy alone**
(deterministic fallback encoders) so it works with zero model downloads; real
models are opt-in.

## When to use which surface

- **MCP tools** (preferred inside Claude) — the `hybridrag` server exposes
  `hybridrag_search`, `hybridrag_add_text`, `hybridrag_add_html`,
  `hybridrag_update_text`, `hybridrag_delete`, `hybridrag_list_docs`, and
  `hybridrag_stats`. Configured via `.mcp.json`; the index lives in
  `$HYBRIDRAG_STORAGE` (default `.hybridrag_index`).
- **CLI** — for screenshot/PDF ingestion and serving, which need extra
  dependencies.

## MCP workflow

1. `hybridrag_stats` — see whether an index already exists and how big it is.
2. `hybridrag_add_text` / `hybridrag_add_html` — index documents (each call
   persists). Pass a stable `doc_id` so re-indexing is traceable.
3. `hybridrag_search` — query. Returns fused, ranked hits with per-modality
   `components`. Set `modality: "text"` or `"vision"` to force one; omit to let
   the router blend both. Prefer text-only for code, logs, and JSON.
4. `hybridrag_update_text` / `hybridrag_delete` — when a source document
   changed or should be dropped, update or delete it by `doc_id` instead of
   rebuilding the index; only that document is re-embedded. `hybridrag_list_docs`
   shows what is currently indexed.

Ground your answer in the returned `text`/`image_path` and cite `doc_id`.

## CLI reference

```bash
pip install -e ".[mcp]"                 # MCP server deps
hybridrag add-text  --storage .idx --id doc1 --file README.md --title Readme
hybridrag ingest-url --storage .idx --id wiki --url https://example.com   # needs [render]
hybridrag ingest-pdf --storage .idx --id paper --pdf paper.pdf            # needs [render]
hybridrag search    --storage .idx --query "revenue table" -k 5
hybridrag add-text  --storage .idx --id doc1 --file README.md --replace    # upsert in place
hybridrag delete    --storage .idx --id doc1                               # remove a document
hybridrag list-docs --storage .idx
hybridrag eval      --dataset examples/datasets/sample.json               # text vs vision vs hybrid
hybridrag stats     --storage .idx
```

## Notes

- Keep `$HYBRIDRAG_STORAGE` consistent between writes and reads — the MCP
  server and CLI must point at the same directory to share an index.
- Vision ingestion (`ingest-url`, `ingest-pdf`) needs the `[render]` extra
  (Playwright + PyMuPDF); text and HTML indexing run on numpy alone.
