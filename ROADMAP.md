# HybridRAG Roadmap

HybridRAG is developed incrementally. This file tracks what exists and what's next.

## Done (v0.1.0)
- [x] Dual-modality engine (text + pixel) with independent vector stores
- [x] Character-window chunking with soft boundaries and overlap
- [x] HTML→text and PDF→text extraction
- [x] Screenshot rendering (Playwright) and page tiling (Pillow)
- [x] Pluggable encoders: hashing fallback, sentence-transformers, VLM/CLIP
- [x] numpy + optional FAISS vector store with persistence
- [x] Query router (per-query text/vision weighting)
- [x] Reciprocal Rank Fusion across modalities
- [x] CLI (`add-text`, `ingest-url`, `ingest-pdf`, `search`, `stats`, `serve`)
- [x] FastAPI search server
- [x] pytest suite that runs on numpy alone

## Done (v0.2.0)
- [x] **Evaluation harness** (`hybridrag.eval`) — compare text-only vs
      pixel-only vs hybrid on one corpus: recall@k, precision@k, nDCG@k, hit@k,
      MRR, latency, and index footprint, via the `hybridrag eval` CLI. JSON
      dataset format with binary/graded qrels + a built-in offline sample.

## Done (v0.3.0)
- [x] **MCP server + Claude skill** — `hybridrag.mcp_server` exposes
      `hybridrag_search`, `hybridrag_add_text`, `hybridrag_add_html`, and
      `hybridrag_stats` over the Model Context Protocol (`hybridrag-mcp`), with a
      project-scoped `.mcp.json` and a `.claude/skills/hybridrag` skill so Claude
      can index and search documents mid-conversation. Dependency-light handler
      layer runs on numpy alone; the `mcp` runtime is an opt-in extra.

## Done (v0.4.0)
- [x] **Incremental updates & deletes keyed by `doc_id`** — `delete`,
      `list-docs`, and `add-text --replace` (upsert) on the CLI;
      `HybridRAG.delete()` / `upsert_text()` / `upsert_html()` / `doc_ids()` on
      the engine; `delete_where()` / `delete_doc()` on the store (compacts
      vectors + metadata, rebuilds FAISS from survivors); and
      `hybridrag_delete` / `hybridrag_update_text` / `hybridrag_list_docs` MCP
      tools. Only the changed document is re-embedded.

## Done (v0.5.0)
- [x] **Per-query cost & storage model** (`hybridrag.eval.cost`) — measure vector
      bytes exactly from the index, model raw artifact bytes (screenshots, tiles,
      stored text), one-time indexing $, and $/query per modality, then project
      total storage + monthly $ to any corpus size (the "10M pages → N TB"
      question). Tunable `CostModel` unit prices; `hybridrag cost`, `hybridrag
      eval --cost`, and a `hybridrag_cost` MCP tool.

## Next
- [ ] **Cross-encoder reranking** of the fused candidate set.
- [ ] **Qwen-VL embedding adapter** with batched GPU inference.
- [ ] **Async / batched ingestion** for large corpora.
- [ ] **Selective pixel indexing** — heuristics to render only pages that are
      visually rich (tables/figures), saving storage and GPU.
- [ ] **Hybrid answer synthesis** — feed top text + top tiles to a VLM reader.
- [ ] **Benchmarks & screenshots** from a real corpus in the README.

## Ideas / research
- Learned router (small classifier) instead of keyword heuristics.
- Score-calibrated fusion as an alternative to RRF.
- Tile-level deduplication to cut storage further.
