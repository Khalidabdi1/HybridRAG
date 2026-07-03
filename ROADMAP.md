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

## Done (v0.6.0)
- [x] **Cross-encoder reranking of the fused candidate set** (`hybridrag.retrieve.rerank`)
      — a `Reranker` interface with a dependency-free BM25 `LexicalReranker` default and
      an opt-in `sentence-transformers` `CrossEncoderReranker`. The engine fuses a deeper
      candidate pool (`rerank_top_n`) and re-scores it against the query, blending the
      reranker signal with the fusion score (`rerank_blend`). Exposed via
      `search(rerank=...)`, `hybridrag search --rerank/--no-rerank`, and a `rerank`
      parameter on the `hybridrag_search` MCP tool.

## Done (v0.7.0)
- [x] **Selective pixel indexing** (`hybridrag.pipeline.select`) — index a page
      into the vision modality only when it's visually rich enough to earn it.
      Pre-render text/HTML scoring (tabular rows, numeric grids, media tags) skips
      rendering entirely for text-native pages; image scoring (ruled lines, colour
      saturation, mid-tones) judges a rendered page. `pixel_selection`
      auto/always/never policy, `HybridRAG.should_index_pixels()`, a `hybridrag
      richness` CLI command, and a `hybridrag_richness` MCP tool. Directly cuts
      storage and GPU on code/log/JSON/prose pages.

## Done (v0.8.0)
- [x] **Hybrid answer synthesis** (`hybridrag.synth`) — the RAG "final readout"
      as a pluggable step. A dependency-free `ExtractiveReader` selects the best
      sentences from retrieved chunks (BM25) and returns a grounded, **cited**
      answer that never hallucinates, with relevant figures surfaced as
      `visual_evidence`; an opt-in `LLMReader` wraps any Claude/Qwen-VL callable
      and feeds it the top text chunks + tile images over the bounded hit set.
      `HybridRAG.answer()`, a `hybridrag answer` CLI command, and a
      `hybridrag_answer` MCP tool.

## Done (v0.9.0)
- [x] **Batched & parallel ingestion for large corpora**
      (`hybridrag.pipeline.ingest`) — `BatchIngestor` buffers prepared chunks and
      tiles *across* documents and flushes them to the store in fixed-size
      embedding batches (the shape a batched GPU encoder needs), while
      per-document prep (HTML→text, chunking, tiling, the selective-pixel
      decision) runs on a thread pool so the next document is prepared while the
      current batch embeds. Store writes stay single-threaded and in input order,
      so batched output is byte-for-byte identical to serial. `IngestDoc` /
      `IngestStats`, `HybridRAG.add_documents(...)`, a `hybridrag ingest-batch`
      CLI command (JSON/JSONL manifest, `--batch-size` / `--workers` / `--replace`),
      and a `hybridrag_add_batch` MCP tool. Directly attacks disadvantage #2
      (slow indexing). Runs on numpy alone.

## Done (v0.10.0)
- [x] **Qwen-VL embedding adapter with batched GPU inference**
      (`hybridrag.embed.vision`) — a `QwenVLEmbedder` for the Qwen2-VL / GME
      family (e.g. `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`) that aligns an
      instruction-prefixed text query with a page image in one shared space, the
      retrieval backbone pixel-only systems are built on. Both vision backends
      now subclass a new `BatchedVisionEmbedder`, encoding tiles/queries in
      `vision_batch_size` GPU mini-batches (order- and value-preserving, so
      batching only bounds memory). `vision_backend` (`auto`/`clip`/`qwen`) auto-
      selects the wrapper from the model id. Batching machinery + backend
      selection are covered by numpy-only tests.

## Next
- [ ] **Benchmarks & screenshots** from a real corpus in the README.
- [ ] **Learned router** (small classifier) instead of keyword heuristics.

## Ideas / research
- Learned router (small classifier) instead of keyword heuristics.
- Score-calibrated fusion as an alternative to RRF.
- Tile-level deduplication to cut storage further.
