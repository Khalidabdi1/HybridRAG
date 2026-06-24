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

## Next
- [ ] **Per-query cost model** — extend the harness with $/query and storage
      projections (vector + screenshot bytes) across modalities.
- [ ] **Cross-encoder reranking** of the fused candidate set.
- [ ] **Qwen-VL embedding adapter** with batched GPU inference.
- [ ] **Async / batched ingestion** for large corpora.
- [ ] **Incremental updates & deletes** keyed by `doc_id`.
- [ ] **Selective pixel indexing** — heuristics to render only pages that are
      visually rich (tables/figures), saving storage and GPU.
- [ ] **Hybrid answer synthesis** — feed top text + top tiles to a VLM reader.
- [ ] **Benchmarks & screenshots** from a real corpus in the README.

## Ideas / research
- Learned router (small classifier) instead of keyword heuristics.
- Score-calibrated fusion as an alternative to RRF.
- Tile-level deduplication to cut storage further.
