# Changelog

All notable changes to HybridRAG are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.12.0] - 2026-07-05
### Added
- **Tile-level deduplication** (`hybridrag.pipeline.dedup`) — collapse
  content-identical image tiles to a single embedded + stored representative,
  directly attacking the storage bill (disadvantage #1) and VLM inference cost
  (disadvantage #4). A document's running header/footer/logo band tiles into one
  identical copy per page; embedding and storing each one is waste.
  - **`TileDeduplicator`.** Content-hashes every tile and keeps one
    representative per group; each other occurrence is recorded on the rep as
    `meta["occurrences"]` (+ `duplicate_count`), so citations still resolve to
    every page the tile appears on. Returns a `DedupStats` (input/unique/
    duplicate counts, dedup ratio, estimated bytes saved).
  - **Scope keeps deletes correct.** `scope="doc"` (default) only collapses
    tiles within a single `doc_id`, so a representative is never shared across
    documents and `delete(doc_id)` stays clean; `scope="global"` collapses
    across documents for the maximum saving.
  - **Exact and perceptual matching.** `method="exact"` (default) SHA-256s the
    tile bytes — no dependencies; `method="ahash"` folds visually-identical but
    byte-different copies (re-encoded PNGs, format changes) via an 8×8 average
    hash (needs Pillow, degrades gracefully to exact). Path-less/metadata-only
    tiles always stay distinct.
  - **Wired through ingestion.** `HybridRAG.add_tiles()` deduplicates before
    embedding when `tile_dedup` is on (default `"doc"`); `BatchIngestor` reports
    `tiles_deduplicated`; `stats()` surfaces cumulative `tiles_deduplicated` and
    `dedup_bytes_saved`. Non-duplicated tiles pass through untouched, so search,
    fusion, rerank, and delete are unchanged.
  - **Config.** `tile_dedup` (`"doc"` / `"global"` / `"off"`),
    `tile_dedup_method` (`"exact"` / `"ahash"`).
  - **Surfaces.** `hybridrag dedup --dir <tiles>` (dry-run analysis of rendered
    tiles: unique vs duplicate, storage saved) and a `hybridrag_dedup` MCP tool.
  - Fully covered by numpy-only tests (`tests/test_dedup.py`); the perceptual
    path is Pillow-gated like the rest of the vision suite.

## [0.11.0] - 2026-07-04
### Added
- **Learned query router** (`hybridrag.retrieve.router`) — a small
  **logistic-regression classifier** that replaces the hand-tuned keyword
  weights, the top open roadmap item. Directly sharpens disadvantage #4 (GPU
  cost): a better router sends fewer queries down the expensive vision path.
  - **`LearnedRouter`.** Predicts a calibrated `P(vision-relevant)` from
    `extract_features(query)` — interpretable, embedding-free signals
    (text/vision cue densities, code structure, numeric-token density, query
    length) — and maps it to per-modality weights that never zero a modality
    (recall preserved). Ships **pre-trained** on an embedded, human-labelled seed
    set (`DEFAULT_TRAINING_EXAMPLES`) so it works out of the box.
  - **Deterministic training.** `fit()` runs full-batch gradient descent on
    cross-entropy with L2, initialised at zero — no random seed — so the default
    weights are reproducible and the whole thing runs on **numpy alone**.
    `save()` / `load()` / `to_dict()` / `from_dict()` persist trained weights.
  - **Unlike fixed constants, it's retrainable** on your own labelled query logs
    — the point of a learned router.
  - **`HeuristicRouter`** wraps the original `route()` behind the same interface;
    `build_router(config)` selects between them. `route()` and `RouteDecision`
    stay backward-compatible (`RouteDecision.vision_probability` is the only new,
    optional field).
  - **Config.** `router_model` (`"heuristic"` / `"learned"`),
    `router_weights_path` (load trained weights).
  - **Surfaces.** `hybridrag route` (inspect a query's weights) and
    `hybridrag train-router` (fit + save) CLI commands; a `hybridrag_route` MCP
    tool that explains the split without searching. Engine routes through
    `self.router`.
  - **Tests.** `tests/test_learned_router.py` covers feature extraction, training
    determinism, seed-set separation, weight mapping, persistence round-trip,
    `build_router`, and engine wiring; plus `hybridrag_route` MCP tests. Full
    suite: **160 passed, 3 skipped**; `ruff` clean. Runs on numpy alone.

## [0.10.0] - 2026-07-03
### Added
- **Qwen-VL embedding adapter with batched GPU inference**
  (`hybridrag.embed.vision`) — makes the vision modality real with the same
  model family pixel-only systems are built on, and directly softens
  disadvantage #6 (lock-in to one VLM) and #4 (GPU cost).
  - **`QwenVLEmbedder`.** Targets the Qwen2-VL / GME embedding family (e.g.
    `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`) via `get_image_embeddings` /
    `get_text_embeddings`, mapping an **instruction-prefixed** text query and a
    page image into one shared space. Adapts to the signature variance across
    GME checkpoints (keyword `images=`/`texts=`+`instruction=`, positional
    fallback) and casts to fp16 on CUDA.
  - **`BatchedVisionEmbedder`.** Both vision backends now subclass it and
    implement per-batch `_encode_images` / `_encode_texts`; the base slices
    input into `vision_batch_size` chunks, encodes each, concatenates in input
    order, and L2-normalizes once. Batched output is **identical, row-for-row**,
    to a single un-chunked call — batching only bounds GPU memory on long pages
    and large query sets, it never changes results. `iter_batches` helper.
  - **`VLMVisionEmbedder`** (CLIP-style dual-head) refactored onto the same
    batched base; both wrappers take a `batch_size`.
  - **Backend selection.** `vision_backend` config (`auto`/`clip`/`qwen`) with
    an `auto` heuristic that picks the Qwen-VL adapter when the model id mentions
    `qwen`/`gme`, else the CLIP wrapper. New config: `vision_backend`,
    `vision_batch_size`, `vision_query_instruction`.
  - **Tests.** The batching machinery (chunking, order preservation,
    normalization, empty inputs) and backend resolution are covered by
    numpy-only tests via a fake recording embedder — no torch required.

## [0.9.0] - 2026-07-01
### Added
- **Batched & parallel ingestion for large corpora** (`hybridrag.pipeline.ingest`)
  — directly attacks disadvantage #2 (indexing is slow). The naive engine path
  embeds one document at a time (`add_text` → one `encode()` per document), which
  wastes a real encoder's throughput; `BatchIngestor` amortizes it.
  - **Batched embedding.** Prepared chunks and tiles are buffered *across*
    documents and flushed to the vector store in fixed-size batches, so each
    `encode()` call sees `batch_size` units — the exact shape a batched GPU
    encoder wants, and it cuts per-call overhead on the numpy fallback too.
  - **Parallel preparation.** Per-document prep (HTML→text, chunking, tiling
    metadata, the selective-pixel richness decision) runs on a `ThreadPoolExecutor`
    with a bounded look-ahead window, so the next document is prepared while the
    current batch embeds. Store writes stay single-threaded and in input order, so
    batched output is **byte-for-byte identical** to serial ingestion.
  - `IngestDoc` (text / html / image_paths / title / meta, with `from_dict`) and
    `IngestStats` (documents, chunks/tiles added, vision_skipped, batch counts,
    seconds, docs/s) dataclasses. Selective pixel indexing applies per document,
    and `upsert=True` re-embeds only changed `doc_id`s.
  - Engine: `HybridRAG.add_documents(docs, batch_size=, vision_batch_size=,
    max_workers=, upsert=, on_progress=)`.
  - CLI: `hybridrag ingest-batch --manifest <JSON array | JSONL>` with
    `--batch-size`, `--workers`, `--replace`, `--vision-selection`, `--json`, and
    a streaming progress line.
  - MCP: a `hybridrag_add_batch` tool so Claude can index a whole corpus in one
    batched call.
  - Tests: `tests/test_ingest.py` (batching, serial/parallel equivalence,
    equivalence with `add_text`, html, meta, upsert, progress, the vision path +
    selection) plus `hybridrag_add_batch` MCP tests.

## [0.8.0] - 2026-06-30
### Added
- **Hybrid answer synthesis** (`hybridrag.synth`) — the RAG "final readout" that
  turns fused search hits into a grounded, *cited* answer, making it a pluggable
  step instead of an always-on VLM call (Pixel RAG's most expensive operation).
  - `ExtractiveReader` (default, **numpy alone**) ranks the sentences inside the
    retrieved chunks by BM25 against the query, stitches the best few into a short
    answer, and attaches a `[n]` citation to each span. Every word is copied
    verbatim from an indexed chunk, so there is nothing to hallucinate. Relevant
    tables/charts with no readable text are returned as `visual_evidence` so a
    caller can still surface the figure.
  - `LLMReader` (opt-in) wraps any `generate(prompt, image_paths) -> str`
    callable (Claude, Qwen-VL, a local model). It builds a grounded,
    citation-instructed prompt from the top text chunks and passes the top tile
    images through for a VLM to read — only ever over the handful of fused hits,
    so the heavy model's cost stays bounded.
  - `Answer` / `Citation` dataclasses with `to_dict()` and a `formatted()`
    human-readable rendering (answer + numbered source list).
  - Engine: `HybridRAG.answer(query, top_k=, modality=, rerank=, reader=)`
    retrieves then synthesizes; an explicit `reader=` overrides config.
  - Config: `reader_model` (`"extractive"` / `"none"`), `answer_max_sentences`,
    `answer_top_k`.
  - CLI: `hybridrag answer --query ...` (with `--json`, `--modality`, `--rerank`).
  - MCP: a `hybridrag_answer` tool so Claude gets a cited answer in one call.
  - Tests: `tests/test_synth.py`.

## [0.7.0] - 2026-06-29
### Added
- **Selective pixel indexing** (`hybridrag.pipeline.select`) — render, tile, and
  embed a page into the vision index *only when it is visually rich enough to
  earn it*, directly cutting Pixel RAG's two worst costs (storage and GPU) on
  text-native pages (prose, code, logs, JSON) that the text index already covers.
  - Two heuristic signal sources, cheapest first: `score_text_richness(text,
    html)` reads extracted text/HTML **before** rendering (tabular rows, numeric
    grids, column alignment, and media tags `<table>/<svg>/<canvas>/<img>/
    <figure>/chart`) so a "no" skips rendering entirely; `score_image_richness`
    scores a rendered page (numpy array / PIL image / PNG path) from ruled-line
    density (tables), colour saturation (charts/figures), and mid-tone density
    (photos) — the cues that survive only in pixels.
  - `PixelSelector` applies the policy and returns a `SelectionDecision`
    (`index_pixels`, `score`, per-signal breakdown, human-readable `reason`).
  - Config: `pixel_selection` (`"auto"` / `"always"` / `"never"`),
    `pixel_selection_threshold`, `pixel_selection_text_weight`. `"always"` keeps
    classic Pixel RAG behaviour; `"never"` is text-only.
  - Engine: `HybridRAG.should_index_pixels(text=, html=, image=)` and
    `add_tiles_if_rich(...)`.
  - CLI: `ingest-url` / `ingest-pdf` now skip the vision path for text-native
    pages (with a `--vision-selection` override and a skipped-page count), and a
    new `hybridrag richness` command inspects a page's score and decision.
  - MCP: a `hybridrag_richness` tool so Claude can explain/simulate the decision.
  - Runs on **numpy alone** (Pillow only needed to load an image from a path);
    tests in `tests/test_select.py`.

## [0.6.0] - 2026-06-28
### Added
- **Cross-encoder reranking of the fused candidate set** (`hybridrag.retrieve.rerank`)
  — RRF fuses by *rank* and never looks at query/document content together, so it
  can't tell a strong answer from a mediocre one at adjacent ranks. A reranker
  re-scores the small fused candidate set against the query *jointly* for higher
  precision, the classic cross-encoder pattern (too slow corpus-wide, ideal here).
  - `Reranker` interface with two backends, mirroring the embedder design:
    `LexicalReranker` — a dependency-free **BM25** scorer over the candidate set
    (IDF-weighted, length-normalized; genuinely discriminative offline, not a
    placeholder) — and `CrossEncoderReranker`, an opt-in
    `sentence-transformers` `CrossEncoder` wrapper.
  - `rerank_results()` blends the reranker signal with the fusion score, both
    min-max normalized (`combined = blend*rerank + (1-blend)*fusion`), records it
    under `components['rerank']`, and never demotes image-only tiles that have no
    text to score — they keep their fusion standing.
  - Engine: `HybridRAG.search(..., rerank=...)` fuses a deeper candidate pool
    (`rerank_top_n`) then reranks before returning `top_k`; per-query override.
  - Config: `enable_rerank`, `rerank_model` (`"lexical"` / `"none"` / a
    CrossEncoder id), `rerank_top_n`, `rerank_blend`.
  - CLI: `hybridrag search --rerank` / `--no-rerank`.
  - MCP: a `rerank` parameter on `hybridrag_search`.
  - Tests: `tests/test_rerank.py`.

## [0.5.0] - 2026-06-27
### Added
- **Per-query cost & storage model** (`hybridrag.eval.cost`) — turns Pixel RAG's
  central weakness (storage and GPU cost) into measurable numbers per modality.
  - `CostModel`: tunable unit prices (storage $/GB-month, text vs vision embed $,
    render $/page, per-query $) and artifact sizes (screenshot, tile, text bytes)
    with documented order-of-magnitude defaults; override via a JSON file.
  - `estimate_cost(engine)` → `CostReport` with a `ModeCost` breakdown for text /
    vision / hybrid: vector bytes (measured exactly), modelled artifact bytes,
    one-time index $, and $/query; one screenshot is attributed per distinct
    rendered page and one tile per vision unit.
  - `CostReport.project(pages)` and `projection_table()` extrapolate total storage
    and monthly $ to any corpus size (e.g. 10M pages → terabytes), with the
    vision/text storage ratio called out.
  - CLI: `hybridrag cost` (on a persisted index or a dataset, `--cost-model`,
    `--project-pages`) and a `--cost`/`--project-pages` flag on `hybridrag eval`.
  - MCP: `hybridrag_cost` tool so Claude can reason about $/storage mid-chat.
  - Tests: `tests/test_cost.py`.

## [0.4.0] - 2026-06-26
### Added
- **Incremental updates & deletes keyed by `doc_id`** — re-embed only the
  document that changed instead of rebuilding the whole index, directly
  addressing Pixel RAG's costly "change one line → re-screenshot/re-tile/
  re-embed the page" update path.
  - `VectorStore.delete_where()`, `delete_doc()`, and `doc_ids()` — predicate
    deletion that compacts vectors + metadata and rebuilds the FAISS index from
    the survivors.
  - Engine: `HybridRAG.delete(doc_id)` (removes units across both modalities),
    `upsert_text()` / `upsert_html()` (delete-then-add in place), and
    `doc_ids()`.
  - CLI: `hybridrag delete`, `hybridrag list-docs`, and a `--replace` (upsert)
    flag on `add-text`.
  - MCP tools: `hybridrag_delete`, `hybridrag_update_text`, and
    `hybridrag_list_docs`.
  - `stats()` now reports a `documents` count.
  - Tests: delete/upsert coverage in `test_store.py`, `test_engine.py`, and
    `test_mcp_server.py`.

## [0.3.0] - 2026-06-25
### Added
- **MCP server** (`hybridrag.mcp_server`) exposing HybridRAG to Claude Code,
  Claude Desktop, and the Claude Agent SDK over the Model Context Protocol.
  - Tools: `hybridrag_search` (fused text+pixel search, optional forced
    modality), `hybridrag_add_text`, `hybridrag_add_html`, `hybridrag_stats`.
  - Dependency-light, unit-tested handler layer (`call_tool`) that runs on numpy
    alone; the `mcp` runtime is an opt-in extra and `main()` degrades gracefully
    with a clear install hint when it is absent.
  - Persistent index keyed by `$HYBRIDRAG_STORAGE` (default `.hybridrag_index`);
    every mutating tool call saves.
- `hybridrag-mcp` console entry point and `pip install -e ".[mcp]"` extra.
- Project-scoped `.mcp.json` so Claude Code auto-discovers the server in-repo.
- **Claude skill** at `.claude/skills/hybridrag/SKILL.md` describing when and how
  to index/search and which modality to prefer.
- Tests: `test_mcp_server.py`.

## [0.2.0] - 2026-06-24
### Added
- Evaluation harness (`hybridrag.eval`) comparing text-only, vision-only, and
  hybrid retrieval on the same corpus.
  - Ranking metrics (`eval/metrics.py`): recall@k, precision@k, nDCG@k (graded),
    hit@k, and MRR — pure, dependency-free, unit-tested.
  - Dataset container and JSON loader (`eval/dataset.py`) with binary or graded
    relevance judgements, plus `build_engine()` to index a dataset.
  - `evaluate()` harness (`eval/harness.py`) reporting per-mode metrics, mean
    latency, and index footprint; empty modalities are noted, not dropped.
  - Built-in offline sample dataset (`eval/sample.py`, runs on numpy alone).
- `hybridrag eval` CLI command (`--dataset`, `--text-model`, `--vision-model`,
  `-k`, `--modes`, `--json`).
- Example dataset JSON at `examples/datasets/sample.json`.
- Tests: `test_metrics.py`, `test_eval.py`.

## [0.1.0] - 2026-06-23
### Added
- Initial release of HybridRAG: a hybrid text + pixel (vision) RAG system.
- `HybridRAG` engine with independent text and vision vector stores, search,
  fusion, and JSON+npy persistence.
- Pluggable embedders (`embed/`): deterministic hashing fallback (numpy-only),
  sentence-transformers text encoder, and a VLM/CLIP vision encoder.
- Ingestion pipeline (`pipeline/`): HTML/PDF text extraction, character-window
  chunking, Playwright screenshot rendering, and page tiling.
- Retrieval (`retrieve/`): keyword query router for per-modality weighting and
  Reciprocal Rank Fusion across modalities.
- `VectorStore` with numpy backend and optional FAISS.
- `hybridrag` CLI (`add-text`, `ingest-url`, `ingest-pdf`, `search`, `stats`,
  `serve`) and a FastAPI search server.
- Test suite that runs without any model downloads; quickstart example.
- Documentation: README with architecture and CLI diagrams, ROADMAP.
