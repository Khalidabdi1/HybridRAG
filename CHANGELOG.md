# Changelog

All notable changes to HybridRAG are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/).

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
