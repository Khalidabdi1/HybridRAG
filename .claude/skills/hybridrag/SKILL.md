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
  `hybridrag_search`, `hybridrag_answer`, `hybridrag_add_text`,
  `hybridrag_add_html`, `hybridrag_add_batch`, `hybridrag_update_text`,
  `hybridrag_delete`, `hybridrag_list_docs`, `hybridrag_stats`,
  `hybridrag_cost`, `hybridrag_richness`, `hybridrag_route`, and
  `hybridrag_dedup`. Configured via `.mcp.json`; the index lives in
  `$HYBRIDRAG_STORAGE` (default `.hybridrag_index`).
- **CLI** — for screenshot/PDF ingestion and serving, which need extra
  dependencies.

## MCP workflow

1. `hybridrag_stats` — see whether an index already exists and how big it is.
2. `hybridrag_add_text` / `hybridrag_add_html` — index documents (each call
   persists). Pass a stable `doc_id` so re-indexing is traceable. For **many
   documents at once**, prefer `hybridrag_add_batch` (a `documents` list of
   `{doc_id, text|html, title?}`): chunks are embedded in batches, so it is far
   cheaper than one add call per document. Set `replace: true` to upsert.
3. `hybridrag_search` — query. Returns fused, ranked hits with per-modality
   `components`. Set `modality: "text"` or `"vision"` to force one; omit to let
   the router blend both. Prefer text-only for code, logs, and JSON. Pass
   `rerank: true` to re-score the fused candidates against the query (BM25 /
   cross-encoder) for higher precision on keyword-specific questions; the chosen
   hits then carry a `components.rerank` value. Pass `fusion: "calibrated"` to
   fuse on normalized raw scores instead of rank (RRF) — this keeps confidence
   magnitude, so a strongly-matching doc leads a weakly-matching one by more than
   one rank; leave it unset for the default `rrf`.
4. `hybridrag_update_text` / `hybridrag_delete` — when a source document
   changed or should be dropped, update or delete it by `doc_id` instead of
   rebuilding the index; only that document is re-embedded. `hybridrag_list_docs`
   shows what is currently indexed.

Ground your answer in the returned `text`/`image_path` and cite `doc_id`.

**Getting a cited answer in one call.** When the user asks a *question* (rather
than for a list of passages), prefer `hybridrag_answer` over `hybridrag_search`:
it retrieves and synthesizes a grounded answer whose every claim carries a `[n]`
citation back to an indexed chunk (nothing is invented). It returns `text`,
`citations` (with `doc_id`/`page`/`snippet`), and `visual_evidence` — relevant
tables/charts that were retrieved but carry no readable text, which you may want
to surface to the user. Accepts the same `modality` and `rerank` options as
search.

When the user asks what an index *costs* — storage, $/query, or "how big would N
pages be?" — call `hybridrag_cost` (optionally with `project_pages`). It returns
vector + artifact bytes, indexing and per-query $ for text vs vision vs hybrid,
and a storage projection that quantifies why pixel-only RAG is far pricier.

**Selective pixel indexing.** HybridRAG renders + tiles + embeds a page into the
vision index only when it's visually rich enough to earn it — prose, code, logs,
and JSON stay text-only, saving storage and GPU. Call `hybridrag_richness` with a
page's `text` and/or `html` to see the verdict (`index_pixels`, a richness
`score`, the per-signal breakdown, and a `reason`) — useful for explaining why a
code/log page should not be pixel-indexed while a financial-table page should.
The `pixel_selection` config knob (`auto`/`always`/`never`) and the
`--vision-selection` CLI flag control the policy at ingest time.

**Tile deduplication.** A document's repeating header/footer/logo band tiles into
one byte-identical copy per page, so HybridRAG deduplicates tiles before
embedding — each unique tile is embedded and stored once (cutting vision storage
*and* VLM inference), with every duplicate recorded as `occurrences` metadata so
citations still resolve to each page. It is on by default (`tile_dedup` config:
`doc`/`global`/`off`) and runs inside ingestion. To estimate the saving on
already-rendered tiles, call `hybridrag_dedup` with `image_paths` (or run
`hybridrag dedup --dir <tiles>`): it reports unique vs duplicate tiles, the dedup
ratio, and bytes saved, without indexing anything. Use it to quantify the storage
argument for pixel-heavy corpora.

**The query router.** By default `hybridrag_search`/`hybridrag_answer` let a
router weight text vs vision per query (never zeroing a modality, so recall is
preserved). Two routers exist: the default `heuristic` keyword rules, and a
`learned` numpy logistic-regression classifier that outputs a calibrated
`P(vision-relevant)`. Call `hybridrag_route` with a `query` (and optional
`model: "learned"`) to see how a query *would* be split — the per-modality
weights and, for the learned router, the probability — without running a search.
Use it to explain why a code/log/JSON query stays text-heavy while a
chart/table/layout query pulls in pixels. Select the router for an index via the
`router_model` config; retrain the learned router on your own labelled queries
with `hybridrag train-router`.

## CLI reference

```bash
pip install -e ".[mcp]"                 # MCP server deps
hybridrag add-text  --storage .idx --id doc1 --file README.md --title Readme
hybridrag ingest-url --storage .idx --id wiki --url https://example.com   # needs [render]
hybridrag ingest-pdf --storage .idx --id paper --pdf paper.pdf            # needs [render]
hybridrag ingest-batch --storage .idx --manifest corpus.jsonl --batch-size 128 --workers 4  # large corpora
hybridrag search    --storage .idx --query "revenue table" -k 5
hybridrag answer    --storage .idx --query "what was Q3 revenue?"          # grounded, cited answer
hybridrag add-text  --storage .idx --id doc1 --file README.md --replace    # upsert in place
hybridrag delete    --storage .idx --id doc1                               # remove a document
hybridrag list-docs --storage .idx
hybridrag eval      --dataset examples/datasets/sample.json               # text vs vision vs hybrid
hybridrag cost      --storage .idx --project-pages 10000000               # storage + $/query model
hybridrag richness  --file page.html                                      # selective-indexing verdict
hybridrag dedup     --dir .idx/tiles                                      # tile duplicates + storage saved
hybridrag route     --query "which chart shows revenue" --model learned    # per-query modality weights
hybridrag train-router --examples queries.jsonl --out router.json          # fit the learned router
hybridrag stats     --storage .idx
```

## Notes

- Keep `$HYBRIDRAG_STORAGE` consistent between writes and reads — the MCP
  server and CLI must point at the same directory to share an index.
- Vision ingestion (`ingest-url`, `ingest-pdf`) needs the `[render]` extra
  (Playwright + PyMuPDF); text and HTML indexing run on numpy alone.
