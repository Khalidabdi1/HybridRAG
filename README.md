<div align="center">

# HybridRAG

**Search documents by what they _say_ and by how they _look_ — in one index.**

Text RAG is cheap, fast, and easy to update. Pixel RAG preserves tables, charts,
and layout. HybridRAG indexes **both** modalities, then **fuses** the results so
you get the strengths of each without committing to the weaknesses of either.

[Why Hybrid?](#why-hybrid) · [Architecture](#architecture) · [Quickstart](#quickstart) · [CLI](#cli) · [How it works](#how-it-works) · [Roadmap](#roadmap)

</div>

---

## Motivation

[Pixel RAG](https://github.com/StarTrail-org/PixelRAG) makes a compelling case:
render each page to a screenshot, slice it into tiles, embed the tiles with a
vision-language model (VLM), and retrieve visually. It never parses HTML, so it
keeps tables, charts, and complex layouts intact — and it rides the rapid
progress of VLMs like Qwen-VL.

That power comes with real costs:

| Pixel-only pain point | Why it hurts | How HybridRAG responds |
|---|---|---|
| **Storage** — screenshots + tiles + tile embeddings | 10M pages → tens/hundreds of TB | Text is cheap; store pixels only where they add value |
| **Slow indexing** — render → tile → VLM per page | Far longer than `HTML → chunk → embed` | Text path is fast; vision path runs in parallel and is optional per source |
| **Hard updates** — one line change re-screenshots the page | Text RAG updates a single chunk | Per-modality stores update independently |
| **GPU cost** — VLMs are heavy at index, rerank, and read | Expensive at scale | The **query router** skips vision when text suffices |
| **Text-native content** — code, logs, JSON | Vision is slower *and* worse here | Routed straight to the text index |
| **Lock-in to one VLM** | Whole system rebuilt if the model changes | Pluggable encoders behind a stable interface |

> The future isn't *only* text RAG, nor *only* pixel RAG — it's **hybrid**:
> extract the text, keep the image, search both, merge the results.
> That is exactly what HybridRAG implements.

## Why Hybrid

- **Recall where it counts.** A figure-only answer is found by pixels; a code
  snippet is found by text. Fusion surfaces a document that *either* modality
  ranks highly, and **reinforces** documents that *both* agree on.
- **Pay for vision only when it pays off.** The router reads the query and
  shifts weight toward text or pixels — code/log/JSON queries barely touch the
  GPU; "which chart shows…" leans visual.
- **No vendor lock-in.** Text and vision encoders sit behind small interfaces.
  Swap MiniLM for E5, or CLIP for Qwen-VL, without touching the pipeline.
- **Runs anywhere, immediately.** The core depends on **numpy only**. Deterministic
  fallback encoders let you try the full pipeline — ingest, fuse, serve — with
  zero model downloads, then opt into real models when you're ready.

## Architecture

![HybridRAG architecture](docs/images/architecture.svg)

A document flows down **two parallel pipelines**:

```
                         ┌───────────────────────────────────────────────┐
  Source (web/PDF/img) ──┤ TEXT:   extract → chunk → text-embed → index   │──┐
                         │ PIXEL:  render  → tile  → vision-embed → index │──┤
                         └───────────────────────────────────────────────┘  │
                                                                             ▼
        Query ──► Router (per-modality weights) ──► search both ──► RRF Fusion ──► Results
```

## Quickstart

```bash
git clone https://github.com/Khalidabdi1/HybridRAG.git
cd HybridRAG
pip install -e .            # core only — numpy
python examples/quickstart.py
```

```python
from hybridrag import HybridConfig
from hybridrag.engine import HybridRAG

rag = HybridRAG(HybridConfig())

# Index text (chunked + embedded automatically)
rag.add_text("py", "Python is dynamically typed and garbage-collected ...", title="Python")
rag.add_text("fin", "The chart and table show quarterly revenue growth ...", title="Report")

# Search — both modalities are queried and fused
for hit in rag.search("how is python typed?", top_k=3):
    print(hit.doc_id, round(hit.score, 4), hit.components)
```

The same `rag` can index image tiles too:

```python
from hybridrag.pipeline.render import tile_image      # needs: pip install -e ".[render]"

tiles = tile_image("page.png", doc_id="report", out_dir=".idx/tiles")
rag.add_tiles(tiles)
```

### Installing real models (optional extras)

```bash
pip install -e ".[text]"     # sentence-transformers (real text embeddings)
pip install -e ".[vision]"   # torch + transformers (CLIP / Qwen-VL style VLM)
pip install -e ".[render]"   # playwright + pymupdf (screenshots, PDF rendering)
pip install -e ".[serve]"    # FastAPI search server
pip install -e ".[faiss]"    # FAISS-backed vector store
pip install -e ".[all]"      # everything
```

Then point the config at real models:

```python
cfg = HybridConfig(
    text_model="sentence-transformers/all-MiniLM-L6-v2",
    vision_model="openai/clip-vit-base-patch32",   # or a Qwen-VL embedding model
)
rag = HybridRAG(cfg)
```

## CLI

![HybridRAG CLI demo](docs/images/cli-demo.svg)

```bash
# Index raw text / a file
hybridrag add-text   --storage .idx --id readme --file README.md --title Readme

# Screenshot + tile a web page (needs [render])
hybridrag ingest-url --storage .idx --id wiki --url https://en.wikipedia.org/wiki/RAG

# Extract text AND render+tile a PDF (both modalities)
hybridrag ingest-pdf --storage .idx --id paper --pdf paper.pdf

# Search (fused). Force a single modality with --modality text|vision
hybridrag search     --storage .idx --query "quarterly revenue table" -k 5
hybridrag stats      --storage .idx

# Serve a search API (needs [serve])
hybridrag serve      --storage .idx --port 8000
#  GET /search?q=...&k=5   ·   POST /search {"query": "...", "top_k": 5}

# Benchmark text-only vs vision-only vs hybrid on a dataset
hybridrag eval       --dataset examples/datasets/sample.json -k 1,5,10
```

## Use with Claude (MCP + Skill)

HybridRAG ships an [MCP](https://modelcontextprotocol.io) server so Claude Code,
Claude Desktop, or the Claude Agent SDK can index and search documents *during a
conversation* — and a [Claude skill](.claude/skills/hybridrag/SKILL.md) that
teaches Claude when and how to use it.

```bash
pip install -e ".[mcp]"        # installs the `mcp` runtime
hybridrag-mcp                  # or: python -m hybridrag.mcp_server
```

The repo includes a project-scoped [`.mcp.json`](.mcp.json), so inside this
directory Claude Code discovers the server automatically. To register it
elsewhere:

```bash
claude mcp add hybridrag -- python -m hybridrag.mcp_server
```

The server exposes four tools over a persistent index (set by
`$HYBRIDRAG_STORAGE`, default `.hybridrag_index`):

| Tool | Purpose |
| --- | --- |
| `hybridrag_search` | Fused text+pixel search; force a modality with `modality`. |
| `hybridrag_add_text` | Chunk and index a raw text document. |
| `hybridrag_add_html` | Extract text from HTML, then chunk and index it. |
| `hybridrag_stats` | Report index size, models, and dimensions. |

The bundled **skill** (`.claude/skills/hybridrag/`) is picked up automatically by
Claude Code in this repo; it tells Claude to prefer text-only retrieval for code,
logs, and JSON, and to blend in pixels for tables, charts, and complex layouts.

## Evaluation

The whole HybridRAG thesis is empirical: *does* fusing pixels with text help, and
what does it cost? The built-in harness answers that on **your** corpus with
standard ranking metrics (recall@k, precision@k, nDCG@k, hit@k, MRR), measured
latency, and the index footprint — for each modality side by side.

```bash
hybridrag eval                 # runs the built-in offline sample
```

```
Dataset: hybridrag-sample   metrics @k=10   (text_units=12, vision_units=0)
mode    recall  prec   nDCG   hit    MRR    ms/q  units
------  ------  -----  -----  -----  -----  ----  -----
text    0.917   0.092  0.804  0.917  0.767  0.10  12
vision  0.000   0.000  0.000  0.000  0.000  0.00  0
hybrid  0.917   0.092  0.804  0.917  0.767  0.10  12
  note[vision]: no indexed units for this modality — skipped
```

The built-in sample is intentionally **text-only** so it runs with numpy alone —
it measures a lexical baseline and proves the harness end to end. To get a real
three-way comparison, point it at a dataset with `image_path` documents and turn
on real encoders:

```bash
hybridrag eval --dataset my_corpus.json \
  --text-model sentence-transformers/all-MiniLM-L6-v2 \
  --vision-model openai/clip-vit-base-patch32 \
  -k 1,5,10 --json
```

Datasets are plain JSON — a corpus plus relevance judgements (`qrels`):

```json
{
  "name": "my_corpus",
  "documents": [
    {"doc_id": "d1", "text": "Python raises ZeroDivisionError when ..."},
    {"doc_id": "d2", "image_path": "tiles/revenue_chart.png"}
  ],
  "queries": [
    {"id": "q1", "query": "divide by zero error", "relevant": {"d1": 1.0}},
    {"id": "q2", "query": "quarterly revenue chart", "relevant": ["d2"]}
  ]
}
```

Relevance may be binary (`["d2"]`) or graded (`{"d1": 2.0, "d2": 1.0}`, used by
nDCG). The same API is available in Python:

```python
from hybridrag.eval import evaluate, build_engine, EvalDataset

ds = EvalDataset.from_file("my_corpus.json")
report = evaluate(build_engine(ds), ds.queries, ks=(1, 5, 10))
print(report.table())
```

## How it works

### 1. Two stores, one engine
`HybridRAG` keeps a `VectorStore` per modality (numpy by default, FAISS optional).
They are populated and **updated independently** — re-embedding a changed text
chunk never touches the pixel index, and vice versa.

### 2. The query router
Before searching, [`retrieve/router.py`](src/hybridrag/retrieve/router.py) reads
the query for cues — `code`, `stack trace`, `json`, `{ } ;` lean **text**;
`chart`, `table`, `diagram`, `layout` lean **vision** — and returns per-modality
weights. It never hard-disables a modality (recall is preserved); it shifts
emphasis so you don't pay GPU cost for a query that text answers better.

### 3. Reciprocal Rank Fusion
Cosine scores from a text encoder and a vision encoder are **not** comparable,
so [`retrieve/fusion.py`](src/hybridrag/retrieve/fusion.py) fuses by *rank*, not
raw score: each list contributes `weight / (k + rank)` per document. Results are
grouped by `doc_id`, so a document found by **both** modalities is reinforced,
and every hit reports its per-modality `components` for transparency.

### 4. Pluggable encoders
Everything implements `TextEmbedder` / `VisionEmbedder`
([`embed/base.py`](src/hybridrag/embed/base.py)). The default `hash` encoders are
deterministic and dependency-free (great for tests and demos); the real
`sentence-transformers` and VLM backends drop in by changing one config string.

## Project layout

```
src/hybridrag/
├── engine.py           # HybridRAG: ingest + search + persistence
├── config.py           # HybridConfig (all knobs, JSON-serializable)
├── types.py            # Chunk, Tile, Document, SearchResult, Modality
├── embed/              # text & vision encoders (+ hashing fallback)
├── index/              # VectorStore (numpy / FAISS)
├── pipeline/           # extract (HTML/PDF→text, chunk) + render (screenshot, tile)
├── retrieve/           # router (per-query weights) + fusion (RRF)
├── eval/               # metrics, datasets, harness (text vs vision vs hybrid)
├── serve/              # FastAPI search API
├── mcp_server.py       # MCP server for Claude (`hybridrag-mcp`)
└── cli.py              # `hybridrag` command
.claude/skills/hybridrag/  # Claude skill describing the engine
.mcp.json               # project-scoped MCP server registration
tests/                  # pytest suite (runs on numpy alone)
examples/quickstart.py  # end-to-end demo
examples/datasets/      # sample evaluation dataset (JSON)
```

## Development

```bash
pip install -e ".[dev]"
pytest -q          # full suite runs without any model downloads
ruff check .
```

## Roadmap

See [ROADMAP.md](ROADMAP.md). Near-term: cross-encoder reranking of fused
results, async batched ingestion, incremental updates/deletes keyed by
`doc_id`, and a real Qwen-VL embedding adapter. The
[MCP server + Claude skill](#use-with-claude-mcp--skill) landed in v0.3; the
[evaluation harness](#evaluation) (text-only / pixel-only / hybrid on one
corpus) landed in v0.2.

## Acknowledgements

Inspired by [PixelRAG](https://github.com/StarTrail-org/PixelRAG) by StarTrail —
HybridRAG keeps its visual insight and pairs it with classic text retrieval.

## License

[MIT](LICENSE)
