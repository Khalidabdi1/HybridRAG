# Changelog

All notable changes to HybridRAG are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/).

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
