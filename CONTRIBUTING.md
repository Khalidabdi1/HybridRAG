# Contributing to HybridRAG

Thanks for your interest! HybridRAG is built to be approachable: the core runs
on numpy alone, so you can develop and test without GPUs or model downloads.

## Setup

```bash
pip install -e ".[dev]"
pytest -q          # should pass on a clean checkout
ruff check .
```

## Guidelines

- **Keep the core light.** New heavy dependencies belong in an optional extra in
  `pyproject.toml`, imported lazily inside the function that needs them (see how
  `pipeline/render.py` handles Playwright).
- **Stay behind the interfaces.** New encoders implement `TextEmbedder` /
  `VisionEmbedder` (`embed/base.py`); new stores match `VectorStore`'s API.
- **Add tests.** Every module under `src/hybridrag/` should be exercisable with
  the numpy-only fallbacks.
- **Run `ruff` and `pytest`** before opening a PR.

## Good first issues

See [ROADMAP.md](ROADMAP.md) — the evaluation harness, a Qwen-VL adapter, and
incremental updates/deletes are all self-contained starting points.
