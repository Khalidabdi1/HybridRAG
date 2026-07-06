"""Command line interface for HybridRAG.

Examples::

    hybridrag add-text --storage .idx --id doc1 --file README.md --title Readme
    hybridrag ingest-url --storage .idx --id wiki --url https://example.com
    hybridrag ingest-pdf --storage .idx --id paper --pdf paper.pdf
    hybridrag add-text --storage .idx --id doc1 --file README.md --replace
    hybridrag search --storage .idx --query "revenue table" -k 5
    hybridrag answer --storage .idx --query "what was Q3 revenue?"
    hybridrag delete --storage .idx --id doc1
    hybridrag list-docs --storage .idx
    hybridrag stats --storage .idx
    hybridrag serve --storage .idx --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from .config import HybridConfig
from .engine import HybridRAG
from .types import Modality


def _load_or_new(storage: str) -> HybridRAG:
    if os.path.exists(os.path.join(storage, "config.json")):
        return HybridRAG.load(storage)
    return HybridRAG(HybridConfig(storage_dir=storage))


def _cmd_add_text(args: argparse.Namespace) -> int:
    engine = _load_or_new(args.storage)
    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = args.text or sys.stdin.read()
    if args.replace:
        result = engine.upsert_text(args.id, text, title=args.title or "")
        n = result["chunks_added"]
        engine.save(args.storage)
        print(
            f"Replaced doc {args.id!r}: removed {result['text_removed']} text "
            f"+ {result['vision_removed']} vision unit(s), indexed {n} chunk(s)"
        )
        return 0
    n = engine.add_text(args.id, text, title=args.title or "")
    engine.save(args.storage)
    print(f"Indexed {n} text chunk(s) into {args.storage}")
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    engine = HybridRAG.load(args.storage)
    result = engine.delete(args.id)
    engine.save(args.storage)
    total = result["text_removed"] + result["vision_removed"]
    if total == 0:
        print(f"No units found for doc {args.id!r}.")
    else:
        print(
            f"Deleted doc {args.id!r}: {result['text_removed']} text "
            f"+ {result['vision_removed']} vision unit(s)"
        )
    return 0


def _cmd_list_docs(args: argparse.Namespace) -> int:
    engine = HybridRAG.load(args.storage)
    doc_ids = sorted(engine.doc_ids())
    if args.json:
        print(json.dumps(doc_ids, indent=2))
        return 0
    if not doc_ids:
        print("No documents indexed.")
        return 0
    for d in doc_ids:
        print(d)
    return 0


def _apply_vision_selection(engine: HybridRAG, args: argparse.Namespace) -> None:
    """Let a CLI ``--vision-selection`` flag override the index policy for this run."""
    mode = getattr(args, "vision_selection", None)
    if mode:
        engine.config.pixel_selection = mode
        engine.selector = engine.selector.from_config(engine.config)


def _cmd_ingest_url(args: argparse.Namespace) -> int:
    from .pipeline.render import render_url_to_png, tile_image

    engine = _load_or_new(args.storage)
    _apply_vision_selection(engine, args)
    shots_dir = os.path.join(args.storage, "shots")
    tiles_dir = os.path.join(args.storage, "tiles")
    os.makedirs(shots_dir, exist_ok=True)
    png = os.path.join(shots_dir, f"{args.id}.png")
    render_url_to_png(args.url, png, viewport_width=engine.config.viewport_width,
                      scale=engine.config.render_scale)
    # Decide from the rendered page before paying for tiling + embedding.
    decision = engine.should_index_pixels(image=png)
    if decision.index_pixels:
        tiles = tile_image(png, args.id, tiles_dir, page=0,
                           tile_height=engine.config.tile_height,
                           overlap=engine.config.tile_overlap)
        n_tiles = engine.add_tiles(tiles)
        engine.save(args.storage)
        print(f"Indexed {n_tiles} tile(s) from {args.url} ({decision.reason})")
    else:
        engine.save(args.storage)
        print(f"Skipped vision for {args.url}: not visually rich ({decision.reason})")
    return 0


def _cmd_ingest_pdf(args: argparse.Namespace) -> int:
    from .pipeline.extract import extract_pdf_text_by_page
    from .pipeline.render import render_pdf_to_pngs, tile_image

    engine = _load_or_new(args.storage)
    _apply_vision_selection(engine, args)
    page_texts: List[str] = []
    n_text = 0
    if not args.no_text:
        for page_no, page_text in enumerate(extract_pdf_text_by_page(args.pdf)):
            page_texts.append(page_text)
            n_text += engine.add_text(args.id, page_text, page=page_no)
    n_tiles = 0
    n_skipped = 0
    if not args.no_vision:
        shots_dir = os.path.join(args.storage, "shots", args.id)
        tiles_dir = os.path.join(args.storage, "tiles")
        for page_no, png in enumerate(render_pdf_to_pngs(args.pdf, shots_dir,
                                                         scale=engine.config.render_scale or 1.5)):
            page_text = page_texts[page_no] if page_no < len(page_texts) else None
            # Skip tiling + embedding for text-native pages the text index covers.
            decision = engine.should_index_pixels(text=page_text, image=png)
            if not decision.index_pixels:
                n_skipped += 1
                continue
            tiles = tile_image(png, args.id, tiles_dir, page=page_no,
                               tile_height=engine.config.tile_height,
                               overlap=engine.config.tile_overlap)
            n_tiles += engine.add_tiles(tiles)
    engine.save(args.storage)
    msg = f"Indexed {n_text} chunk(s) and {n_tiles} tile(s) from {args.pdf}"
    if n_skipped:
        msg += f" ({n_skipped} text-native page(s) skipped for vision)"
    print(msg)
    return 0


def _read_manifest(path: str) -> List[dict]:
    """Load a batch manifest: a JSON array file, or JSONL (one object per line)."""
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    stripped = content.lstrip()
    if stripped.startswith("["):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON manifest must be an array of document objects")
        return data
    docs = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            docs.append(json.loads(line))
    return docs


def _cmd_ingest_batch(args: argparse.Namespace) -> int:
    engine = _load_or_new(args.storage)
    if args.vision_selection:
        engine.config.pixel_selection = args.vision_selection
        engine.selector = engine.selector.__class__(
            mode=args.vision_selection,
            threshold=engine.config.pixel_selection_threshold,
            text_weight=engine.config.pixel_selection_text_weight,
        )
    docs = _read_manifest(args.manifest)
    total = len(docs)

    last = [0.0]

    def progress(stats):
        # Print a throughput line every ~10% so large runs show life.
        frac = stats.documents / total if total else 1.0
        if frac - last[0] >= 0.1 or stats.documents == total:
            last[0] = frac
            print(
                f"  {stats.documents}/{total} docs  "
                f"{stats.chunks_added} chunks  {stats.tiles_added} tiles  "
                f"{stats.docs_per_second:.0f} docs/s",
                file=sys.stderr,
            )

    stats = engine.add_documents(
        docs,
        batch_size=args.batch_size,
        max_workers=args.workers,
        upsert=args.replace,
        on_progress=progress if not args.json else None,
    )
    engine.save(args.storage)
    if args.json:
        print(json.dumps(stats.to_dict(), indent=2))
    else:
        d = stats.to_dict()
        print(
            f"Ingested {d['documents']} document(s): {d['chunks_added']} chunk(s), "
            f"{d['tiles_added']} tile(s)"
            + (f", {d['vision_skipped']} text-native page(s) skipped for vision"
               if d['vision_skipped'] else "")
            + f" in {d['seconds']}s ({d['docs_per_second']:.0f} docs/s, "
            f"{d['text_batches']} text + {d['vision_batches']} vision batch(es))"
        )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    engine = HybridRAG.load(args.storage)
    modality = Modality(args.modality) if args.modality else None
    results = engine.search(
        args.query, top_k=args.k, modality=modality, rerank=args.rerank,
        fusion=args.fusion,
    )
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0
    if not results:
        print("No results.")
        return 0
    for i, r in enumerate(results, 1):
        comps = " ".join(f"{k}={v:.4f}" for k, v in r.components.items())
        print(f"{i:>2}. [{r.modality.value}] doc={r.doc_id} score={r.score:.4f} ({comps})")
        if r.text:
            snippet = r.text[:160].replace("\n", " ")
            print(f"    {snippet}")
        if r.image_path:
            print(f"    image: {r.image_path} (page {r.page})")
    return 0


def _cmd_answer(args: argparse.Namespace) -> int:
    engine = HybridRAG.load(args.storage)
    modality = Modality(args.modality) if args.modality else None
    ans = engine.answer(
        args.query, top_k=args.k, modality=modality, rerank=args.rerank,
        fusion=args.fusion,
    )
    if args.json:
        print(json.dumps(ans.to_dict(), indent=2))
        return 0
    print(ans.formatted())
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    engine = HybridRAG.load(args.storage)
    print(json.dumps(engine.stats(), indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .eval import build_engine, evaluate, sample_dataset
    from .eval.dataset import EvalDataset

    if args.dataset:
        dataset = EvalDataset.from_file(args.dataset)
    else:
        dataset = sample_dataset()

    config = HybridConfig(
        text_model=args.text_model,
        vision_model=args.vision_model,
    )
    engine = build_engine(dataset, config)
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    report = evaluate(engine, dataset.queries, ks=ks, modes=modes, dataset_name=dataset.name)

    cost_report = None
    if getattr(args, "cost", False):
        from .eval import estimate_cost
        cost_report = estimate_cost(engine, dataset_name=dataset.name)

    if args.json:
        out = report.to_dict()
        if cost_report is not None:
            out["cost"] = cost_report.to_dict()
            out["cost"]["projection"] = {
                "pages": args.project_pages,
                **cost_report.project(args.project_pages),
            }
        print(json.dumps(out, indent=2))
    else:
        print(report.table(primary_k=args.primary_k))
        if cost_report is not None:
            print()
            print(cost_report.table())
            print()
            print(cost_report.projection_table(args.project_pages))
    return 0


def _cmd_cost(args: argparse.Namespace) -> int:
    from .eval import CostModel, build_engine, estimate_cost, sample_dataset
    from .eval.dataset import EvalDataset

    cm = CostModel()
    if args.cost_model:
        with open(args.cost_model, "r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        known = set(CostModel().to_dict())
        cm = CostModel(**{k: v for k, v in overrides.items() if k in known})

    # Prefer a real, persisted index; otherwise build one from a dataset/sample.
    if args.storage and os.path.exists(os.path.join(args.storage, "config.json")):
        engine = HybridRAG.load(args.storage)
        name = args.storage
    else:
        if args.dataset:
            dataset = EvalDataset.from_file(args.dataset)
        else:
            dataset = sample_dataset()
        engine = build_engine(dataset, HybridConfig(
            text_model=args.text_model, vision_model=args.vision_model))
        name = dataset.name

    report = estimate_cost(engine, cost_model=cm, dataset_name=name)

    if args.json:
        out = report.to_dict()
        out["projection"] = {"pages": args.project_pages, **report.project(args.project_pages)}
        print(json.dumps(out, indent=2))
        return 0
    print(report.table())
    print()
    print(report.projection_table(args.project_pages))
    return 0


def _cmd_richness(args: argparse.Namespace) -> int:
    from .pipeline.select import PixelSelector

    text = None
    html = None
    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        if args.file.lower().endswith((".html", ".htm")):
            html = content
        else:
            text = content
    elif args.text:
        text = args.text

    selector = PixelSelector(mode=args.mode, threshold=args.threshold)
    decision = selector.decide(text=text, html=html, image=args.image)
    if args.json:
        print(json.dumps(decision.to_dict(), indent=2))
        return 0
    verdict = "INDEX pixels" if decision.index_pixels else "SKIP pixels"
    print(f"{verdict}  (score={decision.score:.3f}, {decision.reason})")
    if decision.signals:
        sig = "  ".join(f"{k}={v:.3f}" for k, v in decision.signals.items())
        print(f"  signals: {sig}")
    return 0


def _cmd_dedup(args: argparse.Namespace) -> int:
    from .pipeline.dedup import TileDeduplicator, scan_tiles, tiles_from_paths

    if args.dir:
        tiles = scan_tiles(args.dir)
    elif args.images:
        tiles = tiles_from_paths(args.images)
    else:
        print("Provide --dir or --images.", file=sys.stderr)
        return 2
    dedup = TileDeduplicator(method=args.method, scope=args.scope, tile_bytes=args.tile_bytes)
    _unique, stats = dedup.deduplicate(tiles)
    d = stats.to_dict()
    if args.json:
        print(json.dumps(d, indent=2))
        return 0
    if d["input_tiles"] == 0:
        print("No tiles found.")
        return 0
    saved_mb = d["bytes_saved"] / (1024 * 1024)
    print(
        f"{d['input_tiles']} tile(s) -> {d['unique_tiles']} unique "
        f"({d['duplicate_tiles']} duplicate, {d['dedup_ratio'] * 100:.1f}% saved) "
        f"across {d['groups_with_duplicates']} repeated group(s)"
    )
    print(
        f"  method={d['method']} scope={d['scope']}  "
        f"est. artifact storage saved ~{saved_mb:.1f} MB"
    )
    return 0


def _cmd_route(args: argparse.Namespace) -> int:
    from .retrieve.router import HeuristicRouter, LearnedRouter

    if args.model == "learned":
        router = (
            LearnedRouter.load(args.weights) if args.weights else LearnedRouter.default()
        )
    else:
        router = HeuristicRouter()
    decision = router.route(args.query)
    if args.json:
        print(json.dumps(decision.to_dict(), indent=2))
        return 0
    print(
        f"[{router.model}] text_weight={decision.text_weight:.3f}  "
        f"vision_weight={decision.vision_weight:.3f}"
    )
    print(f"  {decision.reason}")
    return 0


def _cmd_train_router(args: argparse.Namespace) -> int:
    from .retrieve.router import DEFAULT_TRAINING_EXAMPLES, LearnedRouter

    if args.examples:
        examples = []
        for row in _read_manifest(args.examples):
            q = row.get("query")
            label = row.get("label", row.get("vision_label"))
            if q is None or label is None:
                raise ValueError("each example needs `query` and `label` (0..1)")
            examples.append((q, float(label)))
    else:
        examples = list(DEFAULT_TRAINING_EXAMPLES)

    router = LearnedRouter().fit(examples, iterations=args.iterations, lr=args.lr)
    router.save(args.out)
    print(
        f"Trained learned router on {len(examples)} example(s) -> {args.out}"
    )
    weights = {n: round(float(w), 4) for n, w in zip(
        ["bias", "text_cue", "vision_cue", "codeish", "numeric", "length"],
        router.weights,
    )}
    print(f"  weights: {weights}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .serve.api import create_app

    app = create_app(args.storage)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hybridrag", description="Hybrid text + pixel RAG")
    sub = p.add_subparsers(dest="command", required=True)

    def add_storage(sp):
        sp.add_argument("--storage", default=".hybridrag", help="index storage directory")

    sp = sub.add_parser("add-text", help="chunk and index raw text")
    add_storage(sp)
    sp.add_argument("--id", required=True, help="document id")
    sp.add_argument("--file", help="read text from a file")
    sp.add_argument("--text", help="inline text (else stdin)")
    sp.add_argument("--title", help="document title")
    sp.add_argument("--replace", action="store_true",
                    help="upsert: delete any existing units for --id first")
    sp.set_defaults(func=_cmd_add_text)

    sp = sub.add_parser("delete", help="remove all units for a document id")
    add_storage(sp)
    sp.add_argument("--id", required=True, help="document id to delete")
    sp.set_defaults(func=_cmd_delete)

    sp = sub.add_parser("list-docs", help="list indexed document ids")
    add_storage(sp)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_list_docs)

    def add_vision_selection(sp):
        sp.add_argument("--vision-selection", choices=["auto", "always", "never"],
                        help="override pixel-selection policy for this ingest "
                             "(auto=index pixels only for visually rich pages)")

    sp = sub.add_parser("ingest-url", help="screenshot + tile a web page (needs [render])")
    add_storage(sp)
    sp.add_argument("--id", required=True)
    sp.add_argument("--url", required=True)
    add_vision_selection(sp)
    sp.set_defaults(func=_cmd_ingest_url)

    sp = sub.add_parser("ingest-pdf", help="extract text and render+tile a PDF (needs [render])")
    add_storage(sp)
    sp.add_argument("--id", required=True)
    sp.add_argument("--pdf", required=True)
    sp.add_argument("--no-text", action="store_true", help="skip text extraction")
    sp.add_argument("--no-vision", action="store_true", help="skip rendering/tiling")
    add_vision_selection(sp)
    sp.set_defaults(func=_cmd_ingest_pdf)

    sp = sub.add_parser(
        "ingest-batch",
        help="batch-ingest many documents from a manifest (fast path for large corpora)",
    )
    add_storage(sp)
    sp.add_argument("--manifest", required=True,
                    help="JSON array or JSONL file; each object has doc_id + "
                         "text/html and optional image_paths")
    sp.add_argument("--batch-size", type=int, default=128,
                    help="units per embedding batch (default 128)")
    sp.add_argument("--workers", type=int, default=1,
                    help="threads preparing documents while batches embed (default 1)")
    sp.add_argument("--replace", action="store_true",
                    help="upsert: delete existing units for each doc_id before adding")
    add_vision_selection(sp)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_ingest_batch)

    sp = sub.add_parser("richness", help="score a page's visual richness and the index decision")
    sp.add_argument("--file", help="text or .html file to score (pre-render signal)")
    sp.add_argument("--text", help="inline text to score")
    sp.add_argument("--image", help="rendered page image (PNG/JPG) to score")
    sp.add_argument("--mode", default="auto", choices=["auto", "always", "never"])
    sp.add_argument("--threshold", type=float, default=0.35)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_richness)

    sp = sub.add_parser("search", help="search the index")
    add_storage(sp)
    sp.add_argument("--query", required=True)
    sp.add_argument("-k", type=int, default=10)
    sp.add_argument("--modality", choices=["text", "vision"], help="force one modality")
    sp.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                    help="rerank the fused candidates (BM25/cross-encoder) before returning")
    sp.add_argument("--no-rerank", dest="rerank", action="store_false",
                    help="disable reranking even if the index config enables it")
    sp.add_argument("--fusion", choices=["rrf", "calibrated"], default=None,
                    help="fusion strategy (default: the index config's fusion_method)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_search)

    sp = sub.add_parser("answer", help="retrieve and synthesize a grounded, cited answer")
    add_storage(sp)
    sp.add_argument("--query", required=True)
    sp.add_argument("-k", type=int, default=5, help="hits to feed the reader")
    sp.add_argument("--modality", choices=["text", "vision"], help="force one modality")
    sp.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                    help="rerank the fused candidates before synthesis")
    sp.add_argument("--no-rerank", dest="rerank", action="store_false",
                    help="disable reranking even if the index config enables it")
    sp.add_argument("--fusion", choices=["rrf", "calibrated"], default=None,
                    help="fusion strategy (default: the index config's fusion_method)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_answer)

    sp = sub.add_parser("stats", help="print index stats")
    add_storage(sp)
    sp.set_defaults(func=_cmd_stats)

    sp = sub.add_parser("eval", help="compare text/vision/hybrid retrieval on a dataset")
    sp.add_argument("--dataset", help="dataset JSON file (default: built-in sample)")
    sp.add_argument("--text-model", default="hash", help='text encoder id (default "hash")')
    sp.add_argument("--vision-model", default="hash", help='vision encoder id (default "hash")')
    sp.add_argument("-k", default="1,3,5,10", help="comma-separated cutoffs")
    sp.add_argument("--primary-k", type=int, help="k used for the printed table (default: max)")
    sp.add_argument("--modes", default="text,vision,hybrid", help="comma-separated modes")
    sp.add_argument("--cost", action="store_true",
                    help="also print the storage/$ cost model and a projection")
    sp.add_argument("--project-pages", type=int, default=1_000_000,
                    help="corpus size to project storage cost to (default 1,000,000)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_eval)

    sp = sub.add_parser("cost", help="model storage + $/query cost per modality and project it")
    add_storage(sp)
    sp.add_argument("--dataset", help="dataset JSON to build an index from if --storage is empty")
    sp.add_argument("--text-model", default="hash", help='text encoder id (default "hash")')
    sp.add_argument("--vision-model", default="hash", help='vision encoder id (default "hash")')
    sp.add_argument("--cost-model", help="JSON file of CostModel field overrides (real prices)")
    sp.add_argument("--project-pages", type=int, default=10_000_000,
                    help="corpus size to project storage to (default 10,000,000)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_cost)

    sp = sub.add_parser(
        "dedup",
        help="analyse rendered tiles for content duplicates (storage/GPU a dedup would save)",
    )
    sp.add_argument("--dir", help="directory of rendered tile images to scan (recursive)")
    sp.add_argument("--images", nargs="+", help="explicit tile image paths instead of --dir")
    sp.add_argument("--method", default="exact", choices=["exact", "ahash"],
                    help="exact byte hash (default) or perceptual average-hash (needs Pillow)")
    sp.add_argument("--scope", default="doc", choices=["doc", "global"],
                    help="collapse within each doc_id (default) or across all documents")
    sp.add_argument("--tile-bytes", type=int, default=200_000,
                    help="assumed bytes per tile file for the storage-saved estimate")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_dedup)

    sp = sub.add_parser("route", help="show the per-query modality weights the router picks")
    sp.add_argument("--query", required=True)
    sp.add_argument("--model", default="heuristic", choices=["heuristic", "learned"],
                    help="which router to inspect (default heuristic)")
    sp.add_argument("--weights", help="trained learned-router weights JSON (for --model learned)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_route)

    sp = sub.add_parser("train-router",
                        help="train the learned router on labelled queries and save its weights")
    sp.add_argument("--examples",
                    help="JSON array or JSONL of {query, label} (label=P(vision) in 0..1); "
                         "omit to train on the built-in seed set")
    sp.add_argument("--out", default="router.json", help="where to write the trained weights")
    sp.add_argument("--iterations", type=int, default=4000)
    sp.add_argument("--lr", type=float, default=0.5)
    sp.set_defaults(func=_cmd_train_router)

    sp = sub.add_parser("serve", help="serve a search API (needs [serve])")
    add_storage(sp)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=_cmd_serve)

    return p


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
