"""Command line interface for HybridRAG.

Examples::

    hybridrag add-text --storage .idx --id doc1 --file README.md --title Readme
    hybridrag ingest-url --storage .idx --id wiki --url https://example.com
    hybridrag ingest-pdf --storage .idx --id paper --pdf paper.pdf
    hybridrag search --storage .idx --query "revenue table" -k 5
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
    n = engine.add_text(args.id, text, title=args.title or "")
    engine.save(args.storage)
    print(f"Indexed {n} text chunk(s) into {args.storage}")
    return 0


def _cmd_ingest_url(args: argparse.Namespace) -> int:
    from .pipeline.render import render_url_to_png, tile_image

    engine = _load_or_new(args.storage)
    shots_dir = os.path.join(args.storage, "shots")
    tiles_dir = os.path.join(args.storage, "tiles")
    os.makedirs(shots_dir, exist_ok=True)
    png = os.path.join(shots_dir, f"{args.id}.png")
    render_url_to_png(args.url, png, viewport_width=engine.config.viewport_width,
                      scale=engine.config.render_scale)
    tiles = tile_image(png, args.id, tiles_dir, page=0,
                       tile_height=engine.config.tile_height,
                       overlap=engine.config.tile_overlap)
    n_tiles = engine.add_tiles(tiles)
    engine.save(args.storage)
    print(f"Indexed {n_tiles} tile(s) from {args.url}")
    return 0


def _cmd_ingest_pdf(args: argparse.Namespace) -> int:
    from .pipeline.extract import extract_pdf_text_by_page
    from .pipeline.render import render_pdf_to_pngs, tile_image

    engine = _load_or_new(args.storage)
    n_text = 0
    if not args.no_text:
        for page_no, page_text in enumerate(extract_pdf_text_by_page(args.pdf)):
            n_text += engine.add_text(args.id, page_text, page=page_no)
    n_tiles = 0
    if not args.no_vision:
        shots_dir = os.path.join(args.storage, "shots", args.id)
        tiles_dir = os.path.join(args.storage, "tiles")
        for page_no, png in enumerate(render_pdf_to_pngs(args.pdf, shots_dir,
                                                         scale=engine.config.render_scale or 1.5)):
            tiles = tile_image(png, args.id, tiles_dir, page=page_no,
                               tile_height=engine.config.tile_height,
                               overlap=engine.config.tile_overlap)
            n_tiles += engine.add_tiles(tiles)
    engine.save(args.storage)
    print(f"Indexed {n_text} chunk(s) and {n_tiles} tile(s) from {args.pdf}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    engine = HybridRAG.load(args.storage)
    modality = Modality(args.modality) if args.modality else None
    results = engine.search(args.query, top_k=args.k, modality=modality)
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


def _cmd_stats(args: argparse.Namespace) -> int:
    engine = HybridRAG.load(args.storage)
    print(json.dumps(engine.stats(), indent=2))
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
    sp.set_defaults(func=_cmd_add_text)

    sp = sub.add_parser("ingest-url", help="screenshot + tile a web page (needs [render])")
    add_storage(sp)
    sp.add_argument("--id", required=True)
    sp.add_argument("--url", required=True)
    sp.set_defaults(func=_cmd_ingest_url)

    sp = sub.add_parser("ingest-pdf", help="extract text and render+tile a PDF (needs [render])")
    add_storage(sp)
    sp.add_argument("--id", required=True)
    sp.add_argument("--pdf", required=True)
    sp.add_argument("--no-text", action="store_true", help="skip text extraction")
    sp.add_argument("--no-vision", action="store_true", help="skip rendering/tiling")
    sp.set_defaults(func=_cmd_ingest_pdf)

    sp = sub.add_parser("search", help="search the index")
    add_storage(sp)
    sp.add_argument("--query", required=True)
    sp.add_argument("-k", type=int, default=10)
    sp.add_argument("--modality", choices=["text", "vision"], help="force one modality")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_search)

    sp = sub.add_parser("stats", help="print index stats")
    add_storage(sp)
    sp.set_defaults(func=_cmd_stats)

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
