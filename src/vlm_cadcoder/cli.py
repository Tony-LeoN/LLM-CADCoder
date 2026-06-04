from __future__ import annotations

import argparse
from pathlib import Path

from vlm_cadcoder.dataflow.artifact_store import ArtifactStore
from vlm_cadcoder.dataflow.paths import DataFlowPaths
from vlm_cadcoder.dataflow.pdf_render import render_pdf_pages
from vlm_cadcoder.dataflow.sample_index import find_pdf_step_pairs, write_samples_csv


def build_sample_index(args: argparse.Namespace) -> None:
    records = find_pdf_step_pairs(args.raw_dir)
    write_samples_csv(records, args.output)
    print(f"Wrote {len(records)} samples to {args.output}")


def render_pdf(args: argparse.Namespace) -> None:
    store = ArtifactStore(DataFlowPaths(root=Path(args.dataflow_root)))
    rendered = render_pdf_pages(
        pdf_path=args.pdf,
        sample_id=args.sample_id,
        store=store,
        dpi=args.dpi,
        skip_multipage=args.skip_multipage,
    )
    print(f"Rendered {len(rendered)} pages")


def clean_layout(args: argparse.Namespace) -> None:
    from vlm_cadcoder.dataflow.layout_clean import clean_layout_page

    result = clean_layout_page(
        image_path=args.image,
        dataflow_root=args.dataflow_root,
        sample_id=args.sample_id,
        page=args.page,
        output_stem=args.output_stem,
        save_crops=args.save_crops,
        save_overlay=args.save_overlay,
    )
    print(f"Wrote clean page to {result.clean_image_path}")
    print(f"Wrote layout metadata to {result.layout_path}")
    print(f"Detected {len(result.regions)} removable layout regions")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vlm-cadcoder")
    subparsers = parser.add_subparsers(required=True)

    index_parser = subparsers.add_parser("build-sample-index")
    index_parser.add_argument("--raw-dir", default="DataFlow/01.RawPDFWithSTEP")
    index_parser.add_argument("--output", default="data/samples.csv")
    index_parser.set_defaults(func=build_sample_index)

    render_parser = subparsers.add_parser("render-pdf")
    render_parser.add_argument("--pdf", required=True)
    render_parser.add_argument("--sample-id", required=True)
    render_parser.add_argument("--dataflow-root", default="DataFlow")
    render_parser.add_argument("--dpi", type=int, default=600)
    render_parser.add_argument("--skip-multipage", action="store_true")
    render_parser.set_defaults(func=render_pdf)

    clean_parser = subparsers.add_parser("clean-layout")
    clean_parser.add_argument("--image", required=True)
    clean_parser.add_argument("--sample-id")
    clean_parser.add_argument("--page", type=int, default=1)
    clean_parser.add_argument("--dataflow-root", default="DataFlow")
    clean_parser.add_argument("--output-stem")
    clean_parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    clean_parser.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=True)
    clean_parser.set_defaults(func=clean_layout)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
