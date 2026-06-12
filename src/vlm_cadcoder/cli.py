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


def render_pdf_batch(args: argparse.Namespace) -> None:
    from vlm_cadcoder.dataflow.pdf_batch import render_pdf_directory

    store = ArtifactStore(DataFlowPaths(root=Path(args.dataflow_root)))
    summary = render_pdf_directory(
        raw_dir=args.raw_dir,
        store=store,
        dpi=args.dpi,
        skip_multipage=args.skip_multipage,
        skip_existing=args.skip_existing,
        recursive=args.recursive,
        fail_fast=args.fail_fast,
    )
    print(
        "Rendered "
        f"{summary.rendered_pdf_count} PDFs / {summary.rendered_page_count} pages; "
        f"skipped {summary.skipped_count}; failed {summary.failed_count}"
    )
    for record in summary.records:
        if record.error:
            print(f"[failed] {record.sample_id}: {record.error}")
        elif record.skipped:
            print(f"[skipped] {record.sample_id}")
        else:
            print(f"[rendered] {record.sample_id}: {record.rendered_pages} page(s)")

    if summary.failed_count:
        raise SystemExit(1)


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


def clean_layout_batch(args: argparse.Namespace) -> None:
    from vlm_cadcoder.dataflow.layout_batch import clean_layout_directory

    summary = clean_layout_directory(
        raw_png_dir=args.raw_png_dir,
        dataflow_root=args.dataflow_root,
        dpi=args.dpi,
        skip_existing=args.skip_existing,
        save_crops=args.save_crops,
        save_overlay=args.save_overlay,
        fail_fast=args.fail_fast,
    )
    print(
        f"Cleaned {summary.cleaned_count} page(s); "
        f"skipped {summary.skipped_count}; failed {summary.failed_count}"
    )
    for record in summary.records:
        prefix = "[cleaned]"
        detail = f"{record.regions} removable region(s)"
        if record.error:
            prefix = "[failed]"
            detail = record.error
        elif record.skipped:
            prefix = "[skipped]"
            detail = "existing clean page"
        print(f"{prefix} {record.sample_id} page {record.page:03d}: {detail}")

    if summary.failed_count:
        raise SystemExit(1)


def filter_view_detections_cli(args: argparse.Namespace) -> None:
    from vlm_cadcoder.dataflow.view_filter import ViewFilterConfig, filter_view_detections_file

    dataflow_root = Path(args.dataflow_root)
    if args.input_json:
        detection_path = Path(args.input_json)
        output_path = args.output_json
    else:
        if not args.sample_id:
            raise SystemExit("--sample-id is required when --input-json is not provided")
        detection_dir = dataflow_root / "05.ViewDetection" / args.sample_id
        raw_path = detection_dir / f"page_{args.page:03d}_views_raw.json"
        detection_path = raw_path if raw_path.exists() else detection_dir / f"page_{args.page:03d}_views.json"
        output_path = args.output_json or detection_dir / f"page_{args.page:03d}_views.json"

    clean_image_path = Path(args.clean_image) if args.clean_image else None
    layout_path = Path(args.layout_json) if args.layout_json else None
    result = filter_view_detections_file(
        detection_path=detection_path,
        clean_image_path=clean_image_path,
        layout_path=layout_path,
        dataflow_root=dataflow_root,
        output_path=output_path,
        config=ViewFilterConfig(
            min_score=args.min_score,
            top_strip_score=args.top_strip_score,
            dense_ink_ratio=args.dense_ink_ratio,
            dense_thick_ink_ratio=args.dense_thick_ink_ratio,
        ),
        save_overlay=args.save_overlay,
    )
    print(f"Wrote filtered view detections to {result.filtered_path}")
    print(f"Wrote rejected view report to {result.rejected_path}")
    if result.raw_path:
        print(f"Preserved raw detections at {result.raw_path}")
    if result.overlay_path:
        print(f"Wrote filter overlay to {result.overlay_path}")
    print(f"Accepted {len(result.accepted_views)} view(s); rejected {len(result.rejected_views)} candidate(s)")


def filter_view_detections_batch_cli(args: argparse.Namespace) -> None:
    from vlm_cadcoder.dataflow.view_filter import ViewFilterConfig
    from vlm_cadcoder.dataflow.view_filter_batch import filter_view_detection_directory

    summary = filter_view_detection_directory(
        dataflow_root=args.dataflow_root,
        view_detection_dir=args.view_detection_dir,
        config=ViewFilterConfig(
            min_score=args.min_score,
            top_strip_score=args.top_strip_score,
            dense_ink_ratio=args.dense_ink_ratio,
            dense_thick_ink_ratio=args.dense_thick_ink_ratio,
        ),
        save_overlay=args.save_overlay,
        fail_fast=args.fail_fast,
    )
    print(f"Filtered {summary.filtered_count} page(s); failed {summary.failed_count}")
    for record in summary.records:
        if record.error:
            print(f"[failed] {record.sample_id} page {record.page:03d}: {record.error}")
        else:
            print(
                f"[filtered] {record.sample_id} page {record.page:03d}: "
                f"accepted {record.accepted_views}, rejected {record.rejected_candidates}"
            )

    if summary.failed_count:
        raise SystemExit(1)


def audit_single_views_cli(args: argparse.Namespace) -> None:
    from vlm_cadcoder.dataflow.single_view_audit import audit_single_views

    summary = audit_single_views(
        dataflow_root=args.dataflow_root,
        output_csv=args.output_csv,
        output_json=args.output_json,
        page=args.page,
    )
    print(
        f"Audited {summary.total_count} sample(s); "
        f"consistent {summary.consistent_count}; needs review {summary.review_count}"
    )
    if summary.csv_path:
        print(f"Wrote CSV audit report to {summary.csv_path}")
    if summary.json_path:
        print(f"Wrote JSON audit report to {summary.json_path}")
    for record in summary.records:
        if record.needs_manual_review:
            reasons = ", ".join(record.review_reasons)
            print(
                f"[review] {record.sample_id}: "
                f"05={record.detected_view_count}, 06={record.exported_view_count}, reasons={reasons}"
            )


def classify_views_cli(args: argparse.Namespace) -> None:
    from vlm_cadcoder.dataflow.view_classification import classify_view_samples

    summary = classify_view_samples(
        dataflow_root=args.dataflow_root,
        sample_id=args.sample_id,
        page=args.page,
        include_copy=args.include_copy,
        fail_fast=args.fail_fast,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(
        f"Classified {summary.classified_count} sample(s); "
        f"skipped {summary.skipped_count}; failed {summary.failed_count}"
    )
    if summary.csv_path:
        print(f"Wrote classification summary CSV to {summary.csv_path}")
    if summary.json_path:
        print(f"Wrote classification summary JSON to {summary.json_path}")
    for record in summary.records:
        if record.skipped:
            print(f"[skipped] {record.sample_id}: copy sample")
        elif record.error:
            print(f"[failed] {record.sample_id}: {record.error}")
        else:
            print(f"[classified] {record.sample_id}: {record.classified_views} view(s) -> {record.output_path}")

    if summary.failed_count:
        raise SystemExit(1)


def build_drawing_ir_cli(args: argparse.Namespace) -> None:
    from vlm_cadcoder.dataflow.drawing_ir_builder import build_drawing_ir_samples

    summary = build_drawing_ir_samples(
        dataflow_root=args.dataflow_root,
        sample_id=args.sample_id,
        page=args.page,
        include_copy=args.include_copy,
        fail_fast=args.fail_fast,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(
        f"Built DrawingIR for {summary.built_count} sample(s); "
        f"skipped {summary.skipped_count}; failed {summary.failed_count}"
    )
    if summary.csv_path:
        print(f"Wrote DrawingIR summary CSV to {summary.csv_path}")
    if summary.json_path:
        print(f"Wrote DrawingIR summary JSON to {summary.json_path}")
    for record in summary.records:
        if record.skipped:
            print(f"[skipped] {record.sample_id}: copy sample")
        elif record.error:
            print(f"[failed] {record.sample_id}: {record.error}")
        else:
            print(f"[built] {record.sample_id}: {record.view_count} view(s) -> {record.output_path}")

    if summary.failed_count:
        raise SystemExit(1)


def build_view2cad_prototype_cli(args: argparse.Namespace) -> None:
    from vlm_cadcoder.cad.view2cad_prototype import View2CADPrototypeConfig, build_view2cad_prototype

    result = build_view2cad_prototype(
        sample_id=args.sample_id,
        config=View2CADPrototypeConfig(
            dataflow_root=Path(args.dataflow_root),
            external_crop_set=args.external_crop_set,
            experiments_root=Path(args.experiments_root),
            output_set=args.output_set,
        ),
    )
    print(f"Wrote external crop manifest to {result.manifest_path}")
    print(f"Wrote minimal DrawingIR to {result.drawing_ir_path}")
    print(f"Wrote modeling plan to {result.modeling_plan_path}")
    print(f"Wrote CadQuery prompt to {result.cadquery_prompt_path}")


def build_cadquery_draft_cli(args: argparse.Namespace) -> None:
    from vlm_cadcoder.cad.cadquery_draft import CadQueryDraftConfig, build_cadquery_draft

    result = build_cadquery_draft(
        sample_id=args.sample_id,
        config=CadQueryDraftConfig(
            dataflow_root=Path(args.dataflow_root),
            input_set=args.input_set,
            output_set=args.output_set,
            part_family=args.part_family,
        ),
    )
    print(f"Wrote CadQuery parameter review to {result.parameter_path}")
    print(f"Wrote CadQuery draft script to {result.script_path}")


def generate_cadquery_llm_cli(args: argparse.Namespace) -> None:
    from vlm_cadcoder.cad.llm_cadquery import generate_cadquery_with_llm

    result = generate_cadquery_with_llm(
        sample_id=args.sample_id,
        model_name=args.model,
        dataflow_root=args.dataflow_root,
        model_config_path=args.model_config,
        input_set=args.input_set,
        output_set=args.output_set,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Wrote raw LLM response to {result.raw_response_path}")
    print(f"Wrote sanitized CadQuery script to {result.script_path}")
    if result.error:
        print(f"Model error: {result.error}")


def sanitize_cadquery_llm_cli(args: argparse.Namespace) -> None:
    from vlm_cadcoder.cad.llm_cadquery import sanitize_cadquery_file

    path = sanitize_cadquery_file(args.input, args.output)
    print(f"Wrote sanitized CadQuery script to {path}")


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

    render_batch_parser = subparsers.add_parser("render-pdf-batch")
    render_batch_parser.add_argument("--raw-dir", default="DataFlow/01.RawPDFWithSTEP")
    render_batch_parser.add_argument("--dataflow-root", default="DataFlow")
    render_batch_parser.add_argument("--dpi", type=int, default=600)
    render_batch_parser.add_argument("--skip-multipage", action="store_true")
    render_batch_parser.add_argument("--skip-existing", action="store_true")
    render_batch_parser.add_argument("--recursive", action="store_true")
    render_batch_parser.add_argument("--fail-fast", action="store_true")
    render_batch_parser.set_defaults(func=render_pdf_batch)

    clean_parser = subparsers.add_parser("clean-layout")
    clean_parser.add_argument("--image", required=True)
    clean_parser.add_argument("--sample-id")
    clean_parser.add_argument("--page", type=int, default=1)
    clean_parser.add_argument("--dataflow-root", default="DataFlow")
    clean_parser.add_argument("--output-stem")
    clean_parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    clean_parser.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=True)
    clean_parser.set_defaults(func=clean_layout)

    clean_batch_parser = subparsers.add_parser("clean-layout-batch")
    clean_batch_parser.add_argument("--raw-png-dir", default="DataFlow/02.RawPNG")
    clean_batch_parser.add_argument("--dataflow-root", default="DataFlow")
    clean_batch_parser.add_argument("--dpi", type=int, default=600)
    clean_batch_parser.add_argument("--skip-existing", action="store_true")
    clean_batch_parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    clean_batch_parser.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=True)
    clean_batch_parser.add_argument("--fail-fast", action="store_true")
    clean_batch_parser.set_defaults(func=clean_layout_batch)

    filter_views_parser = subparsers.add_parser("filter-view-detections")
    filter_views_parser.add_argument("--sample-id")
    filter_views_parser.add_argument("--page", type=int, default=1)
    filter_views_parser.add_argument("--dataflow-root", default="DataFlow")
    filter_views_parser.add_argument("--input-json")
    filter_views_parser.add_argument("--output-json")
    filter_views_parser.add_argument("--clean-image")
    filter_views_parser.add_argument("--layout-json")
    filter_views_parser.add_argument("--min-score", type=float, default=0.5)
    filter_views_parser.add_argument("--top-strip-score", type=float, default=0.6)
    filter_views_parser.add_argument("--dense-ink-ratio", type=float, default=0.16)
    filter_views_parser.add_argument("--dense-thick-ink-ratio", type=float, default=0.14)
    filter_views_parser.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=True)
    filter_views_parser.set_defaults(func=filter_view_detections_cli)

    filter_views_batch_parser = subparsers.add_parser("filter-view-detections-batch")
    filter_views_batch_parser.add_argument("--dataflow-root", default="DataFlow")
    filter_views_batch_parser.add_argument("--view-detection-dir")
    filter_views_batch_parser.add_argument("--min-score", type=float, default=0.5)
    filter_views_batch_parser.add_argument("--top-strip-score", type=float, default=0.6)
    filter_views_batch_parser.add_argument("--dense-ink-ratio", type=float, default=0.16)
    filter_views_batch_parser.add_argument("--dense-thick-ink-ratio", type=float, default=0.14)
    filter_views_batch_parser.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=True)
    filter_views_batch_parser.add_argument("--fail-fast", action="store_true")
    filter_views_batch_parser.set_defaults(func=filter_view_detections_batch_cli)

    audit_single_views_parser = subparsers.add_parser("audit-single-views")
    audit_single_views_parser.add_argument("--dataflow-root", default="DataFlow")
    audit_single_views_parser.add_argument("--page", type=int, default=1)
    audit_single_views_parser.add_argument("--output-csv")
    audit_single_views_parser.add_argument("--output-json")
    audit_single_views_parser.set_defaults(func=audit_single_views_cli)

    classify_views_parser = subparsers.add_parser("classify-views")
    classify_views_parser.add_argument("--sample-id")
    classify_views_parser.add_argument("--dataflow-root", default="DataFlow")
    classify_views_parser.add_argument("--page", type=int, default=1)
    classify_views_parser.add_argument("--include-copy", action="store_true")
    classify_views_parser.add_argument("--fail-fast", action="store_true")
    classify_views_parser.add_argument("--output-csv")
    classify_views_parser.add_argument("--output-json")
    classify_views_parser.set_defaults(func=classify_views_cli)

    drawing_ir_parser = subparsers.add_parser("build-drawing-ir")
    drawing_ir_parser.add_argument("--sample-id")
    drawing_ir_parser.add_argument("--dataflow-root", default="DataFlow")
    drawing_ir_parser.add_argument("--page", type=int, default=1)
    drawing_ir_parser.add_argument("--include-copy", action="store_true")
    drawing_ir_parser.add_argument("--fail-fast", action="store_true")
    drawing_ir_parser.add_argument("--output-csv")
    drawing_ir_parser.add_argument("--output-json")
    drawing_ir_parser.set_defaults(func=build_drawing_ir_cli)

    view2cad_parser = subparsers.add_parser("build-view2cad-prototype")
    view2cad_parser.add_argument("--sample-id", required=True)
    view2cad_parser.add_argument("--dataflow-root", default="DataFlow")
    view2cad_parser.add_argument("--external-crop-set", default="testView2CAD")
    view2cad_parser.add_argument("--experiments-root", default="experiments/external_crops")
    view2cad_parser.add_argument("--output-set", default="testView2CAD")
    view2cad_parser.set_defaults(func=build_view2cad_prototype_cli)

    cadquery_parser = subparsers.add_parser("build-cadquery-draft")
    cadquery_parser.add_argument("--sample-id", required=True)
    cadquery_parser.add_argument("--dataflow-root", default="DataFlow")
    cadquery_parser.add_argument("--input-set", default="testView2CAD")
    cadquery_parser.add_argument("--output-set", default="testView2CAD")
    cadquery_parser.add_argument("--part-family", default="rectangular_plate")
    cadquery_parser.set_defaults(func=build_cadquery_draft_cli)

    generate_parser = subparsers.add_parser("generate-cadquery-llm")
    generate_parser.add_argument("--sample-id", required=True)
    generate_parser.add_argument("--model", default="qwen2_5_vl_3b")
    generate_parser.add_argument("--dataflow-root", default="DataFlow")
    generate_parser.add_argument("--model-config", default="configs/models.json")
    generate_parser.add_argument("--input-set", default="testView2CAD")
    generate_parser.add_argument("--output-set", default="testView2CAD")
    generate_parser.add_argument("--max-new-tokens", type=int, default=4096)
    generate_parser.set_defaults(func=generate_cadquery_llm_cli)

    sanitize_parser = subparsers.add_parser("sanitize-cadquery-llm")
    sanitize_parser.add_argument("--input", required=True)
    sanitize_parser.add_argument("--output")
    sanitize_parser.set_defaults(func=sanitize_cadquery_llm_cli)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
