from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_cadcoder.utils.json_utils import read_json, write_json


@dataclass(frozen=True)
class ViewFeatureExtractionResult:
    sample_id: str
    output_path: Path
    view_count: int
    feature_count: int


@dataclass(frozen=True)
class ViewFeatureExtractionRecord:
    sample_id: str
    output_path: Path | None
    view_count: int = 0
    feature_count: int = 0
    skipped: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ViewFeatureExtractionSummary:
    records: list[ViewFeatureExtractionRecord]
    csv_path: Path | None = None
    json_path: Path | None = None

    @property
    def extracted_count(self) -> int:
        return sum(1 for record in self.records if record.output_path is not None and not record.error and not record.skipped)

    @property
    def skipped_count(self) -> int:
        return sum(1 for record in self.records if record.skipped)

    @property
    def failed_count(self) -> int:
        return sum(1 for record in self.records if record.error is not None)


def extract_view_features_sample(
    *,
    sample_id: str,
    dataflow_root: str | Path = "DataFlow",
    output_path: str | Path | None = None,
) -> ViewFeatureExtractionResult:
    root = Path(dataflow_root)
    drawing_ir_path = root / "10.StructuredCADRepresentation" / sample_id / "drawing_ir.json"
    drawing_ir = _read_drawing_ir(drawing_ir_path, sample_id)

    views = _views_by_id(drawing_ir)
    semantic_candidates: list[dict[str, Any]] = []
    grouped_candidates: dict[str, list[dict[str, Any]]] = {view_id: [] for view_id in views}
    skipped_components: list[dict[str, Any]] = []

    for component in drawing_ir.get("feature_candidates") or []:
        if not isinstance(component, dict) or component.get("type") != "geometry_component":
            continue
        view_id = str(component.get("view_id") or "")
        view = views.get(view_id)
        if view is None:
            skipped_components.append(
                {
                    "source_candidate_id": str(component.get("id") or ""),
                    "reason": "candidate_view_missing_from_drawing_ir",
                    "view_id": view_id,
                }
            )
            continue
        try:
            semantic = _classify_geometry_component(component, view)
        except (TypeError, ValueError) as exc:
            skipped_components.append(
                {
                    "source_candidate_id": str(component.get("id") or ""),
                    "reason": str(exc),
                    "view_id": view_id,
                }
            )
            continue
        semantic_candidates.append(semantic)
        grouped_candidates.setdefault(view_id, []).append(semantic)

    view_blocks = [
        {
            "view_id": view_id,
            "view_type": str(view.get("type") or "unknown"),
            "is_primary": bool(view.get("is_primary")),
            "bbox_on_page": view.get("bbox_on_page") or view.get("bbox"),
            "crop_size": _crop_size(view),
            "geometry_core": view.get("geometry_core") if isinstance(view.get("geometry_core"), dict) else {},
            "candidate_count": len(grouped_candidates.get(view_id, [])),
            "feature_candidates": grouped_candidates.get(view_id, []),
        }
        for view_id, view in views.items()
    ]

    payload = {
        "schema": "view_feature_extraction",
        "version": "0.1.0",
        "sample_id": sample_id,
        "source_drawing_ir": _stage_path(root, drawing_ir_path),
        "method": {
            "name": "rule_based_geometry_component_classifier",
            "version": "0.1.0",
            "role": "conservative_semantic_candidate_layer_for_manual_review",
        },
        "views": view_blocks,
        "feature_candidates": semantic_candidates,
        "skipped_components": skipped_components,
        "quality": _quality_block(
            drawing_ir=drawing_ir,
            views=view_blocks,
            semantic_candidates=semantic_candidates,
            skipped_components=skipped_components,
        ),
    }

    target = Path(output_path) if output_path else root / "08.Multi-viewFeatureExtraction" / sample_id / "view_features.json"
    write_json(target, payload)
    return ViewFeatureExtractionResult(
        sample_id=sample_id,
        output_path=target,
        view_count=len(view_blocks),
        feature_count=len(semantic_candidates),
    )


def extract_view_features_samples(
    *,
    dataflow_root: str | Path = "DataFlow",
    sample_id: str | None = None,
    include_copy: bool = False,
    fail_fast: bool = False,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
) -> ViewFeatureExtractionSummary:
    root = Path(dataflow_root)
    sample_ids = [sample_id] if sample_id else _iter_drawing_ir_sample_ids(root / "10.StructuredCADRepresentation")
    records: list[ViewFeatureExtractionRecord] = []

    for current_sample_id in sample_ids:
        if not current_sample_id:
            continue
        if not include_copy and _looks_like_copy_sample(current_sample_id):
            records.append(ViewFeatureExtractionRecord(sample_id=current_sample_id, output_path=None, skipped=True))
            continue
        try:
            result = extract_view_features_sample(sample_id=current_sample_id, dataflow_root=root)
            records.append(
                ViewFeatureExtractionRecord(
                    sample_id=current_sample_id,
                    output_path=result.output_path,
                    view_count=result.view_count,
                    feature_count=result.feature_count,
                )
            )
        except Exception as exc:  # pragma: no cover - sample-specific data failures are reported in summary
            if fail_fast:
                raise
            records.append(ViewFeatureExtractionRecord(sample_id=current_sample_id, output_path=None, error=str(exc)))

    out_root = root / "08.Multi-viewFeatureExtraction"
    csv_path = Path(output_csv) if output_csv else out_root / "view_feature_summary.csv"
    json_path = Path(output_json) if output_json else out_root / "view_feature_summary.json"
    _write_summary_csv(csv_path, records)
    _write_summary_json(json_path, records)
    return ViewFeatureExtractionSummary(records=records, csv_path=csv_path, json_path=json_path)


def _read_drawing_ir(path: Path, sample_id: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing DrawingIR file: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"DrawingIR JSON must be an object: {path}")
    if data.get("schema") != "drawing_ir":
        raise ValueError(f"Expected drawing_ir schema in {path}")
    if data.get("sample_id") != sample_id:
        raise ValueError(f"DrawingIR sample_id mismatch in {path}")
    if not isinstance(data.get("views"), list):
        raise ValueError(f"DrawingIR views must be a list: {path}")
    if not isinstance(data.get("feature_candidates"), list):
        raise ValueError(f"DrawingIR feature_candidates must be a list: {path}")
    return data


def _views_by_id(drawing_ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    views: dict[str, dict[str, Any]] = {}
    for view in drawing_ir.get("views") or []:
        if not isinstance(view, dict):
            continue
        view_id = str(view.get("id") or view.get("view_id") or "")
        if view_id:
            views[view_id] = view
    return views


def _classify_geometry_component(component: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    bbox = _bbox(component)
    metrics = _component_metrics(bbox, int(component.get("area_px") or 0), _crop_size(view))
    feature_type, confidence, reasons = _semantic_label(metrics)
    source_id = str(component.get("id") or "geometry_component")

    return {
        "id": f"{source_id}_{feature_type}",
        "type": feature_type,
        "semantic_status": "candidate",
        "view_id": str(component.get("view_id") or ""),
        "source_candidate_id": source_id,
        "source_type": "geometry_component",
        "bbox": bbox,
        "bbox_on_page": component.get("bbox_on_page"),
        "area_px": int(component.get("area_px") or 0),
        "confidence": confidence,
        "needs_manual_review": True,
        "review_reasons": ["rule_based_semantic_candidates_need_validation", *reasons],
        "metrics": metrics,
        "source": component.get("source"),
        "evidence": list(component.get("evidence") or []) + [f"semantic_rule={feature_type}"],
    }


def _semantic_label(metrics: dict[str, Any]) -> tuple[str, float, list[str]]:
    width = metrics["bbox_width"]
    height = metrics["bbox_height"]
    min_side = min(width, height)
    aspect = metrics["aspect_ratio"]
    bbox_area_ratio = metrics["bbox_area_ratio"]
    fill_ratio = metrics["fill_ratio"]

    if min_side <= 4 or metrics["ink_area_px"] < 12 or bbox_area_ratio < 0.00025:
        return "annotation_residue_candidate", 0.2, ["tiny_or_sparse_component"]
    if bbox_area_ratio >= 0.22 or (metrics["crop_width_ratio"] >= 0.65 and metrics["crop_height_ratio"] >= 0.45):
        return "outer_profile_candidate", 0.48, ["large_component_relative_to_view"]
    if 0.70 <= aspect <= 1.45 and 0.001 <= bbox_area_ratio <= 0.12 and fill_ratio <= 0.75:
        return "hole_candidate", 0.42, ["near_square_internal_component"]
    if (aspect >= 2.6 or aspect <= 0.38) and 0.001 <= bbox_area_ratio <= 0.16:
        return "slot_candidate", 0.4, ["elongated_internal_component"]
    return "unknown_geometry_candidate", 0.25, ["geometry_component_not_matched_by_mvp_rules"]


def _component_metrics(bbox: list[int], ink_area: int, crop_size: dict[str, int]) -> dict[str, Any]:
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    bbox_area = width * height
    crop_width = max(1, int(crop_size.get("width") or width))
    crop_height = max(1, int(crop_size.get("height") or height))
    crop_area = crop_width * crop_height
    safe_ink_area = max(0, int(ink_area or bbox_area))

    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area_px": bbox_area,
        "ink_area_px": safe_ink_area,
        "aspect_ratio": round(width / height, 4),
        "fill_ratio": round(min(safe_ink_area, bbox_area) / bbox_area, 4),
        "bbox_area_ratio": round(bbox_area / crop_area, 6),
        "crop_width_ratio": round(width / crop_width, 6),
        "crop_height_ratio": round(height / crop_height, 6),
    }


def _bbox(component: dict[str, Any]) -> list[int]:
    values = component.get("bbox")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(f"Geometry component is missing bbox: {component.get('id')}")
    x1, y1, x2, y2 = [int(value) for value in values]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Geometry component bbox must be xyxy with positive area: {component.get('id')}")
    return [x1, y1, x2, y2]


def _crop_size(view: dict[str, Any]) -> dict[str, int]:
    value = view.get("crop_size")
    if isinstance(value, dict) and value.get("width") and value.get("height"):
        return {"width": int(value["width"]), "height": int(value["height"])}
    bbox = view.get("bbox_on_page") or view.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return {"width": int(bbox[2]) - int(bbox[0]), "height": int(bbox[3]) - int(bbox[1])}
    return {"width": 1, "height": 1}


def _quality_block(
    *,
    drawing_ir: dict[str, Any],
    views: list[dict[str, Any]],
    semantic_candidates: list[dict[str, Any]],
    skipped_components: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    source_quality = drawing_ir.get("quality") if isinstance(drawing_ir.get("quality"), dict) else {}
    if source_quality.get("ready_for_feature_extraction") is False:
        reasons.append("source_drawing_ir_not_ready_for_feature_extraction")
    if not semantic_candidates:
        reasons.append("no_semantic_feature_candidates")
    if skipped_components:
        reasons.append("skipped_geometry_components")
    if semantic_candidates:
        reasons.append("rule_based_semantic_candidates_need_validation")

    views_with_candidates = [view["view_id"] for view in views if view["candidate_count"] > 0]
    blocked_views = [
        view["view_id"]
        for view in views
        if not (view.get("geometry_core") or {}).get("ready_for_feature_extraction") and view["candidate_count"] == 0
    ]

    return {
        "view_count": len(views),
        "views_with_candidates": views_with_candidates,
        "views_without_candidates": [view["view_id"] for view in views if view["candidate_count"] == 0],
        "geometry_core_blocked_view_ids": blocked_views,
        "input_component_count": _input_component_count(drawing_ir),
        "semantic_candidate_count": len(semantic_candidates),
        "skipped_component_count": len(skipped_components),
        "needs_manual_review": bool(reasons),
        "review_reasons": reasons,
        "ready_for_constraint_binding": False,
        "blocking_items": [
            "semantic_features_not_validated",
            "dimensions_not_extracted",
            "dimension_geometry_binding_not_built",
            "cross_view_relations_not_built",
        ],
    }


def _input_component_count(drawing_ir: dict[str, Any]) -> int:
    return sum(
        1
        for candidate in drawing_ir.get("feature_candidates") or []
        if isinstance(candidate, dict) and candidate.get("type") == "geometry_component"
    )


def _iter_drawing_ir_sample_ids(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "drawing_ir.json").exists())


def _looks_like_copy_sample(sample_id: str) -> bool:
    lowered = sample_id.lower()
    return lowered.endswith("-copy") or lowered.endswith("_copy") or " copy" in lowered


def _write_summary_csv(path: Path, records: list[ViewFeatureExtractionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "output_path", "view_count", "feature_count", "skipped", "error"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "output_path": record.output_path.as_posix() if record.output_path else "",
                    "view_count": record.view_count,
                    "feature_count": record.feature_count,
                    "skipped": record.skipped,
                    "error": record.error or "",
                }
            )


def _write_summary_json(path: Path, records: list[ViewFeatureExtractionRecord]) -> None:
    write_json(
        path,
        {
            "records": [
                {
                    "sample_id": record.sample_id,
                    "output_path": record.output_path.as_posix() if record.output_path else None,
                    "view_count": record.view_count,
                    "feature_count": record.feature_count,
                    "skipped": record.skipped,
                    "error": record.error,
                }
                for record in records
            ]
        },
    )


def _stage_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.as_posix()
    return (Path(root.name) / relative).as_posix()
