from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from vlm_cadcoder.utils.json_utils import write_json


@dataclass(frozen=True)
class DrawingIRBuildResult:
    sample_id: str
    page: int
    output_path: Path
    view_count: int


@dataclass(frozen=True)
class DrawingIRBuildRecord:
    sample_id: str
    output_path: Path | None
    view_count: int = 0
    skipped: bool = False
    error: str | None = None


@dataclass(frozen=True)
class DrawingIRBuildSummary:
    records: list[DrawingIRBuildRecord]
    csv_path: Path | None = None
    json_path: Path | None = None

    @property
    def built_count(self) -> int:
        return sum(1 for record in self.records if record.output_path is not None and not record.error and not record.skipped)

    @property
    def skipped_count(self) -> int:
        return sum(1 for record in self.records if record.skipped)

    @property
    def failed_count(self) -> int:
        return sum(1 for record in self.records if record.error is not None)


def build_drawing_ir_sample(
    *,
    sample_id: str,
    dataflow_root: str | Path = "DataFlow",
    page: int = 1,
    output_path: str | Path | None = None,
    geometry_core_audit_path: str | Path | None = None,
    extract_feature_candidates: bool = True,
) -> DrawingIRBuildResult:
    root = Path(dataflow_root)
    detection_path = root / "05.ViewDetection" / sample_id / f"page_{page:03d}_views.json"
    classification_path = root / "07.ViewClassification" / sample_id / f"page_{page:03d}_view_classification.json"
    single_view_dir = root / "06.SingleViews" / sample_id
    clean_page_path = root / "04.CleanPNG" / sample_id / f"page_{page:03d}_clean.png"
    audit_path = Path(geometry_core_audit_path) if geometry_core_audit_path else root / "06.SingleViews" / "geometry_core_audit.json"

    detection = _read_strict_json(detection_path)
    classification = _read_strict_json(classification_path)
    geometry_core_audit = _read_optional_geometry_core_audit(audit_path)
    _validate_detection_payload(detection, detection_path)
    _validate_classification_payload(classification, sample_id, page, classification_path)
    accepted_views = _accepted_detection_views(detection)
    classified_views = classification["views"]
    geometry_core_records = _geometry_core_records_by_view(geometry_core_audit, sample_id)

    views: list[dict[str, Any]] = []
    stale_skipped_views: list[dict[str, Any]] = []
    for view in classified_views:
        view_id = str(view.get("view_id") or "")
        if not view_id:
            raise ValueError(f"Classified view in {classification_path} is missing view_id")
        if not (single_view_dir / view_id / "view_metadata.json").exists():
            stale_skipped_views.append(_skipped_stale_classification_view(view))
            continue
        views.append(
            _build_view_ir(
                root=root,
                sample_id=sample_id,
                single_view_dir=single_view_dir,
                detection_path=detection_path,
                classification_path=classification_path,
                classification_view=view,
                accepted_views=accepted_views,
                geometry_core_record=geometry_core_records.get(view_id),
                geometry_core_audit_source=_path_or_none(root, audit_path),
            )
        )
    feature_candidates, feature_extraction = _build_feature_candidates(
        root=root,
        views=views,
        enabled=extract_feature_candidates,
    )

    skipped_views = list(classification.get("skipped_views") or []) + stale_skipped_views
    drawing_ir = {
        "schema": "drawing_ir",
        "version": "0.1.0",
        "sample_id": sample_id,
        "page": page,
        "sheet": {
            "coordinate_system": classification.get("coordinate_system", "page_pixel_xyxy"),
            "image_size": classification.get("image_size") or detection.get("image_size"),
            "clean_page_image": _path_or_none(root, clean_page_path),
            "input_mode": "detected_views_plus_classification",
            "source_stages": {
                "view_detection": _stage_path(root, detection_path),
                "single_views": _stage_path(root, single_view_dir),
                "view_classification": _stage_path(root, classification_path),
                "geometry_core_audit": _path_or_none(root, audit_path),
            },
        },
        "views": views,
        "dimensions": [],
        "feature_candidates": feature_candidates,
        "constraints": [],
        "view_relations": [],
        "skipped_views": skipped_views,
        "provenance": {
            "builder": {
                "name": "drawing_ir_builder",
                "version": "0.1.0",
            },
            "view_detection_filter": detection.get("filter"),
            "classification_method": classification.get("method"),
            "classification_input_filter": classification.get("input_filter"),
            "skipped_views": skipped_views,
            "source_files": {
                "view_detection": _stage_path(root, detection_path),
                "view_classification": _stage_path(root, classification_path),
                "geometry_core_audit": _path_or_none(root, audit_path),
            },
        },
        "quality": _build_quality_block(
            views=views,
            skipped_views=skipped_views,
            geometry_core_audit=geometry_core_audit,
            feature_extraction=feature_extraction,
        ),
    }

    target = Path(output_path) if output_path else root / "10.StructuredCADRepresentation" / sample_id / "drawing_ir.json"
    write_json(target, drawing_ir)
    return DrawingIRBuildResult(sample_id=sample_id, page=page, output_path=target, view_count=len(views))


def build_drawing_ir_samples(
    *,
    dataflow_root: str | Path = "DataFlow",
    sample_id: str | None = None,
    page: int = 1,
    include_copy: bool = False,
    fail_fast: bool = False,
    geometry_core_audit_path: str | Path | None = None,
    extract_feature_candidates: bool = True,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
) -> DrawingIRBuildSummary:
    root = Path(dataflow_root)
    sample_ids = [sample_id] if sample_id else _iter_classified_sample_ids(root / "07.ViewClassification", page)
    records: list[DrawingIRBuildRecord] = []

    for current_sample_id in sample_ids:
        if not current_sample_id:
            continue
        if not include_copy and _looks_like_copy_sample(current_sample_id):
            records.append(DrawingIRBuildRecord(sample_id=current_sample_id, output_path=None, skipped=True))
            continue
        try:
            result = build_drawing_ir_sample(
                sample_id=current_sample_id,
                dataflow_root=root,
                page=page,
                geometry_core_audit_path=geometry_core_audit_path,
                extract_feature_candidates=extract_feature_candidates,
            )
            records.append(
                DrawingIRBuildRecord(
                    sample_id=current_sample_id,
                    output_path=result.output_path,
                    view_count=result.view_count,
                )
            )
        except Exception as exc:  # pragma: no cover - sample-specific data failures are reported in summary
            if fail_fast:
                raise
            records.append(DrawingIRBuildRecord(sample_id=current_sample_id, output_path=None, error=str(exc)))

    out_root = root / "10.StructuredCADRepresentation"
    csv_path = Path(output_csv) if output_csv else out_root / "drawing_ir_summary.csv"
    json_path = Path(output_json) if output_json else out_root / "drawing_ir_summary.json"
    _write_summary_csv(csv_path, records)
    _write_summary_json(json_path, records)
    return DrawingIRBuildSummary(records=records, csv_path=csv_path, json_path=json_path)


def _skipped_stale_classification_view(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "view_id": str(view.get("view_id") or ""),
        "reason": "missing_single_view_metadata",
        "bbox_on_page": view.get("bbox_on_page") or view.get("bbox"),
        "source": "07.ViewClassification",
    }


def _build_view_ir(
    *,
    root: Path,
    sample_id: str,
    single_view_dir: Path,
    detection_path: Path,
    classification_path: Path,
    classification_view: dict[str, Any],
    accepted_views: list[dict[str, Any]],
    geometry_core_record: dict[str, Any] | None,
    geometry_core_audit_source: str | None,
) -> dict[str, Any]:
    view_id = str(classification_view.get("view_id") or "")
    if not view_id:
        raise ValueError(f"Classified view in {classification_path} is missing view_id")

    bbox = _bbox_from_classified_view(classification_view, classification_path)
    view_dir = single_view_dir / view_id
    metadata_path = view_dir / "view_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing view_metadata.json for {sample_id}/{view_id}: {metadata_path}")
    metadata = _read_view_metadata_json(metadata_path)
    detection_view = _match_detection_view(bbox, accepted_views)
    if detection_view is None:
        raise ValueError(f"Classified view {sample_id}/{view_id} does not match any accepted 05.ViewDetection bbox")
    detector = _build_detector_block(classification_view, metadata, detection_view)
    image_clean = classification_view.get("image_clean") or _path_or_none(root, view_dir / "clean_view_with_annotations.png")
    view_type = str(classification_view.get("type") or "unknown")
    type_confidence = float(classification_view.get("confidence") or 0.0)
    needs_manual_review = bool(classification_view.get("needs_manual_review"))
    review_reasons = list(classification_view.get("reasons") or [])
    geometry_core = _build_geometry_core_block(root, view_dir, geometry_core_record, geometry_core_audit_source)

    return {
        "id": view_id,
        "type": view_type,
        "type_source": "heuristic_view_classifier",
        "type_confidence": type_confidence,
        "type_candidates": [
            {
                "type": view_type,
                "confidence": type_confidence,
                "source": "07.ViewClassification",
                "needs_manual_review": needs_manual_review,
                "reasons": review_reasons,
            }
        ],
        "is_primary": bool(classification_view.get("is_primary")),
        "needs_manual_review": needs_manual_review,
        "review_reasons": review_reasons,
        "bbox": bbox,
        "bbox_on_page": bbox,
        "crop_size": _crop_size(classification_view, metadata, bbox),
        "image_clean": _normalize_path(root, Path(str(image_clean))) if image_clean else None,
        "image_raw": _path_or_none(root, view_dir / "raw_view_with_annotations.png"),
        "image_geometry_core": geometry_core["paths"]["geometry_core"],
        "geometry_core": geometry_core,
        "detector": detector,
        "source": {
            "view_detection": _stage_path(root, detection_path),
            "view_classification": _stage_path(root, classification_path),
            "view_metadata": _path_or_none(root, metadata_path),
            "accepted_detection": detection_view,
        },
    }


def _build_detector_block(
    classification_view: dict[str, Any],
    metadata: dict[str, Any],
    detection_view: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata_detector = metadata.get("detector") if isinstance(metadata.get("detector"), dict) else {}
    score = classification_view.get("detector_score")
    if score is None and detection_view:
        score = detection_view.get("score")
    if score is None:
        score = metadata_detector.get("score")

    return {
        "score": float(score) if score is not None else None,
        "source": (detection_view or {}).get("source") or metadata_detector.get("name"),
        "source_view_id": (detection_view or {}).get("source_view_id") or metadata.get("view_id"),
        "accepted_view_id": (detection_view or {}).get("view_id"),
    }


def _build_quality_block(
    *,
    views: list[dict[str, Any]],
    skipped_views: list[dict[str, Any]],
    geometry_core_audit: dict[str, Any] | None,
    feature_extraction: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not views:
        reasons.append("no_classified_views")
    if any(view.get("needs_manual_review") for view in views):
        reasons.append("view_classification_needs_manual_review")
    if skipped_views:
        reasons.append("skipped_single_view_crops")
    if geometry_core_audit is None:
        reasons.append("missing_geometry_core_audit")

    ready_view_ids = [
        str(view["id"])
        for view in views
        if view.get("geometry_core", {}).get("ready_for_feature_extraction")
    ]
    blocked_view_ids = [
        str(view["id"])
        for view in views
        if not view.get("geometry_core", {}).get("ready_for_feature_extraction")
    ]
    if blocked_view_ids:
        reasons.append("geometry_core_not_ready")
    if feature_extraction.get("status") == "unavailable":
        reasons.append(str(feature_extraction.get("reason") or "feature_extraction_unavailable"))

    ready_for_feature_extraction = bool(views) and not blocked_view_ids and feature_extraction.get("status") != "unavailable"
    blocking_items = [
        "dimensions_not_extracted",
        "semantic_features_not_validated",
        "constraints_not_built",
    ]
    if blocked_view_ids:
        blocking_items.insert(0, "geometry_core_quality_gate_blocked")
    if not feature_extraction.get("candidate_count"):
        blocking_items.insert(0, "low_level_feature_candidates_not_extracted")

    return {
        "view_count": len(views),
        "skipped_view_count": len(skipped_views),
        "needs_manual_review": bool(reasons),
        "review_reasons": reasons,
        "ready_for_feature_extraction": ready_for_feature_extraction,
        "feature_extraction_input_view_count": len(ready_view_ids),
        "geometry_core_blocked_view_count": len(blocked_view_ids),
        "feature_candidate_count": int(feature_extraction.get("candidate_count") or 0),
        "geometry_core": {
            "audit_available": geometry_core_audit is not None,
            "ready_view_ids": ready_view_ids,
            "blocked_view_ids": blocked_view_ids,
            "tier_counts": (geometry_core_audit or {}).get("summary", {}).get("tier_counts"),
        },
        "feature_extraction": feature_extraction,
        "ready_for_cad_generation": False,
        "blocking_items": blocking_items,
    }


def _read_strict_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required JSON file: {path}") from exc

    decoder = json.JSONDecoder()
    try:
        data, end = decoder.raw_decode(text)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if text[end:].strip():
        raise ValueError(f"Trailing data after JSON document in {path}")
    return data


def _read_view_metadata_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required JSON file: {path}") from exc

    decoder = json.JSONDecoder()
    try:
        data, end = decoder.raw_decode(text)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
    return data


def _read_optional_geometry_core_audit(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = _read_strict_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Geometry-core audit JSON must be an object: {path}")
    records = data.get("records")
    if records is not None and not isinstance(records, list):
        raise ValueError(f"Geometry-core audit records must be a list: {path}")
    return data


def _geometry_core_records_by_view(audit: dict[str, Any] | None, sample_id: str) -> dict[str, dict[str, Any]]:
    records = audit.get("records") if audit else None
    if not isinstance(records, list):
        return {}

    by_view: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("sample_id") != sample_id:
            continue
        view_id = record.get("view_id")
        if isinstance(view_id, str):
            by_view[view_id] = record
    return by_view


def _build_geometry_core_block(
    root: Path,
    view_dir: Path,
    record: dict[str, Any] | None,
    audit_source: str | None,
) -> dict[str, Any]:
    fallback_paths = {
        "geometry_core": _path_or_none(root, view_dir / "geometry_core.png"),
        "geometry_core_mask": _path_or_none(root, view_dir / "geometry_core_mask.png"),
        "geometry_core_prob": _path_or_none(root, view_dir / "geometry_core_prob.png"),
    }
    if record is None:
        return {
            "quality_tier": None,
            "needs_manual_review": True,
            "review_reasons": ["missing_geometry_core_audit_record"],
            "ready_for_feature_extraction": False,
            "paths": fallback_paths,
            "metrics": {},
            "source": audit_source,
        }

    paths = record.get("paths") if isinstance(record.get("paths"), dict) else {}
    geometry_core_path = paths.get("geometry_core") or record.get("geometry_core_path") or fallback_paths["geometry_core"]
    mask_path = paths.get("geometry_core_mask") or record.get("mask_path") or fallback_paths["geometry_core_mask"]
    probability_path = paths.get("geometry_core_prob") or record.get("probability_path") or fallback_paths["geometry_core_prob"]

    normalized_paths = {
        "geometry_core": _normalize_stage_path(root, geometry_core_path),
        "geometry_core_mask": _normalize_stage_path(root, mask_path),
        "geometry_core_prob": _normalize_stage_path(root, probability_path),
    }
    quality_tier = record.get("quality_tier")
    review_reasons = list(record.get("review_reasons") or [])
    geometry_core_exists = bool(
        normalized_paths["geometry_core"] and _resolve_stage_path(root, normalized_paths["geometry_core"]).exists()
    )
    if not geometry_core_exists:
        review_reasons.append("missing_geometry_core_image")
    if quality_tier != "A" and not review_reasons:
        review_reasons.append("geometry_core_quality_tier_not_a")

    return {
        "quality_tier": quality_tier,
        "needs_manual_review": bool(record.get("needs_manual_review")) or quality_tier != "A" or not geometry_core_exists,
        "review_reasons": review_reasons,
        "ready_for_feature_extraction": quality_tier == "A" and geometry_core_exists,
        "manual_quality_label": record.get("manual_quality_label"),
        "manual_notes": record.get("manual_notes"),
        "paths": normalized_paths,
        "metrics": {
            "clean_black_ratio": record.get("clean_black_ratio"),
            "geometry_black_ratio": record.get("geometry_black_ratio"),
            "retained_ink_ratio": record.get("retained_ink_ratio"),
            "missing_ink_ratio": record.get("missing_ink_ratio"),
            "excess_ink_ratio": record.get("excess_ink_ratio"),
            "geometry_component_count": record.get("geometry_component_count"),
        },
        "source": audit_source,
    }


def _build_feature_candidates(
    *,
    root: Path,
    views: list[dict[str, Any]],
    enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not enabled:
        return [], {"status": "disabled", "method": None, "candidate_count": 0, "errors": []}

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return [], {
            "status": "unavailable",
            "reason": "missing_pillow_or_numpy",
            "method": "geometry_core_connected_components",
            "candidate_count": 0,
            "errors": [],
        }

    candidates: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for view in views:
        geometry_core = view.get("geometry_core")
        if not isinstance(geometry_core, dict) or not geometry_core.get("ready_for_feature_extraction"):
            continue
        path_text = (geometry_core.get("paths") or {}).get("geometry_core")
        if not isinstance(path_text, str) or not path_text:
            continue
        image_path = _resolve_stage_path(root, path_text)
        try:
            image = Image.open(image_path).convert("L")
            mask = np.asarray(image) < 250
        except Exception as exc:  # pragma: no cover - corrupt view artifacts are reported in IR quality
            errors.append({"view_id": str(view.get("id")), "path": path_text, "error": str(exc)})
            continue

        for index, component in enumerate(_connected_component_bboxes(mask), start=1):
            x1, y1, x2, y2, pixel_count = component
            width = x2 - x1
            height = y2 - y1
            if pixel_count < 12 or width < 2 or height < 2:
                continue
            candidates.append(
                {
                    "id": f"{view['id']}_geometry_component_{index:03d}",
                    "type": "geometry_component",
                    "semantic_status": "unclassified",
                    "view_id": view["id"],
                    "count": 1,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_on_page": _component_bbox_on_page(view, [x1, y1, x2, y2]),
                    "area_px": int(pixel_count),
                    "source": "geometry_core_connected_components",
                    "confidence": 0.3,
                    "evidence": [
                        f"geometry_core_quality={geometry_core.get('quality_tier')}",
                        path_text,
                    ],
                }
            )

    return candidates, {
        "status": "ok",
        "method": "geometry_core_connected_components",
        "candidate_count": len(candidates),
        "errors": errors,
    }


def _connected_component_bboxes(mask) -> list[tuple[int, int, int, int, int]]:
    import numpy as np

    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x]:
            continue
        min_x = max_x = x
        min_y = max_y = y
        pixel_count = 0
        stack = [(y, x)]
        visited[y, x] = True
        while stack:
            current_y, current_x = stack.pop()
            pixel_count += 1
            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)
            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)
            for next_y in range(max(0, current_y - 1), min(mask.shape[0], current_y + 2)):
                for next_x in range(max(0, current_x - 1), min(mask.shape[1], current_x + 2)):
                    if mask[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        components.append((min_x, min_y, max_x + 1, max_y + 1, pixel_count))
    return components


def _component_bbox_on_page(view: dict[str, Any], component_bbox: list[int]) -> list[int] | None:
    view_bbox = view.get("bbox_on_page") or view.get("bbox")
    if not isinstance(view_bbox, list) or len(view_bbox) != 4:
        return None
    return [
        int(view_bbox[0]) + component_bbox[0],
        int(view_bbox[1]) + component_bbox[1],
        int(view_bbox[0]) + component_bbox[2],
        int(view_bbox[1]) + component_bbox[3],
    ]


def _validate_detection_payload(detection: Any, path: Path) -> None:
    if not isinstance(detection, dict):
        raise ValueError(f"Detection JSON must be an object: {path}")
    views = detection.get("views")
    if not isinstance(views, list):
        raise ValueError(f"Missing views list in {path}")
    for index, view in enumerate(views, start=1):
        if not isinstance(view, dict):
            raise ValueError(f"Detection view #{index} in {path} must be an object")
        filter_data = view.get("filter")
        if isinstance(filter_data, dict) and filter_data.get("accepted") is False:
            continue
        bbox = view.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Accepted detection view #{index} in {path} is missing bbox")
        if not view.get("view_id") and not view.get("source_view_id"):
            raise ValueError(f"Accepted detection view #{index} in {path} is missing view_id/source_view_id")


def _validate_classification_payload(classification: Any, sample_id: str, page: int, path: Path) -> None:
    if not isinstance(classification, dict):
        raise ValueError(f"View classification JSON must be an object: {path}")
    if classification.get("sample_id") != sample_id:
        raise ValueError(f"View classification sample_id mismatch in {path}")
    if int(classification.get("page") or 0) != page:
        raise ValueError(f"View classification page mismatch in {path}")
    if not isinstance(classification.get("image_size"), dict):
        raise ValueError(f"Missing image_size object in {path}")
    if not isinstance(classification.get("views"), list):
        raise ValueError(f"Missing views list in {path}")


def _accepted_detection_views(detection: dict[str, Any]) -> list[dict[str, Any]]:
    views = detection.get("views")
    if not isinstance(views, list):
        return []
    accepted: list[dict[str, Any]] = []
    for view in views:
        if not isinstance(view, dict):
            continue
        filter_data = view.get("filter")
        if isinstance(filter_data, dict) and filter_data.get("accepted") is False:
            continue
        bbox = view.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            item = dict(view)
            item["bbox"] = [int(value) for value in bbox]
            accepted.append(item)
    return accepted


def _match_detection_view(bbox: list[int], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    scored = [(_intersection_over_smaller_bbox(bbox, candidate["bbox"]), candidate) for candidate in candidates]
    score, candidate = max(scored, key=lambda item: item[0])
    return candidate if score >= 0.85 else None


def _intersection_over_smaller_bbox(left: list[int], right: list[int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / min(left_area, right_area)


def _bbox_from_classified_view(view: dict[str, Any], path: Path) -> list[int]:
    bbox = view.get("bbox_on_page") or view.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Classified view in {path} is missing bbox_on_page")
    return [int(value) for value in bbox]


def _crop_size(view: dict[str, Any], metadata: dict[str, Any], bbox: list[int]) -> dict[str, int]:
    value = view.get("crop_size")
    if isinstance(value, dict) and value.get("width") and value.get("height"):
        return {"width": int(value["width"]), "height": int(value["height"])}

    metadata_value = metadata.get("crop_size")
    if isinstance(metadata_value, dict) and metadata_value.get("width") and metadata_value.get("height"):
        return {"width": int(metadata_value["width"]), "height": int(metadata_value["height"])}

    return {"width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1]}


def _iter_classified_sample_ids(root: Path, page: int) -> list[str]:
    if not root.exists():
        return []
    filename = f"page_{page:03d}_view_classification.json"
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / filename).exists())


def _looks_like_copy_sample(sample_id: str) -> bool:
    lowered = sample_id.lower()
    return lowered.endswith("-copy") or lowered.endswith("_copy") or " copy" in lowered


def _write_summary_csv(path: Path, records: list[DrawingIRBuildRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "output_path", "view_count", "skipped", "error"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "output_path": record.output_path.as_posix() if record.output_path else "",
                    "view_count": record.view_count,
                    "skipped": record.skipped,
                    "error": record.error or "",
                }
            )


def _write_summary_json(path: Path, records: list[DrawingIRBuildRecord]) -> None:
    write_json(
        path,
        {
            "records": [
                {
                    "sample_id": record.sample_id,
                    "output_path": record.output_path.as_posix() if record.output_path else None,
                    "view_count": record.view_count,
                    "skipped": record.skipped,
                    "error": record.error,
                }
                for record in records
            ]
        },
    )


def _path_or_none(root: Path, path: Path) -> str | None:
    return _stage_path(root, path) if path.exists() else None


def _normalize_path(root: Path, path: Path) -> str:
    if path.is_absolute():
        return path.as_posix()
    if path.parts and path.parts[0] == root.name:
        return path.as_posix()
    return _stage_path(root, root / path)


def _normalize_stage_path(root: Path, path_text: Any) -> str | None:
    if not isinstance(path_text, str) or not path_text:
        return None
    return _normalize_path(root, Path(path_text))


def _resolve_stage_path(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == root.name:
        return root.parent / path
    return root / path


def _stage_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.as_posix()
    return (Path(root.name) / relative).as_posix()
