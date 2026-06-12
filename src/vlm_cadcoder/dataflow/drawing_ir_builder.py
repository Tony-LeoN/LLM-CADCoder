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
) -> DrawingIRBuildResult:
    root = Path(dataflow_root)
    detection_path = root / "05.ViewDetection" / sample_id / f"page_{page:03d}_views.json"
    classification_path = root / "07.ViewClassification" / sample_id / f"page_{page:03d}_view_classification.json"
    single_view_dir = root / "06.SingleViews" / sample_id
    clean_page_path = root / "04.CleanPNG" / sample_id / f"page_{page:03d}_clean.png"

    detection = _read_strict_json(detection_path)
    classification = _read_strict_json(classification_path)
    _validate_detection_payload(detection, detection_path)
    _validate_classification_payload(classification, sample_id, page, classification_path)
    accepted_views = _accepted_detection_views(detection)
    classified_views = classification["views"]

    views = [
        _build_view_ir(
            root=root,
            sample_id=sample_id,
            single_view_dir=single_view_dir,
            detection_path=detection_path,
            classification_path=classification_path,
            classification_view=view,
            accepted_views=accepted_views,
        )
        for view in classified_views
    ]

    skipped_views = classification.get("skipped_views") or []
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
            },
        },
        "views": views,
        "dimensions": [],
        "feature_candidates": [],
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
            },
        },
        "quality": _build_quality_block(views, skipped_views),
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
            result = build_drawing_ir_sample(sample_id=current_sample_id, dataflow_root=root, page=page)
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


def _build_view_ir(
    *,
    root: Path,
    sample_id: str,
    single_view_dir: Path,
    detection_path: Path,
    classification_path: Path,
    classification_view: dict[str, Any],
    accepted_views: list[dict[str, Any]],
) -> dict[str, Any]:
    view_id = str(classification_view.get("view_id") or "")
    if not view_id:
        raise ValueError(f"Classified view in {classification_path} is missing view_id")

    bbox = _bbox_from_classified_view(classification_view, classification_path)
    view_dir = single_view_dir / view_id
    metadata_path = view_dir / "view_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing view_metadata.json for {sample_id}/{view_id}: {metadata_path}")
    metadata = _read_strict_json(metadata_path)
    detection_view = _match_detection_view(bbox, accepted_views)
    if detection_view is None:
        raise ValueError(f"Classified view {sample_id}/{view_id} does not match any accepted 05.ViewDetection bbox")
    detector = _build_detector_block(classification_view, metadata, detection_view)
    image_clean = classification_view.get("image_clean") or _path_or_none(root, view_dir / "clean_view_with_annotations.png")
    view_type = str(classification_view.get("type") or "unknown")
    type_confidence = float(classification_view.get("confidence") or 0.0)
    needs_manual_review = bool(classification_view.get("needs_manual_review"))
    review_reasons = list(classification_view.get("reasons") or [])

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


def _build_quality_block(views: list[dict[str, Any]], skipped_views: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    if not views:
        reasons.append("no_classified_views")
    if any(view.get("needs_manual_review") for view in views):
        reasons.append("view_classification_needs_manual_review")
    if skipped_views:
        reasons.append("skipped_single_view_crops")

    return {
        "view_count": len(views),
        "skipped_view_count": len(skipped_views),
        "needs_manual_review": bool(reasons),
        "review_reasons": reasons,
        "ready_for_feature_extraction": bool(views),
        "ready_for_cad_generation": False,
        "blocking_items": [
            "dimensions_not_extracted",
            "features_not_extracted",
            "constraints_not_built",
        ],
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


def _stage_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.as_posix()
    return (Path(root.name) / relative).as_posix()
