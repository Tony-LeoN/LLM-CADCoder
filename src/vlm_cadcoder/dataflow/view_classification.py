from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vlm_cadcoder.utils.json_utils import write_json

_VIEW_DIR_RE = re.compile(r"^view_\d+$")


@dataclass(frozen=True)
class ViewClassification:
    view_id: str
    view_type: str
    confidence: float
    is_primary: bool
    needs_manual_review: bool
    reasons: list[str]
    bbox_on_page: list[int]
    crop_size: dict[str, int]
    detector_score: float | None
    image_clean: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ViewClassificationResult:
    sample_id: str
    page: int
    output_path: Path
    views: list[ViewClassification]


@dataclass(frozen=True)
class ViewClassificationBatchRecord:
    sample_id: str
    output_path: Path | None
    classified_views: int = 0
    skipped: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ViewClassificationBatchSummary:
    records: list[ViewClassificationBatchRecord]
    csv_path: Path | None = None
    json_path: Path | None = None

    @property
    def classified_count(self) -> int:
        return sum(1 for record in self.records if record.output_path is not None and not record.error and not record.skipped)

    @property
    def skipped_count(self) -> int:
        return sum(1 for record in self.records if record.skipped)

    @property
    def failed_count(self) -> int:
        return sum(1 for record in self.records if record.error is not None)


def classify_single_view_sample(
    *,
    sample_id: str,
    dataflow_root: str | Path = "DataFlow",
    page: int = 1,
    output_path: str | Path | None = None,
) -> ViewClassificationResult:
    root = Path(dataflow_root)
    sample_dir = root / "06.SingleViews" / sample_id
    if not sample_dir.exists():
        raise FileNotFoundError(f"Missing single-view directory: {sample_dir}")

    view_items = _load_view_items(sample_dir)
    if not view_items:
        raise ValueError(f"No view_* directories with metadata found in {sample_dir}")

    detection_data = _load_detection_data(root, sample_id, page)
    page_size = _load_page_size(detection_data, view_items)
    accepted_bboxes = _accepted_detection_bboxes(detection_data)
    classified_items, skipped_views, input_filter = _filter_items_by_accepted_detections(view_items, accepted_bboxes)
    classifications = _classify_view_items(classified_items, page_size)
    target = Path(output_path) if output_path else root / "07.ViewClassification" / sample_id / f"page_{page:03d}_view_classification.json"
    _write_classification_json(target, sample_id, page, page_size, classifications, input_filter, skipped_views)
    return ViewClassificationResult(sample_id=sample_id, page=page, output_path=target, views=classifications)


def classify_view_samples(
    *,
    dataflow_root: str | Path = "DataFlow",
    sample_id: str | None = None,
    page: int = 1,
    include_copy: bool = False,
    fail_fast: bool = False,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
) -> ViewClassificationBatchSummary:
    root = Path(dataflow_root)
    sample_ids = [sample_id] if sample_id else _iter_single_view_sample_ids(root / "06.SingleViews")
    records: list[ViewClassificationBatchRecord] = []

    for current_sample_id in sample_ids:
        if not include_copy and _looks_like_copy_sample(current_sample_id):
            records.append(ViewClassificationBatchRecord(sample_id=current_sample_id, output_path=None, skipped=True))
            continue
        try:
            result = classify_single_view_sample(sample_id=current_sample_id, dataflow_root=root, page=page)
            records.append(
                ViewClassificationBatchRecord(
                    sample_id=current_sample_id,
                    output_path=result.output_path,
                    classified_views=len(result.views),
                )
            )
        except Exception as exc:  # pragma: no cover - exact data errors are sample dependent
            if fail_fast:
                raise
            records.append(ViewClassificationBatchRecord(sample_id=current_sample_id, output_path=None, error=str(exc)))

    out_root = root / "07.ViewClassification"
    csv_path = Path(output_csv) if output_csv else out_root / "view_classification_summary.csv"
    json_path = Path(output_json) if output_json else out_root / "view_classification_summary.json"
    _write_summary_csv(csv_path, records)
    _write_summary_json(json_path, records)
    return ViewClassificationBatchSummary(records=records, csv_path=csv_path, json_path=json_path)


def _load_view_items(sample_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for view_dir in sorted(path for path in sample_dir.iterdir() if path.is_dir() and _VIEW_DIR_RE.match(path.name)):
        metadata_path = view_dir / "view_metadata.json"
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        bbox = _bbox_from_metadata(metadata)
        crop_size = _crop_size_from_metadata(metadata, bbox)
        items.append(
            {
                "view_id": str(metadata.get("view_id") or view_dir.name),
                "view_dir": view_dir,
                "bbox_on_page": bbox,
                "crop_size": crop_size,
                "detector_score": _detector_score(metadata),
                "image_clean": (view_dir / "clean_view_with_annotations.png").as_posix()
                if (view_dir / "clean_view_with_annotations.png").exists()
                else None,
            }
        )
    return items


def _classify_view_items(view_items: list[dict[str, Any]], page_size: tuple[int, int]) -> list[ViewClassification]:
    enriched = [_view_geometry(item, page_size) for item in view_items]
    isometric_ids = _select_isometric_view_ids(enriched, page_size)
    front_id = _select_front_view_id(enriched, isometric_ids)
    results: list[ViewClassification] = []

    for item in enriched:
        view_type, confidence, reasons = _classify_one(item, front_id, isometric_ids)
        results.append(
            ViewClassification(
                view_id=item["view_id"],
                view_type=view_type,
                confidence=confidence,
                is_primary=item["view_id"] == front_id,
                needs_manual_review=confidence < 0.6 or view_type == "unknown",
                reasons=reasons,
                bbox_on_page=item["bbox_on_page"],
                crop_size=item["crop_size"],
                detector_score=item["detector_score"],
                image_clean=item["image_clean"],
            )
        )
    return results


def _classify_one(item: dict[str, Any], front_id: str | None, isometric_ids: set[str]) -> tuple[str, float, list[str]]:
    if item["view_id"] in isometric_ids:
        return "isometric", 0.55, ["right_lower_oblique_view_candidate"]
    if item["view_id"] == front_id:
        return "front", 0.68, ["largest_non_isometric_view"]

    aspect = item["aspect"]
    if aspect >= 3.5:
        return "top", 0.58, ["thin_horizontal_profile"]
    if aspect <= 0.45:
        return "left", 0.58, ["thin_vertical_profile"]
    if item["center_x_ratio"] >= 0.55 and aspect <= 0.9:
        return "left", 0.52, ["right_side_profile_candidate"]
    return "unknown", 0.25, ["ambiguous_geometry"]


def _select_front_view_id(enriched: list[dict[str, Any]], isometric_ids: set[str]) -> str | None:
    candidates = [item for item in enriched if item["view_id"] not in isometric_ids]
    if not candidates:
        candidates = enriched
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["area"])["view_id"]


def _select_isometric_view_ids(enriched: list[dict[str, Any]], page_size: tuple[int, int]) -> set[str]:
    if len(enriched) < 3:
        return set()
    largest_area = max(item["area"] for item in enriched)
    candidates = [
        item
        for item in enriched
        if item["center_x_ratio"] >= 0.52
        and item["center_y_ratio"] >= 0.48
        and item["area"] <= largest_area * 0.75
        and 0.45 <= item["aspect"] <= 2.2
    ]
    if not candidates:
        return set()
    return {max(candidates, key=lambda item: (item["center_x_ratio"] + item["center_y_ratio"], item["area"]))["view_id"]}


def _view_geometry(item: dict[str, Any], page_size: tuple[int, int]) -> dict[str, Any]:
    width, height = page_size
    x1, y1, x2, y2 = item["bbox_on_page"]
    crop_width = max(1, x2 - x1)
    crop_height = max(1, y2 - y1)
    enriched = dict(item)
    enriched.update(
        {
            "area": crop_width * crop_height,
            "aspect": crop_width / crop_height,
            "center_x_ratio": ((x1 + x2) / 2) / max(1, width),
            "center_y_ratio": ((y1 + y2) / 2) / max(1, height),
        }
    )
    return enriched


def _load_detection_data(root: Path, sample_id: str, page: int) -> dict[str, Any] | None:
    detection_path = root / "05.ViewDetection" / sample_id / f"page_{page:03d}_views.json"
    if not detection_path.exists():
        return None
    with detection_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def _load_page_size(detection_data: dict[str, Any] | None, view_items: list[dict[str, Any]]) -> tuple[int, int]:
    if detection_data:
        size = detection_data.get("image_size")
        if isinstance(size, dict) and size.get("width") and size.get("height"):
            return int(size["width"]), int(size["height"])
        if isinstance(size, list) and len(size) == 2:
            return int(size[0]), int(size[1])

    max_x = max(item["bbox_on_page"][2] for item in view_items)
    max_y = max(item["bbox_on_page"][3] for item in view_items)
    return max_x, max_y


def _accepted_detection_bboxes(detection_data: dict[str, Any] | None) -> list[list[int]] | None:
    if detection_data is None:
        return None
    views = detection_data.get("views")
    if not isinstance(views, list):
        return []

    bboxes: list[list[int]] = []
    for view in views:
        if not isinstance(view, dict):
            continue
        bbox = view.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        filter_data = view.get("filter")
        if isinstance(filter_data, dict) and filter_data.get("accepted") is False:
            continue
        bboxes.append([int(value) for value in bbox])
    return bboxes


def _filter_items_by_accepted_detections(
    view_items: list[dict[str, Any]],
    accepted_bboxes: list[list[int]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if accepted_bboxes is None:
        return (
            view_items,
            [],
            {
                "source": "06.SingleViews only",
                "accepted_detection_count": None,
                "skipped_view_count": 0,
            },
        )

    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in view_items:
        if _matches_any_bbox(item["bbox_on_page"], accepted_bboxes):
            kept.append(item)
        else:
            skipped.append(
                {
                    "view_id": item["view_id"],
                    "reason": "not_in_05_accepted_views",
                    "bbox_on_page": item["bbox_on_page"],
                }
            )

    return (
        kept,
        skipped,
        {
            "source": "05.ViewDetection accepted views",
            "accepted_detection_count": len(accepted_bboxes),
            "skipped_view_count": len(skipped),
            "match_rule": "intersection_over_smaller_bbox>=0.85",
        },
    )


def _matches_any_bbox(bbox: list[int], candidates: list[list[int]]) -> bool:
    return any(_intersection_over_smaller_bbox(bbox, candidate) >= 0.85 for candidate in candidates)


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


def _bbox_from_metadata(metadata: dict[str, Any]) -> list[int]:
    values = metadata.get("bbox_on_page") or metadata.get("bbox")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("view_metadata.json must contain bbox_on_page with four values")
    return [int(value) for value in values]


def _crop_size_from_metadata(metadata: dict[str, Any], bbox: list[int]) -> dict[str, int]:
    values = metadata.get("crop_size")
    if isinstance(values, dict) and values.get("width") and values.get("height"):
        return {"width": int(values["width"]), "height": int(values["height"])}
    return {"width": bbox[2] - bbox[0], "height": bbox[3] - bbox[1]}


def _detector_score(metadata: dict[str, Any]) -> float | None:
    detector = metadata.get("detector")
    if isinstance(detector, dict) and detector.get("score") is not None:
        return float(detector["score"])
    return None


def _iter_single_view_sample_ids(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and any(_VIEW_DIR_RE.match(child.name) for child in path.iterdir() if child.is_dir()))


def _looks_like_copy_sample(sample_id: str) -> bool:
    lowered = sample_id.lower()
    return lowered.endswith("-copy") or lowered.endswith("_copy") or " copy" in lowered


def _write_classification_json(
    path: Path,
    sample_id: str,
    page: int,
    page_size: tuple[int, int],
    views: list[ViewClassification],
    input_filter: dict[str, Any],
    skipped_views: list[dict[str, Any]],
) -> None:
    width, height = page_size
    write_json(
        path,
        {
            "sample_id": sample_id,
            "page": page,
            "image_size": {"width": width, "height": height},
            "coordinate_system": "page_pixel_xyxy",
            "method": {
                "name": "heuristic_view_classifier",
                "version": "0.1.0",
                "role": "baseline_for_manual_review_and_vlm_comparison",
            },
            "input_filter": input_filter,
            "views": [
                {
                    "view_id": view.view_id,
                    "type": view.view_type,
                    "confidence": view.confidence,
                    "is_primary": view.is_primary,
                    "needs_manual_review": view.needs_manual_review,
                    "reasons": view.reasons,
                    "bbox_on_page": view.bbox_on_page,
                    "crop_size": view.crop_size,
                    "detector_score": view.detector_score,
                    "image_clean": view.image_clean,
                }
                for view in views
            ],
            "skipped_views": skipped_views,
        },
    )


def _write_summary_csv(path: Path, records: list[ViewClassificationBatchRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "output_path", "classified_views", "skipped", "error"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "output_path": record.output_path.as_posix() if record.output_path else "",
                    "classified_views": record.classified_views,
                    "skipped": record.skipped,
                    "error": record.error or "",
                }
            )


def _write_summary_json(path: Path, records: list[ViewClassificationBatchRecord]) -> None:
    write_json(
        path,
        {
            "records": [
                {
                    "sample_id": record.sample_id,
                    "output_path": record.output_path.as_posix() if record.output_path else None,
                    "classified_views": record.classified_views,
                    "skipped": record.skipped,
                    "error": record.error,
                }
                for record in records
            ]
        },
    )
