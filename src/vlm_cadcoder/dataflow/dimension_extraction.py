from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_cadcoder.utils.json_utils import read_json, write_json


@dataclass(frozen=True)
class DimensionExtractionResult:
    sample_id: str
    output_path: Path
    view_count: int
    dimension_count: int


@dataclass(frozen=True)
class DimensionExtractionRecord:
    sample_id: str
    output_path: Path | None
    view_count: int = 0
    dimension_count: int = 0
    skipped: bool = False
    error: str | None = None


@dataclass(frozen=True)
class DimensionExtractionSummary:
    records: list[DimensionExtractionRecord]
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


def extract_dimensions_sample(
    *,
    sample_id: str,
    dataflow_root: str | Path = "DataFlow",
    prediction_jsonl_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    output_path: str | Path | None = None,
) -> DimensionExtractionResult:
    root = Path(dataflow_root)
    drawing_ir_path = root / "10.StructuredCADRepresentation" / sample_id / "drawing_ir.json"
    drawing_ir = _read_drawing_ir(drawing_ir_path, sample_id)
    views = _views_by_id(drawing_ir)
    view_lookup = _view_lookup(root, views)

    candidates: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {view_id: [] for view_id in views}
    unmatched_records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    source_records = _dimension_ocr_records(
        root=root,
        sample_id=sample_id,
        views=views,
        view_lookup=view_lookup,
        prediction_jsonl_paths=prediction_jsonl_paths or [],
    )
    for record in source_records:
        if not record["view_ids"]:
            unmatched_records.append(record["unmatched"])
            continue
        for view_id in record["view_ids"]:
            view = views[view_id]
            for item in record["dimensions"]:
                candidate = _dimension_candidate(
                    item=item,
                    view=view,
                    view_id=view_id,
                    source=record["source"],
                    index=len(candidates) + 1,
                )
                key = (
                    view_id,
                    candidate["text"],
                    candidate["normalized"],
                    json.dumps(candidate.get("bbox"), sort_keys=True),
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                grouped.setdefault(view_id, []).append(candidate)

    view_blocks = [
        {
            "view_id": view_id,
            "view_type": str(view.get("type") or "unknown"),
            "is_primary": bool(view.get("is_primary")),
            "bbox_on_page": view.get("bbox_on_page") or view.get("bbox"),
            "crop_size": _crop_size(view),
            "image_clean": view.get("image_clean"),
            "dimension_candidate_count": len(grouped.get(view_id, [])),
            "dimension_candidates": grouped.get(view_id, []),
        }
        for view_id, view in views.items()
    ]

    payload = {
        "schema": "dimension_extraction",
        "version": "0.1.0",
        "sample_id": sample_id,
        "source_drawing_ir": _stage_path(root, drawing_ir_path),
        "method": {
            "name": "dimension_ocr_prediction_normalizer",
            "version": "0.1.0",
            "role": "candidate_layer_before_dimension_geometry_binding",
        },
        "inputs": {
            "prediction_jsonl_paths": [_stage_or_raw_path(root, Path(path)) for path in prediction_jsonl_paths or []],
        },
        "views": view_blocks,
        "dimension_candidates": candidates,
        "unmatched_records": unmatched_records,
        "quality": _quality_block(view_blocks, candidates, unmatched_records),
    }

    target = Path(output_path) if output_path else root / "08.Multi-viewFeatureExtraction" / sample_id / "dimension_candidates.json"
    write_json(target, payload)
    return DimensionExtractionResult(
        sample_id=sample_id,
        output_path=target,
        view_count=len(view_blocks),
        dimension_count=len(candidates),
    )


def extract_dimensions_samples(
    *,
    dataflow_root: str | Path = "DataFlow",
    sample_id: str | None = None,
    prediction_jsonl_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    include_copy: bool = False,
    fail_fast: bool = False,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
) -> DimensionExtractionSummary:
    root = Path(dataflow_root)
    sample_ids = [sample_id] if sample_id else _iter_drawing_ir_sample_ids(root / "10.StructuredCADRepresentation")
    records: list[DimensionExtractionRecord] = []

    for current_sample_id in sample_ids:
        if not current_sample_id:
            continue
        if not include_copy and _looks_like_copy_sample(current_sample_id):
            records.append(DimensionExtractionRecord(sample_id=current_sample_id, output_path=None, skipped=True))
            continue
        try:
            result = extract_dimensions_sample(
                sample_id=current_sample_id,
                dataflow_root=root,
                prediction_jsonl_paths=prediction_jsonl_paths,
            )
            records.append(
                DimensionExtractionRecord(
                    sample_id=current_sample_id,
                    output_path=result.output_path,
                    view_count=result.view_count,
                    dimension_count=result.dimension_count,
                )
            )
        except Exception as exc:  # pragma: no cover - sample-specific data failures are reported in summary
            if fail_fast:
                raise
            records.append(DimensionExtractionRecord(sample_id=current_sample_id, output_path=None, error=str(exc)))

    out_root = root / "08.Multi-viewFeatureExtraction"
    csv_path = Path(output_csv) if output_csv else out_root / "dimension_extraction_summary.csv"
    json_path = Path(output_json) if output_json else out_root / "dimension_extraction_summary.json"
    _write_summary_csv(csv_path, records)
    _write_summary_json(json_path, records)
    return DimensionExtractionSummary(records=records, csv_path=csv_path, json_path=json_path)


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


def _dimension_ocr_records(
    *,
    root: Path,
    sample_id: str,
    views: dict[str, dict[str, Any]],
    view_lookup: dict[str, str],
    prediction_jsonl_paths: list[str | Path] | tuple[str | Path, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for jsonl_path in prediction_jsonl_paths:
        path = Path(jsonl_path)
        for line_number, record in _read_jsonl(path):
            if not isinstance(record, dict) or record.get("task") != "dimension_ocr":
                continue
            record_sample_id = record.get("sample_id")
            if record_sample_id and record_sample_id != sample_id:
                continue
            prediction = record.get("prediction") if isinstance(record.get("prediction"), dict) else {}
            dimensions = prediction.get("dimensions")
            if not isinstance(dimensions, list):
                continue
            input_images = [str(item) for item in record.get("input_images") or []]
            view_ids = _matched_view_ids(root, input_images, view_lookup)
            if not view_ids and record_sample_id == sample_id and len(views) == 1:
                view_ids = list(views)
            source = {
                "type": "dimension_ocr_prediction",
                "task": "dimension_ocr",
                "model": record.get("model"),
                "prediction_jsonl": _stage_or_raw_path(root, path),
                "line_number": line_number,
                "input_images": input_images,
                "record_sample_id": record_sample_id,
            }
            records.append(
                {
                    "view_ids": view_ids,
                    "dimensions": [item for item in dimensions if isinstance(item, dict)],
                    "source": source,
                    "unmatched": {
                        "reason": "dimension_ocr_record_not_matched_to_drawing_ir_view",
                        "prediction_jsonl": _stage_or_raw_path(root, path),
                        "line_number": line_number,
                        "input_images": input_images,
                        "record_sample_id": record_sample_id,
                    },
                }
            )
    return records


def _dimension_candidate(
    *,
    item: dict[str, Any],
    view: dict[str, Any],
    view_id: str,
    source: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    text = str(item.get("text") or item.get("normalized") or "").strip()
    normalized = str(item.get("normalized") or text).strip()
    dimension_type = _dimension_type(str(item.get("type") or "unknown"), normalized)
    quantity = _leading_quantity(normalized)
    value = _dimension_value(normalized, dimension_type, quantity)
    bbox = _bbox_or_none(item.get("bbox"))
    confidence = item.get("confidence")

    review_reasons = ["dimension_ocr_candidate_needs_validation", "dimension_geometry_binding_not_built"]
    if not bbox:
        review_reasons.append("missing_dimension_bbox")
    if dimension_type == "unknown":
        review_reasons.append("unknown_dimension_type")

    return {
        "id": f"{view_id}_dimension_{index:03d}",
        "type": "dimension_candidate",
        "semantic_status": "candidate",
        "view_id": view_id,
        "text": text,
        "normalized": normalized,
        "dimension_type": dimension_type,
        "bbox": bbox,
        "bbox_on_page": _bbox_on_page(view, bbox),
        "value": value,
        "quantity": quantity,
        "unit": _unit_for_type(dimension_type),
        "tolerance": _tolerance(normalized),
        "confidence": float(confidence) if confidence is not None else 0.35,
        "needs_manual_review": True,
        "review_reasons": review_reasons,
        "source": source,
        "evidence": [f"raw_text={text}", f"normalized={normalized}"],
    }


def _dimension_type(source_type: str, normalized: str) -> str:
    lowered_source = source_type.lower().strip()
    if lowered_source and lowered_source != "unknown":
        return lowered_source
    upper = normalized.upper().replace(" ", "")
    if any(symbol in normalized for symbol in ("Φ", "Ø", "⌀")) or "DIA" in upper:
        return "diameter"
    if re.search(r"(^|[^A-Z])R\s*\d", upper):
        return "radius"
    if re.search(r"(^|\d[-X×]?)M\d", upper):
        return "thread"
    if re.search(r"(^|\d[-X×]?)C\d", upper):
        return "chamfer"
    if "°" in normalized or "DEG" in upper:
        return "angle"
    if "RA" in upper or "粗糙" in normalized:
        return "surface_roughness"
    if "±" in normalized or re.search(r"\d", normalized):
        return "linear"
    return "unknown"


def _leading_quantity(text: str) -> int | None:
    compact = text.strip().replace("×", "x")
    match = re.match(r"^(\d+)\s*(?:x|-)\s*(.+)$", compact, flags=re.IGNORECASE)
    if not match:
        return None
    rest = match.group(2).strip().upper()
    if rest.startswith(("Φ", "Ø", "⌀", "M", "C", "R")) or re.match(r"^\d", rest):
        return int(match.group(1))
    return None


def _dimension_value(text: str, dimension_type: str, quantity: int | None) -> float | None:
    compact = text.strip().replace("×", "x")
    if dimension_type == "diameter":
        match = re.search(r"[ΦØ⌀]\s*([0-9]+(?:\.[0-9]+)?)", compact, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    if dimension_type == "thread":
        match = re.search(r"M\s*([0-9]+(?:\.[0-9]+)?)", compact, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    if dimension_type == "chamfer":
        match = re.search(r"C\s*([0-9]+(?:\.[0-9]+)?)", compact, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    if dimension_type == "radius":
        match = re.search(r"R\s*([0-9]+(?:\.[0-9]+)?)", compact, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    numbers = [float(value) for value in re.findall(r"[0-9]+(?:\.[0-9]+)?", compact)]
    if not numbers:
        return None
    if quantity is not None and len(numbers) >= 2 and int(numbers[0]) == quantity:
        return numbers[1]
    return numbers[0]


def _tolerance(text: str) -> dict[str, float] | None:
    symmetric = re.search(r"±\s*([0-9]+(?:\.[0-9]+)?)", text)
    if symmetric:
        value = float(symmetric.group(1))
        return {"plus": value, "minus": value}
    return None


def _bbox_or_none(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    x1, y1, x2, y2 = [int(item) for item in value]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_on_page(view: dict[str, Any], bbox: list[int] | None) -> list[int] | None:
    if bbox is None:
        return None
    view_bbox = view.get("bbox_on_page") or view.get("bbox")
    if not isinstance(view_bbox, list) or len(view_bbox) != 4:
        return None
    return [
        int(view_bbox[0]) + bbox[0],
        int(view_bbox[1]) + bbox[1],
        int(view_bbox[0]) + bbox[2],
        int(view_bbox[1]) + bbox[3],
    ]


def _unit_for_type(dimension_type: str) -> str:
    if dimension_type == "angle":
        return "deg"
    if dimension_type == "surface_roughness":
        return "um"
    return "mm"


def _quality_block(
    views: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    unmatched_records: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = ["dimension_candidates_need_validation", "dimension_geometry_binding_not_built"]
    if not candidates:
        reasons.append("no_dimension_candidates")
    if unmatched_records:
        reasons.append("unmatched_dimension_ocr_records")

    return {
        "view_count": len(views),
        "views_with_dimensions": [view["view_id"] for view in views if view["dimension_candidate_count"] > 0],
        "views_without_dimensions": [view["view_id"] for view in views if view["dimension_candidate_count"] == 0],
        "dimension_candidate_count": len(candidates),
        "unmatched_record_count": len(unmatched_records),
        "needs_manual_review": True,
        "review_reasons": reasons,
        "ready_for_dimension_geometry_binding": False,
        "blocking_items": [
            "dimension_candidates_not_validated",
            "dimension_geometry_binding_not_built",
            "cross_view_relations_not_built",
        ],
    }


def _view_lookup(root: Path, views: dict[str, dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for view_id, view in views.items():
        image_clean = view.get("image_clean")
        if isinstance(image_clean, str) and image_clean:
            for key in _path_keys(root, image_clean):
                lookup[key] = view_id
    return lookup


def _matched_view_ids(root: Path, input_images: list[str], view_lookup: dict[str, str]) -> list[str]:
    matched: list[str] = []
    for image in input_images:
        for key in _path_keys(root, image):
            view_id = view_lookup.get(key)
            if view_id and view_id not in matched:
                matched.append(view_id)
    return matched


def _path_keys(root: Path, path_text: str) -> set[str]:
    normalized = path_text.replace("\\", "/").strip()
    keys = {normalized.lower()}
    path = Path(normalized)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    elif path.parts and path.parts[0] == root.name:
        candidates.append(root.parent / path)
    else:
        candidates.append(root / path)

    for candidate in candidates:
        keys.add(candidate.as_posix().lower())
        try:
            keys.add(candidate.relative_to(root.parent).as_posix().lower())
        except ValueError:
            pass
        try:
            keys.add((Path(root.name) / candidate.relative_to(root)).as_posix().lower())
        except ValueError:
            pass
    return keys


def _read_jsonl(path: Path) -> list[tuple[int, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction JSONL file: {path}")
    records: list[tuple[int, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            records.append((line_number, json.loads(line)))
    return records


def _crop_size(view: dict[str, Any]) -> dict[str, int]:
    value = view.get("crop_size")
    if isinstance(value, dict) and value.get("width") and value.get("height"):
        return {"width": int(value["width"]), "height": int(value["height"])}
    bbox = view.get("bbox_on_page") or view.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return {"width": int(bbox[2]) - int(bbox[0]), "height": int(bbox[3]) - int(bbox[1])}
    return {"width": 1, "height": 1}


def _iter_drawing_ir_sample_ids(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "drawing_ir.json").exists())


def _looks_like_copy_sample(sample_id: str) -> bool:
    lowered = sample_id.lower()
    return lowered.endswith("-copy") or lowered.endswith("_copy") or " copy" in lowered


def _write_summary_csv(path: Path, records: list[DimensionExtractionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "output_path", "view_count", "dimension_count", "skipped", "error"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "output_path": record.output_path.as_posix() if record.output_path else "",
                    "view_count": record.view_count,
                    "dimension_count": record.dimension_count,
                    "skipped": record.skipped,
                    "error": record.error or "",
                }
            )


def _write_summary_json(path: Path, records: list[DimensionExtractionRecord]) -> None:
    write_json(
        path,
        {
            "schema": "dimension_extraction_summary",
            "version": "0.1.0",
            "records": [
                {
                    "sample_id": record.sample_id,
                    "output_path": record.output_path.as_posix() if record.output_path else None,
                    "view_count": record.view_count,
                    "dimension_count": record.dimension_count,
                    "skipped": record.skipped,
                    "error": record.error,
                }
                for record in records
            ],
        },
    )


def _stage_or_raw_path(root: Path, path: Path) -> str:
    try:
        return _stage_path(root, path)
    except ValueError:
        return path.as_posix()


def _stage_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.as_posix()
    return (Path(root.name) / relative).as_posix()
