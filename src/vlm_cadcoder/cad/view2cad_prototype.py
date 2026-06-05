from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_cadcoder.utils.json_utils import read_json, write_json


@dataclass(frozen=True)
class View2CADPrototypeResult:
    sample_id: str
    manifest_path: Path
    drawing_ir_path: Path
    modeling_plan_path: Path
    cadquery_prompt_path: Path


@dataclass(frozen=True)
class View2CADPrototypeConfig:
    dataflow_root: Path = Path("DataFlow")
    external_crop_set: str = "testView2CAD"
    experiments_root: Path = Path("experiments/external_crops")
    output_set: str = "testView2CAD"


def build_view2cad_prototype(
    sample_id: str,
    config: View2CADPrototypeConfig | None = None,
) -> View2CADPrototypeResult:
    cfg = config or View2CADPrototypeConfig()
    root = Path(cfg.dataflow_root)

    manifest = _build_manifest(sample_id, cfg)
    predictions = _collect_predictions(Path(cfg.experiments_root), sample_id)
    drawing_ir = _build_minimal_drawing_ir(sample_id, manifest, predictions)
    modeling_plan = _build_modeling_plan(sample_id, manifest, drawing_ir, predictions)

    structured_dir = root / "10.StructuredCADRepresentation" / cfg.output_set / sample_id
    cad_dir = root / "11.CADProgram" / cfg.output_set / sample_id
    manifest_path = structured_dir / "external_crop_manifest.json"
    drawing_ir_path = structured_dir / "minimal_drawing_ir.json"
    modeling_plan_path = structured_dir / "modeling_plan.json"
    cadquery_prompt_path = cad_dir / "cadquery_generation_prompt.md"

    write_json(manifest_path, manifest)
    write_json(drawing_ir_path, drawing_ir)
    write_json(modeling_plan_path, modeling_plan)
    _write_cadquery_prompt(cadquery_prompt_path, drawing_ir, modeling_plan)

    return View2CADPrototypeResult(
        sample_id=sample_id,
        manifest_path=manifest_path,
        drawing_ir_path=drawing_ir_path,
        modeling_plan_path=modeling_plan_path,
        cadquery_prompt_path=cadquery_prompt_path,
    )


def _build_manifest(sample_id: str, cfg: View2CADPrototypeConfig) -> dict[str, Any]:
    root = Path(cfg.dataflow_root)
    crop_sample_dir = root / "06.SingleViews" / cfg.external_crop_set / sample_id
    crop_img_dir = crop_sample_dir / "cut-img"
    crop_json_dir = crop_sample_dir / "cut-json"
    clean_image = root / "04.CleanPNG" / cfg.external_crop_set / f"{sample_id}.png"
    step_path = _find_step_path(root / "01.RawPDFWithSTEP" / cfg.external_crop_set, sample_id)

    views = []
    for crop_json_path in sorted(crop_json_dir.glob("*.json")):
        crop_meta = read_json(crop_json_path)
        image_name = crop_meta.get("imagePath") or f"{crop_json_path.stem}.png"
        image_path = crop_img_dir / image_name
        views.append(_build_view_manifest(crop_json_path, image_path, crop_meta))

    missing_inputs = []
    if step_path is None:
        missing_inputs.append("step_ground_truth")
    if not clean_image.exists():
        missing_inputs.append("clean_image")
    if not crop_img_dir.exists():
        missing_inputs.append("external_crop_images")
    if not crop_json_dir.exists():
        missing_inputs.append("external_crop_json")
    if not views:
        missing_inputs.append("view_crops")

    return {
        "sample_id": sample_id,
        "source": {
            "external_crop_set": cfg.external_crop_set,
            "external_crop_root": _path(crop_sample_dir),
            "clean_image": _path(clean_image) if clean_image.exists() else None,
            "step_ground_truth": _path(step_path) if step_path else None,
            "experiments_root": _path(Path(cfg.experiments_root)),
        },
        "views": views,
        "missing_inputs": missing_inputs,
        "usage_note": (
            "External crops are prototype inputs for downstream View2CAD validation; "
            "they are not evidence for automatic view detection performance."
        ),
    }


def _build_view_manifest(crop_json_path: Path, image_path: Path, crop_meta: dict[str, Any]) -> dict[str, Any]:
    min_x = int(round(float(crop_meta.get("group_min_x", 0))))
    min_y = int(round(float(crop_meta.get("group_min_y", 0))))
    width = int(crop_meta.get("imageWidth", 0))
    height = int(crop_meta.get("imageHeight", 0))
    label_counts: dict[str, int] = {}
    for shape in crop_meta.get("shapes", []):
        label = str(shape.get("label", "unknown"))
        label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "id": crop_json_path.stem,
        "type": "unknown",
        "bbox": [min_x, min_y, min_x + width, min_y + height],
        "image_raw": _path(image_path),
        "source_json": _path(crop_json_path),
        "image_size": [width, height],
        "external_crop": True,
        "metadata": {
            "group_id": crop_meta.get("group_id"),
            "pmi_label_counts": label_counts,
            "num_shapes": len(crop_meta.get("shapes", [])),
        },
    }


def _find_step_path(directory: Path, sample_id: str) -> Path | None:
    if not directory.exists():
        return None
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and candidate.stem == sample_id and candidate.suffix.lower() in {".step", ".stp"}:
            return candidate
    return None


def _collect_predictions(experiments_root: Path, sample_id: str) -> list[dict[str, Any]]:
    if not experiments_root.exists():
        return []

    records: list[dict[str, Any]] = []
    for jsonl_path in sorted(experiments_root.rglob("predictions.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                input_images = [str(path) for path in record.get("input_images", [])]
                if any(sample_id in image_path for image_path in input_images):
                    item = {
                        "task": record.get("task"),
                        "model": record.get("model"),
                        "input_images": input_images,
                        "prediction": record.get("prediction"),
                        "is_json_valid": record.get("is_json_valid"),
                        "latency_sec": record.get("latency_sec"),
                        "source_prediction_file": _path(jsonl_path),
                    }
                    records.append(item)
    return records


def _build_minimal_drawing_ir(
    sample_id: str,
    manifest: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    dimensions = _collect_dimensions(predictions)
    feature_candidates = _collect_feature_candidates(predictions)

    return {
        "schema": "minimal_drawing_ir",
        "version": "0.1.0",
        "sample_id": sample_id,
        "sheet": {
            "coordinate_system": "image_xy_top_left",
            "clean_image": manifest["source"]["clean_image"],
            "step_ground_truth": manifest["source"]["step_ground_truth"],
            "input_mode": "external_crops_plus_clean_image",
        },
        "views": manifest["views"],
        "dimensions": dimensions,
        "feature_candidates": feature_candidates,
        "constraints": [],
        "provenance": {
            "external_crop_manifest": "external_crop_manifest.json",
            "num_model_prediction_records": len(predictions),
            "prediction_tasks": sorted({str(item.get("task")) for item in predictions if item.get("task")}),
        },
        "status": {
            "ready_for_modeling_plan": bool(manifest["views"]),
            "needs_human_review": True,
            "review_reason": (
                "Dimensions, view types, feature locations, and cross-view bindings are not yet verified."
            ),
        },
    }


def _collect_dimensions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimensions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counter = 1
    for record in predictions:
        if record.get("task") != "dimension_ocr":
            continue
        prediction = record.get("prediction") or {}
        for item in prediction.get("dimensions", []):
            text = str(item.get("text") or "")
            normalized = str(item.get("normalized") or text)
            key = (text, normalized)
            if not text or key in seen:
                continue
            seen.add(key)
            quantity = _leading_quantity(normalized)
            dimensions.append(
                {
                    "id": f"d{counter:03d}",
                    "text": text,
                    "normalized": normalized,
                    "dimension_type": item.get("type", "unknown"),
                    "bbox": None,
                    "view_id": None,
                    "value": _dimension_value(normalized, quantity),
                    "quantity": quantity,
                    "unit": "mm",
                    "source": {
                        "task": record.get("task"),
                        "model": record.get("model"),
                        "input_images": record.get("input_images", []),
                    },
                }
            )
            counter += 1
    return dimensions


def _collect_feature_candidates(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    for record in predictions:
        if record.get("task") != "feature_count":
            continue
        prediction = record.get("prediction") or {}
        feature_counts = prediction.get("feature_counts", {})
        for feature_type, count_value in feature_counts.items():
            count = int(count_value or 0)
            counts[feature_type] = max(counts.get(feature_type, 0), count)
            if count > 0:
                sources.setdefault(feature_type, []).append(
                    {
                        "model": record.get("model"),
                        "input_images": record.get("input_images", []),
                    }
                )
        for item in prediction.get("evidence", []):
            text = str(item)
            for feature_type, count in feature_counts.items():
                if int(count or 0) > 0:
                    evidence.setdefault(feature_type, [])
                    if text not in evidence[feature_type]:
                        evidence[feature_type].append(text)

    candidates = []
    for index, feature_type in enumerate(sorted(counts), start=1):
        candidates.append(
            {
                "id": f"f{index:03d}",
                "type": feature_type,
                "view_id": None,
                "count": counts[feature_type],
                "evidence": evidence.get(feature_type, []),
                "source": sources.get(feature_type, []),
                "status": "candidate" if counts[feature_type] > 0 else "not_detected",
            }
        )
    return candidates


def _build_modeling_plan(
    sample_id: str,
    manifest: dict[str, Any],
    drawing_ir: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    nonzero_features = {
        item["type"]: item["count"] for item in drawing_ir["feature_candidates"] if int(item.get("count") or 0) > 0
    }
    operations: list[dict[str, Any]] = [
        {
            "id": "op001",
            "operation": "create_base_solid",
            "cadquery_hint": "cq.Workplane(...).box(...) or sketch/extrude from the main view",
            "status": "needs_human_or_model_confirmation",
            "requires": ["base_profile", "extrusion_depth", "main_view"],
        }
    ]

    next_id = 2
    operation_specs = [
        ("through_hole", "cut_through_hole_pattern", ["hole_diameter", "hole_centers", "target_face"]),
        ("counterbore", "cut_counterbore_or_spotface_pattern", ["counterbore_diameter", "counterbore_depth"]),
        ("blind_hole", "cut_blind_hole_pattern", ["hole_diameter", "hole_depth", "hole_centers"]),
        ("side_hole", "cut_side_hole_pattern", ["side_face", "hole_axis", "hole_centers"]),
        ("slot", "cut_slot", ["slot_profile", "slot_position"]),
        ("chamfer", "apply_chamfer", ["edge_selection", "chamfer_size"]),
        ("fillet", "apply_fillet", ["edge_selection", "fillet_radius"]),
    ]
    for feature_type, operation, required_parameters in operation_specs:
        count = int(nonzero_features.get(feature_type, 0))
        if count <= 0:
            continue
        operations.append(
            {
                "id": f"op{next_id:03d}",
                "operation": operation,
                "feature_type": feature_type,
                "count": count,
                "status": "candidate_from_vlm",
                "requires": required_parameters,
            }
        )
        next_id += 1

    if _has_thread_evidence(drawing_ir, predictions):
        operations.append(
            {
                "id": f"op{next_id:03d}",
                "operation": "represent_threaded_holes",
                "feature_type": "thread",
                "status": "candidate_from_dimension_text",
                "requires": ["thread_spec", "thread_axis", "thread_locations"],
                "cadquery_hint": "Model minor holes geometrically and keep thread metadata in the IR.",
            }
        )

    return {
        "schema": "view2cad_modeling_plan",
        "version": "0.1.0",
        "sample_id": sample_id,
        "target_backend": "CadQuery",
        "reference_step": manifest["source"]["step_ground_truth"],
        "input_artifacts": {
            "minimal_drawing_ir": "minimal_drawing_ir.json",
            "external_crop_manifest": "external_crop_manifest.json",
            "clean_image": manifest["source"]["clean_image"],
        },
        "readiness": {
            "can_prompt_for_cadquery": bool(drawing_ir["views"] and (drawing_ir["dimensions"] or nonzero_features)),
            "can_execute_without_review": False,
            "blocking_items": [
                "base solid dimensions are not fully bound to views",
                "feature positions are not yet bound to image geometry",
                "view types and projection relationships are not confirmed",
            ],
        },
        "operations": operations,
        "validation_plan": [
            "Run CadQuery script and export STEP.",
            "Compare operation success and exported STEP availability.",
            "Compare feature counts against minimal DrawingIR candidates.",
            "Use reference STEP for later geometry/topology validation after a CAD checker is added.",
        ],
    }


def _has_thread_evidence(drawing_ir: dict[str, Any], predictions: list[dict[str, Any]]) -> bool:
    for dimension in drawing_ir["dimensions"]:
        if "M" in str(dimension.get("normalized", "")).upper():
            return True
    for record in predictions:
        prediction = record.get("prediction") or {}
        evidence_text = " ".join(str(item) for item in prediction.get("evidence", []))
        if "M" in evidence_text.upper():
            return True
    return False


def _write_cadquery_prompt(path: Path, drawing_ir: dict[str, Any], modeling_plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt = (
        "# CadQuery Generation Prompt\n\n"
        "You are generating a prototype CadQuery script from a minimal drawing IR.\n"
        "Use the script as a prototype only. Do not invent missing dimensions silently; "
        "define uncertain values as named parameters and mark them for review.\n\n"
        "## Requirements\n\n"
        "- Generate Python CadQuery code.\n"
        "- Use parameter names for every drawing-derived dimension.\n"
        "- Include comments for dimensions or feature locations that need human confirmation.\n"
        "- Export a STEP file when executed.\n"
        "- Keep thread features as geometric pilot holes plus metadata comments unless full thread geometry is required.\n\n"
        "## Minimal Drawing IR\n\n"
        "```json\n"
        f"{json.dumps(drawing_ir, ensure_ascii=False, indent=2)}\n"
        "```\n\n"
        "## Modeling Plan\n\n"
        "```json\n"
        f"{json.dumps(modeling_plan, ensure_ascii=False, indent=2)}\n"
        "```\n"
    )
    path.write_text(prompt, encoding="utf-8", newline="\n")


def _first_number(text: str) -> float | None:
    token = ""
    started = False
    for char in text:
        if char.isdigit() or (char == "." and started):
            token += char
            started = True
        elif started:
            break
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _dimension_value(text: str, quantity: int | None) -> float | None:
    if quantity is None:
        return _first_number(text)

    stripped = text.strip()
    digits = ""
    for char in stripped:
        if char.isdigit():
            digits += char
            continue
        break
    tail = stripped[len(digits) :].lstrip()
    if tail and tail[0] in {"x", "X", "-", "×"}:
        return _first_number(tail[1:])
    return _first_number(text)


def _leading_quantity(text: str) -> int | None:
    stripped = text.strip()
    digits = ""
    for char in stripped:
        if char.isdigit():
            digits += char
            continue
        break
    if not digits:
        return None
    tail = stripped[len(digits) :].lstrip()
    if tail and tail[0] in {"x", "X", "-", "×"}:
        return int(digits)
    return None


def _path(path: Path) -> str:
    return path.as_posix()
