from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_cadcoder.utils.json_utils import read_json, write_json


@dataclass(frozen=True)
class CadQueryDraftResult:
    sample_id: str
    parameter_path: Path
    script_path: Path


@dataclass(frozen=True)
class CadQueryDraftConfig:
    dataflow_root: Path = Path("DataFlow")
    input_set: str = "testView2CAD"
    output_set: str = "testView2CAD"
    part_family: str = "rectangular_plate"


def build_cadquery_draft(
    sample_id: str,
    config: CadQueryDraftConfig | None = None,
) -> CadQueryDraftResult:
    cfg = config or CadQueryDraftConfig()
    root = Path(cfg.dataflow_root)
    structured_dir = root / "10.StructuredCADRepresentation" / cfg.input_set / sample_id
    cad_dir = root / "11.CADProgram" / cfg.output_set / sample_id
    drawing_ir = read_json(structured_dir / "minimal_drawing_ir.json")
    modeling_plan = read_json(structured_dir / "modeling_plan.json")

    parameter_review = _build_parameter_review(sample_id, drawing_ir, modeling_plan, cfg.part_family)
    script = _render_cadquery_script(sample_id, parameter_review)

    parameter_path = cad_dir / "cadquery_parameters.json"
    script_path = cad_dir / "cadquery_draft.py"
    write_json(parameter_path, parameter_review)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8", newline="\n")
    return CadQueryDraftResult(sample_id=sample_id, parameter_path=parameter_path, script_path=script_path)


def _build_parameter_review(
    sample_id: str,
    drawing_ir: dict[str, Any],
    modeling_plan: dict[str, Any],
    part_family: str,
) -> dict[str, Any]:
    dimensions = drawing_ir.get("dimensions", [])
    features = {item.get("type"): int(item.get("count") or 0) for item in drawing_ir.get("feature_candidates", [])}
    evidence_text = " | ".join(
        evidence
        for item in drawing_ir.get("feature_candidates", [])
        for evidence in item.get("evidence", [])
        if isinstance(evidence, str)
    )

    parameters = [
        _parameter("base_length", 50.0, "mm", "needs_review", "rectangular_plate_default", "Overall length."),
        _parameter("base_width", 45.0, "mm", "needs_review", "rectangular_plate_default", "Overall width."),
        _parameter(
            "base_depth",
            _dimension_value_by_text(dimensions, "29", fallback=29.0),
            "mm",
            "candidate",
            "dimension_ocr_or_default",
            "Extrusion depth / thickness.",
        ),
        _parameter("hole_pitch_x", 38.0, "mm", "needs_review", "rectangular_plate_default", "X pitch of 4-hole pattern."),
        _parameter("hole_pitch_y", 30.0, "mm", "needs_review", "rectangular_plate_default", "Y pitch of 4-hole pattern."),
        _parameter(
            "through_hole_diameter",
            _best_diameter(evidence_text, preferred=[5.5, 4.5, 3.4]),
            "mm",
            "candidate",
            "feature_evidence",
            "Diameter for the visible through-hole pattern.",
        ),
        _parameter(
            "counterbore_diameter",
            _best_diameter(evidence_text, preferred=[8.0, 6.0]),
            "mm",
            "needs_review",
            "feature_evidence",
            "Counterbore or spotface diameter, if present.",
        ),
        _parameter(
            "counterbore_depth",
            _number_after_depth(evidence_text, fallback=4.6),
            "mm",
            "needs_review",
            "feature_evidence",
            "Counterbore depth, if present.",
        ),
        _parameter("chamfer_size", 2.0, "mm", "needs_review", "feature_evidence", "Corner chamfer size."),
    ]

    return {
        "schema": "cadquery_parameter_review",
        "version": "0.1.0",
        "sample_id": sample_id,
        "part_family": part_family,
        "source_artifacts": {
            "minimal_drawing_ir": "DataFlow/10.StructuredCADRepresentation/testView2CAD/"
            f"{sample_id}/minimal_drawing_ir.json",
            "modeling_plan": "DataFlow/10.StructuredCADRepresentation/testView2CAD/"
            f"{sample_id}/modeling_plan.json",
            "reference_step": drawing_ir.get("sheet", {}).get("step_ground_truth"),
        },
        "parameters": parameters,
        "feature_counts": {
            "through_hole": features.get("through_hole", 0),
            "counterbore": features.get("counterbore", 0),
            "chamfer": features.get("chamfer", 0),
        },
        "operation_ids": [item.get("id") for item in modeling_plan.get("operations", [])],
        "review": {
            "required": True,
            "reason": "Base dimensions and feature positions are not yet bound by a constraint graph.",
        },
    }


def _render_cadquery_script(sample_id: str, parameter_review: dict[str, Any]) -> str:
    values = {item["name"]: item["value"] for item in parameter_review["parameters"]}
    feature_counts = parameter_review.get("feature_counts", {})
    use_counterbore = int(feature_counts.get("counterbore", 0) or 0) > 0
    chamfer_count = int(feature_counts.get("chamfer", 0) or 0)
    hole_points = [
        (-values["hole_pitch_x"] / 2, -values["hole_pitch_y"] / 2),
        (-values["hole_pitch_x"] / 2, values["hole_pitch_y"] / 2),
        (values["hole_pitch_x"] / 2, -values["hole_pitch_y"] / 2),
        (values["hole_pitch_x"] / 2, values["hole_pitch_y"] / 2),
    ]
    hole_call = (
        f".cboreHole(through_hole_diameter, counterbore_diameter, counterbore_depth)"
        if use_counterbore
        else ".hole(through_hole_diameter)"
    )
    chamfer_block = (
        "if chamfer_size > 0:\n"
        "    # REVIEW: edge selector assumes four long outside edges of a rectangular plate.\n"
        "    result = result.edges('|Z').chamfer(chamfer_size)\n"
        if chamfer_count > 0
        else ""
    )

    return f'''from __future__ import annotations

from pathlib import Path

import cadquery as cq


# Prototype generated from minimal DrawingIR for {sample_id}.
# REVIEW REQUIRED: parameters below are candidates, not final bound constraints.
base_length = {values["base_length"]!r}
base_width = {values["base_width"]!r}
base_depth = {values["base_depth"]!r}
hole_pitch_x = {values["hole_pitch_x"]!r}
hole_pitch_y = {values["hole_pitch_y"]!r}
through_hole_diameter = {values["through_hole_diameter"]!r}
counterbore_diameter = {values["counterbore_diameter"]!r}
counterbore_depth = {values["counterbore_depth"]!r}
chamfer_size = {values["chamfer_size"]!r}

hole_points = {hole_points!r}

result = cq.Workplane("XY").box(base_length, base_width, base_depth)

# REVIEW: hole direction assumes the annotated main face is normal to Z.
result = result.faces(">Z").workplane().pushPoints(hole_points){hole_call}

{chamfer_block}# Thread callouts are kept as metadata until thread axes and locations are bound.
thread_metadata = {{
    "status": "not_geometrically_modeled",
    "reason": "thread locations require view and constraint binding",
}}

output_path = Path(__file__).with_name("{sample_id}_cadquery_draft.step")
cq.exporters.export(result, str(output_path))
print(output_path)
'''


def _parameter(name: str, value: float, unit: str, status: str, source: str, note: str) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "status": status,
        "source": source,
        "note": note,
    }


def _dimension_value_by_text(dimensions: list[dict[str, Any]], needle: str, fallback: float) -> float:
    for item in dimensions:
        if needle in str(item.get("normalized", "")) and item.get("value") is not None:
            return float(item["value"])
    return fallback


def _best_diameter(text: str, preferred: list[float]) -> float:
    diameters = [float(match) for match in re.findall(r"[ΦØ]\s*(\d+(?:\.\d+)?)", text)]
    for value in preferred:
        if value in diameters:
            return value
    return diameters[0] if diameters else preferred[0]


def _number_after_depth(text: str, fallback: float) -> float:
    match = re.search(r"(?:深|depth)\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return float(match.group(1)) if match else fallback
