from __future__ import annotations

from pathlib import Path

from vlm_cadcoder.cad.cadquery_draft import CadQueryDraftConfig, build_cadquery_draft
from vlm_cadcoder.utils.json_utils import read_json, write_json


def test_build_cadquery_draft_writes_parameters_and_script(tmp_path: Path) -> None:
    sample_id = "part-001"
    structured_dir = tmp_path / "DataFlow" / "10.StructuredCADRepresentation" / "testView2CAD" / sample_id
    structured_dir.mkdir(parents=True)
    write_json(
        structured_dir / "minimal_drawing_ir.json",
        {
            "sample_id": sample_id,
            "sheet": {"step_ground_truth": "DataFlow/01.RawPDFWithSTEP/testView2CAD/part-001.STEP"},
            "dimensions": [{"normalized": "29.00 ±0.05", "value": 29.0}],
            "feature_candidates": [
                {
                    "type": "through_hole",
                    "count": 4,
                    "evidence": ["4 x Φ5.5 完全贯穿 / Φ8 深 4.6"],
                },
                {
                    "type": "counterbore",
                    "count": 4,
                    "evidence": ["4 x Φ5.5 完全贯穿 / Φ8 深 4.6"],
                },
                {"type": "chamfer", "count": 4, "evidence": ["4-C2"]},
            ],
        },
    )
    write_json(
        structured_dir / "modeling_plan.json",
        {
            "sample_id": sample_id,
            "operations": [{"id": "op001"}, {"id": "op002"}],
        },
    )

    result = build_cadquery_draft(
        sample_id,
        CadQueryDraftConfig(dataflow_root=tmp_path / "DataFlow"),
    )

    parameters = read_json(result.parameter_path)
    script = result.script_path.read_text(encoding="utf-8")

    assert _parameter_value(parameters, "base_depth") == 29.0
    assert _parameter_value(parameters, "through_hole_diameter") == 5.5
    assert _parameter_value(parameters, "counterbore_diameter") == 8.0
    assert _parameter_value(parameters, "counterbore_depth") == 4.6
    assert parameters["review"]["required"] is True
    assert "cq.Workplane" in script
    assert "cq.exporters.export" in script
    assert ".cboreHole(through_hole_diameter, counterbore_diameter, counterbore_depth)" in script


def _parameter_value(parameter_review: dict, name: str) -> float:
    for item in parameter_review["parameters"]:
        if item["name"] == name:
            return float(item["value"])
    raise AssertionError(f"Missing parameter: {name}")
