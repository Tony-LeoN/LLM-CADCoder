from __future__ import annotations

import json
from pathlib import Path

from vlm_cadcoder.cad.view2cad_prototype import View2CADPrototypeConfig, build_view2cad_prototype
from vlm_cadcoder.utils.json_utils import read_json


def test_build_view2cad_prototype_collects_external_crops_and_predictions(tmp_path: Path) -> None:
    sample_id = "part-001"
    _write_test_inputs(tmp_path, sample_id)

    result = build_view2cad_prototype(
        sample_id,
        View2CADPrototypeConfig(
            dataflow_root=tmp_path / "DataFlow",
            experiments_root=tmp_path / "experiments" / "external_crops",
        ),
    )

    manifest = read_json(result.manifest_path)
    drawing_ir = read_json(result.drawing_ir_path)
    modeling_plan = read_json(result.modeling_plan_path)

    assert manifest["source"]["step_ground_truth"].endswith("part-001.STEP")
    assert manifest["views"][0]["bbox"] == [100, 200, 740, 680]
    assert manifest["views"][0]["metadata"]["pmi_label_counts"] == {"PMI_Line": 1, "PMI_Chamfer": 1}
    assert drawing_ir["dimensions"][0]["normalized"] == "4-M3"
    assert drawing_ir["dimensions"][0]["quantity"] == 4
    assert drawing_ir["dimensions"][0]["value"] == 3.0
    assert _feature_count(drawing_ir, "through_hole") == 4
    assert _feature_count(drawing_ir, "counterbore") == 4
    assert modeling_plan["readiness"]["can_prompt_for_cadquery"] is True
    assert any(item["operation"] == "represent_threaded_holes" for item in modeling_plan["operations"])
    assert result.cadquery_prompt_path.exists()


def _write_test_inputs(root: Path, sample_id: str) -> None:
    dataflow = root / "DataFlow"
    crop_root = dataflow / "06.SingleViews" / "testView2CAD" / sample_id
    crop_img = crop_root / "cut-img"
    crop_json = crop_root / "cut-json"
    crop_img.mkdir(parents=True)
    crop_json.mkdir(parents=True)
    (crop_img / f"{sample_id}_crop_1.png").write_bytes(b"fake png")
    (dataflow / "04.CleanPNG" / "testView2CAD").mkdir(parents=True)
    (dataflow / "04.CleanPNG" / "testView2CAD" / f"{sample_id}.png").write_bytes(b"fake clean png")
    (dataflow / "01.RawPDFWithSTEP" / "testView2CAD").mkdir(parents=True)
    (dataflow / "01.RawPDFWithSTEP" / "testView2CAD" / f"{sample_id}.STEP").write_text("STEP", encoding="utf-8")

    crop_meta = {
        "group_min_x": 100,
        "group_min_y": 200,
        "imageWidth": 640,
        "imageHeight": 480,
        "imagePath": f"{sample_id}_crop_1.png",
        "shapes": [{"label": "PMI_Line"}, {"label": "PMI_Chamfer"}],
    }
    (crop_json / f"{sample_id}_crop_1.json").write_text(json.dumps(crop_meta), encoding="utf-8")

    prediction_dir = root / "experiments" / "external_crops" / "run"
    prediction_dir.mkdir(parents=True)
    records = [
        {
            "task": "dimension_ocr",
            "model": "mock",
            "input_images": [f"DataFlow/06.SingleViews/testView2CAD/{sample_id}/cut-img/{sample_id}_crop_1.png"],
            "prediction": {
                "dimensions": [
                    {
                        "text": "4-M3",
                        "normalized": "4-M3",
                        "type": "thread",
                    }
                ]
            },
            "is_json_valid": True,
        },
        {
            "task": "feature_count",
            "model": "mock",
            "input_images": [f"DataFlow/06.SingleViews/testView2CAD/{sample_id}/cut-img/{sample_id}_crop_1.png"],
            "prediction": {
                "feature_counts": {
                    "through_hole": 4,
                    "counterbore": 4,
                    "chamfer": 0,
                },
                "evidence": ["4 x Phi 4.5 through / Phi 8 depth 4.6"],
            },
            "is_json_valid": True,
        },
    ]
    (prediction_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _feature_count(drawing_ir: dict, feature_type: str) -> int:
    for candidate in drawing_ir["feature_candidates"]:
        if candidate["type"] == feature_type:
            return int(candidate["count"])
    raise AssertionError(f"Missing feature type: {feature_type}")
