from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from vlm_cadcoder.dataflow.dimension_geometry_binding import bind_dimensions_to_geometry_sample


def test_bind_dimensions_to_geometry_sample_writes_rule_scaffold(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_inputs(dataflow, "Part-Bind")

    result = bind_dimensions_to_geometry_sample(sample_id="Part-Bind", dataflow_root=dataflow)

    assert result.output_path == dataflow / "09.Cross-viewGeometricReasoning" / "Part-Bind" / "dimension_geometry_bindings.json"
    assert result.dimension_count == 2
    assert result.binding_candidate_count == 2

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert data["schema"] == "dimension_geometry_binding"
    assert data["method"]["name"] == "vlm_assisted_dimension_geometry_binding_mvp"
    assert data["method"]["vlm_model"] is None
    assert data["views"][0]["view_id"] == "view_001"

    bindings = data["binding_candidates"]
    assert bindings[0]["dimension_id"] == "view_001_dimension_001"
    assert bindings[0]["target_feature_id"] == "view_001_feature_001"
    assert bindings[0]["binding_type"] == "diameter_of_hole"
    assert bindings[0]["source"] == "rule_candidate_scaffold"
    assert bindings[0]["rule_support"]["rank"] == 1
    assert bindings[0]["needs_vlm_review"] is True

    assert data["unbound_dimensions"] == []
    assert data["quality"]["ready_for_constraint_graph"] is False


def test_bind_dimensions_to_geometry_cli_writes_single_sample_output(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_inputs(dataflow, "Part-CLI")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vlm_cadcoder.cli",
            "bind-dimensions-to-geometry",
            "--sample-id",
            "Part-CLI",
            "--dataflow-root",
            str(dataflow),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Bound dimensions for 1 sample(s)" in result.stdout
    assert (
        dataflow / "09.Cross-viewGeometricReasoning" / "Part-CLI" / "dimension_geometry_bindings.json"
    ).exists()


def test_bind_dimensions_to_geometry_falls_back_to_drawing_ir_components(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_inputs(dataflow, "Part-Fallback")
    feature_path = dataflow / "08.Multi-viewFeatureExtraction" / "Part-Fallback" / "view_features.json"
    feature_data = json.loads(feature_path.read_text(encoding="utf-8"))
    feature_data["feature_candidates"] = []
    feature_path.write_text(json.dumps(feature_data), encoding="utf-8")

    drawing_ir_path = dataflow / "10.StructuredCADRepresentation" / "Part-Fallback" / "drawing_ir.json"
    drawing_ir = json.loads(drawing_ir_path.read_text(encoding="utf-8"))
    drawing_ir["feature_candidates"] = [
        {
            "id": "view_001_geometry_component_001",
            "type": "geometry_component",
            "view_id": "view_001",
            "bbox": [190, 180, 230, 220],
            "bbox_on_page": [290, 380, 330, 420],
            "area_px": 400,
        }
    ]
    drawing_ir_path.write_text(json.dumps(drawing_ir), encoding="utf-8")

    result = bind_dimensions_to_geometry_sample(sample_id="Part-Fallback", dataflow_root=dataflow)

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert result.binding_candidate_count >= 1
    assert data["inputs"]["feature_candidate_source"] == "drawing_ir_geometry_components"
    assert data["binding_candidates"][0]["target_feature_id"] == "view_001_geometry_component_001"


def _write_inputs(dataflow: Path, sample_id: str) -> None:
    drawing_ir_dir = dataflow / "10.StructuredCADRepresentation" / sample_id
    drawing_ir_dir.mkdir(parents=True)
    drawing_ir = {
        "schema": "drawing_ir",
        "version": "0.1.0",
        "sample_id": sample_id,
        "views": [
            {
                "id": "view_001",
                "type": "front",
                "bbox_on_page": [100, 200, 500, 600],
                "crop_size": {"width": 400, "height": 400},
                "image_clean": f"DataFlow/06.SingleViews/{sample_id}/view_001/clean_view_with_annotations.png",
            }
        ],
        "feature_candidates": [],
    }
    (drawing_ir_dir / "drawing_ir.json").write_text(json.dumps(drawing_ir), encoding="utf-8")

    feature_dir = dataflow / "08.Multi-viewFeatureExtraction" / sample_id
    feature_dir.mkdir(parents=True)
    view_features = {
        "schema": "view_feature_extraction",
        "sample_id": sample_id,
        "views": [{"view_id": "view_001"}],
        "feature_candidates": [
            {
                "id": "view_001_feature_001",
                "type": "hole_candidate",
                "view_id": "view_001",
                "bbox": [190, 180, 230, 220],
                "bbox_on_page": [290, 380, 330, 420],
                "confidence": 0.42,
            },
            {
                "id": "view_001_feature_002",
                "type": "outer_profile_candidate",
                "view_id": "view_001",
                "bbox": [20, 20, 360, 360],
                "bbox_on_page": [120, 220, 460, 560],
                "confidence": 0.48,
            },
        ],
    }
    (feature_dir / "view_features.json").write_text(json.dumps(view_features), encoding="utf-8")

    dimensions = {
        "schema": "dimension_extraction",
        "sample_id": sample_id,
        "views": [{"view_id": "view_001"}],
        "dimension_candidates": [
            {
                "id": "view_001_dimension_001",
                "view_id": "view_001",
                "text": "φ20",
                "normalized": "φ20",
                "dimension_type": "diameter",
                "bbox": [175, 155, 245, 175],
                "bbox_on_page": [275, 355, 345, 375],
                "value": 20.0,
                "quantity": None,
            },
            {
                "id": "view_001_dimension_002",
                "view_id": "view_001",
                "text": "80",
                "normalized": "80",
                "dimension_type": "linear",
                "bbox": [30, 370, 80, 390],
                "bbox_on_page": [130, 570, 180, 590],
                "value": 80.0,
                "quantity": None,
            },
        ],
    }
    (feature_dir / "dimension_candidates.json").write_text(json.dumps(dimensions), encoding="utf-8")
