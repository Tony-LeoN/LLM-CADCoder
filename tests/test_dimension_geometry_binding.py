from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vlm_cadcoder.dataflow import dimension_geometry_binding as binding_module
from vlm_cadcoder.dataflow.dimension_geometry_binding import bind_dimensions_to_geometry_sample
from vlm_cadcoder.models.base import ModelResponse


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


def test_bind_dimensions_to_geometry_writes_numbered_overlay(tmp_path: Path) -> None:
    pytest.importorskip("PIL.Image")
    dataflow = tmp_path / "DataFlow"
    _write_inputs(dataflow, "Part-Overlay")

    result = bind_dimensions_to_geometry_sample(sample_id="Part-Overlay", dataflow_root=dataflow)

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    request = data["vlm_requests"][0]
    overlay_path = dataflow / "09.Cross-viewGeometricReasoning" / "Part-Overlay" / "overlays" / "view_001_binding_overlay.png"
    assert overlay_path.exists()
    assert request["overlay_image"] == "DataFlow/09.Cross-viewGeometricReasoning/Part-Overlay/overlays/view_001_binding_overlay.png"
    assert request["resolved_overlay_image"] == overlay_path.as_posix()
    assert request["visual_labels"]["dimensions"][0]["label"] == "D1"
    assert request["visual_labels"]["features"][0]["label"] == "G1"
    assert "\"label\": \"D1\"" in request["prompt"]


def test_bind_dimensions_to_geometry_keeps_fallback_features_visible_without_rule_match(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_inputs(dataflow, "Part-Visible-Fallback")
    feature_path = dataflow / "08.Multi-viewFeatureExtraction" / "Part-Visible-Fallback" / "view_features.json"
    feature_data = json.loads(feature_path.read_text(encoding="utf-8"))
    feature_data["feature_candidates"] = []
    feature_path.write_text(json.dumps(feature_data), encoding="utf-8")

    drawing_ir_path = dataflow / "10.StructuredCADRepresentation" / "Part-Visible-Fallback" / "drawing_ir.json"
    drawing_ir = json.loads(drawing_ir_path.read_text(encoding="utf-8"))
    drawing_ir["feature_candidates"] = [
        {
            "id": "view_001_geometry_component_001",
            "type": "geometry_component",
            "view_id": "view_001",
            "bbox": [20, 20, 360, 360],
            "bbox_on_page": [120, 220, 460, 560],
            "area_px": 10000,
        }
    ]
    drawing_ir_path.write_text(json.dumps(drawing_ir), encoding="utf-8")

    dimensions_path = dataflow / "08.Multi-viewFeatureExtraction" / "Part-Visible-Fallback" / "dimension_candidates.json"
    dimensions = json.loads(dimensions_path.read_text(encoding="utf-8"))
    dimensions["dimension_candidates"][0]["dimension_type"] = "geometric_tolerance"
    dimensions["dimension_candidates"][1]["dimension_type"] = "unknown"
    dimensions_path.write_text(json.dumps(dimensions), encoding="utf-8")

    result = bind_dimensions_to_geometry_sample(sample_id="Part-Visible-Fallback", dataflow_root=dataflow)

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    request = data["vlm_requests"][0]
    assert result.binding_candidate_count == 0
    assert request["rule_candidate_count"] == 0
    assert request["visual_labels"]["features"] == [
        {
            "label": "G1",
            "id": "view_001_geometry_component_001",
            "type": "unknown_geometry_candidate",
            "bbox": [20, 20, 360, 360],
            "bbox_on_page": [120, 220, 460, 560],
        }
    ]
    assert "view_001_geometry_component_001" in request["prompt"]


def test_bind_dimensions_to_geometry_maps_vlm_overlay_labels_to_feature_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_inputs(dataflow, "Part-VLM-Labels")
    model_config = tmp_path / "models.json"
    model_config.write_text(json.dumps({"models": {"fake_vlm": {}}}), encoding="utf-8")

    class FakeModel:
        def generate(self, images, prompt, generation_config=None):
            return ModelResponse(
                text="{}",
                parsed_json={
                    "bindings": [
                        {
                            "dimension_id": "view_001_dimension_001",
                            "target_feature_ids": ["G1"],
                            "binding_type": "diameter_of_hole",
                            "confidence": 0.91,
                            "evidence": ["uses visible label"],
                        },
                        {
                            "dimension_id": "view_001_dimension_002",
                            "target_feature_ids": ["G999"],
                            "binding_type": "linear_extent",
                            "confidence": 0.8,
                            "evidence": ["invalid label should not be accepted"],
                        },
                    ],
                    "unbound_dimensions": [],
                    "ambiguous_bindings": [],
                },
                latency_sec=0.01,
            )

    monkeypatch.setattr(binding_module, "build_model", lambda _name, _config: FakeModel())

    result = bind_dimensions_to_geometry_sample(
        sample_id="Part-VLM-Labels",
        dataflow_root=dataflow,
        model_name="fake_vlm",
        model_config_path=model_config,
    )

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    valid, invalid = data["vlm_binding_candidates"]
    assert valid["target_feature_labels"] == ["G1"]
    assert valid["target_feature_ids"] == ["view_001_feature_001"]
    assert valid["target_feature_id"] == "view_001_feature_001"
    assert valid["semantic_status"] == "candidate"
    assert invalid["target_feature_labels"] == ["G999"]
    assert invalid["target_feature_ids"] == []
    assert invalid["target_feature_id"] is None
    assert invalid["invalid_target_feature_labels"] == ["G999"]
    assert invalid["semantic_status"] == "invalid_candidate"
    assert "vlm_target_label_not_in_request" in invalid["review_reasons"]


def _write_inputs(dataflow: Path, sample_id: str) -> None:
    clean_view_path = dataflow / "06.SingleViews" / sample_id / "view_001" / "clean_view_with_annotations.png"
    clean_view_path.parent.mkdir(parents=True)
    clean_view_path.write_bytes(base64.b64decode(_ONE_PIXEL_PNG_BASE64))

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


_ONE_PIXEL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAZAAAAGQCAIAAAAP3aGbAAACcElEQVR4nO3BMQEAAADCoPVPbQwfoAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwZQF1AAGPd+bkAAAAAElF"
    "TkSuQmCC"
)
