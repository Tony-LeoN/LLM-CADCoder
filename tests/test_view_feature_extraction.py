from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from vlm_cadcoder.dataflow.view_feature_extraction import (
    extract_view_features_sample,
    extract_view_features_samples,
)


def test_extract_view_features_sample_promotes_geometry_components_to_semantic_candidates(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_drawing_ir(dataflow, "Part-Feature")

    result = extract_view_features_sample(sample_id="Part-Feature", dataflow_root=dataflow)

    assert result.output_path == dataflow / "08.Multi-viewFeatureExtraction" / "Part-Feature" / "view_features.json"
    assert result.view_count == 1
    assert result.feature_count == 4

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert data["schema"] == "view_feature_extraction"
    assert data["version"] == "0.1.0"
    assert data["sample_id"] == "Part-Feature"
    assert data["source_drawing_ir"] == "DataFlow/10.StructuredCADRepresentation/Part-Feature/drawing_ir.json"
    assert data["method"]["name"] == "rule_based_geometry_component_classifier"

    flat_candidates = data["feature_candidates"]
    assert {candidate["type"] for candidate in flat_candidates} == {
        "outer_profile_candidate",
        "hole_candidate",
        "slot_candidate",
        "annotation_residue_candidate",
    }
    assert {candidate["semantic_status"] for candidate in flat_candidates} == {"candidate"}
    assert {candidate["source_type"] for candidate in flat_candidates} == {"geometry_component"}
    assert {candidate["needs_manual_review"] for candidate in flat_candidates} == {True}
    assert {candidate["source_candidate_id"] for candidate in flat_candidates} == {
        "view_001_geometry_component_001",
        "view_001_geometry_component_002",
        "view_001_geometry_component_003",
        "view_001_geometry_component_004",
    }

    view = data["views"][0]
    assert view["view_id"] == "view_001"
    assert view["view_type"] == "front"
    assert view["candidate_count"] == 4
    assert data["quality"]["input_component_count"] == 4
    assert data["quality"]["semantic_candidate_count"] == 4
    assert data["quality"]["needs_manual_review"] is True
    assert "rule_based_semantic_candidates_need_validation" in data["quality"]["review_reasons"]


def test_extract_view_features_samples_writes_batch_summary(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_drawing_ir(dataflow, "Part-Batch")

    summary = extract_view_features_samples(dataflow_root=dataflow)

    assert summary.extracted_count == 1
    assert summary.failed_count == 0
    assert summary.skipped_count == 0
    assert (dataflow / "08.Multi-viewFeatureExtraction" / "view_feature_summary.csv").exists()
    assert (dataflow / "08.Multi-viewFeatureExtraction" / "view_feature_summary.json").exists()

    summary_json = json.loads((dataflow / "08.Multi-viewFeatureExtraction" / "view_feature_summary.json").read_text())
    assert summary_json["records"][0]["sample_id"] == "Part-Batch"
    assert summary_json["records"][0]["feature_count"] == 4


def test_extract_view_features_sample_skips_malformed_geometry_components(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_drawing_ir(dataflow, "Part-Malformed")
    drawing_ir_path = dataflow / "10.StructuredCADRepresentation" / "Part-Malformed" / "drawing_ir.json"
    drawing_ir = json.loads(drawing_ir_path.read_text(encoding="utf-8"))
    drawing_ir["feature_candidates"].append(
        {
            "id": "view_001_geometry_component_bad",
            "type": "geometry_component",
            "semantic_status": "unclassified",
            "view_id": "view_001",
            "bbox": [20, 20, 20, 25],
        }
    )
    drawing_ir_path.write_text(json.dumps(drawing_ir), encoding="utf-8")

    result = extract_view_features_sample(sample_id="Part-Malformed", dataflow_root=dataflow)

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert result.feature_count == 4
    assert data["quality"]["skipped_component_count"] == 1
    assert data["skipped_components"] == [
        {
            "source_candidate_id": "view_001_geometry_component_bad",
            "reason": "Geometry component bbox must be xyxy with positive area: view_001_geometry_component_bad",
            "view_id": "view_001",
        }
    ]


def test_extract_view_features_cli_writes_single_sample_output(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_drawing_ir(dataflow, "Part-CLI")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vlm_cadcoder.cli",
            "extract-view-features",
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
    assert "Extracted view features for 1 sample(s)" in result.stdout
    assert (dataflow / "08.Multi-viewFeatureExtraction" / "Part-CLI" / "view_features.json").exists()


def _write_drawing_ir(dataflow: Path, sample_id: str) -> None:
    target = dataflow / "10.StructuredCADRepresentation" / sample_id
    target.mkdir(parents=True)
    payload = {
        "schema": "drawing_ir",
        "version": "0.1.0",
        "sample_id": sample_id,
        "page": 1,
        "views": [
            {
                "id": "view_001",
                "type": "front",
                "bbox": [100, 120, 500, 320],
                "bbox_on_page": [100, 120, 500, 320],
                "crop_size": {"width": 400, "height": 200},
                "geometry_core": {
                    "quality_tier": "A",
                    "ready_for_feature_extraction": True,
                },
            }
        ],
        "feature_candidates": [
            {
                "id": "view_001_geometry_component_001",
                "type": "geometry_component",
                "semantic_status": "unclassified",
                "view_id": "view_001",
                "bbox": [10, 20, 360, 170],
                "bbox_on_page": [110, 140, 460, 290],
                "area_px": 5000,
                "source": "geometry_core_connected_components",
                "confidence": 0.3,
                "evidence": ["geometry_core_quality=A"],
            },
            {
                "id": "view_001_geometry_component_002",
                "type": "geometry_component",
                "semantic_status": "unclassified",
                "view_id": "view_001",
                "bbox": [110, 70, 134, 94],
                "bbox_on_page": [210, 190, 234, 214],
                "area_px": 280,
                "source": "geometry_core_connected_components",
                "confidence": 0.3,
                "evidence": ["geometry_core_quality=A"],
            },
            {
                "id": "view_001_geometry_component_003",
                "type": "geometry_component",
                "semantic_status": "unclassified",
                "view_id": "view_001",
                "bbox": [180, 80, 260, 98],
                "bbox_on_page": [280, 200, 360, 218],
                "area_px": 360,
                "source": "geometry_core_connected_components",
                "confidence": 0.3,
                "evidence": ["geometry_core_quality=A"],
            },
            {
                "id": "view_001_geometry_component_004",
                "type": "geometry_component",
                "semantic_status": "unclassified",
                "view_id": "view_001",
                "bbox": [20, 10, 23, 13],
                "bbox_on_page": [120, 130, 123, 133],
                "area_px": 5,
                "source": "geometry_core_connected_components",
                "confidence": 0.3,
                "evidence": ["geometry_core_quality=A"],
            },
        ],
        "quality": {
            "ready_for_feature_extraction": True,
            "feature_candidate_count": 4,
        },
    }
    (target / "drawing_ir.json").write_text(json.dumps(payload), encoding="utf-8")
