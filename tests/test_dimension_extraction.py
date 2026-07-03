from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from vlm_cadcoder.dataflow.dimension_extraction import extract_dimensions_sample, extract_dimensions_samples


def test_extract_dimensions_sample_collects_dimension_ocr_predictions(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_drawing_ir(dataflow, "Part-Dim")
    predictions_path = tmp_path / "predictions.jsonl"
    _write_prediction_jsonl(predictions_path, "Part-Dim")

    result = extract_dimensions_sample(
        sample_id="Part-Dim",
        dataflow_root=dataflow,
        prediction_jsonl_paths=[predictions_path],
    )

    assert result.output_path == dataflow / "08.Multi-viewFeatureExtraction" / "Part-Dim" / "dimension_candidates.json"
    assert result.view_count == 1
    assert result.dimension_count == 3

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert data["schema"] == "dimension_extraction"
    assert data["sample_id"] == "Part-Dim"
    assert data["source_drawing_ir"] == "DataFlow/10.StructuredCADRepresentation/Part-Dim/drawing_ir.json"
    assert data["method"]["name"] == "dimension_ocr_prediction_normalizer"

    candidates = data["dimension_candidates"]
    assert [candidate["normalized"] for candidate in candidates] == [
        "4 x Φ 4.5 完全贯穿",
        "29 ±0.05",
        "4xC2",
    ]
    assert candidates[0]["dimension_type"] == "diameter"
    assert candidates[0]["quantity"] == 4
    assert candidates[0]["value"] == 4.5
    assert candidates[0]["view_id"] == "view_001"
    assert candidates[0]["bbox_on_page"] == [110, 130, 150, 150]
    assert candidates[1]["dimension_type"] == "linear"
    assert candidates[1]["value"] == 29.0
    assert candidates[1]["tolerance"] == {"plus": 0.05, "minus": 0.05}
    assert candidates[2]["dimension_type"] == "chamfer"
    assert candidates[2]["quantity"] == 4
    assert candidates[2]["value"] == 2.0
    assert data["quality"]["dimension_candidate_count"] == 3
    assert data["quality"]["ready_for_dimension_geometry_binding"] is False


def test_extract_dimensions_samples_writes_batch_summary(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_drawing_ir(dataflow, "Part-Batch")
    predictions_path = tmp_path / "predictions.jsonl"
    _write_prediction_jsonl(predictions_path, "Part-Batch")

    summary = extract_dimensions_samples(dataflow_root=dataflow, prediction_jsonl_paths=[predictions_path])

    assert summary.extracted_count == 1
    assert summary.failed_count == 0
    assert summary.skipped_count == 0
    assert (dataflow / "08.Multi-viewFeatureExtraction" / "dimension_extraction_summary.csv").exists()
    summary_json = json.loads(
        (dataflow / "08.Multi-viewFeatureExtraction" / "dimension_extraction_summary.json").read_text(encoding="utf-8")
    )
    assert summary_json["records"][0]["sample_id"] == "Part-Batch"
    assert summary_json["records"][0]["dimension_count"] == 3


def test_extract_dimensions_matches_absolute_prediction_image_to_relative_drawing_ir_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    dataflow = Path("DataFlow")
    sample_id = "Part-Absolute"
    _write_drawing_ir(dataflow, sample_id)
    absolute_image = (
        tmp_path / "DataFlow" / "06.SingleViews" / sample_id / "view_001" / "clean_view_with_annotations.png"
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_prediction_jsonl(predictions_path, sample_id, input_images=[absolute_image.as_posix()])

    result = extract_dimensions_sample(
        sample_id=sample_id,
        dataflow_root=dataflow,
        prediction_jsonl_paths=[predictions_path],
    )

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert result.dimension_count == 3
    assert data["dimension_candidates"][0]["view_id"] == "view_001"
    assert data["unmatched_records"] == []


def test_extract_dimensions_splits_compound_diameter_and_chamfer_callout(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    sample_id = "Part-Compound"
    _write_drawing_ir(dataflow, sample_id)
    predictions_path = tmp_path / "predictions.jsonl"
    _write_prediction_jsonl(
        predictions_path,
        sample_id,
        dimensions=[
            {
                "text": "φ52完全贯穿孔口倒角C0.5",
                "normalized": "φ52 fully penetrated hole chamfer C0.5",
                "type": "diameter",
            }
        ],
    )

    result = extract_dimensions_sample(
        sample_id=sample_id,
        dataflow_root=dataflow,
        prediction_jsonl_paths=[predictions_path],
    )

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    candidates = data["dimension_candidates"]
    assert result.dimension_count == 2
    assert [(candidate["text"], candidate["dimension_type"], candidate["value"]) for candidate in candidates] == [
        ("φ52完全贯穿孔口倒角C0.5", "diameter", 52.0),
        ("C0.5", "chamfer", 0.5),
    ]
    assert candidates[1]["source"]["compound_parent_text"] == "φ52完全贯穿孔口倒角C0.5"
    assert "compound_dimension_callout_split" in candidates[1]["review_reasons"]


def test_extract_dimensions_does_not_split_standalone_chamfer_callout(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    sample_id = "Part-Chamfer"
    _write_drawing_ir(dataflow, sample_id)
    predictions_path = tmp_path / "predictions.jsonl"
    _write_prediction_jsonl(
        predictions_path,
        sample_id,
        dimensions=[
            {
                "text": "C0.5",
                "normalized": "C0.5 chamfer",
                "type": "chamfer",
            }
        ],
    )

    result = extract_dimensions_sample(
        sample_id=sample_id,
        dataflow_root=dataflow,
        prediction_jsonl_paths=[predictions_path],
    )

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert result.dimension_count == 1
    assert data["dimension_candidates"][0]["text"] == "C0.5"
    assert "compound_parent_text" not in data["dimension_candidates"][0]["source"]


def test_extract_dimensions_cli_writes_single_sample_output(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_drawing_ir(dataflow, "Part-CLI")
    predictions_path = tmp_path / "predictions.jsonl"
    _write_prediction_jsonl(predictions_path, "Part-CLI")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vlm_cadcoder.cli",
            "extract-dimensions",
            "--sample-id",
            "Part-CLI",
            "--dataflow-root",
            str(dataflow),
            "--prediction-jsonl",
            str(predictions_path),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Extracted dimensions for 1 sample(s)" in result.stdout
    assert (dataflow / "08.Multi-viewFeatureExtraction" / "Part-CLI" / "dimension_candidates.json").exists()


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
                "bbox_on_page": [100, 120, 500, 320],
                "crop_size": {"width": 400, "height": 200},
                "image_clean": f"DataFlow/06.SingleViews/{sample_id}/view_001/clean_view_with_annotations.png",
                "geometry_core": {"quality_tier": "A", "ready_for_feature_extraction": True},
            }
        ],
        "dimensions": [],
        "feature_candidates": [],
        "constraints": [],
        "view_relations": [],
        "quality": {"ready_for_feature_extraction": True},
    }
    (target / "drawing_ir.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_prediction_jsonl(
    path: Path,
    sample_id: str,
    input_images: list[str] | None = None,
    dimensions: list[dict[str, object]] | None = None,
) -> None:
    record = {
        "sample_id": sample_id,
        "task": "dimension_ocr",
        "model": "qwen2_5_vl_3b",
        "input_images": input_images
        or [f"DataFlow/06.SingleViews/{sample_id}/view_001/clean_view_with_annotations.png"],
        "prediction": {
            "dimensions": dimensions
            or [
                {
                    "text": "4 x Φ 4.5 完全贯穿",
                    "normalized": "4 x Φ 4.5 完全贯穿",
                    "type": "unknown",
                    "bbox": [10, 10, 50, 30],
                },
                {
                    "text": "29 ±0.05",
                    "normalized": "29 ±0.05",
                    "type": "linear",
                },
                {
                    "text": "4xC2",
                    "normalized": "4xC2",
                    "type": "unknown",
                },
                {
                    "text": "4xC2",
                    "normalized": "4xC2",
                    "type": "unknown",
                },
            ]
        },
        "error": None,
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
