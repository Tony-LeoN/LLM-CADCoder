from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_cadcoder.dataflow.drawing_ir_builder import build_drawing_ir_sample, build_drawing_ir_samples


def test_build_drawing_ir_sample_promotes_view_classification_to_schema(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    accepted_bbox = [100, 120, 700, 420]
    rejected_bbox = [760, 80, 900, 160]
    _write_detection(dataflow, "Part-A", accepted_bbox=accepted_bbox, rejected_bbox=rejected_bbox)
    _write_single_view(dataflow, "Part-A", "view_003", bbox=accepted_bbox)
    _write_classification(dataflow, "Part-A", accepted_bbox=accepted_bbox, skipped_bbox=rejected_bbox)

    result = build_drawing_ir_sample(sample_id="Part-A", dataflow_root=dataflow)

    assert result.output_path == dataflow / "10.StructuredCADRepresentation" / "Part-A" / "drawing_ir.json"
    assert result.view_count == 1

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert data["schema"] == "drawing_ir"
    assert data["version"] == "0.1.0"
    assert data["sample_id"] == "Part-A"
    assert data["sheet"]["input_mode"] == "detected_views_plus_classification"
    view = data["views"][0]
    assert view["id"] == "view_003"
    assert view["type"] == "front"
    assert view["type_source"] == "heuristic_view_classifier"
    assert view["type_confidence"] == 0.68
    assert view["type_candidates"] == [
        {
            "type": "front",
            "confidence": 0.68,
            "source": "07.ViewClassification",
            "needs_manual_review": False,
            "reasons": ["largest_non_isometric_view"],
        }
    ]
    assert view["bbox"] == accepted_bbox
    assert view["image_clean"] == "DataFlow/06.SingleViews/Part-A/view_003/clean_view_with_annotations.png"
    assert view["detector"] == {
        "score": 0.93,
        "source": "sketchsegment_view_detector",
        "source_view_id": "view_003",
        "accepted_view_id": "view_001",
    }
    assert view["source"]["view_metadata"] == "DataFlow/06.SingleViews/Part-A/view_003/view_metadata.json"
    assert data["dimensions"] == []
    assert data["feature_candidates"] == []
    assert data["constraints"] == []
    assert data["view_relations"] == []
    assert data["skipped_views"][0]["reason"] == "not_in_05_accepted_views"
    assert data["provenance"]["skipped_views"][0]["reason"] == "not_in_05_accepted_views"
    assert data["quality"]["needs_manual_review"] is True
    assert "skipped_single_view_crops" in data["quality"]["review_reasons"]
    assert view["source"]["accepted_detection"]["filter"] == {"accepted": True, "reject_reasons": []}


def test_build_drawing_ir_sample_rejects_trailing_json_data(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    bbox = [100, 120, 700, 420]
    _write_detection(dataflow, "Part-B", accepted_bbox=bbox)
    _write_single_view(dataflow, "Part-B", "view_001", bbox=bbox)
    _write_classification(dataflow, "Part-B", accepted_bbox=bbox)

    classification_path = dataflow / "07.ViewClassification" / "Part-B" / "page_001_view_classification.json"
    classification_path.write_text(classification_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Trailing data"):
        build_drawing_ir_sample(sample_id="Part-B", dataflow_root=dataflow)


def test_build_drawing_ir_sample_requires_single_view_metadata(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    bbox = [100, 120, 700, 420]
    _write_detection(dataflow, "Part-Missing", accepted_bbox=bbox)
    _write_classification(dataflow, "Part-Missing", accepted_bbox=bbox)

    with pytest.raises(FileNotFoundError, match="view_metadata.json"):
        build_drawing_ir_sample(sample_id="Part-Missing", dataflow_root=dataflow)


def test_build_drawing_ir_sample_requires_matching_accepted_detection(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_detection(dataflow, "Part-Mismatch", accepted_bbox=[10, 20, 80, 90])
    _write_single_view(dataflow, "Part-Mismatch", "view_001", bbox=[100, 120, 700, 420])
    _write_classification(dataflow, "Part-Mismatch", accepted_bbox=[100, 120, 700, 420])

    with pytest.raises(ValueError, match="does not match any accepted 05"):
        build_drawing_ir_sample(sample_id="Part-Mismatch", dataflow_root=dataflow)


def test_build_drawing_ir_samples_writes_batch_summary(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    bbox = [100, 120, 700, 420]
    _write_detection(dataflow, "Part-C", accepted_bbox=bbox)
    _write_single_view(dataflow, "Part-C", "view_001", bbox=bbox)
    _write_classification(dataflow, "Part-C", accepted_bbox=bbox)

    summary = build_drawing_ir_samples(dataflow_root=dataflow)

    assert summary.built_count == 1
    assert summary.failed_count == 0
    assert summary.skipped_count == 0
    assert (dataflow / "10.StructuredCADRepresentation" / "drawing_ir_summary.csv").exists()
    assert (dataflow / "10.StructuredCADRepresentation" / "drawing_ir_summary.json").exists()


def _write_detection(
    dataflow: Path,
    sample_id: str,
    *,
    accepted_bbox: list[int],
    rejected_bbox: list[int] | None = None,
) -> None:
    target = dataflow / "05.ViewDetection" / sample_id
    target.mkdir(parents=True)
    payload = {
        "sample_id": sample_id,
        "page": 1,
        "image_size": {"width": 1000, "height": 700},
        "views": [
            {
                "view_id": "view_001",
                "source_view_id": "view_003",
                "label": "view_with_annotations",
                "bbox": accepted_bbox,
                "score": 0.93,
                "source": "sketchsegment_view_detector",
                "filter": {"accepted": True, "reject_reasons": []},
            }
        ],
    }
    (target / "page_001_views.json").write_text(json.dumps(payload), encoding="utf-8")
    if rejected_bbox:
        (target / "page_001_rejected_views.json").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "page": 1,
                    "rejected_views": [
                        {
                            "view_id": "view_002",
                            "bbox": rejected_bbox,
                            "filter": {"accepted": False, "reject_reasons": ["test"]},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )


def _write_single_view(dataflow: Path, sample_id: str, view_id: str, *, bbox: list[int]) -> None:
    view_dir = dataflow / "06.SingleViews" / sample_id / view_id
    view_dir.mkdir(parents=True)
    x1, y1, x2, y2 = bbox
    (view_dir / "view_metadata.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "view_id": view_id,
                "bbox_on_page": bbox,
                "crop_size": {"width": x2 - x1, "height": y2 - y1},
                "coordinate_system": "page_pixel_xyxy",
                "detector": {
                    "name": "sketchsegment_view_detector",
                    "score": 0.93,
                },
            }
        ),
        encoding="utf-8",
    )
    (view_dir / "clean_view_with_annotations.png").write_bytes(b"not-a-real-png")


def _write_classification(
    dataflow: Path,
    sample_id: str,
    *,
    accepted_bbox: list[int],
    skipped_bbox: list[int] | None = None,
) -> None:
    target = dataflow / "07.ViewClassification" / sample_id
    target.mkdir(parents=True)
    payload = {
        "sample_id": sample_id,
        "page": 1,
        "image_size": {"width": 1000, "height": 700},
        "coordinate_system": "page_pixel_xyxy",
        "method": {
            "name": "heuristic_view_classifier",
            "version": "0.1.0",
        },
        "input_filter": {
            "source": "05.ViewDetection accepted views",
            "accepted_detection_count": 1,
            "skipped_view_count": 1 if skipped_bbox else 0,
        },
        "views": [
            {
                "view_id": "view_003" if skipped_bbox else "view_001",
                "type": "front",
                "confidence": 0.68,
                "is_primary": True,
                "needs_manual_review": False,
                "reasons": ["largest_non_isometric_view"],
                "bbox_on_page": accepted_bbox,
                "crop_size": {"width": accepted_bbox[2] - accepted_bbox[0], "height": accepted_bbox[3] - accepted_bbox[1]},
                "detector_score": 0.93,
                "image_clean": f"DataFlow/06.SingleViews/{sample_id}/{'view_003' if skipped_bbox else 'view_001'}/clean_view_with_annotations.png",
            }
        ],
        "skipped_views": [
            {
                "view_id": "view_002",
                "reason": "not_in_05_accepted_views",
                "bbox_on_page": skipped_bbox,
            }
        ]
        if skipped_bbox
        else [],
    }
    (target / "page_001_view_classification.json").write_text(json.dumps(payload), encoding="utf-8")
