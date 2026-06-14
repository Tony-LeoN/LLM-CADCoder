from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

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


def test_build_drawing_ir_sample_records_stale_classification_views_as_skipped(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    bbox = [100, 120, 700, 420]
    _write_detection(dataflow, "Part-Stale", accepted_bbox=bbox)
    _write_classification(dataflow, "Part-Stale", accepted_bbox=bbox)

    result = build_drawing_ir_sample(sample_id="Part-Stale", dataflow_root=dataflow)

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert result.view_count == 0
    assert data["skipped_views"] == [
        {
            "view_id": "view_001",
            "reason": "missing_single_view_metadata",
            "bbox_on_page": bbox,
            "source": "07.ViewClassification",
        }
    ]
    assert "no_classified_views" in data["quality"]["review_reasons"]


def test_build_drawing_ir_sample_tolerates_extra_metadata_closing_brace(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    bbox = [100, 120, 700, 420]
    _write_detection(dataflow, "Part-Metadata", accepted_bbox=bbox)
    _write_single_view(dataflow, "Part-Metadata", "view_001", bbox=bbox)
    _write_classification(dataflow, "Part-Metadata", accepted_bbox=bbox)
    metadata_path = dataflow / "06.SingleViews" / "Part-Metadata" / "view_001" / "view_metadata.json"
    metadata_path.write_text(metadata_path.read_text(encoding="utf-8") + "}\n", encoding="utf-8")

    result = build_drawing_ir_sample(sample_id="Part-Metadata", dataflow_root=dataflow)

    assert result.view_count == 1


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


def test_build_drawing_ir_sample_extracts_a_tier_geometry_components(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    bbox = [0, 0, 100, 80]
    _write_detection(dataflow, "Part-Feature", accepted_bbox=bbox)
    _write_single_view(dataflow, "Part-Feature", "view_001", bbox=bbox)
    geometry_path = dataflow / "06.SingleViews" / "Part-Feature" / "view_001" / "geometry_core.png"
    _write_geometry_core(geometry_path, rectangles=[(10, 10, 30, 20), (55, 35, 72, 52)])
    _write_geometry_core_audit(dataflow, "Part-Feature", "view_001", tier="A", geometry_path=geometry_path)
    _write_classification(dataflow, "Part-Feature", accepted_bbox=bbox)

    result = build_drawing_ir_sample(sample_id="Part-Feature", dataflow_root=dataflow)

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    view = data["views"][0]
    assert view["geometry_core"]["quality_tier"] == "A"
    assert view["geometry_core"]["ready_for_feature_extraction"] is True
    assert view["image_geometry_core"] == "DataFlow/06.SingleViews/Part-Feature/view_001/geometry_core.png"
    assert data["quality"]["ready_for_feature_extraction"] is True
    assert data["quality"]["feature_extraction_input_view_count"] == 1
    assert data["quality"]["feature_candidate_count"] == 2
    assert {candidate["type"] for candidate in data["feature_candidates"]} == {"geometry_component"}
    assert {candidate["semantic_status"] for candidate in data["feature_candidates"]} == {"unclassified"}
    assert data["feature_candidates"][0]["view_id"] == "view_001"
    assert data["feature_candidates"][0]["bbox"]


def test_build_drawing_ir_sample_blocks_non_a_geometry_core_from_feature_candidates(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    bbox = [0, 0, 100, 80]
    _write_detection(dataflow, "Part-Blocked", accepted_bbox=bbox)
    _write_single_view(dataflow, "Part-Blocked", "view_001", bbox=bbox)
    geometry_path = dataflow / "06.SingleViews" / "Part-Blocked" / "view_001" / "geometry_core.png"
    _write_geometry_core(geometry_path, rectangles=[(10, 10, 30, 20)])
    _write_geometry_core_audit(
        dataflow,
        "Part-Blocked",
        "view_001",
        tier="C",
        geometry_path=geometry_path,
        review_reasons=["manual_bad_geometry_core"],
    )
    _write_classification(dataflow, "Part-Blocked", accepted_bbox=bbox)

    result = build_drawing_ir_sample(sample_id="Part-Blocked", dataflow_root=dataflow)

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    view = data["views"][0]
    assert view["geometry_core"]["quality_tier"] == "C"
    assert view["geometry_core"]["ready_for_feature_extraction"] is False
    assert data["feature_candidates"] == []
    assert data["quality"]["ready_for_feature_extraction"] is False
    assert data["quality"]["geometry_core_blocked_view_count"] == 1
    assert "geometry_core_not_ready" in data["quality"]["review_reasons"]


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


def _write_geometry_core(path: Path, *, rectangles: list[tuple[int, int, int, int]]) -> None:
    image = Image.new("L", (100, 80), 255)
    draw = ImageDraw.Draw(image)
    for rectangle in rectangles:
        draw.rectangle(rectangle, fill=0)
    image.save(path)


def _write_geometry_core_audit(
    dataflow: Path,
    sample_id: str,
    view_id: str,
    *,
    tier: str,
    geometry_path: Path,
    review_reasons: list[str] | None = None,
) -> None:
    target = dataflow / "06.SingleViews" / "geometry_core_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    relative_geometry_path = Path("DataFlow") / geometry_path.relative_to(dataflow)
    payload = {
        "schema": "geometry_core_audit",
        "version": "0.1.0",
        "summary": {"total_count": 1, "tier_counts": {tier: 1}, "review_count": 0},
        "records": [
            {
                "sample_id": sample_id,
                "view_id": view_id,
                "view_type": "front",
                "quality_tier": tier,
                "needs_manual_review": tier != "A",
                "review_reasons": review_reasons or [],
                "geometry_component_count": len(review_reasons or []) + 1,
                "geometry_core_path": relative_geometry_path.as_posix(),
                "mask_path": None,
                "probability_path": None,
                "paths": {
                    "geometry_core": relative_geometry_path.as_posix(),
                    "geometry_core_mask": None,
                    "geometry_core_prob": None,
                },
            }
        ],
    }
    target.write_text(json.dumps(payload), encoding="utf-8")
