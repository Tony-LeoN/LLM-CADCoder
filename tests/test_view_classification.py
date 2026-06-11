from __future__ import annotations

import json
from pathlib import Path

from vlm_cadcoder.dataflow.view_classification import classify_single_view_sample, classify_view_samples


def test_classify_single_view_sample_writes_baseline_labels(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    bboxes = [
        [100, 100, 700, 300],
        [740, 120, 820, 330],
        [120, 340, 690, 390],
        [680, 430, 930, 620],
    ]
    _write_detection(dataflow, "Plate-A", image_size=(1000, 700), bboxes=bboxes)
    _write_view(dataflow, "Plate-A", "view_001", bbox=bboxes[0], score=0.95)
    _write_view(dataflow, "Plate-A", "view_002", bbox=bboxes[1], score=0.91)
    _write_view(dataflow, "Plate-A", "view_003", bbox=bboxes[2], score=0.83)
    _write_view(dataflow, "Plate-A", "view_004", bbox=bboxes[3], score=0.96)

    result = classify_single_view_sample(sample_id="Plate-A", dataflow_root=dataflow)

    assert result.output_path == dataflow / "07.ViewClassification" / "Plate-A" / "page_001_view_classification.json"
    assert [view.view_type for view in result.views] == ["front", "left", "top", "isometric"]
    assert result.views[0].is_primary is True
    assert result.views[0].needs_manual_review is False
    assert result.output_path.exists()

    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert data["views"][3]["type"] == "isometric"
    assert data["method"]["name"] == "heuristic_view_classifier"
    assert data["input_filter"]["source"] == "05.ViewDetection accepted views"


def test_classify_single_view_sample_skips_06_views_rejected_by_05(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    accepted = [[100, 100, 700, 300], [120, 360, 690, 420]]
    rejected = [[790, 510, 930, 620]]
    _write_detection(dataflow, "Plate-R", image_size=(1000, 700), bboxes=accepted, rejected_bboxes=rejected)
    _write_view(dataflow, "Plate-R", "view_001", bbox=accepted[0])
    _write_view(dataflow, "Plate-R", "view_002", bbox=accepted[1])
    _write_view(dataflow, "Plate-R", "view_003", bbox=rejected[0])

    result = classify_single_view_sample(sample_id="Plate-R", dataflow_root=dataflow)

    assert [view.view_id for view in result.views] == ["view_001", "view_002"]
    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert data["skipped_views"] == [
        {
            "view_id": "view_003",
            "reason": "not_in_05_accepted_views",
            "bbox_on_page": rejected[0],
        }
    ]


def test_classify_view_samples_skips_copy_samples_by_default(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    bbox = [100, 100, 700, 300]
    _write_detection(dataflow, "Plate-A", image_size=(1000, 700), bboxes=[bbox])
    _write_view(dataflow, "Plate-A", "view_001", bbox=bbox)
    _write_detection(dataflow, "Plate-A-copy", image_size=(1000, 700), bboxes=[bbox])
    _write_view(dataflow, "Plate-A-copy", "view_001", bbox=bbox)

    summary = classify_view_samples(dataflow_root=dataflow)

    assert summary.classified_count == 1
    assert summary.skipped_count == 1
    assert summary.failed_count == 0
    by_sample = {record.sample_id: record for record in summary.records}
    assert by_sample["Plate-A-copy"].skipped is True
    assert (dataflow / "07.ViewClassification" / "view_classification_summary.csv").exists()
    assert (dataflow / "07.ViewClassification" / "view_classification_summary.json").exists()


def _write_detection(
    dataflow: Path,
    sample_id: str,
    *,
    image_size: tuple[int, int],
    bboxes: list[list[int]],
    rejected_bboxes: list[list[int]] | None = None,
) -> None:
    target = dataflow / "05.ViewDetection" / sample_id
    target.mkdir(parents=True)
    width, height = image_size
    (target / "page_001_views.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "page": 1,
                "image_size": {"width": width, "height": height},
                "views": [
                    {
                        "view_id": f"view_{index:03d}",
                        "bbox": bbox,
                        "filter": {"accepted": True, "reject_reasons": []},
                    }
                    for index, bbox in enumerate(bboxes, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )
    if rejected_bboxes:
        (target / "page_001_rejected_views.json").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "page": 1,
                    "rejected_views": [
                        {
                            "view_id": f"view_{index + len(bboxes):03d}",
                            "bbox": bbox,
                            "filter": {"accepted": False, "reject_reasons": ["test_rejected"]},
                        }
                        for index, bbox in enumerate(rejected_bboxes, start=1)
                    ],
                }
            ),
            encoding="utf-8",
        )


def _write_view(
    dataflow: Path,
    sample_id: str,
    view_id: str,
    *,
    bbox: list[int],
    score: float = 0.9,
) -> None:
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
                "detector": {"name": "sketchsegment_view_detector", "score": score},
                "coordinate_system": "page_pixel_xyxy",
            }
        ),
        encoding="utf-8",
    )
    (view_dir / "clean_view_with_annotations.png").write_bytes(b"not-a-real-png")
