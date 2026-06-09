from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from vlm_cadcoder.dataflow.view_filter_batch import filter_view_detection_directory


def test_filter_view_detection_directory_skips_rejected_reports(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_sample(
        dataflow=dataflow,
        sample_id="A-Part",
        detection_name="page_001_views.json",
        source_view_id="view_007",
    )
    _write_sample(
        dataflow=dataflow,
        sample_id="B-Part",
        detection_name="page_001_views_raw.json",
        source_view_id="view_009",
    )
    rejected_path = dataflow / "05.ViewDetection" / "A-Part" / "page_001_rejected_views.json"
    rejected_path.write_text(
        json.dumps({"sample_id": "A-Part", "rejected_views": [{"bbox": [1, 2, 3, 4]}]}),
        encoding="utf-8",
    )

    summary = filter_view_detection_directory(dataflow_root=dataflow, save_overlay=False)

    assert [(record.sample_id, record.page) for record in summary.records] == [
        ("A-Part", 1),
        ("B-Part", 1),
    ]
    assert summary.filtered_count == 2
    assert summary.failed_count == 0

    a_filtered = json.loads(
        (dataflow / "05.ViewDetection" / "A-Part" / "page_001_views.json").read_text(encoding="utf-8")
    )
    b_filtered = json.loads(
        (dataflow / "05.ViewDetection" / "B-Part" / "page_001_views.json").read_text(encoding="utf-8")
    )

    assert a_filtered["views"][0]["source_view_id"] == "view_007"
    assert b_filtered["views"][0]["source_view_id"] == "view_009"
    assert summary.records[0].rejected_candidates == 0
    assert summary.records[1].accepted_views == 1


def test_filter_view_detection_directory_reports_failures_without_fail_fast(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    bad_dir = dataflow / "05.ViewDetection" / "Bad-Part"
    bad_dir.mkdir(parents=True)
    (bad_dir / "page_001_views.json").write_text("{", encoding="utf-8")

    summary = filter_view_detection_directory(dataflow_root=dataflow, save_overlay=False)

    assert summary.filtered_count == 0
    assert summary.failed_count == 1
    assert summary.records[0].sample_id == "Bad-Part"
    assert summary.records[0].error


def _write_sample(
    *,
    dataflow: Path,
    sample_id: str,
    detection_name: str,
    source_view_id: str,
) -> None:
    image_path = dataflow / "04.CleanPNG" / sample_id / "page_001_clean.png"
    detection_path = dataflow / "05.ViewDetection" / sample_id / detection_name
    image_path.parent.mkdir(parents=True)
    detection_path.parent.mkdir(parents=True)

    image = Image.new("RGB", (600, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 180, 500, 320), outline="black", width=3)
    draw.line((120, 330, 480, 330), fill="black", width=2)
    image.save(image_path)

    detection_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "page": 1,
                "image_size": {"width": 600, "height": 600},
                "views": [
                    {
                        "view_id": source_view_id,
                        "label": "view_with_annotations",
                        "bbox": [100, 180, 500, 320],
                        "score": 0.91,
                        "source": "sketchsegment_view_detector",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
