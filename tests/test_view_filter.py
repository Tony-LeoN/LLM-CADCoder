from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from vlm_cadcoder.cli import filter_view_detections_cli
from vlm_cadcoder.dataflow.view_filter import ViewFilterConfig, filter_view_detections_file


def test_filter_rejects_header_and_dense_text_but_keeps_line_view(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    sample_id = "Part-A"
    image_path = dataflow / "04.CleanPNG" / sample_id / "page_001_clean.png"
    detection_path = dataflow / "05.ViewDetection" / sample_id / "page_001_views.json"
    image_path.parent.mkdir(parents=True)
    detection_path.parent.mkdir(parents=True)

    image = Image.new("RGB", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    _draw_header_table(draw, (120, 20, 880, 90))
    _draw_line_view(draw, (150, 230, 850, 360))
    _draw_dense_text_block(draw, (720, 760, 940, 910))
    image.save(image_path)

    detection_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "page": 1,
                "image_size": {"width": 1000, "height": 1000},
                "views": [
                    {
                        "view_id": "view_001",
                        "label": "view_with_annotations",
                        "bbox": [120, 20, 880, 90],
                        "score": 0.37,
                        "source": "sketchsegment_view_detector",
                    },
                    {
                        "view_id": "view_002",
                        "label": "view_with_annotations",
                        "bbox": [150, 230, 850, 360],
                        "score": 0.22,
                        "source": "sketchsegment_view_detector",
                    },
                    {
                        "view_id": "view_003",
                        "label": "view_with_annotations",
                        "bbox": [720, 760, 940, 910],
                        "score": 0.66,
                        "source": "sketchsegment_view_detector",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = filter_view_detections_file(
        detection_path=detection_path,
        clean_image_path=image_path,
        dataflow_root=dataflow,
        config=ViewFilterConfig(min_score=0.5),
    )

    assert [view["source_view_id"] for view in result.accepted_views] == ["view_002"]
    assert result.accepted_views[0]["view_id"] == "view_001"
    assert [view["source_view_id"] for view in result.rejected_views] == ["view_001", "view_003"]
    assert result.rejected_views[0]["filter"]["reject_reasons"] == ["top_strip_low_score"]
    assert result.rejected_views[1]["filter"]["reject_reasons"] == ["dense_text_or_stamp"]


def test_filter_preserves_raw_json_and_writes_rejected_report(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    sample_id = "Part-B"
    image_path = dataflow / "04.CleanPNG" / sample_id / "page_001_clean.png"
    detection_path = dataflow / "05.ViewDetection" / sample_id / "page_001_views.json"
    image_path.parent.mkdir(parents=True)
    detection_path.parent.mkdir(parents=True)

    image = Image.new("RGB", (600, 600), "white")
    draw = ImageDraw.Draw(image)
    _draw_line_view(draw, (100, 180, 500, 320))
    image.save(image_path)
    detection_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "page": 1,
                "image_size": {"width": 600, "height": 600},
                "views": [
                    {
                        "view_id": "view_009",
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

    result = filter_view_detections_file(
        detection_path=detection_path,
        clean_image_path=image_path,
        dataflow_root=dataflow,
        save_overlay=False,
    )

    raw_path = dataflow / "05.ViewDetection" / sample_id / "page_001_views_raw.json"
    rejected_path = dataflow / "05.ViewDetection" / sample_id / "page_001_rejected_views.json"
    filtered = json.loads(detection_path.read_text(encoding="utf-8"))
    rejected = json.loads(rejected_path.read_text(encoding="utf-8"))

    assert raw_path.exists()
    assert result.filtered_path == detection_path
    assert filtered["views"][0]["view_id"] == "view_001"
    assert filtered["views"][0]["source_view_id"] == "view_009"
    assert rejected["rejected_views"] == []


def test_layout_overlap_rejects_removed_table_candidate(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    sample_id = "Part-C"
    image_path = dataflow / "04.CleanPNG" / sample_id / "page_001_clean.png"
    detection_path = dataflow / "05.ViewDetection" / sample_id / "page_001_views.json"
    layout_path = dataflow / "03.LayoutAnalysis" / sample_id / "page_001_layout.json"
    image_path.parent.mkdir(parents=True)
    detection_path.parent.mkdir(parents=True)
    layout_path.parent.mkdir(parents=True)

    Image.new("RGB", (800, 800), "white").save(image_path)
    detection_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "page": 1,
                "image_size": {"width": 800, "height": 800},
                "views": [
                    {
                        "view_id": "view_001",
                        "label": "view_with_annotations",
                        "bbox": [520, 590, 780, 760],
                        "score": 0.92,
                        "source": "sketchsegment_view_detector",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    layout_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "page": 1,
                "regions": [
                    {
                        "id": "r006",
                        "type": "title_or_tolerance_table",
                        "bbox": [500, 560, 790, 780],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = filter_view_detections_file(
        detection_path=detection_path,
        clean_image_path=image_path,
        layout_path=layout_path,
        dataflow_root=dataflow,
        save_overlay=False,
    )

    assert result.accepted_views == []
    assert result.rejected_views[0]["filter"]["reject_reasons"] == ["layout_region_overlap:title_or_tolerance_table"]


def test_cli_reads_raw_detections_but_writes_filtered_views(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    sample_id = "Part-D"
    image_path = dataflow / "04.CleanPNG" / sample_id / "page_001_clean.png"
    detection_dir = dataflow / "05.ViewDetection" / sample_id
    raw_path = detection_dir / "page_001_views_raw.json"
    filtered_path = detection_dir / "page_001_views.json"
    image_path.parent.mkdir(parents=True)
    detection_dir.mkdir(parents=True)

    image = Image.new("RGB", (600, 600), "white")
    draw = ImageDraw.Draw(image)
    _draw_line_view(draw, (100, 180, 500, 320))
    image.save(image_path)
    raw_payload = {
        "sample_id": sample_id,
        "page": 1,
        "image_size": {"width": 600, "height": 600},
        "views": [
            {
                "view_id": "view_009",
                "label": "view_with_annotations",
                "bbox": [100, 180, 500, 320],
                "score": 0.91,
                "source": "sketchsegment_view_detector",
            }
        ],
    }
    raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")
    filtered_path.write_text("{}", encoding="utf-8")

    filter_view_detections_cli(
        SimpleNamespace(
            sample_id=sample_id,
            page=1,
            dataflow_root=str(dataflow),
            input_json=None,
            output_json=None,
            clean_image=None,
            layout_json=None,
            min_score=0.5,
            top_strip_score=0.6,
            dense_ink_ratio=0.16,
            dense_thick_ink_ratio=0.14,
            save_overlay=False,
        )
    )

    assert json.loads(raw_path.read_text(encoding="utf-8")) == raw_payload
    filtered = json.loads(filtered_path.read_text(encoding="utf-8"))
    assert filtered["views"][0]["source_view_id"] == "view_009"
    assert filtered["filter"]["accepted_count"] == 1


def _draw_header_table(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle(box, outline="black", width=3)
    for index in range(1, 5):
        x = x1 + (x2 - x1) * index // 5
        draw.line((x, y1, x, y2), fill="black", width=2)
    draw.line((x1, (y1 + y2) // 2, x2, (y1 + y2) // 2), fill="black", width=2)


def _draw_line_view(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cy = (y1 + y2) // 2
    draw.rectangle((x1 + 70, cy - 20, x2 - 70, cy + 20), outline="black", width=3)
    for x in (x1 + 190, x2 - 190):
        draw.ellipse((x - 14, cy - 14, x + 14, cy + 14), outline="black", width=3)
        draw.line((x, cy - 40, x, cy + 40), fill="black", width=1)
    draw.line((x1 + 70, y2 - 15, x2 - 70, y2 - 15), fill="black", width=2)
    draw.text((x1 + 330, y2 - 45), "375", fill="black")


def _draw_dense_text_block(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1 + 5, y1 + 20, x1 + 55, y2 - 15), fill="black")
    draw.rectangle((x1 + 90, y1 + 15, x2 - 10, y1 + 55), fill="black")
    draw.rectangle((x1 + 85, y1 + 75, x2 - 30, y1 + 105), fill="black")
    draw.rectangle((x1 + 100, y1 + 125, x2 - 20, y2 - 10), fill="black")
