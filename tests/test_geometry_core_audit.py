from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from vlm_cadcoder.dataflow.geometry_core_audit import audit_geometry_core


def test_audit_geometry_core_writes_metrics_and_contact_sheet(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-A" / "view_001"
    view_dir.mkdir(parents=True)
    _write_clean_view(view_dir / "clean_view_with_annotations.png")
    _write_geometry_core(view_dir / "geometry_core.png")
    Image.new("L", (160, 120), 255).save(view_dir / "geometry_core_mask.png")
    Image.new("L", (160, 120), 255).save(view_dir / "geometry_core_prob.png")

    summary = audit_geometry_core(dataflow_root=dataflow, use_view_classification=False)

    assert summary.total_count == 1
    assert summary.records[0].sample_id == "Part-A"
    assert summary.records[0].view_id == "view_001"
    assert summary.records[0].has_geometry_core is True
    assert summary.records[0].size_matches is True
    assert summary.records[0].geometry_black_ratio > 0
    assert summary.records[0].retained_ink_ratio is not None
    assert summary.records[0].quality_tier in {"A", "B"}
    assert summary.csv_path == dataflow / "06.SingleViews" / "geometry_core_audit.csv"
    assert summary.json_path == dataflow / "06.SingleViews" / "geometry_core_audit.json"
    assert summary.contact_sheet_path == dataflow / "06.SingleViews" / "geometry_core_audit_contact_sheet.png"
    assert summary.csv_path.exists()
    assert summary.json_path.exists()
    assert summary.contact_sheet_path.exists()

    with summary.csv_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["sample_id"] == "Part-A"
    assert row["quality_tier"] in {"A", "B"}

    data = json.loads(summary.json_path.read_text(encoding="utf-8"))
    assert data["summary"]["total_count"] == 1
    assert data["records"][0]["paths"]["geometry_core"].endswith("geometry_core.png")


def test_audit_geometry_core_flags_missing_outputs_and_skips_copy_by_default(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    missing_view_dir = dataflow / "06.SingleViews" / "Part-B" / "view_001"
    missing_view_dir.mkdir(parents=True)
    _write_clean_view(missing_view_dir / "clean_view_with_annotations.png")

    copy_view_dir = dataflow / "06.SingleViews" / "Part-B-copy" / "view_001"
    copy_view_dir.mkdir(parents=True)
    _write_clean_view(copy_view_dir / "clean_view_with_annotations.png")
    _write_geometry_core(copy_view_dir / "geometry_core.png")

    summary = audit_geometry_core(dataflow_root=dataflow, use_view_classification=False, save_contact_sheet=False)

    assert [record.sample_id for record in summary.records] == ["Part-B"]
    record = summary.records[0]
    assert record.has_geometry_core is False
    assert record.quality_tier == "C"
    assert record.needs_manual_review is True
    assert "missing_geometry_core" in record.review_reasons
    assert summary.contact_sheet_path is None


def test_audit_geometry_core_uses_view_classification_filter_by_default(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    for view_id in ("view_001", "view_002", "view_003", "view_004"):
        view_dir = dataflow / "06.SingleViews" / "Part-C" / view_id
        view_dir.mkdir(parents=True)
        _write_clean_view(view_dir / "clean_view_with_annotations.png")
        if view_id != "view_004":
            _write_geometry_core(view_dir / "geometry_core.png")
            Image.new("L", (160, 120), 255).save(view_dir / "geometry_core_mask.png")
            Image.new("L", (160, 120), 255).save(view_dir / "geometry_core_prob.png")

    classification_dir = dataflow / "07.ViewClassification" / "Part-C"
    classification_dir.mkdir(parents=True)
    (classification_dir / "page_001_view_classification.json").write_text(
        json.dumps(
            {
                "sample_id": "Part-C",
                "page": 1,
                "views": [
                    {"view_id": "view_002", "type": "front", "confidence": 0.68, "is_primary": True},
                    {"view_id": "view_003", "type": "isometric", "confidence": 0.55, "is_primary": False},
                    {"view_id": "view_004", "type": "left", "confidence": 0.58, "is_primary": False},
                ],
                "skipped_views": [
                    {"view_id": "view_001", "reason": "not_in_05_accepted_views"},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = audit_geometry_core(dataflow_root=dataflow, save_contact_sheet=False)

    assert [(record.view_id, record.view_type) for record in summary.records] == [
        ("view_002", "front"),
        ("view_004", "left"),
    ]
    assert summary.tier_counts == {"A": 1, "B": 0, "C": 1}
    assert "missing_geometry_core" in summary.records[1].review_reasons


def test_audit_geometry_core_skips_samples_without_classification_by_default(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-D" / "view_001"
    view_dir.mkdir(parents=True)
    _write_clean_view(view_dir / "clean_view_with_annotations.png")
    _write_geometry_core(view_dir / "geometry_core.png")

    default_summary = audit_geometry_core(dataflow_root=dataflow)
    raw_summary = audit_geometry_core(
        dataflow_root=dataflow,
        use_view_classification=False,
        save_contact_sheet=False,
    )

    assert default_summary.records == []
    assert default_summary.contact_sheet_path is None
    default_data = json.loads(default_summary.json_path.read_text(encoding="utf-8"))
    assert default_data["summary"]["contact_sheet_path"] is None
    assert [record.view_id for record in raw_summary.records] == ["view_001"]


def test_audit_geometry_core_skips_stale_classification_views_missing_in_single_views(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-E" / "view_001"
    view_dir.mkdir(parents=True)
    _write_clean_view(view_dir / "clean_view_with_annotations.png")
    _write_geometry_core(view_dir / "geometry_core.png")

    classification_dir = dataflow / "07.ViewClassification" / "Part-E"
    classification_dir.mkdir(parents=True)
    (classification_dir / "page_001_view_classification.json").write_text(
        json.dumps(
            {
                "sample_id": "Part-E",
                "page": 1,
                "views": [
                    {"view_id": "view_001", "type": "front", "confidence": 0.68, "is_primary": True},
                    {"view_id": "view_002", "type": "left", "confidence": 0.58, "is_primary": False},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = audit_geometry_core(dataflow_root=dataflow, save_contact_sheet=False)

    assert [record.view_id for record in summary.records] == ["view_001"]


def test_audit_geometry_core_applies_manual_overrides(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    for view_id in ("view_001", "view_002", "view_003"):
        view_dir = dataflow / "06.SingleViews" / "Part-F" / view_id
        view_dir.mkdir(parents=True)
        _write_clean_view(view_dir / "clean_view_with_annotations.png")
        _write_geometry_core(view_dir / "geometry_core.png")
        Image.new("L", (160, 120), 255).save(view_dir / "geometry_core_mask.png")
        Image.new("L", (160, 120), 255).save(view_dir / "geometry_core_prob.png")

    classification_dir = dataflow / "07.ViewClassification" / "Part-F"
    classification_dir.mkdir(parents=True)
    (classification_dir / "page_001_view_classification.json").write_text(
        json.dumps(
            {
                "sample_id": "Part-F",
                "page": 1,
                "views": [
                    {"view_id": "view_001", "type": "front", "confidence": 0.68, "is_primary": True},
                    {"view_id": "view_002", "type": "left", "confidence": 0.58, "is_primary": False},
                    {"view_id": "view_003", "type": "top", "confidence": 0.58, "is_primary": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    override_path = dataflow / "06.SingleViews" / "geometry_core_audit_overrides.json"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "sample_id": "Part-F",
                        "view_id": "view_001",
                        "exclude": True,
                        "reason": "manual_false_view",
                    },
                    {
                        "sample_id": "Part-F",
                        "view_id": "view_002",
                        "quality_tier": "C",
                        "reason": "manual_bad_geometry_core",
                    },
                    {
                        "sample_id": "Part-F",
                        "view_id": "view_003",
                        "quality_tier": "A",
                        "reason": "manual_good_geometry_core",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = audit_geometry_core(dataflow_root=dataflow, save_contact_sheet=False)

    by_view = {record.view_id: record for record in summary.records}
    assert set(by_view) == {"view_002", "view_003"}
    assert by_view["view_002"].quality_tier == "C"
    assert by_view["view_002"].manual_quality_label == "C"
    assert "manual_bad_geometry_core" in by_view["view_002"].review_reasons
    assert by_view["view_003"].quality_tier == "A"
    assert by_view["view_003"].manual_quality_label == "A"
    assert by_view["view_003"].needs_manual_review is False
    assert by_view["view_003"].review_reasons == []


def _write_clean_view(path: Path) -> None:
    image = Image.new("L", (160, 120), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 120, 85), outline=0, width=3)
    draw.line((30, 95, 120, 95), fill=0, width=2)
    draw.text((65, 98), "80", fill=0)
    image.save(path)


def _write_geometry_core(path: Path) -> None:
    image = Image.new("L", (160, 120), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 120, 85), outline=0, width=3)
    image.save(path)
