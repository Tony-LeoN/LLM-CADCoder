from __future__ import annotations

import csv
import json
from pathlib import Path

from vlm_cadcoder.dataflow.single_view_audit import audit_single_views


def test_audit_single_views_reports_consistent_official_sample(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_detection(dataflow, "Part-A", views=2, rejected=1)
    _write_single_views(dataflow, "Part-A", views=2)

    summary = audit_single_views(dataflow_root=dataflow)

    assert summary.total_count == 1
    assert summary.review_count == 0
    record = summary.records[0]
    assert record.sample_id == "Part-A"
    assert record.detected_view_count == 2
    assert record.rejected_view_count == 1
    assert record.exported_view_count == 2
    assert record.metadata_count == 2
    assert record.clean_image_count == 2
    assert record.is_05_06_consistent is True
    assert record.needs_manual_review is False
    assert record.review_reasons == []


def test_audit_single_views_flags_mismatches_and_copy_samples(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_detection(dataflow, "Part-B", views=2)
    _write_single_views(dataflow, "Part-B", views=1)
    _write_single_views(dataflow, "Part-B-copy", views=2)

    summary = audit_single_views(dataflow_root=dataflow)
    by_sample = {record.sample_id: record for record in summary.records}

    assert summary.total_count == 2
    assert summary.review_count == 2
    assert by_sample["Part-B"].needs_manual_review is True
    assert "view_count_mismatch" in by_sample["Part-B"].review_reasons
    assert by_sample["Part-B-copy"].official_candidate is False
    assert "copy_sample" in by_sample["Part-B-copy"].review_reasons
    assert "missing_view_detection" in by_sample["Part-B-copy"].review_reasons


def test_audit_single_views_writes_csv_and_json(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_detection(dataflow, "Part-C", views=1)
    _write_single_views(dataflow, "Part-C", views=1)

    csv_path = dataflow / "06.SingleViews" / "audit.csv"
    json_path = dataflow / "06.SingleViews" / "audit.json"

    summary = audit_single_views(
        dataflow_root=dataflow,
        output_csv=csv_path,
        output_json=json_path,
    )

    assert summary.csv_path == csv_path
    assert summary.json_path == json_path
    assert csv_path.exists()
    assert json_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sample_id"] == "Part-C"
    assert rows[0]["detected_view_count"] == "1"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["summary"]["total_count"] == 1
    assert data["records"][0]["sample_id"] == "Part-C"


def _write_detection(dataflow: Path, sample_id: str, *, views: int, rejected: int = 0) -> None:
    target = dataflow / "05.ViewDetection" / sample_id
    target.mkdir(parents=True)
    (target / "page_001_views.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "page": 1,
                "views": [{"view_id": f"view_{index:03d}"} for index in range(1, views + 1)],
            }
        ),
        encoding="utf-8",
    )
    if rejected:
        (target / "page_001_rejected_views.json").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "page": 1,
                    "views": [{"view_id": f"rejected_{index:03d}"} for index in range(1, rejected + 1)],
                }
            ),
            encoding="utf-8",
        )


def _write_single_views(dataflow: Path, sample_id: str, *, views: int) -> None:
    sample_dir = dataflow / "06.SingleViews" / sample_id
    sample_dir.mkdir(parents=True)
    for index in range(1, views + 1):
        view_dir = sample_dir / f"view_{index:03d}"
        view_dir.mkdir()
        (view_dir / "clean_view_with_annotations.png").write_bytes(b"not-a-real-png")
        (view_dir / "view_metadata.json").write_text(
            json.dumps({"sample_id": sample_id, "view_id": f"view_{index:03d}"}),
            encoding="utf-8",
        )
