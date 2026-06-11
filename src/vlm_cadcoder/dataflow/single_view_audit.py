from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vlm_cadcoder.utils.json_utils import write_json

_VIEW_DIR_RE = re.compile(r"^view_\d+$")


@dataclass(frozen=True)
class SingleViewAuditRecord:
    sample_id: str
    page: int
    official_candidate: bool
    has_view_detection: bool
    has_single_views: bool
    detected_view_count: int
    rejected_view_count: int
    exported_view_count: int
    metadata_count: int
    clean_image_count: int
    is_05_06_consistent: bool
    needs_manual_review: bool
    review_reasons: list[str]
    view_detection_path: str | None
    single_views_dir: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["review_reasons"] = list(self.review_reasons)
        return data


@dataclass(frozen=True)
class SingleViewAuditSummary:
    records: list[SingleViewAuditRecord]
    csv_path: Path | None = None
    json_path: Path | None = None

    @property
    def total_count(self) -> int:
        return len(self.records)

    @property
    def review_count(self) -> int:
        return sum(1 for record in self.records if record.needs_manual_review)

    @property
    def consistent_count(self) -> int:
        return sum(1 for record in self.records if record.is_05_06_consistent)

    @property
    def official_count(self) -> int:
        return sum(1 for record in self.records if record.official_candidate)


def audit_single_views(
    *,
    dataflow_root: str | Path = "DataFlow",
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
    page: int = 1,
) -> SingleViewAuditSummary:
    root = Path(dataflow_root)
    view_detection_root = root / "05.ViewDetection"
    single_views_root = root / "06.SingleViews"
    sample_ids = _collect_sample_ids(view_detection_root, single_views_root, page)

    records = [
        _audit_sample(
            root=root,
            sample_id=sample_id,
            page=page,
        )
        for sample_id in sample_ids
    ]
    summary = SingleViewAuditSummary(records=records)

    csv_path = Path(output_csv) if output_csv else single_views_root / "audit_single_views.csv"
    json_path = Path(output_json) if output_json else single_views_root / "audit_single_views.json"
    _write_csv(csv_path, records)
    _write_json(json_path, summary, csv_path)

    return SingleViewAuditSummary(records=records, csv_path=csv_path, json_path=json_path)


def _collect_sample_ids(view_detection_root: Path, single_views_root: Path, page: int) -> list[str]:
    sample_ids: set[str] = set()
    if view_detection_root.exists():
        for sample_dir in view_detection_root.iterdir():
            if sample_dir.is_dir() and _detection_path(sample_dir, page).exists():
                sample_ids.add(sample_dir.name)

    if single_views_root.exists():
        for sample_dir in single_views_root.iterdir():
            if sample_dir.is_dir() and _official_view_dirs(sample_dir):
                sample_ids.add(sample_dir.name)

    return sorted(sample_ids)


def _audit_sample(*, root: Path, sample_id: str, page: int) -> SingleViewAuditRecord:
    detection_path = _detection_path(root / "05.ViewDetection" / sample_id, page)
    single_views_dir = root / "06.SingleViews" / sample_id
    has_view_detection = detection_path.exists()
    has_single_views = single_views_dir.exists() and bool(_official_view_dirs(single_views_dir))
    detected_view_count = _count_views(detection_path)
    rejected_view_count = _count_rejected_views(root / "05.ViewDetection" / sample_id, page)
    exported_view_count, metadata_count, clean_image_count = _count_exported_views(single_views_dir)
    official_candidate = not _looks_like_copy_sample(sample_id)
    is_consistent = (
        has_view_detection
        and has_single_views
        and detected_view_count == exported_view_count
        and exported_view_count == metadata_count
        and exported_view_count == clean_image_count
    )
    review_reasons = _review_reasons(
        official_candidate=official_candidate,
        has_view_detection=has_view_detection,
        has_single_views=has_single_views,
        detected_view_count=detected_view_count,
        exported_view_count=exported_view_count,
        metadata_count=metadata_count,
        clean_image_count=clean_image_count,
    )

    return SingleViewAuditRecord(
        sample_id=sample_id,
        page=page,
        official_candidate=official_candidate,
        has_view_detection=has_view_detection,
        has_single_views=has_single_views,
        detected_view_count=detected_view_count,
        rejected_view_count=rejected_view_count,
        exported_view_count=exported_view_count,
        metadata_count=metadata_count,
        clean_image_count=clean_image_count,
        is_05_06_consistent=is_consistent,
        needs_manual_review=bool(review_reasons),
        review_reasons=review_reasons,
        view_detection_path=detection_path.as_posix() if has_view_detection else None,
        single_views_dir=single_views_dir.as_posix() if has_single_views else None,
    )


def _detection_path(sample_dir: Path, page: int) -> Path:
    return sample_dir / f"page_{page:03d}_views.json"


def _count_views(path: Path) -> int:
    data = _read_json_or_empty(path)
    views = data.get("views")
    return len(views) if isinstance(views, list) else 0


def _count_rejected_views(sample_dir: Path, page: int) -> int:
    path = sample_dir / f"page_{page:03d}_rejected_views.json"
    data = _read_json_or_empty(path)
    views = data.get("views")
    if isinstance(views, list):
        return len(views)
    rejected_views = data.get("rejected_views")
    return len(rejected_views) if isinstance(rejected_views, list) else 0


def _count_exported_views(sample_dir: Path) -> tuple[int, int, int]:
    view_dirs = _official_view_dirs(sample_dir)
    metadata_count = sum(1 for view_dir in view_dirs if (view_dir / "view_metadata.json").exists())
    clean_image_count = sum(1 for view_dir in view_dirs if (view_dir / "clean_view_with_annotations.png").exists())
    return len(view_dirs), metadata_count, clean_image_count


def _official_view_dirs(sample_dir: Path) -> list[Path]:
    if not sample_dir.exists() or not sample_dir.is_dir():
        return []
    return sorted(path for path in sample_dir.iterdir() if path.is_dir() and _VIEW_DIR_RE.match(path.name))


def _looks_like_copy_sample(sample_id: str) -> bool:
    lowered = sample_id.lower()
    return lowered.endswith("-copy") or lowered.endswith("_copy") or " copy" in lowered


def _review_reasons(
    *,
    official_candidate: bool,
    has_view_detection: bool,
    has_single_views: bool,
    detected_view_count: int,
    exported_view_count: int,
    metadata_count: int,
    clean_image_count: int,
) -> list[str]:
    reasons: list[str] = []
    if not official_candidate:
        reasons.append("copy_sample")
    if not has_view_detection:
        reasons.append("missing_view_detection")
    if not has_single_views:
        reasons.append("missing_single_views")
    if has_view_detection and has_single_views and detected_view_count != exported_view_count:
        reasons.append("view_count_mismatch")
    if exported_view_count != metadata_count:
        reasons.append("missing_view_metadata")
    if exported_view_count != clean_image_count:
        reasons.append("missing_clean_view_image")
    if has_view_detection and detected_view_count == 0:
        reasons.append("zero_detected_views")
    if has_single_views and exported_view_count == 0:
        reasons.append("zero_exported_views")
    return reasons


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _write_csv(path: Path, records: list[SingleViewAuditRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "page",
        "official_candidate",
        "has_view_detection",
        "has_single_views",
        "detected_view_count",
        "rejected_view_count",
        "exported_view_count",
        "metadata_count",
        "clean_image_count",
        "is_05_06_consistent",
        "needs_manual_review",
        "review_reasons",
        "view_detection_path",
        "single_views_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.to_dict()
            row["review_reasons"] = ";".join(record.review_reasons)
            writer.writerow(row)


def _write_json(path: Path, summary: SingleViewAuditSummary, csv_path: Path) -> None:
    write_json(
        path,
        {
            "summary": {
                "total_count": summary.total_count,
                "official_count": summary.official_count,
                "consistent_count": summary.consistent_count,
                "review_count": summary.review_count,
                "csv_path": csv_path.as_posix(),
            },
            "records": [record.to_dict() for record in summary.records],
        },
    )
