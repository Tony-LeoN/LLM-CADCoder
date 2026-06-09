from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .view_filter import ViewFilterConfig, filter_view_detections_file

_RAW_NAME_RE = re.compile(r"^page_(\d+)_views_raw\.json$")
_FILTERED_NAME_RE = re.compile(r"^page_(\d+)_views\.json$")


@dataclass(frozen=True)
class BatchViewFilterRecord:
    sample_id: str
    page: int
    input_path: Path
    output_path: Path | None
    accepted_views: int = 0
    rejected_candidates: int = 0
    error: str | None = None


@dataclass(frozen=True)
class BatchViewFilterSummary:
    records: list[BatchViewFilterRecord]

    @property
    def filtered_count(self) -> int:
        return sum(1 for record in self.records if record.error is None)

    @property
    def failed_count(self) -> int:
        return sum(1 for record in self.records if record.error is not None)


def filter_view_detection_directory(
    *,
    dataflow_root: str | Path = "DataFlow",
    view_detection_dir: str | Path | None = None,
    config: ViewFilterConfig | None = None,
    save_overlay: bool = True,
    fail_fast: bool = False,
) -> BatchViewFilterSummary:
    dataflow_root = Path(dataflow_root)
    view_detection_dir = Path(view_detection_dir) if view_detection_dir else dataflow_root / "05.ViewDetection"
    records: list[BatchViewFilterRecord] = []

    for sample_dir, page, input_path, output_path in _iter_detection_inputs(view_detection_dir):
        try:
            result = filter_view_detections_file(
                detection_path=input_path,
                dataflow_root=dataflow_root,
                output_path=output_path,
                config=config,
                save_overlay=save_overlay,
            )
            records.append(
                BatchViewFilterRecord(
                    sample_id=sample_dir.name,
                    page=page,
                    input_path=input_path,
                    output_path=result.filtered_path,
                    accepted_views=len(result.accepted_views),
                    rejected_candidates=len(result.rejected_views),
                )
            )
        except Exception as exc:  # pragma: no cover - exact exception type is data dependent
            if fail_fast:
                raise
            records.append(
                BatchViewFilterRecord(
                    sample_id=sample_dir.name,
                    page=page,
                    input_path=input_path,
                    output_path=output_path,
                    error=str(exc),
                )
            )

    return BatchViewFilterSummary(records=records)


def _iter_detection_inputs(view_detection_dir: Path) -> list[tuple[Path, int, Path, Path]]:
    inputs: list[tuple[Path, int, Path, Path]] = []
    if not view_detection_dir.exists():
        return inputs

    for sample_dir in sorted(path for path in view_detection_dir.iterdir() if path.is_dir()):
        pages: dict[int, tuple[Path, Path]] = {}
        for path in sorted(sample_dir.iterdir()):
            if not path.is_file():
                continue
            raw_match = _RAW_NAME_RE.match(path.name)
            if raw_match:
                page = int(raw_match.group(1))
                pages[page] = (path, sample_dir / f"page_{page:03d}_views.json")
                continue

            filtered_match = _FILTERED_NAME_RE.match(path.name)
            if filtered_match:
                page = int(filtered_match.group(1))
                pages.setdefault(page, (path, path))

        for page in sorted(pages):
            input_path, output_path = pages[page]
            inputs.append((sample_dir, page, input_path, output_path))

    return inputs
