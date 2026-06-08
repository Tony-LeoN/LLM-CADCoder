from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .layout_clean import clean_layout_page


Cleaner = Callable[..., Any]
_RAW_PAGE_RE = re.compile(r"^page_(?P<page>\d{3})_(?P<dpi>\d+)dpi\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class BatchLayoutRecord:
    sample_id: str
    page: int
    image_path: Path
    cleaned: bool = False
    skipped: bool = False
    regions: int = 0
    error: str | None = None


@dataclass(frozen=True)
class BatchLayoutSummary:
    records: list[BatchLayoutRecord]

    @property
    def cleaned_count(self) -> int:
        return sum(1 for record in self.records if record.cleaned and record.error is None)

    @property
    def skipped_count(self) -> int:
        return sum(1 for record in self.records if record.skipped and record.error is None)

    @property
    def failed_count(self) -> int:
        return sum(1 for record in self.records if record.error is not None)


def clean_layout_directory(
    raw_png_dir: str | Path,
    dataflow_root: str | Path = "DataFlow",
    dpi: int = 600,
    skip_existing: bool = False,
    save_crops: bool = True,
    save_overlay: bool = True,
    fail_fast: bool = False,
    cleaner: Cleaner = clean_layout_page,
) -> BatchLayoutSummary:
    """Clean every rendered page PNG under the RawPNG stage."""
    raw_root = Path(raw_png_dir)
    dataflow = Path(dataflow_root)
    records: list[BatchLayoutRecord] = []

    for image_path, sample_id, page in _discover_rendered_pages(raw_root, dpi=dpi):
        output_stem = f"page_{page:03d}"
        clean_path = dataflow / "04.CleanPNG" / sample_id / f"{output_stem}_clean.png"
        if skip_existing and clean_path.exists():
            records.append(
                BatchLayoutRecord(
                    sample_id=sample_id,
                    page=page,
                    image_path=image_path,
                    skipped=True,
                )
            )
            continue

        try:
            result = cleaner(
                image_path=image_path,
                dataflow_root=dataflow,
                sample_id=sample_id,
                page=page,
                output_stem=output_stem,
                save_crops=save_crops,
                save_overlay=save_overlay,
            )
        except Exception as exc:
            if fail_fast:
                raise
            records.append(
                BatchLayoutRecord(
                    sample_id=sample_id,
                    page=page,
                    image_path=image_path,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        records.append(
            BatchLayoutRecord(
                sample_id=sample_id,
                page=page,
                image_path=image_path,
                cleaned=True,
                regions=len(getattr(result, "regions", [])),
            )
        )

    return BatchLayoutSummary(records=records)


def _discover_rendered_pages(raw_root: Path, dpi: int) -> list[tuple[Path, str, int]]:
    pages: list[tuple[Path, str, int]] = []
    for sample_dir in sorted((path for path in raw_root.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
        for image_path in sorted(sample_dir.glob(f"page_*_{dpi}dpi.png"), key=lambda path: path.name.lower()):
            match = _RAW_PAGE_RE.match(image_path.name)
            if match is None or int(match.group("dpi")) != dpi:
                continue
            pages.append((image_path, sample_dir.name, int(match.group("page"))))
    return pages
