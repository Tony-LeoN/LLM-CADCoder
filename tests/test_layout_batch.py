from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vlm_cadcoder.dataflow.layout_batch import clean_layout_directory


def test_clean_layout_directory_processes_rendered_pages_in_order(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    raw_png = dataflow / "02.RawPNG"
    (raw_png / "B-Part").mkdir(parents=True)
    (raw_png / "A-Part").mkdir(parents=True)
    (raw_png / "B-Part" / "page_002_600dpi.png").write_bytes(b"png")
    (raw_png / "A-Part" / "page_001_600dpi.png").write_bytes(b"png")
    (raw_png / "A-Part" / "page_001_300dpi.png").write_bytes(b"png")

    calls: list[dict] = []

    def fake_cleaner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(regions=[])

    summary = clean_layout_directory(
        raw_png_dir=raw_png,
        dataflow_root=dataflow,
        dpi=600,
        cleaner=fake_cleaner,
    )

    assert [(record.sample_id, record.page) for record in summary.records] == [
        ("A-Part", 1),
        ("B-Part", 2),
    ]
    assert summary.cleaned_count == 2
    assert summary.failed_count == 0
    assert calls[0]["sample_id"] == "A-Part"
    assert calls[0]["page"] == 1
    assert calls[0]["output_stem"] == "page_001"
    assert calls[1]["sample_id"] == "B-Part"
    assert calls[1]["page"] == 2
    assert calls[1]["output_stem"] == "page_002"


def test_clean_layout_directory_can_skip_existing_clean_page(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    raw_png = dataflow / "02.RawPNG" / "A-Part"
    raw_png.mkdir(parents=True)
    (raw_png / "page_001_600dpi.png").write_bytes(b"png")
    clean_page = dataflow / "04.CleanPNG" / "A-Part" / "page_001_clean.png"
    clean_page.parent.mkdir(parents=True)
    clean_page.write_bytes(b"png")

    def fake_cleaner(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("cleaner should not be called for existing output")

    summary = clean_layout_directory(
        raw_png_dir=dataflow / "02.RawPNG",
        dataflow_root=dataflow,
        skip_existing=True,
        cleaner=fake_cleaner,
    )

    assert summary.cleaned_count == 0
    assert summary.skipped_count == 1
    assert summary.records[0].skipped is True
