from __future__ import annotations

from pathlib import Path

from vlm_cadcoder.dataflow.artifact_store import ArtifactStore
from vlm_cadcoder.dataflow.paths import DataFlowPaths
from vlm_cadcoder.dataflow.pdf_batch import render_pdf_directory


def test_render_pdf_directory_renders_sorted_pdfs_to_sample_dirs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "DataFlow" / "01.RawPDFWithSTEP"
    raw_dir.mkdir(parents=True)
    (raw_dir / "B-Part.pdf").write_bytes(b"%PDF")
    (raw_dir / "A-Part.pdf").write_bytes(b"%PDF")
    (raw_dir / "ignore.txt").write_text("not a pdf", encoding="utf-8")
    store = ArtifactStore(DataFlowPaths(root=tmp_path / "DataFlow"))

    calls: list[tuple[Path, str, int, bool]] = []

    def fake_renderer(**kwargs):
        calls.append(
            (
                Path(kwargs["pdf_path"]),
                kwargs["sample_id"],
                kwargs["dpi"],
                kwargs["skip_multipage"],
            )
        )
        return [object()]

    summary = render_pdf_directory(
        raw_dir=raw_dir,
        store=store,
        dpi=300,
        skip_multipage=True,
        renderer=fake_renderer,
    )

    assert [record.sample_id for record in summary.records] == ["A-Part", "B-Part"]
    assert summary.rendered_pdf_count == 2
    assert summary.rendered_page_count == 2
    assert summary.failed_count == 0
    assert calls == [
        (raw_dir / "A-Part.pdf", "A-Part", 300, True),
        (raw_dir / "B-Part.pdf", "B-Part", 300, True),
    ]


def test_render_pdf_directory_can_skip_existing_first_page(tmp_path: Path) -> None:
    raw_dir = tmp_path / "DataFlow" / "01.RawPDFWithSTEP"
    raw_dir.mkdir(parents=True)
    (raw_dir / "A-Part.pdf").write_bytes(b"%PDF")
    store = ArtifactStore(DataFlowPaths(root=tmp_path / "DataFlow"))
    existing = store.page_png("A-Part", page=1, dpi=600)
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"png")

    def fake_renderer(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("renderer should not be called for existing output")

    summary = render_pdf_directory(
        raw_dir=raw_dir,
        store=store,
        skip_existing=True,
        renderer=fake_renderer,
    )

    assert summary.rendered_pdf_count == 0
    assert summary.skipped_count == 1
    assert summary.records[0].skipped is True
