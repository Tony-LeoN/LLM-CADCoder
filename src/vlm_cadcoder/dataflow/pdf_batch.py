from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_store import ArtifactStore
from .pdf_render import render_pdf_pages


Renderer = Callable[..., list[Any]]


@dataclass(frozen=True)
class BatchRenderRecord:
    sample_id: str
    pdf_path: Path
    rendered_pages: int
    skipped: bool = False
    error: str | None = None


@dataclass(frozen=True)
class BatchRenderSummary:
    records: list[BatchRenderRecord]

    @property
    def rendered_pdf_count(self) -> int:
        return sum(1 for record in self.records if record.rendered_pages > 0 and record.error is None)

    @property
    def rendered_page_count(self) -> int:
        return sum(record.rendered_pages for record in self.records if record.error is None)

    @property
    def skipped_count(self) -> int:
        return sum(1 for record in self.records if record.skipped and record.error is None)

    @property
    def failed_count(self) -> int:
        return sum(1 for record in self.records if record.error is not None)


def render_pdf_directory(
    raw_dir: str | Path,
    store: ArtifactStore,
    dpi: int = 600,
    skip_multipage: bool = False,
    skip_existing: bool = False,
    recursive: bool = False,
    fail_fast: bool = False,
    renderer: Renderer = render_pdf_pages,
) -> BatchRenderSummary:
    """Render every PDF in a directory into the DataFlow RawPNG stage."""
    root = Path(raw_dir)
    pdf_paths = _discover_pdfs(root, recursive=recursive)
    records: list[BatchRenderRecord] = []

    for pdf_path in pdf_paths:
        sample_id = _sample_id_for_pdf(root, pdf_path, recursive=recursive)
        if skip_existing and store.page_png(sample_id, page=1, dpi=dpi).exists():
            records.append(
                BatchRenderRecord(
                    sample_id=sample_id,
                    pdf_path=pdf_path,
                    rendered_pages=0,
                    skipped=True,
                )
            )
            continue

        try:
            rendered = renderer(
                pdf_path=pdf_path,
                sample_id=sample_id,
                store=store,
                dpi=dpi,
                skip_multipage=skip_multipage,
            )
        except Exception as exc:
            if fail_fast:
                raise
            records.append(
                BatchRenderRecord(
                    sample_id=sample_id,
                    pdf_path=pdf_path,
                    rendered_pages=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        records.append(
            BatchRenderRecord(
                sample_id=sample_id,
                pdf_path=pdf_path,
                rendered_pages=len(rendered),
                skipped=len(rendered) == 0,
            )
        )

    return BatchRenderSummary(records=records)


def _discover_pdfs(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted((path for path in root.glob(pattern) if path.is_file()), key=lambda path: path.as_posix().lower())


def _sample_id_for_pdf(root: Path, pdf_path: Path, recursive: bool) -> str:
    if not recursive:
        return pdf_path.stem
    relative_stem = pdf_path.relative_to(root).with_suffix("")
    return "__".join(relative_stem.parts)
