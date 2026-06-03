from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vlm_cadcoder.utils.json_utils import write_json

from .artifact_store import ArtifactStore


@dataclass(frozen=True)
class RenderedPage:
    sample_id: str
    source_pdf: Path
    page_number: int
    dpi: int
    image_path: Path
    meta_path: Path
    width_px: int
    height_px: int


def render_pdf_pages(
    pdf_path: str | Path,
    sample_id: str,
    store: ArtifactStore,
    dpi: int = 600,
    pages: list[int] | None = None,
    skip_multipage: bool = False,
) -> list[RenderedPage]:
    """Render PDF pages to PNG and write coordinate metadata.

    Page numbers are 1-based in the public API.
    """
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PDF rendering requires PyMuPDF on the server.") from exc

    source_pdf = Path(pdf_path)
    doc = fitz.open(source_pdf)
    if skip_multipage and doc.page_count != 1:
        return []

    page_numbers = pages or list(range(1, doc.page_count + 1))
    output_dir = store.ensure_sample_dir("raw_png", sample_id)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    rendered: list[RenderedPage] = []
    for page_number in page_numbers:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = output_dir / f"page_{page_number:03d}_{dpi}dpi.png"
        meta_path = output_dir / f"page_{page_number:03d}_{dpi}dpi.meta.json"
        pix.save(image_path)

        meta = {
            "sample_id": sample_id,
            "source_pdf": source_pdf.as_posix(),
            "page": page_number,
            "dpi": dpi,
            "scale": zoom,
            "width_px": pix.width,
            "height_px": pix.height,
            "coordinate_system": "image_xy_top_left",
        }
        write_json(meta_path, meta)

        rendered.append(
            RenderedPage(
                sample_id=sample_id,
                source_pdf=source_pdf,
                page_number=page_number,
                dpi=dpi,
                image_path=image_path,
                meta_path=meta_path,
                width_px=pix.width,
                height_px=pix.height,
            )
        )
    return rendered

