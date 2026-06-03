from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import DataFlowPaths


@dataclass
class ArtifactStore:
    paths: DataFlowPaths

    def ensure_sample_dir(self, stage_name: str, sample_id: str) -> Path:
        directory = self.paths.sample_dir(stage_name, sample_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def page_png(self, sample_id: str, page: int = 1, dpi: int = 600) -> Path:
        return self.paths.sample_dir("raw_png", sample_id) / f"page_{page:03d}_{dpi}dpi.png"

    def render_meta(self, sample_id: str, page: int = 1, dpi: int = 600) -> Path:
        return self.paths.sample_dir("raw_png", sample_id) / f"page_{page:03d}_{dpi}dpi.meta.json"

