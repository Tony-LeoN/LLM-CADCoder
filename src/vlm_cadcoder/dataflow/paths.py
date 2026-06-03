from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataFlowPaths:
    root: Path = Path("DataFlow")

    raw_pdf_with_step: str = "01.RawPDFWithSTEP"
    raw_png: str = "02.RawPNG"
    layout_analysis: str = "03.LayoutAnalysis"
    clean_png: str = "04.CleanPNG"
    view_detection: str = "05.ViewDetection"
    single_views: str = "06.SingleViews"
    view_classification: str = "07.ViewClassification"
    feature_extraction: str = "08.Multi-viewFeatureExtraction"
    cross_view_reasoning: str = "09.Cross-viewGeometricReasoning"
    structured_cad: str = "10.StructuredCADRepresentation"
    cad_program: str = "11.CADProgram"

    def stage(self, name: str) -> Path:
        value = getattr(self, name)
        return self.root / value

    def sample_dir(self, stage_name: str, sample_id: str) -> Path:
        return self.stage(stage_name) / sample_id

