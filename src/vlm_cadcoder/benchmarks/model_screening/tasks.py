from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vlm_cadcoder.models.base import ImageInput


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    prompt_path: Path
    image_level: str
    metric: str

    def load_prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")


def default_tasks(prompt_dir: str | Path) -> dict[str, BenchmarkTask]:
    root = Path(prompt_dir)
    return {
        "view_count": BenchmarkTask("view_count", root / "view_count.md", "page_png", "exact_match"),
        "view_classification": BenchmarkTask(
            "view_classification", root / "view_classification.md", "page_png", "accuracy"
        ),
        "dimension_ocr": BenchmarkTask("dimension_ocr", root / "dimension_ocr.md", "annotated_view_png", "text_f1"),
        "feature_count": BenchmarkTask("feature_count", root / "feature_count.md", "clean_view_png", "count_accuracy"),
        "json_stability": BenchmarkTask("json_stability", root / "json_stability.md", "any_image", "parse_rate"),
    }


def build_image_inputs(image_paths: list[str | Path]) -> list[ImageInput]:
    return [ImageInput(path=Path(path)) for path in image_paths]

