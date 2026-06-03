from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    sample_id: str
    tasks: list[str]
    input_images: list[Path] = field(default_factory=list)
    ground_truth: dict[str, Any] = field(default_factory=dict)


def read_split(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            cases.append(
                BenchmarkCase(
                    sample_id=item["sample_id"],
                    tasks=list(item.get("tasks", [])),
                    input_images=[Path(p) for p in item.get("input_images", [])],
                    ground_truth=dict(item.get("ground_truth", {})),
                )
            )
    return cases


def default_page_image(sample_id: str, dpi: int = 600, dataflow_root: str | Path = "DataFlow") -> Path:
    return Path(dataflow_root) / "02.RawPNG" / sample_id / f"page_001_{dpi}dpi.png"

