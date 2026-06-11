from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ViewType = Literal["front", "top", "left", "right", "bottom", "section", "detail", "isometric", "unknown"]


@dataclass
class ViewIR:
    id: str
    type: ViewType
    bbox: list[int]
    image_raw: str | None = None
    image_clean: str | None = None


@dataclass
class DimensionIR:
    id: str
    text: str
    dimension_type: str
    bbox: list[int] | None = None
    view_id: str | None = None
    value: float | None = None
    quantity: int | None = None
    unit: str = "mm"


@dataclass
class FeatureCandidateIR:
    id: str
    type: str
    view_id: str
    count: int | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class DrawingIR:
    sample_id: str
    sheet: dict[str, Any]
    views: list[ViewIR] = field(default_factory=list)
    dimensions: list[DimensionIR] = field(default_factory=list)
    feature_candidates: list[FeatureCandidateIR] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
