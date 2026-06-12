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
    type_source: str | None = None
    type_confidence: float | None = None
    type_candidates: list[dict[str, Any]] = field(default_factory=list)
    is_primary: bool = False
    needs_manual_review: bool = True
    review_reasons: list[str] = field(default_factory=list)
    crop_size: dict[str, int] | None = None
    detector: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)


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
    schema: str = "drawing_ir"
    version: str = "0.1.0"
    views: list[ViewIR] = field(default_factory=list)
    dimensions: list[DimensionIR] = field(default_factory=list)
    feature_candidates: list[FeatureCandidateIR] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    view_relations: list[dict[str, Any]] = field(default_factory=list)
    skipped_views: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
