from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def clamp(self, image_width: int, image_height: int) -> BBox:
        return BBox(
            x1=max(0, min(self.x1, image_width)),
            y1=max(0, min(self.y1, image_height)),
            x2=max(0, min(self.x2, image_width)),
            y2=max(0, min(self.y2, image_height)),
        )

    def pad(self, padding: int, image_width: int, image_height: int) -> BBox:
        return BBox(
            self.x1 - padding,
            self.y1 - padding,
            self.x2 + padding,
            self.y2 + padding,
        ).clamp(image_width, image_height)

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class LayoutRegion:
    region_id: str
    region_type: str
    bbox: BBox
    action: str
    preserve_as_crop: bool
    confidence: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "type": self.region_type,
            "bbox": self.bbox.as_list(),
            "action": self.action,
            "preserve_as_crop": self.preserve_as_crop,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class LayoutCleaningResult:
    sample_id: str
    page: int
    image_size: tuple[int, int]
    layout_path: str
    clean_image_path: str
    mask_path: str
    overlay_path: str | None
    regions: list[LayoutRegion]

