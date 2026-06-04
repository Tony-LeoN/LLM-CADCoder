from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from vlm_cadcoder.utils.json_utils import write_json

from .layout_schema import BBox, LayoutCleaningResult, LayoutRegion


@dataclass(frozen=True)
class LayoutCleanConfig:
    dark_threshold: int = 190
    neutral_delta: int = 36
    horizontal_min_ratio: float = 0.025
    vertical_min_ratio: float = 0.025
    edge_margin_ratio: float = 0.025
    region_padding_px: int = 22
    border_padding_px: int = 12
    left_table_max_x_ratio: float = 0.48
    bottom_table_min_y_ratio: float = 0.78
    bottom_table_min_x_ratio: float = 0.28
    revision_min_x_ratio: float = 0.62
    revision_max_y_ratio: float = 0.16
    min_grid_segments: int = 8


@dataclass(frozen=True)
class LineSegment:
    axis: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def length(self) -> int:
        return self.x2 - self.x1 if self.axis == "h" else self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


def clean_layout_page(
    image_path: str | Path,
    dataflow_root: str | Path = "DataFlow",
    sample_id: str | None = None,
    page: int = 1,
    output_stem: str | None = None,
    config: LayoutCleanConfig | None = None,
    save_crops: bool = True,
    save_overlay: bool = True,
) -> LayoutCleaningResult:
    """Remove page-frame and table regions while preserving image coordinates."""
    cfg = config or LayoutCleanConfig()
    source = Path(image_path)
    resolved_sample_id = sample_id or source.stem
    stem = output_stem or f"page_{page:03d}"

    root = Path(dataflow_root)
    layout_dir = root / "03.LayoutAnalysis" / resolved_sample_id
    clean_dir = root / "04.CleanPNG" / resolved_sample_id
    crop_dir = layout_dir / "regions"
    layout_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    if save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(source).convert("RGB")
    regions = detect_layout_regions(image, cfg)
    clean_image, mask = whiten_regions(image, regions)

    clean_path = clean_dir / f"{stem}_clean.png"
    mask_path = clean_dir / f"{stem}_remove_mask.png"
    overlay_path = layout_dir / f"{stem}_overlay.png" if save_overlay else None
    layout_path = layout_dir / f"{stem}_layout.json"

    clean_image.save(clean_path)
    mask.save(mask_path)
    if overlay_path:
        draw_overlay(image, regions).save(overlay_path)

    region_dicts = []
    for region in regions:
        item = region.to_dict()
        if save_crops and region.preserve_as_crop and region.bbox.area > 0:
            crop_path = crop_dir / f"{region.region_id}_{region.region_type}.png"
            image.crop(tuple(region.bbox.as_list())).save(crop_path)
            item["crop_path"] = crop_path.as_posix()
        region_dicts.append(item)

    width, height = image.size
    write_json(
        layout_path,
        {
            "sample_id": resolved_sample_id,
            "page": page,
            "source_image": source.as_posix(),
            "image_size": [width, height],
            "coordinate_system": "image_xy_top_left",
            "method": {
                "name": "rule_based_line_layout_cleaner",
                "version": "0.1.0",
                "config": cfg.__dict__,
            },
            "regions": region_dicts,
            "outputs": {
                "clean_image": clean_path.as_posix(),
                "remove_mask": mask_path.as_posix(),
                "overlay": overlay_path.as_posix() if overlay_path else None,
            },
        },
    )

    return LayoutCleaningResult(
        sample_id=resolved_sample_id,
        page=page,
        image_size=(width, height),
        layout_path=layout_path.as_posix(),
        clean_image_path=clean_path.as_posix(),
        mask_path=mask_path.as_posix(),
        overlay_path=overlay_path.as_posix() if overlay_path else None,
        regions=regions,
    )


def detect_layout_regions(image: Image.Image, config: LayoutCleanConfig | None = None) -> list[LayoutRegion]:
    cfg = config or LayoutCleanConfig()
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Layout cleaning requires numpy on the server.") from exc

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    neutral_dark = _neutral_dark_mask(rgb, cfg.dark_threshold, cfg.neutral_delta)
    h_segments = _extract_runs(neutral_dark, "h", max(64, int(width * cfg.horizontal_min_ratio)))
    v_segments = _extract_runs(neutral_dark, "v", max(64, int(height * cfg.vertical_min_ratio)))

    regions: list[LayoutRegion] = []
    next_id = _region_id_factory()
    regions.extend(_detect_border_strips(h_segments, v_segments, width, height, cfg, next_id))
    for region_type, bbox, confidence, meta in _detect_table_regions(h_segments, v_segments, width, height, cfg):
        regions.append(
            LayoutRegion(
                region_id=next_id(),
                region_type=region_type,
                bbox=bbox.pad(cfg.region_padding_px, width, height),
                action="remove_from_clean_page",
                preserve_as_crop=True,
                confidence=confidence,
                source="line_run_grid_rules",
                metadata=meta,
            )
        )
    return _deduplicate_regions(regions)


def whiten_regions(image: Image.Image, regions: Iterable[LayoutRegion]) -> tuple[Image.Image, Image.Image]:
    clean = image.convert("RGB").copy()
    mask = Image.new("L", clean.size, 0)
    clean_draw = ImageDraw.Draw(clean)
    mask_draw = ImageDraw.Draw(mask)
    for region in regions:
        if region.action != "remove_from_clean_page" or region.bbox.area == 0:
            continue
        box = tuple(region.bbox.as_list())
        clean_draw.rectangle(box, fill=(255, 255, 255))
        mask_draw.rectangle(box, fill=255)
    return clean, mask


def draw_overlay(image: Image.Image, regions: Iterable[LayoutRegion]) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    colors = {
        "page_border": (255, 0, 0),
        "hole_table": (255, 128, 0),
        "title_or_tolerance_table": (180, 0, 255),
        "revision_table": (0, 160, 255),
    }
    for region in regions:
        color = colors.get(region.region_type, (0, 200, 0))
        box = tuple(region.bbox.as_list())
        draw.rectangle(box, outline=color, width=8)
        draw.text((region.bbox.x1 + 8, region.bbox.y1 + 8), region.region_type, fill=color)
    return overlay


def _neutral_dark_mask(rgb, dark_threshold: int, neutral_delta: int):
    import numpy as np

    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    return (max_channel < dark_threshold) & ((max_channel - min_channel) <= neutral_delta)


def _extract_runs(mask, axis: str, min_length: int) -> list[LineSegment]:
    import numpy as np

    segments: list[LineSegment] = []
    if axis == "h":
        for y, row in enumerate(mask):
            starts, ends = _runs_1d(row)
            for x1, x2 in zip(starts, ends):
                if x2 - x1 >= min_length:
                    segments.append(LineSegment("h", int(x1), y, int(x2), y + 1))
        return segments

    for x, col in enumerate(mask.T):
        starts, ends = _runs_1d(col)
        for y1, y2 in zip(starts, ends):
            if y2 - y1 >= min_length:
                segments.append(LineSegment("v", x, int(y1), x + 1, int(y2)))
    return segments


def _runs_1d(values) -> tuple[list[int], list[int]]:
    import numpy as np

    padded = np.concatenate(([False], values, [False]))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    return transitions[0::2].tolist(), transitions[1::2].tolist()


def _detect_border_strips(
    h_segments: list[LineSegment],
    v_segments: list[LineSegment],
    width: int,
    height: int,
    cfg: LayoutCleanConfig,
    next_id,
) -> list[LayoutRegion]:
    margin_x = int(width * cfg.edge_margin_ratio)
    margin_y = int(height * cfg.edge_margin_ratio)
    long_h = int(width * 0.35)
    long_v = int(height * 0.35)
    regions: list[LayoutRegion] = []

    top = [s for s in h_segments if s.y1 <= margin_y and s.length >= long_h]
    bottom = [s for s in h_segments if s.y1 >= height - margin_y and s.length >= long_h]
    left = [s for s in v_segments if s.x1 <= margin_x and s.length >= long_v]
    right = [s for s in v_segments if s.x1 >= width - margin_x and s.length >= long_v]

    if top:
        y2 = min(height, max(s.y2 for s in top) + cfg.border_padding_px)
        regions.append(_border_region(next_id(), BBox(0, 0, width, y2), {"edge": "top"}))
    if bottom:
        y1 = max(0, min(s.y1 for s in bottom) - cfg.border_padding_px)
        regions.append(_border_region(next_id(), BBox(0, y1, width, height), {"edge": "bottom"}))
    if left:
        x2 = min(width, max(s.x2 for s in left) + cfg.border_padding_px)
        regions.append(_border_region(next_id(), BBox(0, 0, x2, height), {"edge": "left"}))
    if right:
        x1 = max(0, min(s.x1 for s in right) - cfg.border_padding_px)
        regions.append(_border_region(next_id(), BBox(x1, 0, width, height), {"edge": "right"}))
    return regions


def _border_region(region_id: str, bbox: BBox, metadata: dict[str, str]) -> LayoutRegion:
    return LayoutRegion(
        region_id=region_id,
        region_type="page_border",
        bbox=bbox,
        action="remove_from_clean_page",
        preserve_as_crop=False,
        confidence=0.95,
        source="edge_long_line_rules",
        metadata=metadata,
    )


def _detect_table_regions(
    h_segments: list[LineSegment],
    v_segments: list[LineSegment],
    width: int,
    height: int,
    cfg: LayoutCleanConfig,
) -> list[tuple[str, BBox, float, dict[str, int | float]]]:
    all_segments = h_segments + v_segments
    proposals = [
        (
            "hole_table",
            [
                s
                for s in all_segments
                if s.cx <= width * cfg.left_table_max_x_ratio
                and s.cy <= height * 0.88
                and not _is_page_frame_segment(s, width, height, cfg)
            ],
        ),
        (
            "title_or_tolerance_table",
            [
                s
                for s in all_segments
                if s.cy >= height * cfg.bottom_table_min_y_ratio
                and s.cx >= width * cfg.bottom_table_min_x_ratio
                and not _is_page_frame_segment(s, width, height, cfg)
            ],
        ),
        (
            "revision_table",
            [
                s
                for s in all_segments
                if s.cx >= width * cfg.revision_min_x_ratio
                and s.cy <= height * cfg.revision_max_y_ratio
                and not _is_page_frame_segment(s, width, height, cfg)
            ],
        ),
    ]

    regions: list[tuple[str, BBox, float, dict[str, int | float]]] = []
    for region_type, segments in proposals:
        h_count = sum(1 for s in segments if s.axis == "h")
        v_count = sum(1 for s in segments if s.axis == "v")
        if h_count + v_count < cfg.min_grid_segments or h_count < 2 or v_count < 2:
            continue
        bbox = _segments_bbox(segments).clamp(width, height)
        if not _looks_like_table_bbox(region_type, bbox, width, height):
            continue
        confidence = min(0.98, 0.58 + 0.012 * min(h_count + v_count, 35))
        regions.append(
            (
                region_type,
                bbox,
                confidence,
                {
                    "horizontal_segments": h_count,
                    "vertical_segments": v_count,
                    "bbox_area_ratio": round(bbox.area / (width * height), 6),
                },
            )
        )
    return regions


def _is_page_frame_segment(segment: LineSegment, width: int, height: int, cfg: LayoutCleanConfig) -> bool:
    margin_x = int(width * cfg.edge_margin_ratio)
    margin_y = int(height * cfg.edge_margin_ratio)
    if segment.axis == "h":
        return segment.length >= width * 0.75 and (segment.y1 <= margin_y or segment.y2 >= height - margin_y)
    return segment.length >= height * 0.75 and (segment.x1 <= margin_x or segment.x2 >= width - margin_x)


def _segments_bbox(segments: list[LineSegment]) -> BBox:
    return BBox(
        x1=min(s.x1 for s in segments),
        y1=min(s.y1 for s in segments),
        x2=max(s.x2 for s in segments),
        y2=max(s.y2 for s in segments),
    )


def _looks_like_table_bbox(region_type: str, bbox: BBox, width: int, height: int) -> bool:
    area_ratio = bbox.area / (width * height)
    if region_type == "hole_table":
        return bbox.width >= width * 0.08 and bbox.height >= height * 0.18 and area_ratio >= 0.015
    if region_type == "title_or_tolerance_table":
        return bbox.width >= width * 0.25 and bbox.height >= height * 0.05 and area_ratio >= 0.01
    if region_type == "revision_table":
        return bbox.width >= width * 0.08 and bbox.height >= height * 0.015
    return False


def _deduplicate_regions(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    unique: list[LayoutRegion] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for region in regions:
        key = (region.region_type, tuple(region.bbox.as_list()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(region)
    return unique


def _region_id_factory():
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"r{counter:03d}"

    return next_id
