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
    bottom_table_min_y2_ratio: float = 0.92
    bottom_table_min_width_ratio: float = 0.25
    bottom_table_min_area_ratio: float = 0.01
    bottom_table_search_min_y_ratio: float = 0.62
    bottom_table_max_height_ratio: float = 0.38
    bottom_table_max_area_ratio: float = 0.32
    revision_min_x_ratio: float = 0.62
    revision_max_y_ratio: float = 0.16
    min_grid_segments: int = 8
    min_grid_intersections: int = 4
    component_gap_px: int = 14
    component_frame_margin_ratio: float = 0.04
    max_revision_height_ratio: float = 0.085
    max_revision_area_ratio: float = 0.025
    max_hole_table_width_ratio: float = 0.56
    technical_block_min_y_ratio: float = 0.68
    technical_block_max_x_ratio: float = 0.45
    technical_block_anchor_max_x_ratio: float = 0.18
    technical_block_min_width_ratio: float = 0.04
    technical_block_min_height_ratio: float = 0.035
    technical_block_column_gap_px: int = 140
    technical_block_row_gap_px: int = 70
    technical_block_long_h_ratio: float = 0.12
    technical_block_long_v_ratio: float = 0.12
    technical_block_min_pixels: int = 80


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


@dataclass(frozen=True)
class LineComponent:
    segments: list[LineSegment]

    @property
    def horizontal_segments(self) -> list[LineSegment]:
        return [segment for segment in self.segments if segment.axis == "h"]

    @property
    def vertical_segments(self) -> list[LineSegment]:
        return [segment for segment in self.segments if segment.axis == "v"]

    @property
    def bbox(self) -> BBox:
        return _segments_bbox(self.segments)


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
                "name": "rule_based_line_component_layout_cleaner",
                "version": "0.3.0",
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
    components = _build_line_components(h_segments + v_segments, width, height, cfg)
    table_candidates = _detect_table_regions(components, rgb, width, height, cfg)
    table_candidates.extend(_detect_bottom_table_band_regions(h_segments, v_segments, rgb, width, height, cfg))
    for region_type, bbox, confidence, meta in table_candidates:
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
    regions.extend(_detect_technical_requirement_regions(neutral_dark, width, height, cfg, next_id))
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
        "technical_requirements": (0, 170, 80),
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
    components: list[LineComponent],
    rgb,
    width: int,
    height: int,
    cfg: LayoutCleanConfig,
) -> list[tuple[str, BBox, float, dict[str, int | float]]]:
    regions: list[tuple[str, BBox, float, dict[str, int | float]]] = []
    for component in components:
        h_segments = component.horizontal_segments
        v_segments = component.vertical_segments
        h_count = len(h_segments)
        v_count = len(v_segments)
        if h_count + v_count < cfg.min_grid_segments or h_count < 2 or v_count < 2:
            continue
        bbox = component.bbox.clamp(width, height)
        intersections = _count_intersections(h_segments, v_segments, cfg.component_gap_px)
        if intersections < cfg.min_grid_intersections:
            continue
        region_type = _classify_table_component(component, bbox, rgb, width, height, cfg)
        if region_type is None:
            continue
        confidence = min(0.98, 0.56 + 0.008 * min(h_count + v_count, 40) + 0.01 * min(intersections, 10))
        regions.append(
            (
                region_type,
                bbox,
                confidence,
                {
                    "horizontal_segments": h_count,
                    "vertical_segments": v_count,
                    "intersections": intersections,
                    "bbox_area_ratio": round(bbox.area / (width * height), 6),
                    "colored_foreground_ratio": round(
                        _colored_foreground_ratio(rgb, bbox, cfg.dark_threshold, cfg.neutral_delta), 6
                    ),
                },
            )
        )
    return regions


def _detect_bottom_table_band_regions(
    h_segments: list[LineSegment],
    v_segments: list[LineSegment],
    rgb,
    width: int,
    height: int,
    cfg: LayoutCleanConfig,
) -> list[tuple[str, BBox, float, dict[str, int | float]]]:
    min_y = int(height * cfg.bottom_table_search_min_y_ratio)
    bottom_h = [segment for segment in h_segments if segment.y1 >= min_y]
    bottom_v = [segment for segment in v_segments if segment.y1 >= min_y]
    components = _build_line_components(bottom_h + bottom_v, width, height, cfg)
    regions: list[tuple[str, BBox, float, dict[str, int | float]]] = []
    for component in components:
        h_count = len(component.horizontal_segments)
        v_count = len(component.vertical_segments)
        if h_count < 4 or v_count < 4 or h_count + v_count < cfg.min_grid_segments:
            continue
        bbox = component.bbox.clamp(width, height)
        intersections = _count_intersections(component.horizontal_segments, component.vertical_segments, cfg.component_gap_px)
        if intersections < cfg.min_grid_intersections:
            continue
        width_ratio = bbox.width / width
        height_ratio = bbox.height / height
        area_ratio = bbox.area / (width * height)
        y1_ratio = bbox.y1 / height
        y2_ratio = bbox.y2 / height
        if not (
            y1_ratio >= cfg.bottom_table_search_min_y_ratio
            and y2_ratio >= cfg.bottom_table_min_y2_ratio
            and width_ratio >= cfg.bottom_table_min_width_ratio
            and height_ratio <= cfg.bottom_table_max_height_ratio
            and cfg.bottom_table_min_area_ratio <= area_ratio <= cfg.bottom_table_max_area_ratio
        ):
            continue
        confidence = min(0.96, 0.62 + 0.008 * min(h_count + v_count, 35) + 0.008 * min(intersections, 10))
        regions.append(
            (
                "title_or_tolerance_table",
                bbox,
                confidence,
                {
                    "horizontal_segments": h_count,
                    "vertical_segments": v_count,
                    "intersections": intersections,
                    "bbox_area_ratio": round(area_ratio, 6),
                    "source_detector": "bottom_band_grid_rules",
                },
            )
        )
    return regions


def _build_line_components(
    segments: list[LineSegment],
    width: int,
    height: int,
    cfg: LayoutCleanConfig,
) -> list[LineComponent]:
    candidates = [segment for segment in segments if not _is_page_frame_segment(segment, width, height, cfg)]
    if not candidates:
        return []

    groups = _DisjointSet(len(candidates))
    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            if _segments_touch(left, right, cfg.component_gap_px):
                groups.union(left_index, right_index)

    by_root: dict[int, list[LineSegment]] = {}
    for index, segment in enumerate(candidates):
        by_root.setdefault(groups.find(index), []).append(segment)
    return [LineComponent(component_segments) for component_segments in by_root.values()]


def _segments_touch(left: LineSegment, right: LineSegment, gap: int) -> bool:
    if left.axis == "h" and right.axis == "v":
        return _horizontal_vertical_touch(left, right, gap)
    if left.axis == "v" and right.axis == "h":
        return _horizontal_vertical_touch(right, left, gap)
    if left.axis == "h":
        return abs(left.y1 - right.y1) <= gap and _ranges_overlap(left.x1, left.x2, right.x1, right.x2, gap)
    return abs(left.x1 - right.x1) <= gap and _ranges_overlap(left.y1, left.y2, right.y1, right.y2, gap)


def _horizontal_vertical_touch(horizontal: LineSegment, vertical: LineSegment, gap: int) -> bool:
    return (
        horizontal.x1 - gap <= vertical.x1 <= horizontal.x2 + gap
        and vertical.y1 - gap <= horizontal.y1 <= vertical.y2 + gap
    )


def _ranges_overlap(a1: int, a2: int, b1: int, b2: int, gap: int = 0) -> bool:
    return max(a1, b1) <= min(a2, b2) + gap


def _count_intersections(h_segments: list[LineSegment], v_segments: list[LineSegment], gap: int) -> int:
    count = 0
    for h_segment in h_segments:
        for v_segment in v_segments:
            if _horizontal_vertical_touch(h_segment, v_segment, gap):
                count += 1
    return count


def _classify_table_component(
    component: LineComponent,
    bbox: BBox,
    rgb,
    width: int,
    height: int,
    cfg: LayoutCleanConfig,
) -> str | None:
    if not _looks_like_table_bbox(bbox, width, height):
        return None

    area_ratio = bbox.area / (width * height)
    x1_ratio = bbox.x1 / width
    x2_ratio = bbox.x2 / width
    y1_ratio = bbox.y1 / height
    y2_ratio = bbox.y2 / height
    width_ratio = bbox.width / width
    height_ratio = bbox.height / height
    h_count = len(component.horizontal_segments)
    v_count = len(component.vertical_segments)

    if (
        x1_ratio >= cfg.revision_min_x_ratio
        and y2_ratio <= cfg.revision_max_y_ratio
        and height_ratio <= cfg.max_revision_height_ratio
        and area_ratio <= cfg.max_revision_area_ratio
    ):
        return "revision_table"

    if (
        (y1_ratio >= cfg.bottom_table_min_y_ratio or y2_ratio >= cfg.bottom_table_min_y2_ratio)
        and y1_ratio >= cfg.bottom_table_search_min_y_ratio
        and width_ratio >= cfg.bottom_table_min_width_ratio
        and area_ratio >= cfg.bottom_table_min_area_ratio
        and area_ratio <= cfg.bottom_table_max_area_ratio
        and height_ratio <= cfg.bottom_table_max_height_ratio
        and h_count >= 6
        and v_count >= 6
    ):
        return "title_or_tolerance_table"

    if (
        x1_ratio <= 0.08
        and x2_ratio <= cfg.max_hole_table_width_ratio
        and y1_ratio <= 0.22
        and height_ratio >= 0.18
        and width_ratio >= 0.08
        and h_count >= 10
        and v_count >= 6
    ):
        return "hole_table"

    return None


def _is_page_frame_segment(segment: LineSegment, width: int, height: int, cfg: LayoutCleanConfig) -> bool:
    margin_x = int(width * max(cfg.edge_margin_ratio, cfg.component_frame_margin_ratio))
    margin_y = int(height * max(cfg.edge_margin_ratio, cfg.component_frame_margin_ratio))
    if segment.axis == "h":
        return segment.length >= width * 0.75 and (segment.y1 <= margin_y or segment.y2 >= height - margin_y)
    return segment.length >= height * 0.75 and (segment.x1 <= margin_x or segment.x2 >= width - margin_x)


def _detect_technical_requirement_regions(
    neutral_dark,
    width: int,
    height: int,
    cfg: LayoutCleanConfig,
    next_id,
) -> list[LayoutRegion]:
    import numpy as np

    y_offset = int(height * cfg.technical_block_min_y_ratio)
    x_limit = int(width * cfg.technical_block_max_x_ratio)
    y_limit = height - int(height * cfg.edge_margin_ratio)
    if y_offset >= y_limit or x_limit <= 0:
        return []

    local = neutral_dark[y_offset:y_limit, :x_limit].copy()
    _remove_long_runs_from_mask(local, "h", max(12, int(width * cfg.technical_block_long_h_ratio)))
    _remove_long_runs_from_mask(local, "v", max(12, int(height * cfg.technical_block_long_v_ratio)))

    if int(local.sum()) < cfg.technical_block_min_pixels:
        return []

    col_counts = np.asarray(local.sum(axis=0)).reshape(-1)
    active_cols = (col_counts >= 2).nonzero()[0]
    if active_cols.size == 0:
        return []

    col_groups = _group_sorted_indices(active_cols.tolist(), cfg.technical_block_column_gap_px)
    anchor_limit = int(width * cfg.technical_block_anchor_max_x_ratio)
    min_width = int(width * cfg.technical_block_min_width_ratio)
    min_height = int(height * cfg.technical_block_min_height_ratio)

    candidates: list[BBox] = []
    for col_group in col_groups:
        x1 = col_group[0]
        x2 = col_group[-1] + 1
        if x1 > anchor_limit or x2 - x1 < min_width:
            continue
        cluster = local[:, x1:x2]
        row_counts = np.asarray(cluster.sum(axis=1)).reshape(-1)
        active_rows = (row_counts >= 2).nonzero()[0]
        if active_rows.size == 0:
            continue
        for row_group in _group_sorted_indices(active_rows.tolist(), cfg.technical_block_row_gap_px):
            y1 = row_group[0]
            y2 = row_group[-1] + 1
            if y2 - y1 < min_height:
                continue
            pixels_y, pixels_x = np.nonzero(cluster[y1:y2, :])
            if pixels_x.size < cfg.technical_block_min_pixels:
                continue
            candidates.append(
                BBox(
                    x1 + int(pixels_x.min()),
                    y_offset + y1 + int(pixels_y.min()),
                    x1 + int(pixels_x.max()) + 1,
                    y_offset + y1 + int(pixels_y.max()) + 1,
                )
            )

    if not candidates:
        return []

    bbox = _merge_bboxes(candidates).pad(cfg.region_padding_px, width, height)
    return [
        LayoutRegion(
            region_id=next_id(),
            region_type="technical_requirements",
            bbox=bbox,
            action="remove_from_clean_page",
            preserve_as_crop=True,
            confidence=0.78,
            source="bottom_left_text_block_rules",
            metadata={
                "bbox_area_ratio": round(bbox.area / (width * height), 6),
                "candidate_blocks": len(candidates),
            },
        )
    ]


def _remove_long_runs_from_mask(mask, axis: str, min_length: int) -> None:
    if axis == "h":
        for y, row in enumerate(mask):
            starts, ends = _runs_1d(row)
            for x1, x2 in zip(starts, ends):
                if x2 - x1 >= min_length:
                    mask[y, x1:x2] = False
        return

    for x, col in enumerate(mask.T):
        starts, ends = _runs_1d(col)
        for y1, y2 in zip(starts, ends):
            if y2 - y1 >= min_length:
                mask[y1:y2, x] = False


def _group_sorted_indices(values: list[int], max_gap: int) -> list[list[int]]:
    if not values:
        return []
    groups = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _merge_bboxes(bboxes: list[BBox]) -> BBox:
    return BBox(
        x1=min(bbox.x1 for bbox in bboxes),
        y1=min(bbox.y1 for bbox in bboxes),
        x2=max(bbox.x2 for bbox in bboxes),
        y2=max(bbox.y2 for bbox in bboxes),
    )


def _segments_bbox(segments: list[LineSegment]) -> BBox:
    return BBox(
        x1=min(s.x1 for s in segments),
        y1=min(s.y1 for s in segments),
        x2=max(s.x2 for s in segments),
        y2=max(s.y2 for s in segments),
    )


def _looks_like_table_bbox(bbox: BBox, width: int, height: int) -> bool:
    area_ratio = bbox.area / (width * height)
    return bbox.width >= width * 0.08 and bbox.height >= height * 0.015 and area_ratio >= 0.002


def _colored_foreground_ratio(rgb, bbox: BBox, dark_threshold: int, neutral_delta: int) -> float:
    crop = rgb[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
    if crop.size == 0:
        return 0.0
    max_channel = crop.max(axis=2)
    min_channel = crop.min(axis=2)
    dark = max_channel < dark_threshold
    if int(dark.sum()) == 0:
        return 0.0
    colored = dark & ((max_channel - min_channel) > neutral_delta)
    return float(colored.sum() / dark.sum())


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


class _DisjointSet:
    def __init__(self, size: int):
        self.parents = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parents[item]
        if parent != item:
            self.parents[item] = self.find(parent)
        return self.parents[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[right_root] = left_root
