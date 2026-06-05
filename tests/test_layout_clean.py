from __future__ import annotations

from PIL import Image, ImageDraw

from vlm_cadcoder.dataflow.layout_clean import detect_layout_regions


def test_revision_table_component_does_not_swallow_nearby_view_geometry() -> None:
    image = Image.new("RGB", (1000, 1200), "white")
    draw = ImageDraw.Draw(image)
    _draw_table(draw, (720, 20, 960, 100), rows=1, cols=4)
    draw.rectangle((500, 165, 940, 360), outline="black", width=2)
    draw.line((500, 145, 940, 145), fill="blue", width=2)

    regions = detect_layout_regions(image)
    revision_regions = [region for region in regions if region.region_type == "revision_table"]

    assert len(revision_regions) == 1
    assert revision_regions[0].bbox.y2 < 145


def test_central_view_is_not_classified_as_left_hole_table() -> None:
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((180, 140, 470, 420), outline="black", width=2)
    draw.rectangle((650, 160, 700, 430), outline="black", width=2)
    draw.line((160, 500, 780, 500), fill="blue", width=2)

    regions = detect_layout_regions(image)

    assert [region for region in regions if region.region_type == "hole_table"] == []


def test_bottom_table_is_detected_without_fixed_x_start() -> None:
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    _draw_table(draw, (30, 560, 960, 680), rows=5, cols=10)

    regions = detect_layout_regions(image)
    title_regions = [region for region in regions if region.region_type == "title_or_tolerance_table"]

    assert len(title_regions) == 1
    assert title_regions[0].bbox.x1 <= 30


def test_bottom_table_with_colored_stamp_is_still_detected() -> None:
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((360, 585, 850, 660), fill=(120, 0, 0))
    _draw_table(draw, (300, 560, 960, 680), rows=5, cols=8)

    regions = detect_layout_regions(image)
    title_regions = [region for region in regions if region.region_type == "title_or_tolerance_table"]

    assert len(title_regions) == 1


def _draw_table(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    rows: int,
    cols: int,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle(box, outline="black", width=2)
    for row in range(1, rows + 1):
        y = y1 + round((y2 - y1) * row / (rows + 1))
        draw.line((x1, y, x2, y), fill="black", width=2)
    for col in range(1, cols):
        x = x1 + round((x2 - x1) * col / cols)
        draw.line((x, y1, x, y2), fill="black", width=2)
