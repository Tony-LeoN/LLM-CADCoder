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


def test_near_edge_frame_line_does_not_create_giant_bottom_table() -> None:
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    _draw_table(draw, (360, 560, 965, 680), rows=5, cols=8)
    _draw_table(draw, (760, 35, 965, 75), rows=1, cols=3)
    draw.line((965, 35, 965, 680), fill="black", width=2)
    draw.rectangle((120, 120, 520, 410), outline="black", width=2)
    draw.line((520, 220, 965, 220), fill="black", width=2)
    draw.line((620, 75, 620, 560), fill="black", width=2)

    regions = detect_layout_regions(image)
    title_regions = [region for region in regions if region.region_type == "title_or_tolerance_table"]

    assert len(title_regions) == 1
    assert title_regions[0].bbox.y1 >= 520
    assert title_regions[0].bbox.height < 180


def test_bottom_left_technical_requirements_are_extracted_as_text_block() -> None:
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.multiline_text(
        (55, 520),
        "技术要求(烤漆件):\n1. 去除毛刺, 倒钝棱边.\n2. 清除铁锈, 油渍等污迹.\n3. 表面烤漆, 色泽一致.",
        fill="black",
        spacing=8,
    )
    _draw_table(draw, (450, 560, 960, 680), rows=5, cols=8)

    regions = detect_layout_regions(image)
    technical_regions = [region for region in regions if region.region_type == "technical_requirements"]

    assert len(technical_regions) == 1
    assert technical_regions[0].preserve_as_crop is True
    assert technical_regions[0].bbox.x1 < 80
    assert technical_regions[0].bbox.y1 < 540


def test_right_side_technical_requirements_are_extracted_without_fixed_position() -> None:
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 120, 420, 430), outline="black", width=2)
    draw.multiline_text(
        (640, 130),
        "技术条件:\n1. 所有锐边倒钝.\n2. 表面不得有划伤.\n3. 未注公差按图纸要求.",
        fill="black",
        spacing=8,
    )
    _draw_table(draw, (420, 560, 960, 680), rows=5, cols=8)

    regions = detect_layout_regions(image)
    technical_regions = [region for region in regions if region.region_type == "technical_requirements"]

    assert len(technical_regions) == 1
    assert technical_regions[0].bbox.x1 > 600
    assert technical_regions[0].bbox.y1 < 160


def test_sparse_dimension_text_is_not_extracted_as_technical_requirements() -> None:
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((160, 140, 500, 430), outline="black", width=2)
    draw.line((120, 180, 540, 180), fill="black", width=2)
    draw.line((120, 280, 540, 280), fill="black", width=2)
    draw.line((120, 380, 540, 380), fill="black", width=2)
    draw.text((250, 160), "190", fill="black")
    draw.text((260, 260), "90", fill="black")
    draw.text((280, 360), "4-C2", fill="black")
    _draw_table(draw, (450, 560, 960, 680), rows=5, cols=8)

    regions = detect_layout_regions(image)

    assert [region for region in regions if region.region_type == "technical_requirements"] == []


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
