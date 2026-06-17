from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from vlm_cadcoder.dataflow.geometry_core_repair import GeometryCoreRepairConfig, repair_geometry_core_images


def test_repair_geometry_core_bridges_small_horizontal_gap(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-A" / "view_001"
    view_dir.mkdir(parents=True)
    _write_broken_horizontal_line(view_dir / "geometry_core.png")
    _write_broken_horizontal_line(view_dir / "clean_view_with_annotations.png")
    _write_broken_horizontal_line(view_dir / "geometry_core_prob.png")

    summary = repair_geometry_core_images(
        GeometryCoreRepairConfig(
            dataflow_root=dataflow,
            sample_id="Part-A",
            max_gap_px=8,
            min_segment_px=10,
            tiny_area_px=4,
        )
    )

    repaired_path = view_dir / "geometry_core_repaired.png"
    overlay_path = view_dir / "geometry_core_repair_overlay.png"
    meta_path = view_dir / "geometry_core_repair.meta.json"

    assert summary.total_count == 1
    assert summary.repaired_count == 1
    assert repaired_path.exists()
    assert overlay_path.exists()
    assert meta_path.exists()

    repaired = Image.open(repaired_path).convert("L")
    assert repaired.getpixel((28, 16)) == 0

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["schema"] == "geometry_core_repair"
    assert meta["sample_id"] == "Part-A"
    assert meta["view_id"] == "view_001"
    assert meta["inputs"]["clean_view_with_annotations"] == (view_dir / "clean_view_with_annotations.png").as_posix()
    assert meta["inputs"]["geometry_core_prob"] == (view_dir / "geometry_core_prob.png").as_posix()
    assert meta["outputs"]["geometry_core_repaired"] == repaired_path.as_posix()
    assert meta["metrics"]["added_pixel_count"] > 0
    assert meta["metrics"]["horizontal_bridge_count"] >= 1


def test_repair_geometry_core_bridges_supported_vertical_gap(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-Vertical" / "view_001"
    view_dir.mkdir(parents=True)
    _write_broken_vertical_line(view_dir / "geometry_core.png")

    summary = repair_geometry_core_images(
        GeometryCoreRepairConfig(
            dataflow_root=dataflow,
            sample_id="Part-Vertical",
            max_gap_px=8,
            min_segment_px=10,
            tiny_area_px=1,
        )
    )

    repaired = Image.open(view_dir / "geometry_core_repaired.png").convert("L")
    assert summary.repaired_count == 1
    assert repaired.getpixel((32, 28)) == 0
    assert summary.records[0].metrics["vertical_bridge_count"] >= 1


def test_repair_geometry_core_does_not_bridge_unsupported_single_scanline_gap(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-Close-Separate" / "view_001"
    view_dir.mkdir(parents=True)
    _write_single_scanline_close_segments(view_dir / "geometry_core.png")

    summary = repair_geometry_core_images(
        GeometryCoreRepairConfig(
            dataflow_root=dataflow,
            sample_id="Part-Close-Separate",
            max_gap_px=8,
            min_segment_px=10,
            tiny_area_px=1,
        )
    )

    repaired = Image.open(view_dir / "geometry_core_repaired.png").convert("L")
    assert summary.repaired_count == 1
    assert repaired.getpixel((28, 16)) == 255
    assert summary.records[0].metrics["horizontal_bridge_count"] == 0


def test_repair_geometry_core_removes_tiny_components(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-Tiny" / "view_001"
    view_dir.mkdir(parents=True)
    _write_line_with_tiny_component(view_dir / "geometry_core.png")

    summary = repair_geometry_core_images(
        GeometryCoreRepairConfig(
            dataflow_root=dataflow,
            sample_id="Part-Tiny",
            max_gap_px=0,
            min_segment_px=10,
            tiny_area_px=4,
        )
    )

    repaired = Image.open(view_dir / "geometry_core_repaired.png").convert("L")
    assert repaired.getpixel((50, 5)) == 255
    assert summary.records[0].metrics["removed_tiny_component_count"] == 1


def test_repair_geometry_core_skips_complete_existing_outputs(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-B" / "view_001"
    view_dir.mkdir(parents=True)
    _write_broken_horizontal_line(view_dir / "geometry_core.png")
    (view_dir / "geometry_core_repaired.png").write_bytes(b"existing")
    (view_dir / "geometry_core_repair_overlay.png").write_bytes(b"existing-overlay")
    (view_dir / "geometry_core_repair.meta.json").write_text("{}", encoding="utf-8")

    summary = repair_geometry_core_images(
        GeometryCoreRepairConfig(
            dataflow_root=dataflow,
            sample_id="Part-B",
            skip_existing=True,
        )
    )

    assert summary.total_count == 1
    assert summary.skipped_count == 1
    assert (view_dir / "geometry_core_repaired.png").read_bytes() == b"existing"


def test_repair_geometry_core_regenerates_partial_existing_outputs(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-Partial" / "view_001"
    view_dir.mkdir(parents=True)
    _write_broken_horizontal_line(view_dir / "geometry_core.png")
    (view_dir / "geometry_core_repaired.png").write_bytes(b"partial")

    summary = repair_geometry_core_images(
        GeometryCoreRepairConfig(
            dataflow_root=dataflow,
            sample_id="Part-Partial",
            skip_existing=True,
            max_gap_px=8,
            min_segment_px=10,
        )
    )

    assert summary.repaired_count == 1
    assert (view_dir / "geometry_core_repair_overlay.png").exists()
    assert (view_dir / "geometry_core_repair.meta.json").exists()
    assert (view_dir / "geometry_core_repaired.png").read_bytes() != b"partial"


def test_repair_geometry_core_reports_missing_sample_id(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"

    summary = repair_geometry_core_images(
        GeometryCoreRepairConfig(
            dataflow_root=dataflow,
            sample_id="Missing-Part",
        )
    )

    assert summary.total_count == 1
    assert summary.failed_count == 1
    assert "Missing sample directory" in summary.records[0].error


def test_repair_geometry_core_dry_run_does_not_write_view_outputs(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-DryRun" / "view_001"
    view_dir.mkdir(parents=True)
    _write_broken_horizontal_line(view_dir / "geometry_core.png")

    summary = repair_geometry_core_images(
        GeometryCoreRepairConfig(
            dataflow_root=dataflow,
            sample_id="Part-DryRun",
            dry_run=True,
        )
    )

    assert summary.total_count == 1
    assert summary.dry_run_count == 1
    assert not (view_dir / "geometry_core_repaired.png").exists()
    assert not (view_dir / "geometry_core_repair_overlay.png").exists()
    assert not (view_dir / "geometry_core_repair.meta.json").exists()
    assert (dataflow / "06.SingleViews" / "geometry_core_repair_summary.csv").exists()


def test_repair_geometry_core_cli_writes_single_sample_outputs(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-CLI" / "view_001"
    view_dir.mkdir(parents=True)
    _write_broken_horizontal_line(view_dir / "geometry_core.png")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vlm_cadcoder.cli",
            "repair-geometry-core",
            "--sample-id",
            "Part-CLI",
            "--dataflow-root",
            str(dataflow),
            "--max-gap-px",
            "8",
            "--min-segment-px",
            "10",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Geometry-core repair: total 1; repaired 1" in result.stdout
    assert (view_dir / "geometry_core_repaired.png").exists()
    assert (view_dir / "geometry_core_repair_summary.csv").exists() is False
    assert (dataflow / "06.SingleViews" / "geometry_core_repair_summary.csv").exists()


def test_repair_geometry_core_rejects_overwriting_input(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-D" / "view_001"
    view_dir.mkdir(parents=True)
    _write_broken_horizontal_line(view_dir / "geometry_core.png")

    with pytest.raises(ValueError, match="must not overwrite"):
        repair_geometry_core_images(
            GeometryCoreRepairConfig(
                dataflow_root=dataflow,
                sample_id="Part-D",
                output_name="geometry_core.png",
            )
        )


def test_repair_geometry_core_rejects_invalid_parameters(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"

    with pytest.raises(ValueError, match="black_threshold"):
        repair_geometry_core_images(
            GeometryCoreRepairConfig(
                dataflow_root=dataflow,
                black_threshold=300,
            )
        )


def _write_broken_horizontal_line(path: Path) -> None:
    image = Image.new("L", (64, 32), 255)
    draw = ImageDraw.Draw(image)
    for y in range(15, 18):
        draw.line((6, y, 25, y), fill=0)
        draw.line((31, y, 56, y), fill=0)
    image.save(path)


def _write_broken_vertical_line(path: Path) -> None:
    image = Image.new("L", (64, 64), 255)
    draw = ImageDraw.Draw(image)
    for x in range(31, 34):
        draw.line((x, 6, x, 25), fill=0)
        draw.line((x, 31, x, 56), fill=0)
    image.save(path)


def _write_single_scanline_close_segments(path: Path) -> None:
    image = Image.new("L", (64, 32), 255)
    draw = ImageDraw.Draw(image)
    draw.line((6, 16, 25, 16), fill=0)
    draw.line((31, 16, 56, 16), fill=0)
    image.save(path)


def _write_line_with_tiny_component(path: Path) -> None:
    image = Image.new("L", (64, 32), 255)
    draw = ImageDraw.Draw(image)
    draw.line((6, 16, 40, 16), fill=0)
    draw.point((50, 5), fill=0)
    image.save(path)
