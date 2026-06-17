from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from vlm_cadcoder.dataflow.geometry_primitive_repair import GeometryPrimitiveRepairConfig, repair_geometry_primitives


def test_primitive_repair_accepts_broken_circle_and_rejects_isolated_rectangle(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-A" / "view_001"
    view_dir.mkdir(parents=True)
    _write_broken_circle_with_rectangle_frame(view_dir / "geometry_core.png")

    summary = repair_geometry_primitives(
        GeometryPrimitiveRepairConfig(
            dataflow_root=dataflow,
            sample_id="Part-A",
            min_existing_arc_coverage=0.55,
            max_circle_gap_ratio=0.35,
        )
    )

    candidates_path = view_dir / "primitive_candidates.json"
    overlay_path = view_dir / "primitive_repair_overlay.png"
    repaired_path = view_dir / "geometry_core_primitive_repaired.png"
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))

    assert summary.total_count == 1
    assert summary.repaired_count == 1
    assert candidates_path.exists()
    assert overlay_path.exists()
    assert repaired_path.exists()
    assert payload["accepted_candidates"][0]["primitive_type"] == "circle_arc"
    assert payload["accepted_candidates"][0]["action"] == "repair_gap"
    assert payload["accepted_candidates"][0]["metrics"]["existing_arc_coverage"] >= 0.55
    assert payload["rejected_candidates"][0]["primitive_type"] == "rectangle"
    assert "isolated_closed_rectangle" in payload["rejected_candidates"][0]["reject_reasons"]

    repaired = Image.open(repaired_path).convert("L")
    assert repaired.getpixel((113, 60)) == 0


def test_primitive_repair_cli_writes_summary(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-CLI" / "view_001"
    view_dir.mkdir(parents=True)
    _write_broken_circle_with_rectangle_frame(view_dir / "geometry_core.png")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vlm_cadcoder.cli",
            "repair-geometry-primitives",
            "--sample-id",
            "Part-CLI",
            "--dataflow-root",
            str(dataflow),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Geometry primitive repair: total 1; repaired 1" in result.stdout
    assert (view_dir / "primitive_candidates.json").exists()
    assert (dataflow / "06.SingleViews" / "geometry_primitive_repair_summary.csv").exists()


def test_primitive_repair_records_line_gap_candidate(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-Line" / "view_001"
    view_dir.mkdir(parents=True)
    _write_broken_horizontal_line(view_dir / "geometry_core.png")

    repair_geometry_primitives(
        GeometryPrimitiveRepairConfig(
            dataflow_root=dataflow,
            sample_id="Part-Line",
            max_line_gap_px=8,
            min_line_segment_px=10,
            primitive_types=("line",),
        )
    )

    payload = json.loads((view_dir / "primitive_candidates.json").read_text(encoding="utf-8"))
    assert payload["accepted_candidates"][0]["primitive_type"] == "line"
    assert payload["accepted_candidates"][0]["action"] == "repair_gap"
    assert payload["accepted_candidates"][0]["metrics"]["gap_px"] == 5
    repaired = Image.open(view_dir / "geometry_core_primitive_repaired.png").convert("L")
    assert repaired.getpixel((28, 16)) == 0


def test_primitive_repair_does_not_repair_already_closed_circle(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    view_dir = dataflow / "06.SingleViews" / "Part-Closed" / "view_001"
    view_dir.mkdir(parents=True)
    _write_closed_circle(view_dir / "geometry_core.png")

    summary = repair_geometry_primitives(
        GeometryPrimitiveRepairConfig(
            dataflow_root=dataflow,
            sample_id="Part-Closed",
        )
    )

    payload = json.loads((view_dir / "primitive_candidates.json").read_text(encoding="utf-8"))
    assert summary.repaired_count == 1
    assert payload["accepted_candidates"] == []
    assert payload["metrics"]["added_pixel_count"] == 0


def _write_broken_circle_with_rectangle_frame(path: Path) -> None:
    image = Image.new("L", (160, 128), 255)
    draw = ImageDraw.Draw(image)
    draw.arc((44, 24, 116, 96), start=25, end=335, fill=0, width=2)
    draw.rectangle((8, 8, 34, 28), outline=0, width=2)
    image.save(path)


def _write_broken_horizontal_line(path: Path) -> None:
    image = Image.new("L", (64, 32), 255)
    draw = ImageDraw.Draw(image)
    for y in range(15, 18):
        draw.line((6, y, 25, y), fill=0)
        draw.line((31, y, 56, y), fill=0)
    image.save(path)


def _write_closed_circle(path: Path) -> None:
    image = Image.new("L", (96, 96), 255)
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 24, 72, 72), outline=0, width=2)
    image.save(path)
