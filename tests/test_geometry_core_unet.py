from __future__ import annotations

import json
from pathlib import Path

from vlm_cadcoder.dataflow.geometry_core_unet import (
    DEFAULT_INFERENCE_OVERRIDES,
    GeometryCoreUnetConfig,
    generate_geometry_core_images,
)


def test_generate_geometry_core_dry_run_builds_external_infer_command(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_single_view(dataflow, "Part-A", "view_001")
    sketch_root = tmp_path / "SketchPic2ViewPic"

    summary = generate_geometry_core_images(
        GeometryCoreUnetConfig(
            dataflow_root=dataflow,
            sketchpic2viewpic_root=sketch_root,
            dry_run=True,
        )
    )

    assert summary.total_count == 1
    assert summary.dry_run_count == 1
    record = summary.records[0]
    assert record.command[:4] == ["python", "-m", "sketchpic2viewpic", "infer"]
    assert str(sketch_root / "configs" / "unet_baseline.yaml") in record.command
    assert str(sketch_root / "runs" / "unet_tversky_a07_b03" / "checkpoints" / "best.pt") in record.command
    assert str(dataflow / "06.SingleViews" / "Part-A" / "view_001" / "clean_view_with_annotations.png") in record.command
    assert str(dataflow / "06.SingleViews" / "Part-A" / "view_001" / "geometry_core_unet") in record.command
    for override in DEFAULT_INFERENCE_OVERRIDES:
        assert override in record.command
    assert not record.geometry_core_path.exists()


def test_generate_geometry_core_promotes_unet_outputs_to_view_contract(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_single_view(dataflow, "Part-B", "view_002")
    sketch_root = tmp_path / "SketchPic2ViewPic"

    def fake_runner(command: list[str], cwd: Path) -> None:
        assert cwd == sketch_root
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir(parents=True)
        (output_dir / "clean_view_with_annotations_clean.png").write_bytes(b"clean")
        (output_dir / "clean_view_with_annotations_mask.png").write_bytes(b"mask")
        (output_dir / "clean_view_with_annotations_prob.png").write_bytes(b"prob")

    summary = generate_geometry_core_images(
        GeometryCoreUnetConfig(
            dataflow_root=dataflow,
            sketchpic2viewpic_root=sketch_root,
        ),
        runner=fake_runner,
    )

    view_dir = dataflow / "06.SingleViews" / "Part-B" / "view_002"
    assert summary.generated_count == 1
    assert (view_dir / "geometry_core.png").read_bytes() == b"clean"
    assert (view_dir / "geometry_core_mask.png").read_bytes() == b"mask"
    assert (view_dir / "geometry_core_prob.png").read_bytes() == b"prob"

    meta = json.loads((view_dir / "geometry_core.meta.json").read_text(encoding="utf-8"))
    assert meta["schema"] == "geometry_core_generation"
    assert meta["method"]["external_project"] == "SketchPic2ViewPic"
    assert meta["sample_id"] == "Part-B"
    assert meta["view_id"] == "view_002"
    assert meta["outputs"]["geometry_core"] == str(view_dir / "geometry_core.png")


def test_generate_geometry_core_dry_run_uses_absolute_io_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "LLM-CADCoder"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    _write_single_view(repo_root / "DataFlow", "Part-Rel", "view_001")

    summary = generate_geometry_core_images(
        GeometryCoreUnetConfig(
            dataflow_root=Path("DataFlow"),
            sketchpic2viewpic_root=Path("../SketchPic2ViewPic"),
            dry_run=True,
        )
    )

    command = summary.records[0].command
    input_path = command[command.index("--input") + 1]
    output_path = command[command.index("--output") + 1]
    assert input_path == str((repo_root / "DataFlow" / "06.SingleViews" / "Part-Rel" / "view_001" / "clean_view_with_annotations.png").resolve())
    assert output_path == str((repo_root / "DataFlow" / "06.SingleViews" / "Part-Rel" / "view_001" / "geometry_core_unet").resolve())


def test_generate_geometry_core_skips_copy_samples_by_default(tmp_path: Path) -> None:
    dataflow = tmp_path / "DataFlow"
    _write_single_view(dataflow, "Part-C", "view_001")
    _write_single_view(dataflow, "Part-C-copy", "view_001")

    summary = generate_geometry_core_images(
        GeometryCoreUnetConfig(
            dataflow_root=dataflow,
            sketchpic2viewpic_root=tmp_path / "SketchPic2ViewPic",
            dry_run=True,
        )
    )

    assert [record.sample_id for record in summary.records] == ["Part-C"]


def _write_single_view(dataflow: Path, sample_id: str, view_id: str) -> None:
    view_dir = dataflow / "06.SingleViews" / sample_id / view_id
    view_dir.mkdir(parents=True)
    (view_dir / "clean_view_with_annotations.png").write_bytes(b"not-a-real-png")
    (view_dir / "view_metadata.json").write_text(
        json.dumps({"sample_id": sample_id, "view_id": view_id}),
        encoding="utf-8",
    )
