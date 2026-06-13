from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from vlm_cadcoder.utils.json_utils import write_json

DEFAULT_INFERENCE_OVERRIDES = (
    "inference.resize_max_side=3072",
    "inference.resize_min_side=1024",
    "inference.patch_size=768",
    "inference.stride=576",
    "inference.threshold=0.85",
)


Runner = Callable[[list[str], Path], None]


@dataclass(frozen=True)
class GeometryCoreUnetConfig:
    dataflow_root: Path = Path("DataFlow")
    sketchpic2viewpic_root: Path = Path("../SketchPic2ViewPic")
    python_executable: str = "python"
    config_path: Path = Path("configs/unet_baseline.yaml")
    checkpoint_path: Path = Path("runs/unet_tversky_a07_b03/checkpoints/best.pt")
    sample_id: str | None = None
    input_name: str = "clean_view_with_annotations.png"
    output_subdir: str = "geometry_core_unet"
    include_copy: bool = False
    skip_existing: bool = False
    dry_run: bool = False
    fail_fast: bool = False
    overrides: tuple[str, ...] = field(default_factory=lambda: DEFAULT_INFERENCE_OVERRIDES)


@dataclass(frozen=True)
class GeometryCoreRecord:
    sample_id: str
    view_id: str
    input_image: Path
    work_dir: Path
    geometry_core_path: Path
    mask_path: Path
    probability_path: Path
    metadata_path: Path
    command: list[str]
    status: str
    error: str | None = None


@dataclass(frozen=True)
class GeometryCoreSummary:
    total_count: int
    generated_count: int
    skipped_count: int
    dry_run_count: int
    failed_count: int
    records: list[GeometryCoreRecord]


def generate_geometry_core_images(
    config: GeometryCoreUnetConfig,
    runner: Runner | None = None,
) -> GeometryCoreSummary:
    runner = runner or _run_subprocess
    external_root = config.sketchpic2viewpic_root.resolve()
    records: list[GeometryCoreRecord] = []

    for sample_id, view_id, view_dir in _iter_view_dirs(config):
        input_image = view_dir / config.input_name
        work_dir = view_dir / config.output_subdir
        geometry_core_path = view_dir / "geometry_core.png"
        mask_path = view_dir / "geometry_core_mask.png"
        probability_path = view_dir / "geometry_core_prob.png"
        metadata_path = view_dir / "geometry_core.meta.json"
        command = build_sketchpic2viewpic_infer_command(config, input_image, work_dir)

        if not input_image.exists():
            record = _record(
                sample_id,
                view_id,
                input_image,
                work_dir,
                geometry_core_path,
                mask_path,
                probability_path,
                metadata_path,
                command,
                "failed",
                f"Missing input image: {input_image}",
            )
            records.append(record)
            if config.fail_fast:
                break
            continue

        if config.skip_existing and geometry_core_path.exists():
            records.append(
                _record(
                    sample_id,
                    view_id,
                    input_image,
                    work_dir,
                    geometry_core_path,
                    mask_path,
                    probability_path,
                    metadata_path,
                    command,
                    "skipped",
                )
            )
            continue

        if config.dry_run:
            records.append(
                _record(
                    sample_id,
                    view_id,
                    input_image,
                    work_dir,
                    geometry_core_path,
                    mask_path,
                    probability_path,
                    metadata_path,
                    command,
                    "dry_run",
                )
            )
            continue

        try:
            runner(command, external_root)
            _promote_unet_outputs(input_image, work_dir, geometry_core_path, mask_path, probability_path)
            _write_metadata(
                config,
                sample_id,
                view_id,
                input_image,
                work_dir,
                geometry_core_path,
                mask_path,
                probability_path,
                metadata_path,
                command,
            )
            records.append(
                _record(
                    sample_id,
                    view_id,
                    input_image,
                    work_dir,
                    geometry_core_path,
                    mask_path,
                    probability_path,
                    metadata_path,
                    command,
                    "generated",
                )
            )
        except Exception as exc:
            records.append(
                _record(
                    sample_id,
                    view_id,
                    input_image,
                    work_dir,
                    geometry_core_path,
                    mask_path,
                    probability_path,
                    metadata_path,
                    command,
                    "failed",
                    str(exc),
                )
            )
            if config.fail_fast:
                break

    return GeometryCoreSummary(
        total_count=len(records),
        generated_count=sum(record.status == "generated" for record in records),
        skipped_count=sum(record.status == "skipped" for record in records),
        dry_run_count=sum(record.status == "dry_run" for record in records),
        failed_count=sum(record.status == "failed" for record in records),
        records=records,
    )


def build_sketchpic2viewpic_infer_command(config: GeometryCoreUnetConfig, input_image: Path, output_dir: Path) -> list[str]:
    command = [
        config.python_executable,
        "-m",
        "sketchpic2viewpic",
        "infer",
        "--config",
        str(_resolve_external_path(config.sketchpic2viewpic_root, config.config_path)),
        "--checkpoint",
        str(_resolve_external_path(config.sketchpic2viewpic_root, config.checkpoint_path)),
        "--input",
        str(input_image.resolve()),
        "--output",
        str(output_dir.resolve()),
    ]
    for override in config.overrides:
        command.extend(["-o", override])
    return command


def _iter_view_dirs(config: GeometryCoreUnetConfig) -> list[tuple[str, str, Path]]:
    single_views_root = config.dataflow_root / "06.SingleViews"
    if config.sample_id:
        sample_dirs = [single_views_root / config.sample_id]
    elif not single_views_root.exists():
        sample_dirs = []
    else:
        sample_dirs = sorted(path for path in single_views_root.iterdir() if path.is_dir())

    view_dirs: list[tuple[str, str, Path]] = []
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        if sample_id.endswith("-copy") and not config.include_copy:
            continue
        if sample_id == "testView2CAD":
            continue
        for view_dir in sorted(sample_dir.glob("view_*")):
            if view_dir.is_dir():
                view_dirs.append((sample_id, view_dir.name, view_dir))
    return view_dirs


def _promote_unet_outputs(
    input_image: Path,
    work_dir: Path,
    geometry_core_path: Path,
    mask_path: Path,
    probability_path: Path,
) -> None:
    stem = input_image.stem
    expected_clean = work_dir / f"{stem}_clean.png"
    expected_mask = work_dir / f"{stem}_mask.png"
    expected_prob = work_dir / f"{stem}_prob.png"
    missing = [path for path in (expected_clean, expected_mask, expected_prob) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing SketchPic2ViewPic output(s): {missing_text}")

    shutil.copyfile(expected_clean, geometry_core_path)
    shutil.copyfile(expected_mask, mask_path)
    shutil.copyfile(expected_prob, probability_path)


def _write_metadata(
    config: GeometryCoreUnetConfig,
    sample_id: str,
    view_id: str,
    input_image: Path,
    work_dir: Path,
    geometry_core_path: Path,
    mask_path: Path,
    probability_path: Path,
    metadata_path: Path,
    command: list[str],
) -> None:
    write_json(
        metadata_path,
        {
            "schema": "geometry_core_generation",
            "version": "0.1.0",
            "sample_id": sample_id,
            "view_id": view_id,
            "method": {
                "name": "unet_annotation_removal",
                "external_project": "SketchPic2ViewPic",
                "external_project_root": str(config.sketchpic2viewpic_root.resolve()),
                "config_path": str(_resolve_external_path(config.sketchpic2viewpic_root, config.config_path)),
                "checkpoint_path": str(_resolve_external_path(config.sketchpic2viewpic_root, config.checkpoint_path)),
                "overrides": list(config.overrides),
            },
            "inputs": {
                "clean_view_with_annotations": str(input_image),
            },
            "outputs": {
                "geometry_core": str(geometry_core_path),
                "mask": str(mask_path),
                "probability": str(probability_path),
                "work_dir": str(work_dir),
            },
            "command": command,
        },
    )


def _record(
    sample_id: str,
    view_id: str,
    input_image: Path,
    work_dir: Path,
    geometry_core_path: Path,
    mask_path: Path,
    probability_path: Path,
    metadata_path: Path,
    command: list[str],
    status: str,
    error: str | None = None,
) -> GeometryCoreRecord:
    return GeometryCoreRecord(
        sample_id=sample_id,
        view_id=view_id,
        input_image=input_image,
        work_dir=work_dir,
        geometry_core_path=geometry_core_path,
        mask_path=mask_path,
        probability_path=probability_path,
        metadata_path=metadata_path,
        command=command,
        status=status,
        error=error,
    )


def _resolve_external_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def _run_subprocess(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)
