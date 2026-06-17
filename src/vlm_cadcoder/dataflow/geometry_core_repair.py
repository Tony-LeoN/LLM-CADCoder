from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vlm_cadcoder.utils.json_utils import write_json


@dataclass(frozen=True)
class GeometryCoreRepairConfig:
    dataflow_root: Path = Path("DataFlow")
    sample_id: str | None = None
    input_name: str = "geometry_core.png"
    clean_input_name: str = "clean_view_with_annotations.png"
    probability_input_name: str = "geometry_core_prob.png"
    output_name: str = "geometry_core_repaired.png"
    overlay_name: str = "geometry_core_repair_overlay.png"
    metadata_name: str = "geometry_core_repair.meta.json"
    include_copy: bool = False
    skip_existing: bool = False
    dry_run: bool = False
    fail_fast: bool = False
    black_threshold: int = 250
    max_gap_px: int = 12
    min_segment_px: int = 16
    bridge_support_radius: int = 1
    min_bridge_support: int = 2
    tiny_area_px: int = 12
    directions: tuple[str, ...] = field(default_factory=lambda: ("horizontal", "vertical"))


@dataclass(frozen=True)
class GeometryCoreRepairRecord:
    sample_id: str
    view_id: str
    input_path: Path
    repaired_path: Path
    overlay_path: Path
    metadata_path: Path
    status: str
    metrics: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class GeometryCoreRepairSummary:
    records: list[GeometryCoreRepairRecord]
    csv_path: Path | None = None
    json_path: Path | None = None

    @property
    def total_count(self) -> int:
        return len(self.records)

    @property
    def repaired_count(self) -> int:
        return sum(record.status == "repaired" for record in self.records)

    @property
    def skipped_count(self) -> int:
        return sum(record.status == "skipped" for record in self.records)

    @property
    def dry_run_count(self) -> int:
        return sum(record.status == "dry_run" for record in self.records)

    @property
    def failed_count(self) -> int:
        return sum(record.status == "failed" for record in self.records)


def repair_geometry_core_images(
    config: GeometryCoreRepairConfig,
    *,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
) -> GeometryCoreRepairSummary:
    _validate_config(config)
    records: list[GeometryCoreRepairRecord] = []
    view_dirs = _iter_view_dirs(config)
    if config.sample_id and not view_dirs:
        sample_dir = config.dataflow_root / "06.SingleViews" / config.sample_id
        if sample_dir.exists():
            error = f"No view directories found in sample: {sample_dir}"
        else:
            error = f"Missing sample directory: {sample_dir}"
        records.append(
            _record(
                config.sample_id,
                "*",
                sample_dir / config.input_name,
                sample_dir / config.output_name,
                sample_dir / config.overlay_name,
                sample_dir / config.metadata_name,
                "failed",
                error=error,
            )
        )

    for sample_id, view_id, view_dir in view_dirs:
        input_path = view_dir / config.input_name
        clean_path = view_dir / config.clean_input_name
        probability_path = view_dir / config.probability_input_name
        repaired_path = view_dir / config.output_name
        overlay_path = view_dir / config.overlay_name
        metadata_path = view_dir / config.metadata_name

        if not input_path.exists():
            records.append(
                _record(
                    sample_id,
                    view_id,
                    input_path,
                    repaired_path,
                    overlay_path,
                    metadata_path,
                    "failed",
                    error=f"Missing input image: {input_path}",
                )
            )
            if config.fail_fast:
                break
            continue

        if config.skip_existing and _repair_outputs_complete(repaired_path, overlay_path, metadata_path):
            records.append(_record(sample_id, view_id, input_path, repaired_path, overlay_path, metadata_path, "skipped"))
            continue

        if config.dry_run:
            records.append(_record(sample_id, view_id, input_path, repaired_path, overlay_path, metadata_path, "dry_run"))
            continue

        try:
            metrics = _repair_one_image(
                input_path=input_path,
                repaired_path=repaired_path,
                overlay_path=overlay_path,
                config=config,
            )
            _write_metadata(
                config,
                sample_id,
                view_id,
                input_path,
                clean_path,
                probability_path,
                repaired_path,
                overlay_path,
                metadata_path,
                metrics,
            )
            records.append(_record(sample_id, view_id, input_path, repaired_path, overlay_path, metadata_path, "repaired", metrics=metrics))
        except Exception as exc:  # pragma: no cover - corrupt local artifacts are reported in summary
            records.append(_record(sample_id, view_id, input_path, repaired_path, overlay_path, metadata_path, "failed", error=str(exc)))
            if config.fail_fast:
                break

    out_root = config.dataflow_root / "06.SingleViews"
    csv_path = Path(output_csv) if output_csv else out_root / "geometry_core_repair_summary.csv"
    json_path = Path(output_json) if output_json else out_root / "geometry_core_repair_summary.json"
    _write_summary_csv(csv_path, records)
    _write_summary_json(json_path, records)
    return GeometryCoreRepairSummary(records=records, csv_path=csv_path, json_path=json_path)


def _validate_config(config: GeometryCoreRepairConfig) -> None:
    if config.input_name == config.output_name:
        raise ValueError("geometry-core repair output_name must not overwrite input_name")
    if config.black_threshold < 0 or config.black_threshold > 255:
        raise ValueError("black_threshold must be in [0, 255]")
    if config.max_gap_px < 0:
        raise ValueError("max_gap_px must be >= 0")
    if config.min_segment_px < 1:
        raise ValueError("min_segment_px must be >= 1")
    if config.bridge_support_radius < 0:
        raise ValueError("bridge_support_radius must be >= 0")
    if config.min_bridge_support < 1:
        raise ValueError("min_bridge_support must be >= 1")
    if config.tiny_area_px < 1:
        raise ValueError("tiny_area_px must be >= 1")
    unknown_directions = set(config.directions) - {"horizontal", "vertical"}
    if unknown_directions:
        names = ", ".join(sorted(unknown_directions))
        raise ValueError(f"unsupported repair direction(s): {names}")


def _repair_one_image(
    *,
    input_path: Path,
    repaired_path: Path,
    overlay_path: Path,
    config: GeometryCoreRepairConfig,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    image = Image.open(input_path).convert("L")
    original = np.asarray(image) < config.black_threshold
    repaired = original.copy()
    before_components, before_black = _component_stats(repaired)

    horizontal_bridge_count = 0
    vertical_bridge_count = 0
    if "horizontal" in config.directions:
        horizontal_bridge_count = _bridge_horizontal_gaps(
            repaired,
            config.max_gap_px,
            config.min_segment_px,
            config.bridge_support_radius,
            config.min_bridge_support,
        )
    if "vertical" in config.directions:
        vertical_bridge_count = _bridge_vertical_gaps(
            repaired,
            config.max_gap_px,
            config.min_segment_px,
            config.bridge_support_radius,
            config.min_bridge_support,
        )

    removed_component_count, removed_pixel_count = _remove_tiny_components(repaired, config.tiny_area_px)
    after_components, after_black = _component_stats(repaired)
    added = repaired & ~original
    removed = original & ~repaired

    repaired_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(repaired, 0, 255).astype("uint8"), mode="L").save(repaired_path)
    _write_overlay(original, added, removed, overlay_path)

    return {
        "width": int(repaired.shape[1]),
        "height": int(repaired.shape[0]),
        "component_count_before": before_components,
        "component_count_after": after_components,
        "black_pixel_count_before": before_black,
        "black_pixel_count_after": after_black,
        "added_pixel_count": int(added.sum()),
        "removed_pixel_count": int(removed.sum()),
        "horizontal_bridge_count": horizontal_bridge_count,
        "vertical_bridge_count": vertical_bridge_count,
        "removed_tiny_component_count": removed_component_count,
        "removed_tiny_pixel_count": removed_pixel_count,
    }


def _bridge_horizontal_gaps(mask: Any, max_gap: int, min_segment: int, support_radius: int, min_support: int) -> int:
    if max_gap <= 0:
        return 0
    candidates_by_y = _collect_horizontal_gap_candidates(mask, max_gap, min_segment)
    bridge_count = 0
    for y, candidates in candidates_by_y.items():
        for gap_start, gap_end in candidates:
            if _has_supported_gap(candidates_by_y, y, gap_start, gap_end, support_radius, min_support):
                mask[y, gap_start:gap_end] = True
                bridge_count += 1
    return bridge_count


def _bridge_vertical_gaps(mask: Any, max_gap: int, min_segment: int, support_radius: int, min_support: int) -> int:
    if max_gap <= 0:
        return 0
    candidates_by_x = _collect_vertical_gap_candidates(mask, max_gap, min_segment)
    bridge_count = 0
    for x, candidates in candidates_by_x.items():
        for gap_start, gap_end in candidates:
            if _has_supported_gap(candidates_by_x, x, gap_start, gap_end, support_radius, min_support):
                mask[gap_start:gap_end, x] = True
                bridge_count += 1
    return bridge_count


def _collect_horizontal_gap_candidates(mask: Any, max_gap: int, min_segment: int) -> dict[int, list[tuple[int, int]]]:
    candidates: dict[int, list[tuple[int, int]]] = {}
    for y in range(mask.shape[0]):
        row_candidates = _gap_candidates(_true_runs(mask[y, :]), max_gap, min_segment)
        if row_candidates:
            candidates[y] = row_candidates
    return candidates


def _collect_vertical_gap_candidates(mask: Any, max_gap: int, min_segment: int) -> dict[int, list[tuple[int, int]]]:
    candidates: dict[int, list[tuple[int, int]]] = {}
    for x in range(mask.shape[1]):
        column_candidates = _gap_candidates(_true_runs(mask[:, x]), max_gap, min_segment)
        if column_candidates:
            candidates[x] = column_candidates
    return candidates


def _gap_candidates(runs: list[tuple[int, int]], max_gap: int, min_segment: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for left, right in zip(runs, runs[1:]):
        left_start, left_end = left
        right_start, right_end = right
        gap = right_start - left_end
        if 0 < gap <= max_gap and left_end - left_start >= min_segment and right_end - right_start >= min_segment:
            candidates.append((left_end, right_start))
    return candidates


def _has_supported_gap(
    candidates_by_scanline: dict[int, list[tuple[int, int]]],
    scanline: int,
    gap_start: int,
    gap_end: int,
    support_radius: int,
    min_support: int,
) -> bool:
    support_count = 0
    for nearby_scanline in range(scanline - support_radius, scanline + support_radius + 1):
        for nearby_start, nearby_end in candidates_by_scanline.get(nearby_scanline, []):
            if _matching_gap(gap_start, gap_end, nearby_start, nearby_end):
                support_count += 1
                if support_count >= min_support:
                    return True
    return False


def _matching_gap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    if min(left_end, right_end) > max(left_start, right_start):
        return True
    left_center = (left_start + left_end) / 2
    right_center = (right_start + right_end) / 2
    return abs(left_center - right_center) <= 1.0


def _true_runs(values: Any) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _remove_tiny_components(mask: Any, min_area: int) -> tuple[int, int]:
    if min_area <= 1:
        return 0, 0
    visited, components = _connected_components(mask)
    removed_count = 0
    removed_pixels = 0
    for component in components:
        if len(component) >= min_area:
            continue
        removed_count += 1
        removed_pixels += len(component)
        for y, x in component:
            mask[y, x] = False
    return removed_count, removed_pixels


def _repair_outputs_complete(repaired_path: Path, overlay_path: Path, metadata_path: Path) -> bool:
    return repaired_path.exists() and overlay_path.exists() and metadata_path.exists()


def _component_stats(mask: Any) -> tuple[int, int]:
    _, components = _connected_components(mask)
    return len(components), int(mask.sum())


def _connected_components(mask: Any) -> tuple[Any, list[list[tuple[int, int]]]]:
    import numpy as np

    visited = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x]:
            continue
        component: list[tuple[int, int]] = []
        stack = [(y, x)]
        visited[y, x] = True
        while stack:
            current_y, current_x = stack.pop()
            component.append((current_y, current_x))
            for next_y in range(max(0, current_y - 1), min(mask.shape[0], current_y + 2)):
                for next_x in range(max(0, current_x - 1), min(mask.shape[1], current_x + 2)):
                    if mask[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        components.append(component)
    return visited, components


def _write_overlay(original: Any, added: Any, removed: Any, overlay_path: Path) -> None:
    import numpy as np
    from PIL import Image

    overlay = np.full((*original.shape, 3), 255, dtype="uint8")
    overlay[original] = [0, 0, 0]
    overlay[added] = [220, 0, 0]
    overlay[removed] = [170, 170, 170]
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)


def _write_metadata(
    config: GeometryCoreRepairConfig,
    sample_id: str,
    view_id: str,
    input_path: Path,
    clean_path: Path,
    probability_path: Path,
    repaired_path: Path,
    overlay_path: Path,
    metadata_path: Path,
    metrics: dict[str, Any],
) -> None:
    write_json(
        metadata_path,
        {
            "schema": "geometry_core_repair",
            "version": "0.1.0",
            "sample_id": sample_id,
            "view_id": view_id,
            "method": {
                "name": "rule_based_topology_repair",
                "version": "0.1.0",
                "role": "lightweight_postprocess_for_unet_geometry_core",
                "parameters": {
                    "black_threshold": config.black_threshold,
                    "max_gap_px": config.max_gap_px,
                    "min_segment_px": config.min_segment_px,
                    "bridge_support_radius": config.bridge_support_radius,
                    "min_bridge_support": config.min_bridge_support,
                    "tiny_area_px": config.tiny_area_px,
                    "directions": list(config.directions),
                },
            },
            "inputs": {
                "geometry_core": input_path.as_posix(),
                "clean_view_with_annotations": clean_path.as_posix() if clean_path.exists() else None,
                "geometry_core_prob": probability_path.as_posix() if probability_path.exists() else None,
            },
            "outputs": {
                "geometry_core_repaired": repaired_path.as_posix(),
                "repair_overlay": overlay_path.as_posix(),
            },
            "metrics": metrics,
        },
    )


def _iter_view_dirs(config: GeometryCoreRepairConfig) -> list[tuple[str, str, Path]]:
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
        if sample_id == "testView2CAD":
            continue
        if _looks_like_copy_sample(sample_id) and not config.include_copy:
            continue
        for view_dir in sorted(sample_dir.glob("view_*")):
            if view_dir.is_dir():
                view_dirs.append((sample_id, view_dir.name, view_dir))
    return view_dirs


def _looks_like_copy_sample(sample_id: str) -> bool:
    lowered = sample_id.lower()
    return lowered.endswith("-copy") or lowered.endswith("_copy") or " copy" in lowered


def _record(
    sample_id: str,
    view_id: str,
    input_path: Path,
    repaired_path: Path,
    overlay_path: Path,
    metadata_path: Path,
    status: str,
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
) -> GeometryCoreRepairRecord:
    return GeometryCoreRepairRecord(
        sample_id=sample_id,
        view_id=view_id,
        input_path=input_path,
        repaired_path=repaired_path,
        overlay_path=overlay_path,
        metadata_path=metadata_path,
        status=status,
        metrics=metrics,
        error=error,
    )


def _write_summary_csv(path: Path, records: list[GeometryCoreRepairRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "view_id",
                "status",
                "input_path",
                "repaired_path",
                "added_pixel_count",
                "removed_pixel_count",
                "component_count_before",
                "component_count_after",
                "error",
            ],
        )
        writer.writeheader()
        for record in records:
            metrics = record.metrics or {}
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "view_id": record.view_id,
                    "status": record.status,
                    "input_path": record.input_path.as_posix(),
                    "repaired_path": record.repaired_path.as_posix(),
                    "added_pixel_count": metrics.get("added_pixel_count", ""),
                    "removed_pixel_count": metrics.get("removed_pixel_count", ""),
                    "component_count_before": metrics.get("component_count_before", ""),
                    "component_count_after": metrics.get("component_count_after", ""),
                    "error": record.error or "",
                }
            )


def _write_summary_json(path: Path, records: list[GeometryCoreRepairRecord]) -> None:
    write_json(
        path,
        {
            "schema": "geometry_core_repair_summary",
            "version": "0.1.0",
            "summary": {
                "total_count": len(records),
                "repaired_count": sum(record.status == "repaired" for record in records),
                "skipped_count": sum(record.status == "skipped" for record in records),
                "dry_run_count": sum(record.status == "dry_run" for record in records),
                "failed_count": sum(record.status == "failed" for record in records),
            },
            "records": [
                {
                    "sample_id": record.sample_id,
                    "view_id": record.view_id,
                    "status": record.status,
                    "input_path": record.input_path.as_posix(),
                    "repaired_path": record.repaired_path.as_posix(),
                    "overlay_path": record.overlay_path.as_posix(),
                    "metadata_path": record.metadata_path.as_posix(),
                    "metrics": record.metrics,
                    "error": record.error,
                }
                for record in records
            ]
        },
    )
