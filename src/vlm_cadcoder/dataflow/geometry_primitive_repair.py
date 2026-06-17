from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_cadcoder.utils.json_utils import write_json


@dataclass(frozen=True)
class GeometryPrimitiveRepairConfig:
    dataflow_root: Path = Path("DataFlow")
    sample_id: str | None = None
    input_name: str = "geometry_core.png"
    output_name: str = "geometry_core_primitive_repaired.png"
    overlay_name: str = "primitive_repair_overlay.png"
    candidates_name: str = "primitive_candidates.json"
    include_copy: bool = False
    skip_existing: bool = False
    dry_run: bool = False
    fail_fast: bool = False
    black_threshold: int = 250
    min_component_area_px: int = 24
    max_line_gap_px: int = 12
    min_line_segment_px: int = 16
    min_existing_arc_coverage: float = 0.55
    min_circle_gap_ratio: float = 0.05
    max_circle_gap_ratio: float = 0.35
    circle_radius_tolerance: float = 0.18
    min_circle_radius_px: int = 8
    rectangle_fill_ratio_max: float = 0.45
    rectangle_edge_coverage_min: float = 0.45
    primitive_types: tuple[str, ...] = ("circle_arc",)


@dataclass(frozen=True)
class GeometryPrimitiveRepairRecord:
    sample_id: str
    view_id: str
    input_path: Path
    repaired_path: Path
    overlay_path: Path
    candidates_path: Path
    status: str
    accepted_count: int = 0
    rejected_count: int = 0
    error: str | None = None


@dataclass(frozen=True)
class GeometryPrimitiveRepairSummary:
    records: list[GeometryPrimitiveRepairRecord]
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


def repair_geometry_primitives(
    config: GeometryPrimitiveRepairConfig,
    *,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
) -> GeometryPrimitiveRepairSummary:
    _validate_config(config)
    records: list[GeometryPrimitiveRepairRecord] = []
    view_dirs = _iter_view_dirs(config)
    if config.sample_id and not view_dirs:
        sample_dir = config.dataflow_root / "06.SingleViews" / config.sample_id
        if not sample_dir.exists():
            error = f"Missing sample directory: {sample_dir}"
        else:
            error = f"No view directories found in sample: {sample_dir}"
        records.append(
            _record(
                config.sample_id,
                "*",
                sample_dir / config.input_name,
                sample_dir / config.output_name,
                sample_dir / config.overlay_name,
                sample_dir / config.candidates_name,
                "failed",
                error=error,
            )
        )

    for sample_id, view_id, view_dir in view_dirs:
        input_path = view_dir / config.input_name
        repaired_path = view_dir / config.output_name
        overlay_path = view_dir / config.overlay_name
        candidates_path = view_dir / config.candidates_name

        if not input_path.exists():
            records.append(
                _record(
                    sample_id,
                    view_id,
                    input_path,
                    repaired_path,
                    overlay_path,
                    candidates_path,
                    "failed",
                    error=f"Missing input image: {input_path}",
                )
            )
            if config.fail_fast:
                break
            continue

        if config.skip_existing and repaired_path.exists() and overlay_path.exists() and candidates_path.exists():
            records.append(_record(sample_id, view_id, input_path, repaired_path, overlay_path, candidates_path, "skipped"))
            continue

        if config.dry_run:
            records.append(_record(sample_id, view_id, input_path, repaired_path, overlay_path, candidates_path, "dry_run"))
            continue

        try:
            result = _repair_one_view(input_path, repaired_path, overlay_path, candidates_path, config, sample_id, view_id)
            records.append(
                _record(
                    sample_id,
                    view_id,
                    input_path,
                    repaired_path,
                    overlay_path,
                    candidates_path,
                    "repaired",
                    accepted_count=len(result["accepted_candidates"]),
                    rejected_count=len(result["rejected_candidates"]),
                )
            )
        except Exception as exc:  # pragma: no cover - corrupt local images are reported per sample
            records.append(
                _record(
                    sample_id,
                    view_id,
                    input_path,
                    repaired_path,
                    overlay_path,
                    candidates_path,
                    "failed",
                    error=str(exc),
                )
            )
            if config.fail_fast:
                break

    out_root = config.dataflow_root / "06.SingleViews"
    csv_path = Path(output_csv) if output_csv else out_root / "geometry_primitive_repair_summary.csv"
    json_path = Path(output_json) if output_json else out_root / "geometry_primitive_repair_summary.json"
    _write_summary_csv(csv_path, records)
    _write_summary_json(json_path, records)
    return GeometryPrimitiveRepairSummary(records=records, csv_path=csv_path, json_path=json_path)


def _repair_one_view(
    input_path: Path,
    repaired_path: Path,
    overlay_path: Path,
    candidates_path: Path,
    config: GeometryPrimitiveRepairConfig,
    sample_id: str,
    view_id: str,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    image = Image.open(input_path).convert("L")
    original = np.asarray(image) < config.black_threshold
    repaired = original.copy()
    components = _connected_component_pixels(original)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if "line" in config.primitive_types:
        accepted.extend(_repair_line_gaps(repaired, config))

    for index, component in enumerate(components, start=1):
        if len(component) < config.min_component_area_px:
            continue
        bbox = _component_bbox(component)
        rectangle_candidate = _rectangle_candidate(component, bbox, index, config)
        if rectangle_candidate:
            rejected.append(rectangle_candidate)
            continue
        if "circle_arc" not in config.primitive_types:
            continue
        circle_candidate = _circle_arc_candidate(component, bbox, index, config)
        if circle_candidate:
            _draw_circle_gap(repaired, circle_candidate)
            accepted.append(circle_candidate)

    added = repaired & ~original
    repaired_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(repaired, 0, 255).astype("uint8"), mode="L").save(repaired_path)
    _write_overlay(original, added, overlay_path)
    payload = {
        "schema": "geometry_primitive_repair",
        "version": "0.1.0",
        "sample_id": sample_id,
        "view_id": view_id,
        "method": {
            "name": "conservative_primitive_gap_repair",
            "version": "0.1.0",
            "role": "candidate_level_line_arc_repair_without_preserving_annotation_frames",
            "parameters": {
                "primitive_types": list(config.primitive_types),
                "max_line_gap_px": config.max_line_gap_px,
                "min_line_segment_px": config.min_line_segment_px,
                "min_existing_arc_coverage": config.min_existing_arc_coverage,
                "min_circle_gap_ratio": config.min_circle_gap_ratio,
                "max_circle_gap_ratio": config.max_circle_gap_ratio,
                "circle_radius_tolerance": config.circle_radius_tolerance,
            },
        },
        "inputs": {"geometry_core": input_path.as_posix()},
        "outputs": {
            "geometry_core_primitive_repaired": repaired_path.as_posix(),
            "primitive_repair_overlay": overlay_path.as_posix(),
        },
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
        "metrics": {
            "component_count": len(components),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "added_pixel_count": int(added.sum()),
        },
    }
    write_json(candidates_path, payload)
    return payload


def _circle_arc_candidate(
    component: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    index: int,
    config: GeometryPrimitiveRepairConfig,
) -> dict[str, Any] | None:
    xs = [x for _, x in component]
    ys = [y for y, _ in component]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    distances = [math.hypot(x - cx, y - cy) for y, x in component]
    radius = sum(distances) / max(1, len(distances))
    if radius < config.min_circle_radius_px:
        return None
    mean_radial_error = sum(abs(distance - radius) for distance in distances) / max(1, len(distances))
    radial_error = mean_radial_error / max(1.0, radius)
    if radial_error > config.circle_radius_tolerance:
        return None
    angle_bins = _occupied_angle_bins(component, cx, cy)
    coverage = len(angle_bins) / 72
    max_gap = _max_missing_angle_gap(angle_bins, 72) / 72
    if coverage < config.min_existing_arc_coverage or max_gap < config.min_circle_gap_ratio or max_gap > config.max_circle_gap_ratio:
        return None
    x1, y1, x2, y2 = bbox
    return {
        "id": f"primitive_{index:03d}_circle_arc",
        "primitive_type": "circle_arc",
        "action": "repair_gap",
        "bbox": [x1, y1, x2, y2],
        "center": [round(cx, 2), round(cy, 2)],
        "radius": round(radius, 2),
        "confidence": round(min(0.92, 0.35 + coverage - radial_error), 4),
        "metrics": {
            "existing_arc_coverage": round(coverage, 4),
            "min_gap_ratio": round(config.min_circle_gap_ratio, 4),
            "max_gap_ratio": round(max_gap, 4),
            "radial_error": round(radial_error, 4),
        },
        "evidence": ["component_points_fit_circle", "candidate_repairs_existing_arc_gap"],
    }


def _rectangle_candidate(
    component: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    index: int,
    config: GeometryPrimitiveRepairConfig,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    if width < 8 or height < 8:
        return None
    area = width * height
    fill_ratio = len(component) / area
    edge_hits = 0
    for y, x in component:
        on_left_or_right = abs(x - x1) <= 1 or abs(x - (x2 - 1)) <= 1
        on_top_or_bottom = abs(y - y1) <= 1 or abs(y - (y2 - 1)) <= 1
        if on_left_or_right or on_top_or_bottom:
            edge_hits += 1
    perimeter = max(1, 2 * (width + height))
    edge_coverage = min(1.0, edge_hits / perimeter)
    if fill_ratio <= config.rectangle_fill_ratio_max and edge_coverage >= config.rectangle_edge_coverage_min:
        return {
            "id": f"primitive_{index:03d}_rectangle",
            "primitive_type": "rectangle",
            "action": "reject",
            "bbox": [x1, y1, x2, y2],
            "confidence": round(min(0.95, edge_coverage), 4),
            "metrics": {"fill_ratio": round(fill_ratio, 4), "edge_coverage": round(edge_coverage, 4)},
            "reject_reasons": ["isolated_closed_rectangle", "would_preserve_annotation_or_detail_frame"],
        }
    return None


def _draw_circle_gap(mask: Any, candidate: dict[str, Any]) -> None:
    import numpy as np

    cx, cy = candidate["center"]
    radius = float(candidate["radius"])
    height, width = mask.shape
    for angle in range(360):
        x = int(round(cx + radius * math.cos(math.radians(angle))))
        y = int(round(cy + radius * math.sin(math.radians(angle))))
        for yy in range(max(0, y - 1), min(height, y + 2)):
            for xx in range(max(0, x - 1), min(width, x + 2)):
                if math.hypot(xx - cx, yy - cy) <= radius + 1.5:
                    mask[yy, xx] = True


def _repair_line_gaps(mask: Any, config: GeometryPrimitiveRepairConfig) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    if config.max_line_gap_px <= 0:
        return accepted
    for y in range(mask.shape[0]):
        for gap_start, gap_end in _gap_candidates(
            _true_runs(mask[y, :]),
            config.max_line_gap_px,
            config.min_line_segment_px,
        ):
            mask[y, gap_start:gap_end] = True
            accepted.append(
                {
                    "id": f"primitive_line_h_{y}_{gap_start}_{gap_end}",
                    "primitive_type": "line",
                    "action": "repair_gap",
                    "bbox": [gap_start, y, gap_end, y + 1],
                    "confidence": 0.5,
                    "metrics": {"gap_px": gap_end - gap_start, "orientation": "horizontal"},
                    "evidence": ["collinear_horizontal_segments"],
                }
            )
    for x in range(mask.shape[1]):
        for gap_start, gap_end in _gap_candidates(
            _true_runs(mask[:, x]),
            config.max_line_gap_px,
            config.min_line_segment_px,
        ):
            mask[gap_start:gap_end, x] = True
            accepted.append(
                {
                    "id": f"primitive_line_v_{x}_{gap_start}_{gap_end}",
                    "primitive_type": "line",
                    "action": "repair_gap",
                    "bbox": [x, gap_start, x + 1, gap_end],
                    "confidence": 0.5,
                    "metrics": {"gap_px": gap_end - gap_start, "orientation": "vertical"},
                    "evidence": ["collinear_vertical_segments"],
                }
            )
    return accepted


def _gap_candidates(runs: list[tuple[int, int]], max_gap: int, min_segment: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for left, right in zip(runs, runs[1:]):
        left_start, left_end = left
        right_start, right_end = right
        gap = right_start - left_end
        if (
            0 < gap <= max_gap
            and left_end - left_start >= min_segment
            and right_end - right_start >= min_segment
        ):
            candidates.append((left_end, right_start))
    return candidates


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


def _occupied_angle_bins(component: list[tuple[int, int]], cx: float, cy: float, bin_count: int = 72) -> set[int]:
    bins: set[int] = set()
    for y, x in component:
        angle = math.degrees(math.atan2(y - cy, x - cx)) % 360
        bins.add(int(angle / 360 * bin_count))
    return bins


def _max_missing_angle_gap(occupied: set[int], bin_count: int) -> int:
    if not occupied:
        return bin_count
    missing = [index not in occupied for index in range(bin_count)]
    doubled = missing + missing
    best = 0
    current = 0
    for value in doubled:
        if value:
            current += 1
            best = max(best, min(current, bin_count))
        else:
            current = 0
    return best


def _connected_component_pixels(mask: Any) -> list[list[tuple[int, int]]]:
    import numpy as np

    visited = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x]:
            continue
        stack = [(y, x)]
        component: list[tuple[int, int]] = []
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
    return components


def _component_bbox(component: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    ys = [y for y, _ in component]
    xs = [x for _, x in component]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _write_overlay(original: Any, added: Any, overlay_path: Path) -> None:
    import numpy as np
    from PIL import Image

    overlay = np.full((*original.shape, 3), 255, dtype="uint8")
    overlay[original] = [0, 0, 0]
    overlay[added] = [220, 0, 0]
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)


def _validate_config(config: GeometryPrimitiveRepairConfig) -> None:
    if config.input_name == config.output_name:
        raise ValueError("primitive repair output_name must not overwrite input_name")
    if not 0 <= config.black_threshold <= 255:
        raise ValueError("black_threshold must be in [0, 255]")
    if config.max_line_gap_px < 0:
        raise ValueError("max_line_gap_px must be >= 0")
    if config.min_line_segment_px < 1:
        raise ValueError("min_line_segment_px must be >= 1")
    if not 0 < config.min_existing_arc_coverage <= 1:
        raise ValueError("min_existing_arc_coverage must be in (0, 1]")
    if not 0 <= config.min_circle_gap_ratio <= 1:
        raise ValueError("min_circle_gap_ratio must be in [0, 1]")
    if not 0 < config.max_circle_gap_ratio <= 1:
        raise ValueError("max_circle_gap_ratio must be in (0, 1]")
    if config.min_circle_gap_ratio > config.max_circle_gap_ratio:
        raise ValueError("min_circle_gap_ratio must be <= max_circle_gap_ratio")
    unknown_primitives = set(config.primitive_types) - {"line", "circle_arc"}
    if unknown_primitives:
        names = ", ".join(sorted(unknown_primitives))
        raise ValueError(f"unsupported primitive type(s): {names}")


def _iter_view_dirs(config: GeometryPrimitiveRepairConfig) -> list[tuple[str, str, Path]]:
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
        if not config.include_copy and _looks_like_copy_sample(sample_id):
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
    candidates_path: Path,
    status: str,
    accepted_count: int = 0,
    rejected_count: int = 0,
    error: str | None = None,
) -> GeometryPrimitiveRepairRecord:
    return GeometryPrimitiveRepairRecord(
        sample_id=sample_id,
        view_id=view_id,
        input_path=input_path,
        repaired_path=repaired_path,
        overlay_path=overlay_path,
        candidates_path=candidates_path,
        status=status,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        error=error,
    )


def _write_summary_csv(path: Path, records: list[GeometryPrimitiveRepairRecord]) -> None:
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
                "accepted_count",
                "rejected_count",
                "error",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "view_id": record.view_id,
                    "status": record.status,
                    "input_path": record.input_path.as_posix(),
                    "repaired_path": record.repaired_path.as_posix(),
                    "accepted_count": record.accepted_count,
                    "rejected_count": record.rejected_count,
                    "error": record.error or "",
                }
            )


def _write_summary_json(path: Path, records: list[GeometryPrimitiveRepairRecord]) -> None:
    write_json(
        path,
        {
            "schema": "geometry_primitive_repair_summary",
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
                    "candidates_path": record.candidates_path.as_posix(),
                    "accepted_count": record.accepted_count,
                    "rejected_count": record.rejected_count,
                    "error": record.error,
                }
                for record in records
            ],
        },
    )
